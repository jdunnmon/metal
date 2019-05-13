import copy
import logging
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from metal.mmtl.modules import get_base_module
from metal.utils import move_to_device, recursive_merge_dicts, set_seed

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_defaults = {
    "seed": None,
    "device": 0,  # gpu id (int) or -1 for cpu
    "verbose": True,
    "fp16": False,
    "model_weights": None,  # the path to a saved checkpoint to initialize with
    "model_type": None,  # if slice_model, slice_rep_model, None for naive model
    # whether to delete model source model head weights while loading existing weights
    "delete_heads": False,
}


class MetalModel(nn.Module):
    """A dynamically constructed discriminative classifier

    Args:
        tasks: a list of Task objects which bring their own (named) modules

    We currently support up to N input modules -> middle layers -> up to N heads
    TODO: Accept specifications for more exotic structure (e.g., via user-defined graph)
    """

    def __init__(self, tasks, **kwargs):
        self.config = recursive_merge_dicts(model_defaults, kwargs, misses="insert")

        # Set random seed before initializing module weights
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        set_seed(self.config["seed"])

        super().__init__()

        # Build network
        self._build(tasks)
        self.task_map = {task.name: task for task in tasks}

        # Load weights
        if self.config["model_weights"]:
            logger.info("Loading model weights...")
            self.load_weights(self.config["model_weights"])

        # Half precision
        if self.config["fp16"]:
            logger.info("metal_model.py: Using fp16")
            self.half()

        # Move model to device now, then move data to device in forward() or calculate_loss()
        if self.config["device"] >= 0:
            if torch.cuda.is_available():
                if self.config["verbose"]:
                    logger.info("Using GPU...")
                self.to(torch.device(f"cuda:{self.config['device']}"))
            else:
                if self.config["verbose"]:
                    logger.info("No cuda device available. Using cpu instead.")

        # Show network
        if self.config["verbose"]:
            logger.info("\nNetwork architecture:")
            logger.info(self)
            print()
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Total number of parameters: {num_params}")

    def _build(self, tasks):
        """Iterates over tasks, adding their input_modules and head_modules"""
        # TODO: Allow more flexible specification of network structure
        self.input_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.input_module) for task in tasks}
        )
        self.middle_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.middle_module) for task in tasks}
        )
        self.attention_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.attention_module) for task in tasks}
        )
        self.head_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.head_module) for task in tasks}
        )

        self.loss_hat_funcs = {task.name: task.loss_hat_func for task in tasks}
        self.output_hat_funcs = {task.name: task.output_hat_func for task in tasks}

    def forward(self, X, task_names):
        """Returns the outputs of the requested task heads in a dictionary

        The output of each task is the result of passing the input through the
        input_module, middle_module, and head_module for that task, in that order.
        Before calculating any intermediate values, we first check whether a previously
        evaluated task has produced that intermediate result. If so, we use that.

        Args:
            X: a [batch_size, ...] batch from a DataLoader
        Returns:
            output_dict: {task_name (str): output (Tensor)}
        """
        input = move_to_device(X, self.config["device"])
        outputs = {}
        for task_name in task_names:
            # Use get_base_module to remove MetalModel and DataParallel wrppaers in cache keys
            # Handling input module with caching
            input_module = self.input_modules[task_name]
            input_base_module = get_base_module(input_module)
            if input_base_module not in outputs:
                outputs[get_base_module(input_module)] = input_module(input)

            # Handling middle module with caching
            middle_module = self.middle_modules[task_name]
            middle_base_module = get_base_module(middle_module)
            if middle_base_module not in outputs:
                outputs[middle_base_module] = middle_module(outputs[input_base_module])

            # Handling attention module with caching
            attention_module = self.attention_modules[task_name]
            attention_base_module = get_base_module(attention_module)
            if attention_base_module not in outputs:
                outputs[attention_base_module] = attention_module(
                    outputs[middle_base_module]
                )

            # Handling head modules with caching (this may not be necessary)
            head_module = self.head_modules[task_name]
            head_base_module = get_base_module(head_module)
            if head_base_module not in outputs:
                outputs[head_base_module] = head_module(outputs[attention_base_module])

        return {t: outputs[get_base_module(self.head_modules[t])] for t in task_names}

    def calculate_loss(self, X, Ys, payload_name, labels_to_tasks):
        """Returns a dict of {task_name: loss (a FloatTensor scalar)}.

        Args:
            X: an appropriate input for forward(), either a Tensor or tuple
            Ys: a dict of {task_name: labels} where labels is [n, ?]
            labels_to_tasks: a dict of {label_name: task_name} indicating which task
                head to use to calculate the loss for each label_set.
        """
        task_names = set(labels_to_tasks.values())
        outputs = self.forward(X, task_names)
        loss_dict = {}  # Stores the loss by task
        count_dict = {}  # Stores the number of active examples by task

        for label_name, task_name in labels_to_tasks.items():
            if task_name is None:
                continue
            loss_name = f"{task_name}/{payload_name}/{label_name}/loss"
            Y = Ys[label_name]
            out = outputs[task_name]
            # Identify which instances have at least one non-zero target labels
            active = torch.any(Y.detach() != 0, dim=1)
            count_dict[loss_name] = active.sum().item()

            # If there are inactive instances, slice them out to save computation
            # and ignore their contribution to the loss
            if 0 in active:
                Y = Y[active]
                if isinstance(out, torch.Tensor):
                    out = out[active]
                # If the output of the head has multiple fields, slice them all
                elif isinstance(out, dict):
                    out = move_to_device({k: v[active] for k, v in out.items()})

            # Convert to half precision last thing if applicable
            if self.config["fp16"] and Y.dtype == torch.float32:
                out["data"] = out["data"].half()
                Y = Y.half()

            # If no examples in this batch have labels for this task, skip loss calc
            # Active has type torch.uint8; avoid overflow with long()
            if active.long().sum():
                label_loss = self.loss_hat_funcs[task_name](
                    out, move_to_device(Y, self.config["device"])
                )
                assert isinstance(label_loss.item(), float)
                loss_dict[loss_name] = (
                    label_loss * self.task_map[task_name].loss_multiplier
                )

        return loss_dict, count_dict

    @torch.no_grad()
    def calculate_probs(self, X, task_names):
        """Returns a dict of {task_name: probs}

        Args:
            X: instances to feed through the network
            task_names: the names of the tasks for which to calculate outputs
        Returns:
            {task_name: probs}: probs is the output of the output_hat for the given
                task_head

        The type of each entry in probs depends on the task type:
            instance-based tasks: each entry in probs is a [k]-len array
            token-based tasks: each entry is a  [seq_len, k] array
        """
        assert self.eval()
        return {
            t: [probs.cpu().numpy() for probs in self.output_hat_funcs[t](out)]
            for t, out in self.forward(X, task_names).items()
        }

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary."""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def load_weights(self, model_path, delete_heads=False):
        """Load model weights from checkpoint."""
        # NOTE: returning missing, expected requires torch >=1.1
        if self.config["device"] >= 0:
            device = torch.device(f"cuda:{self.config['device']}")
        else:
            device = torch.device("cpu")
        try:
            self.load_state_dict(torch.load(model_path, map_location=device)["model"])
            missing, expected = [], []
        except RuntimeError:
            logger.info("Your destination state dict has different keys for the update key.")
            try:
                logger.info("Loading without strictness.")
                source_state_dict = torch.load(model_path, map_location=device)["model"]
                missing, unexpected = self.load_state_dict(source_state_dict, strict=False)

            except RuntimeError:
                logger.info("Loading with slice heads deleted")
                # use the slicing hack to delete existing heads
                if self.config["delete_heads"]:
                    warnings.warn(
                        "SLICING HACK: Attemping to remove heads in source state dict."
                        "You MUST fine-tune the model to recover original performance."
                    )

                    for module in list(source_state_dict.keys()):
                        if "head_modules" in module:
                            msg = f"Deleting {module} from loaded weights"
                            warnings.warn(msg)
                            del source_state_dict[module]

                    missing, unexpected = self.load_state_dict(source_state_dict, strict=False)
                                  
        return missing, unexpected

    def save_weights(self, model_path):
        """Saves weight in checkpoint directory"""
        raise NotImplementedError

    @torch.no_grad()
    def score(self, payload, metrics=[], verbose=True, **kwargs):
        """Calculate the requested metrics for the given payload

        Args:
            payload: a Payload to score
            metrics: a list of full metric names, a single full metric name, or []:
                list: a list of full metric names supported by the tasks' Scorers.
                    (full metric names are of the form task/payload/labelset/metric)
                    Only these metrics will be calculated and returned.
                []: defaults to all supported metrics for the given payload's Tasks
                str: a single full metric name
                    A single score will be returned instead of a dictionary

        Returns:
            scores: a dict of the form {metric_name: score} corresponding to the
                requested metrics (optionally a single score if metrics is a string
                instead of a list)
        """
        self.eval()
        return_unwrapped = isinstance(metrics, str)

        # If no specific metrics were requested, calculate all available metrics
        if metrics:
            metrics_list = metrics if isinstance(metrics, list) else [metrics]
            assert all(len(metric.split("/")) == 4 for metric in metrics_list)
            target_metrics = defaultdict(list)
            target_tasks = []
            target_labels = []
            for full_metric_name in metrics:
                task_name, payload_name, label_name, metric_name = full_metric_name.split(
                    "/"
                )
                target_tasks.append(task_name)
                target_labels.append(label_name)
                target_metrics[label_name].append(metric_name)
        else:
            target_tasks = set(payload.labels_to_tasks.values())
            target_labels = set(payload.labels_to_tasks.keys())
            target_metrics = {
                label_name: None for label_name in payload.labels_to_tasks
            }

        Ys, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, target_tasks, target_labels, return_preds=True, **kwargs
        )

        metrics_dict = {}
        for label_name, task_name in payload.labels_to_tasks.items():
            if task_name is None:
                continue

            scorer = self.task_map[task_name].scorer
            task_metrics_dict = scorer.score(
                Ys[label_name],
                Ys_probs[task_name],
                Ys_preds[task_name],
                target_metrics=target_metrics[label_name],
            )
            # Expand short metric names into full metric names
            for metric_name, score in task_metrics_dict.items():
                full_metric_name = (
                    f"{task_name}/{payload.name}/{label_name}/{metric_name}"
                )
                metrics_dict[full_metric_name] = score

        # If a single metric was given as a string (not list), return a float
        if return_unwrapped:
            metric, score = metrics_dict.popitem()
            return score
        else:
            return metrics_dict

    @torch.no_grad()
    def predict_with_gold(
        self,
        payload,
        target_tasks=None,
        target_labels=None,
        return_preds=False,
        return_meta=None,
        max_examples=0,
        **kwargs,
    ):
        """Extracts Y and calculates Y_prods, Y_preds for the given payload and tasks

        To get just the probabilities or predictions for a single task, consider using
        predict() or predict_probs().

        Args:
            payload: the Payload to make predictions for
            target_tasks: if not None, predict probs only for the specified tasks;
                otherwise, predict probs for all tasks with corresponding label_sets
                in the payload
            target_labels: if not None, return labels for only the specified label_sets;
                otherwise, return all label_sets
            return_preds: if True, also include preds in return values
            max_examples: if > 0, predict for a maximum of this many examples

        # TODO: consider returning Ys as tensors instead of lists (padded if necessary)
        Returns:
            Ys: a {label_name: Y} dict where Y is an [n] list of labels (often ints)
            Ys_probs: a {task_name: Y_probs} dict where Y_probs is a [n] list of
                probabilities
            Ys_preds: a {task_name: Y_preds} dict where Y_preds is a [n] list of
                predictions
        """
        validate_targets(payload, target_tasks, target_labels)
        if target_tasks is None:
            target_tasks = set(payload.labels_to_tasks.values())

        # filter tasks that are None (if we don't want to evaluate a particular labelset)
        target_tasks = [t for t in target_tasks if t is not None]

        if target_labels is None:
            target_labels = set(payload.labels_to_tasks.keys())

        Ys = defaultdict(list)
        Ys_probs = defaultdict(list)
        Ys_meta = defaultdict(list)

        total = 0
        for batch_num, (Xb, Yb) in enumerate(payload.data_loader):
            Yb_probs = self.calculate_probs(Xb, target_tasks)
            if return_meta:
                for field, meta in Xb.items():
                    if field in return_meta:
                        Ys_meta[field].extend(meta)
            for task_name, yb_probs in Yb_probs.items():
                Ys_probs[task_name].extend(yb_probs)
            for label_name, yb in Yb.items():
                if label_name in target_labels or target_labels is None:
                    Ys[label_name].extend(yb.cpu().numpy())
            total += len(Xb)
            if max_examples > 0 and total >= max_examples:
                break

        if max_examples:
            Ys = {label_name: Y[:max_examples] for label_name, Y in Ys.items()}
            Ys_probs = {
                task_name: Y_probs[:max_examples]
                for task_name, Y_probs in Ys_probs.items()
            }

        return_tuple = (Ys, Ys_probs)

        if return_preds:
            Ys_preds = {
                task_name: [probs_to_preds(y_probs) for y_probs in Y_probs]
                for task_name, Y_probs in Ys_probs.items()
            }
            return_tuple = return_tuple + (Ys_preds,)
           
        if return_meta:
            return_tuple = return_tuple + (Ys_meta,)

        return return_tuple

    # Single-task prediction helpers (for convenience)
    @torch.no_grad()
    def predict_probs(self, payload, task_name=None, **kwargs):
        """Return probabilistic labels for a single task of a payload

        Args:
            payload: a Payload
            task_name: the task to calculate probabilities for; defaults to the name of
                the payload if none
        Returns:
            Y_probs: an [n] list of probabilities
        """
        self.eval()
        _, Ys_probs = self.predict_with_gold(payload, task_name, **kwargs)
        return Ys_probs[task_name]

    @torch.no_grad()
    def predict(self, payload, task_name=None, return_probs=False, **kwargs):
        """Return predicted labels for a single task of a payload

        Args:
            payload: a Payload
            task_name: the task to calculate probabilities for; defaults to the name of
                the payload if none
        Returns:
            Y_probs: an [n] list of probabilities
            Y_preds: an [n] list of predictions
        """
        self.eval()
        _, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, [task_name], return_preds=True, **kwargs
        )
        Y_probs = Ys_probs[task_name]
        Y_preds = Ys_preds[task_name]
        if return_probs:
            return Y_preds, Y_probs
        else:
            return Y_preds


def validate_targets(payload, target_tasks, target_labels):
    if target_tasks:
        for task_name in target_tasks:
            if task_name not in set(payload.labels_to_tasks.values()):
                msg = (
                    f"Could not find the specified task_name {task_name} in "
                    f"payload {payload}."
                )
                raise Exception(msg)

    if target_labels:
        for label_name in target_labels:
            if label_name not in payload.labels_to_tasks:
                msg = (
                    f"Could not find the specified label_set {label_name} in "
                    f"payload {payload}."
                )
                raise Exception(msg)


def probs_to_preds(probs):
    """Identifies the largest probability in each column on the last axis

    We add 1 to the argmax to account for the fact that all labels in MeTaL are
    categorical and the 0 label is reserved for abstaining weak labels.
    """
    # TODO: Consider replacing argmax with a version of the rargmax utility to randomly
    # break ties instead of accepting the first one, or allowing other tie-breaking
    # strategies
    return np.argmax(probs, axis=-1) + 1
