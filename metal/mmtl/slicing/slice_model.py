from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from metal.end_model import IdentityModule
from metal.mmtl.metal_model import MetalModel
from metal.mmtl.modules import unwrap_module
from metal.utils import move_to_device, recursive_merge_dicts, set_seed


def validate_slice_tasks(tasks):
    # validate slice head cardinality
    for t in tasks:
        if t.head_module.module.out_features != 1:
            raise ValueError(
                f"{t.name}'s head_module is invalid. "
                f"SliceModel only supports binary classification "
                "specified by a output_dim=1."
            )

    base_tasks = [t for t in tasks if not t.slice_head_type]
    slice_tasks = [t for t in tasks if t.slice_head_type]

    # validate base task
    if len(base_tasks) != 1:
        raise ValueError(f"SliceModel only supports 1 base task.")
    base_task = base_tasks[0]

    # validate shared body representations
    # TODO: clean up these checks
    for t in slice_tasks:
        same_input = t.input_module is base_task.input_module
        same_middle = (
            (t.middle_module is None and base_task.middle_module is None)
            or t.middle_module is base_task.middle_module
            or (
                isinstance(unwrap_module(t.middle_module.module), IdentityModule)
                and isinstance(unwrap_module(base_task.middle_module), IdentityModule)
            )
        )
        same_attention = (
            (t.attention_module is None and base_task.attention_module is None)
            or t.attention_module is base_task.attention_module
            or (
                isinstance(unwrap_module(t.attention_module.module), IdentityModule)
                and isinstance(
                    unwrap_module(base_task.attention_module), IdentityModule
                )
            )
        )

        has_same_body = same_input and same_middle and same_attention

        if not has_same_body:
            raise ValueError(
                f"Slice tasks must have the same body as base task: "
                f"(same_input={same_input}"
                f", same_middle={same_middle}"
                f", same_attention={same_attention})"
            )

    # validate that one of the "slice_tasks" operates on the entire dataset
    if not any([":BASE" in t.name for t in slice_tasks]):
        raise ValueError(
            "There must be a `slice_task` designated to operate "
            f"on the entire labelset with name '{base_task.name}:BASE'."
        )


class SliceModel(MetalModel):
    """ Slice-aware version of MetalModel.

    At the moment, only supports:
        * Binary classification heads (with output_dim=1, breaking Metal convention)
        * A single base task + an arbitrary number of slice tasks (slice_head_type != None)
    """

    def __init__(self, tasks, **kwargs):
        validate_slice_tasks(tasks)
        super().__init__(tasks, **kwargs)
        self.base_task = [
            t for t in self.task_map.values() if t.slice_head_type is None
        ][0]
        self.slice_pred_tasks = {
            name: t for name, t in self.task_map.items() if t.slice_head_type == "pred"
        }
        self.slice_ind_tasks = {
            name: t for name, t in self.task_map.items() if t.slice_head_type == "ind"
        }

    def forward_body(self, X):
        """ Makes a forward pass through the "body" of the network
        (everything before the head)."""

        input = move_to_device(X, self.config["device"])
        base_task_name = self.base_task.name

        # Extra .module because of DataParallel wrapper!
        input_module = self.input_modules[base_task_name].module
        middle_module = self.middle_modules[base_task_name].module
        attention_module = self.attention_modules[base_task_name].module

        out = attention_module(middle_module(input_module(input)))
        return out

    def forward_heads(self, body, task_names):
        """ Given body (everything before head) representation, return dict of
        task_head outputs for specified task_names """

        return {t: self.head_modules[t].module(body) for t in task_names}

    def forward_attention_logits(self, slice_ind_heads, slice_task_names):
        """ Computes unnomralized slice_head attention weights based on
        `in slice` indicators """

        # slice_preds is the [batch_size, num_slices] concatenated prediction for
        # each slice head
        slice_inds = torch.cat(
            [slice_ind_heads[name]["data"] for name in slice_task_names], dim=1
        )

        # A_weights is the [batch_size, num_slices] unormalized Tensor where
        # more positive values indicate more likely to be in the slice
        A_weights = slice_inds
        return A_weights

    def forward(self, X, task_names):
        """ Perform forward pass with slice-reweighted base representation through
        the base_task head. """

        # Sort names to ensure that the representation is always
        # computed in the same order
        slice_pred_names = sorted(
            [slice_task_name for slice_task_name in self.slice_pred_tasks.keys()]
        )

        slice_ind_names = sorted(
            [slice_task_name for slice_task_name in self.slice_ind_tasks.keys()]
        )

        body = self.forward_body(X)
        slice_pred_heads = self.forward_heads(body, slice_pred_names)
        slice_ind_heads = self.forward_heads(body, slice_ind_names)

        # [batch_size, num_slices] unnormalized attention weights.
        # we then normalize the weights across all heads
        A_weights = self.forward_attention_logits(slice_ind_heads, slice_ind_names)
        A = F.softmax(A_weights, dim=1)

        # slice_weights is the [num_slices, body_dim] concatenated tensor
        # representing linear transforms for the body into the slice prediction values
        slice_weights = torch.cat(
            [
                # TODO: sad! double module (DataParllel + MetalModule)
                self.head_modules[t].module.module.weight
                for t in slice_pred_names
            ],
            dim=0,
        )

        # we reweight the linear mappings `slice_weights` by the
        # attention scores `A` and use this to reweight the base representation
        reweighted_rep = (A @ slice_weights) * body["data"]

        # finally, forward through original task head with reweighted representation
        body["data"] = reweighted_rep

        # compute losses for individual slices + reweighted Y_head
        output = self.forward_heads(body, [self.base_task.name])
        output.update(slice_pred_heads)
        output.update(slice_ind_heads)
        return output

    @torch.no_grad()
    def calculate_attention(self, X):
        """ Retrieve the unnormalized attention weights used in the model over
        slice_heads

        Borrows template from MetalModel.calculate_probs """

        assert self.eval()
        slice_ind_names = sorted(
            [slice_task_name for slice_task_name in self.slice_ind_tasks.keys()]
        )

        body = self.forward_body(X)
        slice_ind_heads = self.forward_heads(body, slice_ind_names)
        A_weights = self.forward_attention_logits(slice_ind_heads, slice_ind_names)
        return A_weights

    def attention_with_gold(self, payload):
        """ Retrieve the attention weights used in the model for viz/debugging

        Borrows template from MetalModel.predict_with_gold """

        Ys = defaultdict(list)
        A_weights = []

        for batch_num, (Xb, Yb) in enumerate(payload.data_loader):
            Ab = self.calculate_attention(Xb)
            A_weights.extend(Ab.cpu().numpy())

            # NOTE: we only have one set of A_weights, but we load every set of
            # Ys to match the expected dictionary format of the Y_dict
            for label_name, yb in Yb.items():
                Ys[label_name].extend(yb.cpu().numpy())

        return Ys, A_weights
