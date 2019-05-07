import copy
import os
import warnings
import logging
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.end_model import IdentityModule
from metal.mmtl.cxr.cxr_datasets import get_cxr_dataset
from metal.mmtl.cxr.cxr_modules import (
    BinaryHead,
    RegressionHead,
    SoftAttentionModule,
    TorchVisionEncoder,
)

from metal.mmtl.cxr.cxr_slices import create_slice_labels
from metal.mmtl.payload import Payload
from metal.mmtl.scorer import Scorer
from metal.mmtl.slicing.tasks import create_slice_task
from metal.mmtl.task import ClassificationTask, RegressionTask
from metal.utils import recursive_merge_dicts, set_seed

# THIS LINE REQUIRED TO FIX ANC_DATA ERROR FROM PYTORCH
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")

# Restoring default
# torch.multiprocessing.set_sharing_strategy('file_descriptor')

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


MASTER_PAYLOAD_TASK_DICT = {
    "CXR8": [
        "ATELECTASIS",
        "CARDIOMEGALY",
        "EFFUSION",
        "INFILTRATION",
        "MASS",
        "NODULE",
        "PNEUMONIA",
        "PNEUMOTHORAX",
        "CONSOLIDATION",
        "EDEMA",
        "EMPHYSEMA",
        "FIBROSIS",
        "PLEURAL_THICKENING",
        "HERNIA",
    ],
    "CXR8-DRAIN": [
        "ATELECTASIS",
        "CARDIOMEGALY",
        "EFFUSION",
        "INFILTRATION",
        "MASS",
        "NODULE",
        "PNEUMONIA",
        "PNEUMOTHORAX",
        "CONSOLIDATION",
        "EDEMA",
        "EMPHYSEMA",
        "FIBROSIS",
        "PLEURAL_THICKENING",
        "HERNIA",
    ],
}

task_defaults = {
    # General
    "pool_payload_tasks": False,  # Pools same task for different payloads if True
    "split_prop": None,
    "splits": ["train", "valid", "test"],
    "subsample": -1,
    "eval_finding":"ALL",
    "seed": None,
    "dl_kwargs": {
        "pin_memory":True,
        "num_workers": 8,
        "batch_size": 16,
        "shuffle": True,  # Used only when split_prop is None; otherwise, use Sampler
    },
    "dataset_kwargs": {"transform_kwargs": {"res": 224}},
    # CNN
    "cnn_model": "densenet121",
    "cnn_kwargs": {"freeze_cnn": False, "pretrained": True, "drop_rate": 0.2},
    "attention_config": {
        "attention_module": None,  # None, soft currently accepted
        "nonlinearity": "tanh",  # tanh, sigmoid currently accepted
    },
    # Auxiliary Tasks and Primary Tasks
    # EYE TRACKING LOSS IS A SLICE, NOT AN AUX!
    "auxiliary_task_dict": {  # A map of each aux. task to the primary task it applies to
        "AUTOENCODE": ["CXR8"]
    },
    "auxiliary_loss_multiplier": 1.0,
    "tasks": None,  # Comma-sep task list e.g. QNLI,QQP
    # Slicing
    "use_slices": True,
    "model_type":None,
    "slice_dict":None, #{  # A map of the slices that apply to each task
        # chest_Drain_canny_seg_neg
        # chest_drain_cnn_neg
        #"CXR8-DRAIN_PNEUMOTHORAX": ["chest_drain_cnn_pos"]
    #},
    "slice_pos_only":["chest_drain_cnn_neg"],
}


def create_tasks_and_payloads(full_task_names, **kwargs):
    assert len(full_task_names) > 0

    config = recursive_merge_dicts(task_defaults, kwargs)

    if config["seed"] is None:
        config["seed"] = np.random.randint(1e6)
        logger.info(f"Using random seed: {config['seed']}")
    set_seed(config["seed"])

    # share cnn encoder for all tasks
    cnn_kwargs = config["cnn_kwargs"]
    cnn_model = TorchVisionEncoder(config["cnn_model"], **cnn_kwargs)
    resolution = config["dataset_kwargs"]["transform_kwargs"]["res"]
    input_shape = (3, resolution, resolution)
    neck_dim = cnn_model.get_frm_output_size(input_shape)
    input_module = cnn_model
    middle_module = IdentityModule()  # None for now

    # Setting up tasks and payloads
    tasks = []
    payloads = []

    # Get payload:primary task dict
    task_payload_dict = defaultdict(list)
    for full_task_name in full_task_names:
        payload_name = full_task_name.split("_")[0]
        task_name = full_task_name.split("_")[1]
        task_payload_dict[payload_name].append(task_name)

    # If "ALL" supplied as task, only one payload per dataset;
    # Else, create separate payloads for each task-dataset combo
    # TODO: GET THIS WORKING TO REPLICATE SINGLE PASS THROUGH DATA
    for k, v in task_payload_dict.items():
        if "ALL" in v:
            if len(v) > 1:
                raise ValueError("Cannot have 'ALL' task with other primary tasks")
            else:
                full_task_names.remove(f"{k}_ALL")
                full_task_names = full_task_names + [
                    f"{k}_{t}" for t in MASTER_PAYLOAD_TASK_DICT[k]
                ]
    # Getting auxiliary task dict
    auxiliary_task_dict = config["auxiliary_task_dict"]

    # Get primary task:payload dict
    payload_task_dict = invert_dict(task_payload_dict)

    for full_task_name in full_task_names:
        # Pull out names of auxiliary tasks to be dealt with in a second step
        # TODO: fix this logic for cases where auxiliary task for task_name has
        # its own payload
        # Right now, supply tasks as DATASET:TASK, by default if all tasks
        # in a dataset are included, it is the same as training on entire dataset
        if config["pool_payload_tasks"]:
            payload_name = full_task_name.split("_")[0]
            dataset_name = payload_name
            task_name = full_task_name.split("_")[1]
            payload_finding = task_name
            if task_name not in auxiliary_task_dict.keys():
                new_payload = f"{payload_name}_train" not in [p.name for p in payloads]
            else:
                new_payload = False
        else:
            dataset_name = full_task_name.split("_")[0]
            if "ALL" in task_payload_dict[dataset_name]:
                payload_name = dataset_name
                payload_finding = "ALL"
            else:
                payload_name = full_task_name
                payload_finding = full_task_name.split("_")[1]
            task_name = full_task_name
            if task_name.split("_")[1] not in auxiliary_task_dict.keys():
                new_payload = f"{payload_name}_train" not in [p.name for p in payloads]
            else:
                new_payload = False

        # Getting dl_kwargs and dataset kwargs
        dataset_kwargs = copy.deepcopy(config["dataset_kwargs"])
        dl_kwargs = copy.deepcopy(config["dl_kwargs"])

        # Each data source has data_loaders to load
        if new_payload:
            datasets = create_cxr_datasets(
                dataset_name=dataset_name,
                splits=config["splits"],
                subsample=config["subsample"],
                pooled=config["pool_payload_tasks"],
                finding=payload_finding,
                eval_finding=config["eval_finding"], 
                verbose=True,
                seed=config["seed"],
                dataset_kwargs=dataset_kwargs,
            )

            # Wrap datasets with DataLoader objects
            data_loaders = create_cxr_dataloaders(
                datasets,
                dl_kwargs=dl_kwargs,
                split_prop=config["split_prop"],
                splits=config["splits"],
                seed=config["seed"],
            )

        # TODO: PUT IN OPTION TO POOL SAME TASK FOR DIFF SETS HERE?

        task_metrics = ["f1", "roc-auc","accuracy"]
        if "PNEUMOTHORAX" in task_name:
            scorer = Scorer(standard_metrics=task_metrics)
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )
        # TODO: Convolutional decoder module
        elif "AUTOENCODE" in task_name:
            pass

        else:
            scorer = Scorer(standard_metrics=task_metrics)
            task = ClassificationTask(
                name=task_name,
                input_module=input_module,
                middle_module=middle_module,
                attention_module=get_attention_module(config, neck_dim),
                head_module=BinaryHead(neck_dim),
                scorer=scorer,
            )

        # AUXILIARY TASKS

        # NONE YET -- MAYBE AUTOENCODING?

        # else:
        #    msg = (
        #        f"Task name {task_name} was not recognized as a primary or "
        #        f"auxiliary task."
        #    )
        #    raise Exception(msg)

        tasks.append(task)

        # Create payloads (and add slices/auxiliary tasks as applicable)
        for split in config["splits"]:
            payload_name_split = f"{payload_name}_{split}"
            if new_payload:
                data_loader = data_loaders[split]
                payload_name_split = f"{payload_name}_{split}"
                payload = Payload(
                    payload_name_split, data_loader, {task_name: task_name}, split
                )
                # Add auxiliary label sets if applicable
                # CXR: NOT TESTED
                for aux_task_name, target_payloads in auxiliary_task_dict.items():
                    if (
                        any([aux_task_name in t for t in full_task_names])
                        and payload_name in target_payloads
                    ):
                        aux_task_func = auxiliary_task_functions[aux_task_name]
                        payload = aux_task_func(payload)

                payloads.append(payload)

            else:
                # If payload exists, get it
                payload_names = [p.name for p in payloads]
                payload = payloads[payload_names.index(payload_name_split)]

                # Add task name to payload -- assumes label set already there
                payload.labels_to_tasks[task_name] = task_name
                assert task_name in payload.data_loader.dataset.labels.keys()

            # Add slice task and label sets if applicable
            # CXR: not tested
            slice_names = (
                config["slice_dict"].get(task_name, []) if config["slice_dict"] else []
            )

            if slice_names:
                if config['model_type'] in ["slice_model"]:
                    loss_multiplier = 1.0 / (2 * len(slice_names))
                else:
                    loss_multiplier = 1.0 / (len(slice_names))
                for slice_name in slice_names:
                    if config["use_slices"]:
                        # TODO: update to add_slice_labels_and_tasks
                        slice_task_name = f"{task_name}:{slice_name}"
                        base_pos_only = slice_name in config["slice_pos_only"]
                        add_slice_labels_and_tasks(
                            payload,
                            tasks,
                            task,
                            slice_name,
                            model_type=config["model_type"],
                            loss_multiplier=loss_multiplier,
                            base_pos_only=base_pos_only,
                            add_task=True,
                        )

    return tasks, payloads


# IN PROGRESS: ADD THSLICE TASKS LIKE IN
# https://github.com/HazyResearch/metal/blob/mmtl_slicing/metal/mmtl/notebooks/Slicing.ipynb
def add_slice_labels_and_tasks(
    pay, tsks, tsk, slice_nm, model_type=None, loss_multiplier=1.0, base_pos_only=False, add_task=True, pred_eval=False
):
    datast = pay.data_loader.dataset
    task_nm = tsk.name
    slice_labels = create_slice_labels(
        datast, base_task_name=task_nm, slice_name=slice_nm,
        base_pos_only=base_pos_only 
    )

    # Changing which labelsets added based on model used
    if model_type in ["slice_model"]:
        slice_head_types = ["ind", "pred"]
    elif model_type in ["slice_rep_model"]:
        slice_head_types = ["ind"]
    else:
        slice_head_types = ["pred"]

    if pred_eval:
        slice_head_types = ["pred"]
    # Adding a labelset slice to the payload
    for slice_head_type in slice_head_types:
        slice_task_name = f"{task_nm}_slice:{slice_nm}:{slice_head_type}"
        labelset_slice_name = f"{task_nm}_slice:{slice_nm}:{slice_head_type}"

        if add_task:
            # Adding slice task
            tsks.append(
                create_slice_task(
                    tsk,
                    slice_task_name,
                    slice_head_type,
                    loss_multiplier=loss_multiplier,
                )
            )

        # Adding label set
        pay.add_label_set(
            slice_task_name, labelset_slice_name, slice_labels[slice_head_type]
        )


def get_attention_module(config, neck_dim):
    # Get attention head
    attention_config = config["attention_config"]
    if attention_config["attention_module"] is None:
        attention_module = IdentityModule()
    elif attention_config["attention_module"] == "soft":
        nonlinearity = attention_config["nonlinearity"]
        if nonlinearity == "tanh":
            nl_fun = nn.Tanh()
        elif nonlinearity == "sigmoid":
            nl_fun = nn.Sigmoid()
        else:
            raise ValueError("Unrecognized attention nonlinearity")
        attention_module = SoftAttentionModule(neck_dim, nonlinearity=nl_fun)
    else:
        raise ValueError("Unrecognized attention layer")

    return attention_module


def create_cxr_datasets(
    dataset_name,
    splits,
    pooled=False,
    finding="ALL",
    eval_finding="ALL",
    subsample=-1,
    verbose=True,
    dataset_kwargs={},
    get_uid=False,
    return_dict=True,
    seed=None,
):
    if verbose:
        logger.info(f"Loading {dataset_name} Dataset")

    datasets = {}
    for split_name in splits:
        # Codebase uses valid but files are saved as dev.tsv
        if split_name == "valid":
            split = "dev"
        else:
            split = split_name
        # Getting all examples for val and test!
        if split_name != "train" and eval_finding:
            logger.debug(f"Using eval finding {eval_finding}")
            finding = eval_finding
            subsample = -1
        else:
            logger.info(f"Using train finding {finding}")
        datasets[split_name] = get_cxr_dataset(
            dataset_name,
            split,
            subsample=subsample,
            finding=finding,
            get_uid=get_uid,
            return_dict=return_dict,
            seed=seed,
            **dataset_kwargs,
        )
    return datasets


def create_cxr_dataloaders(datasets, dl_kwargs, split_prop, splits, seed=123):
    """ Initializes train/dev/test dataloaders given dataset_class"""
    dataloaders = {}

    # When split_prop is not None, we use create an artificial dev set from the train set
    if split_prop and "train" in splits:
        dataloaders["train"], dataloaders["valid"] = datasets["train"].get_dataloader(
            split_prop=split_prop, split_seed=seed, **dl_kwargs
        )

        # Use the dev set as test set if available.
        if "valid" in datasets:
            dataloaders["test"] = datasets["valid"].get_dataloader(**dl_kwargs)

    # When split_prop is None, we use standard train/dev/test splits.
    else:
        for split_name in datasets:
            dl_kwargs = dl_kwargs
            if split_name != 'train':
                dl_kwargs["shuffle"] = False
            # if split_name == 'test':
            #     dl_kwargs['num_workers'] = 0
            dataloaders[split_name] = datasets[split_name].get_dataloader(**dl_kwargs)
    return dataloaders


def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse
