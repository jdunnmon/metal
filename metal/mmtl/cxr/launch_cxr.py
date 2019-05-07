"""
Example command to run all 9 tasks: python launch.py --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --checkpoint_dir ckpt --batch_size 16
"""
import argparse
import copy
import datetime
import json
import logging
import os
from time import strftime

import numpy as np

import faulthandler
from metal.mmtl.cxr.cxr_tasks import (
    add_slice_labels_and_tasks,
    create_tasks_and_payloads,
    task_defaults,
)
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import SliceModel, SliceRepModel
from metal.mmtl.slicing.tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

faulthandler.enable()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    existing_dirs = np.array(
        [
            d
            for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]
    ).astype(np.int)
    if len(existing_dirs) > 0:
        return str(existing_dirs.max() + 1)
    else:
        return "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MetalModel on single or multiple tasks.", add_help=False
    )

    # Model arguments
    # TODO: parse these automatically from model dict
    parser.add_argument(
        "--model_weights",
        type=str,
        default=None,
        help="Pretrained model for weight initialization",
    )

    # Slice only evaluation?
    parser.add_argument(
        "--slice_eval", type=int, default=0, help="True for loading model"
    )

    # Training arguments
    parser.add_argument(
        "--override_train_config",
        type=str,
        default=None,
        help=(
            "Whether to override train_config dict with json loaded from path. "
            "This is used, e.g., for tuning."
        ),
    )

    # Fine tuning with schedule plan
    parser.add_argument(
        "--fine_tune",
        type=str,
        default=None,
        help="Comma separated list of tasks to fine tune"
        )

    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    args = parser.parse_args()

    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")

    # Override config with config stored in json if specified
    if args.override_train_config is not None:
        with open(args.override_train_config, "r") as f:
            override_config = json.loads(f.read())
        config = recursive_merge_dicts(trainer_config, override_config, misses="report")

    # Set intelligent writer_config settings
    trainer_config["writer"] = "tensorboard"  # Always store tensorboard logs

    # Set splits based on split_prop
    if task_config["split_prop"]:
        # Create a valid set from train set
        task_config["split_prop"] = float(task_config["split_prop"])
        # Sampler will be used and handles shuffle automatically
        task_config["dl_kwargs"]["shuffle"] = False
        task_config["splits"] = ["train", "test"]
    else:
        task_config["splits"] = ["train", "valid", "test"]

    # Getting primary task names
    task_names = [task_name for task_name in args.tasks.split(",")]

    # Updating slice dict:
    slice_dict = json.loads(args.slice_dict) if args.slice_dict else {}
    if task_config["use_slices"]: # equivalent to active_task_heads in mmtl_slicing branch
        task_config.update({"slice_dict": slice_dict})
    else:
        task_config.update({"slice_dict": None})

    # Adding slices if needed for slice model
    model_type =  model_config["model_type"]
    if model_type:
        # Ensuring we get correct labelsets
        task_config["model_type"] = model_type

        # Adding BASE to slice dict for SliceModel
        for k in task_config["slice_dict"]:
            task_config["slice_dict"][k].append("BASE")

        # Right now, make sure only one active key in slice dict!
        assert len(task_config["slice_dict"].keys()) <= 1

        # Set base task name as only one in slice dict for now
        base_task_name = list(task_config["slice_dict"].keys())[0]

    # Getting tasks
    tasks, payloads = create_tasks_and_payloads(task_names, **task_config)
    model_config["verbose"] = False
    if model_type:
        base_task = [t for t in tasks if t.name == base_task_name][0]
        tasks = convert_to_slicing_tasks(tasks)
        if model_type == "slice_model":
            logger.info("Initializing SliceModel...")
            model = SliceModel(tasks, base_task=base_task, **model_config)
        elif model_type == "slice_rep_model":
            logger.info("Initializing SliceRepModel...")
            model = SliceRepModel(tasks, base_task=base_task, **model_config)
    else:
        logger.info("Initializing MetalModel...")
        model = MetalModel(tasks, **model_config)

    if args.model_weights:
        model.load_weights(args.model_weights)

    # Defining training schedule
    if args.fine_tune:
        tasks_to_tune = args.fine_tune.split(',')
        tasks_to_freeze = [t.name for t in tasks if t.name not in tasks_to_tune]
        trainer_config["train_schedule_plan"] = {
                "plan": {
                    "-1": tasks_to_freeze,
                    },
                "freeze": trainer_config["freeze"],
                }
                
    # add metadata to trainer_config that will be logged to disk
    trainer_config["n_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if trainer_config["verbose"]:
        logger.debug(f"Task config:\n{task_config}")
        logger.debug(f"Model config:\n{model_config}")
        logger.debug(f"Trainer config:\n{trainer_config}")

    # Overwrite run_dir to only use one checkpoint dir
    # if args.run_dir is None:
    #    trainer_config["writer_config"]["run_dir"] = strftime("%Y_%m_%d")

    if args.run_name is None:
        trainer_config["writer_config"]["run_name"] = task_config[
            "tasks"
        ]  # + strftime("_%H_%M_%S")

    trainer = MultitaskTrainer(**trainer_config)
    # Force early instantiation of writer to write all three configs to dict
    trainer._set_writer()

    # trainer_config will get written automatically right before training
    trainer.writer.write_config(model_config, "model_config")
    trainer.writer.write_config(task_config, "task_config")

    if not args.slice_eval:
        trainer.train_model(model, payloads, train_schedule_plan=trainer_config['train_schedule_plan'])
    else:
        logger.info("Running slice evaluation only...")
        # Writing output to log where model path was
        slice_subdir = "/".join(args.model_weights.split("/")[:-1])
        trainer.writer.log_subdir = slice_subdir

    # evaluating slices after run complete
    slice_output = {}
    test_splits = trainer_config["metrics_config"]["test_split"]
    if isinstance(test_splits, str):
        test_splits = [test_splits]
    for split in test_splits:
        # Creating payloads for evaluation
        payload_names = [p.split for p in payloads]
        payload_ind = payload_names.index(split)
        slice_payload = copy.deepcopy(payloads[payload_ind])
        main_payload = copy.deepcopy(payloads[payload_ind])
        slice_output[split] = {}
        slice_eval_dict = task_config["slice_dict"]
        if "CXR8-DRAIN_PNEUMOTHORAX" not in slice_eval_dict.keys():
            slice_eval_dict.update({"CXR8-DRAIN_PNEUMOTHORAX": ["chest_drain_cnn_neg"]})
        elif "chest_drain_cnn_neg" not in slice_eval_dict["chest_drain_cnn_neg"]:
            slice_eval_dict["chest_drain_cnn_neg"].append("chest_drain_cnn_neg")
        else:
            logger.info('Negative drain slice already in problem...')
        # Retargeting slices
        for tsk, slices in slice_eval_dict.items():
            for slc in slices:
                #if not task_config["use_slices"]:
                if "{tsk}_slice:{slc}:pred" not in main_payload.labels_to_tasks:
                    tsk_obj = tasks[[t.name for t in tasks].index(tsk)]
                    add_slice_labels_and_tasks(
                        main_payload, tasks, tsk_obj, slc, 
                        base_pos_only=task_config['slice_pos_only'],
                        add_task=False,
                        pred_eval=True
                    )
                main_payload._retarget_labelset(f"{tsk}_slice:{slc}:pred", tsk)
        # Scoring model
        main_dict = model.score(main_payload)
        main_dict = {k: v for k, v in main_dict.items() if ":" in k}
        # Printing results
        logger.info(f"Evaluating slice performance on {split} split")
        logger.info("Using main task heads:")
        logger.info(main_dict)
        # Storing results
        slice_output[split]["MAIN"] = main_dict
        if task_config["use_slices"]:
            slc_dict = model.score(slice_payload)
            slc_dict = {k: v for k, v in slc_dict.items() if ":pred" in k}
            logger.info("Using slice task heads:")
            logger.info(slc_dict)
            slice_output[split]["SLICE"] = slc_dict

    # Writing slice evaluation
    slice_metrics_path = os.path.join(trainer.writer.log_subdir, "slice_metrics.json")
    logger.info(f"Writing slice metrics to {slice_metrics_path}")
    with open(slice_metrics_path, "w") as f:
        json.dump(slice_output, f, indent=1)
