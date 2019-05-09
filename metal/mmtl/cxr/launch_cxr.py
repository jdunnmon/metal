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
from metal.mmtl.slicing.slice_model import SliceModel, SliceRepModel, SliceEnsembleModel
from metal.mmtl.slicing.tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

faulthandler.enable()

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,filename=f'logger_out/run_log_{strftime("%Y%m%d-%H%M%S")}')
logging.getLogger().addHandler(logging.StreamHandler())

# Overwrite defaults
task_defaults["attention"] = False
model_defaults["verbose"] = False
model_defaults["delete_heads"] = False

# by default, log best epoch (not best)
trainer_defaults["checkpoint"] = True
trainer_defaults["checkpoint_config"]["checkpoint_best"] = True
trainer_defaults["writer"] = "tensorboard"

# Model configs
model_configs = {
    "naive": {"model_class": MetalModel, "active_slice_heads": {}},
    "hard_param": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
    "manual": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
    "soft_param": {
        "model_class": SliceModel,
        "active_slice_heads": {"pred": True, "ind": True},
    },
    "soft_param_rep": {
        "model_class": SliceRepModel,
        "active_slice_heads": {"pred": False, "ind": True},
    },
    "soft_param_ens": {
        "model_class": SliceEnsembleModel,
        "active_slice_heads": {"pred": True, "ind": True},
    },
    
}

def main(args):
    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")

    # Getting primary task names
    task_names = [task_name for task_name in args.tasks.split(",")]
    main_to_slice_task_names = copy.deepcopy(task_names) # done b/c _ALL is popped

    # Override config with config stored in json if specified
    if args.override_train_config is not None:
        with open(args.override_train_config, "r") as f:
            override_config = json.loads(f.read())
        config = recursive_merge_dicts(trainer_config, override_config, misses="report")

    # Default name for log directory to task names
    if args.run_name is None:
        run_name = f"{args.model_type}_{args.tasks}"
        trainer_config["writer_config"]["run_name"] = run_name

    # Get model configs
    config = model_configs[model_config["model_type"]]
    model_class =  config["model_class"]
    active_slice_heads = config["active_slice_heads"]

    # Updating slice dict:
    slice_dict = json.loads(args.slice_dict) if args.slice_dict else {}
    if active_slice_heads:
        task_config.update({"slice_dict": slice_dict})
        task_config["active_slice_heads"] = active_slice_heads
    else:
        task_config.update({"slice_dict": None})

    # Adding BASE slices if needed for slice model
    if args.model_type in ["soft_param", "soft_param_rep", "soft_param_ens"]:
        # Ensuring we get correct labelsets
        task_config["model_type"] = args.model_type

        # Adding BASE to slice dict for SliceModel
        for k in task_config["slice_dict"]:
            task_config["slice_dict"][k].append("BASE")

        # Right now, make sure only one active key in slice dict!
        assert len(task_config["slice_dict"].keys()) <= 1
    

    # Getting tasks
    tasks, payloads = create_tasks_and_payloads(task_names, **task_config)
    model_config["verbose"] = False

    if args.model_type == "manual":
        slice_loss_mult = (
            json.loads(args.slice_loss_mult) if args.slice_loss_mult else {}
        )
        for task in tasks:
            if task.name in slice_loss_mult.keys():
                task.loss_multiplier = slice_loss_mult[task.name]
                print(
                    "Override {} loss multiplier with{}.".format(
                        task.name, slice_loss_mult[task.name]
                    )
                )

    # Converting to slicing tasks if needed:
    base_task_name = list(slice_dict.keys())[0]
    base_task = [t for t in tasks if t.name == base_task_name][0]
    if active_slice_heads:
        tasks = convert_to_slicing_tasks(tasks)

    # Create evaluation payload with test_slices -> primary task head
    logger.info("Creating main_to_slice payloads for evaluation...")
    task_config_main_to_slice = copy.deepcopy(task_config)
    # Updating to have negative drain slice regardless of what else is happening...
    task_config_main_to_slice["slice_dict"] = copy.deepcopy(slice_dict)
    if "CXR8-DRAIN_PNEUMOTHORAX" in task_config_main_to_slice["slice_dict"]:
       task_config_main_to_slice["slice_dict"]["CXR8-DRAIN_PNEUMOTHORAX"].append("chest_drain_cnn_neg")
    else:
        task_config_main_to_slice["slice_dict"].update({"CXR8-DRAIN_PNEUMOTHORAX": ["chest_drain_cnn_neg"]})
    task_config_main_to_slice['slice_pos_only'] = ['NONE']

    task_config_main_to_slice["active_slice_heads"] = {
        # turn pred labelsets on, and use model's value for ind head
        "pred": True,
        "ind": active_slice_heads.get("ind", False),
    }
    slice_tasks, main_to_slice_payloads = create_tasks_and_payloads(main_to_slice_task_names, **task_config_main_to_slice)
    pred_labelsets = [
        labelset
        for labelset in main_to_slice_payloads[0].labels_to_tasks.keys()
        if "pred" in labelset
    ]

    # Only eval "pred" labelsets on main task head -- continue eval of inds on ind-heads
    for p in main_to_slice_payloads[1:]:  # remap val and test payloads
        p.remap_labelsets(
            {pred_labelset: base_task_name for pred_labelset in pred_labelsets}
        )

    # Option to validate with slices
    if args.validate_on_slices:
        print("Will compute validation scores for slices based on main head.")
        payloads[1] = main_to_slice_payloads[1]

    # Initializing model
    logger.info(f"Initializing {model_class.__name__}...")
    model = model_class(base_task=base_task, tasks=tasks, **model_config)

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

    # Force early instantiation of writer to write all three configs to dict
    trainer = MultitaskTrainer(**trainer_config)
    trainer._set_writer()

    # trainer_config will get written automatically right before training
    trainer.writer.write_config(model_config, "model_config")
    trainer.writer.write_config(task_config, "task_config")

    # keeping mode to run only slice eval
    if not args.slice_eval:
        trainer.train_model(model, payloads, train_schedule_plan=trainer_config['train_schedule_plan'])
    else:
        logger.info("Running slice evaluation only...")
        # Writing output to log where model path was
        slice_subdir = "/".join(args.model_weights.split("/")[:-1])
        trainer.writer.log_subdir = slice_subdir

    # evaluating slices after run complete
    model.eval()
    slice_output = {}
    test_splits = trainer_config["metrics_config"]["test_split"]
    if isinstance(test_splits, str):
        test_splits = [test_splits]
    for split in test_splits:
        # Creating payloads for evaluation
        payload_names = [p.split for p in payloads]
        payload_ind = payload_names.index(split)
        slice_payload = copy.deepcopy(payloads[payload_ind])
        main_payload = copy.deepcopy(main_to_slice_payloads[payload_ind])
        slice_output[split] = {}

        # Scoring head with slices 
        main_dict = model.score(main_payload)
        main_dict = {k: v for k, v in main_dict.items() if ":" in k}
        # Printing results
        logger.info(f"Evaluating slice performance on {split} split")
        logger.info("Using main task heads:")
        logger.info(main_dict)
        # Storing results
        slice_output[split]["MAIN"] = main_dict
        if task_config["active_slice_heads"]:
            slc_dict = model.score(slice_payload)
            slc_dict = {k: v for k, v in slc_dict.items() if ((":pred" in k) or (":ind" in k))}
            logger.info("Using slice task heads:")
            logger.info(slc_dict)
            slice_output[split]["SLICE"] = slc_dict

    # Writing slice evaluation
    slice_metrics_path = os.path.join(trainer.writer.log_subdir, "slice_metrics.json")
    logger.info(f"Writing slice metrics to {slice_metrics_path}")
    with open(slice_metrics_path, "w") as f:
        json.dump(slice_output, f, indent=1)

def get_parser():
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

    # Option to validate on slices
    parser.add_argument(
        "--validate_on_slices",
        type=bool,
        default=False,
        help="Whether to map eval main head on validation set during training",
    )

    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args) 
