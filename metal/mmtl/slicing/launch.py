"""
Creates slice/baseline/ablation model, trains, and evaluates on
corresponding slice prediction labelsets.

python compare_to_baseline.py --seed 1 --tasks RTE --slice_dict '{"RTE": ["dash_semicolon", "more_people", "BASE"]}' --model_type naive --n_epochs 1
"""

import argparse
import copy
import json
from pprint import pprint

import numpy as np

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import SliceModel
from metal.mmtl.slicing.tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

# Overwrite defaults
task_defaults["attention"] = False
model_defaults["verbose"] = False
model_defaults["delete_heads"] = True  # mainly load the base representation weights
trainer_defaults["writer"] = "tensorboard"
trainer_defaults["metrics_config"][
    "test_split"
] = "valid"  # for GLUE, don't have real test set

# Model configs
model_configs = {
    "naive": {"model_class": MetalModel, "use_slice_heads": False},
    "hard_param": {"model_class": MetalModel, "use_slice_heads": True},
    "soft_param": {"model_class": SliceModel, "use_slice_heads": True},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch slicing models/baselines on GLUE tasks", add_help=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=np.random.randint(1e6),
        help="A single seed to use for trainer, model, and task configs",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["naive", "hard_param", "soft_param"],
        help="Model to run and evaluate",
    )
    parser.add_argument(
        "--train_schedule_plan", type=str, default=None, help="Training schedule"
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
    args = parser.parse_args()

    task_names = args.tasks.split(",")
    assert len(task_names) == 1
    base_task_name = task_names[0]

    # Default name for log directory to task names
    if args.run_name is None:
        run_name = f"{args.model_type}_{args.tasks}"
        trainer_config["writer_config"]["run_name"] = run_name

    # Get model configs
    config = model_configs[args.model_type]
    use_slice_heads = config["use_slice_heads"]
    model_class = config["model_class"]

    # Create tasks and payloads
    slice_dict = json.loads(args.slice_dict)
    if use_slice_heads:
        task_config.update({"slice_dict": slice_dict})
    else:
        task_config.update({"slice_dict": None})

    # Load train schedule
    if args.train_schedule_plan is not None:
        train_schedule_plan = json.loads(args.train_schedule_plan)
    else:
        train_schedule_plan = None

    tasks, payloads = create_glue_tasks_payloads(task_names, **task_config)
    if use_slice_heads:
        tasks = convert_to_slicing_tasks(tasks)

    # Initialize and train model
    model = model_class(tasks, **model_config)

    trainer = MultitaskTrainer(**trainer_config)
    trainer.train_model(model, payloads, train_schedule_plan)

    # Create evaluation payload with test_slices -> primary task head
    task_config.update({"slice_dict": slice_dict})
    slice_tasks, slice_payloads = create_glue_tasks_payloads(task_names, **task_config)
    eval_payload = slice_payloads[1]
    pred_labelsets = [
        labelset
        for labelset in eval_payload.labels_to_tasks.keys()
        if "pred" in labelset or "_gold" in labelset
    ]
    eval_payload.remap_labelsets(
        {pred_labelset: base_task_name for pred_labelset in pred_labelsets}
    )

    model.eval()
    slice_metrics = model.score(eval_payload)
    pprint(slice_metrics)
    if trainer.writer:
        trainer.writer.write_metrics(slice_metrics, "slice_metrics.json")
