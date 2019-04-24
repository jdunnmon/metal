import argparse
import copy
import json

import numpy as np

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import SliceModel
from metal.mmtl.slicing.slicing_tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts


"""
python compare_to_baseline.py --seed 1 --tasks RTE --slice_dict '{"RTE": ["dash_semicolon", "more_people", "BASE"]}'
"""


def get_baseline_scores(tasks, seed, eval_payload):
    # Baseline Model
    task_name = tasks[0].name
    model = MetalModel(tasks, seed=seed, verbose=False)
    trainer = MultitaskTrainer(seed=seed)
    trainer.train_model(
        model,
        payloads,
        checkpoint_metric="{}/{}/{}_gold/accuracy".format(
            task_name, eval_payload.name, task_name
        ),
        checkpoint_metric_mode="max",
        checkoint_best=True,
        writer="tensorboard",
        optimizer="adamax",
        lr=5e-5,
        l2=1e-3,
        log_every=0.1,
        score_every=0.1,
        n_epochs=10,
        progress_bar=True,
        checkpoint_best=True,
        checkpoint_cleanup=False,
    )
    return model.score(eval_payload)


def get_model_scores(tasks_slice, payloads_slice, seed, eval_payload):
    task_name = tasks_slice[0].name
    slicing_tasks = convert_to_slicing_tasks(tasks_slice)
    slice_model = SliceModel(slicing_tasks, seed=seed, verbose=False)
    task_metrics = []
    prefix = "{}/{}_".format(task_name, task_name)
    for p_name, metric in [("train", "loss"), ("valid", "accuracy")]:
        for label_name in eval_payload.labels_to_tasks.keys():
            task_metrics += [prefix + p_name + "/{}/{}".format(label_name, metric)]
    print(task_metrics)
    trainer = MultitaskTrainer(seed=seed)
    trainer.train_model(
        slice_model,
        payloads_slice,
        task_metrics=task_metrics,
        checkpoint_metric="{}/{}/{}_gold/accuracy".format(
            task_name, eval_payload.name, task_name
        ),
        checkpoint_metric_mode="max",
        checkoint_best=True,
        writer="tensorboard",
        optimizer="adamax",
        lr=1e-5,
        l2=1e-3,
        log_every=0.1,
        score_every=0.1,
        n_epochs=20,
        progress_bar=True,
        checkpoint_best=True,
        checkpoint_cleanup=False,
    )
    return slice_model.score(eval_payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare specified model with MetalModel on single tasks",
        add_help=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=np.random.randint(1e6),
        help="A single seed to use for trainer, model, and task configs",
    )
    parser.add_argument(
        "--model_type", type=str, default="metal", help="Baseline model type"
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
    task_name = task_names[0]

    # Create tasks and payloads
    task_config["slice_dict"] = None
    task_config["attention"] = False
    tasks, payloads = create_glue_tasks_payloads(task_names, **task_config)

    # Create slicing tasks and payloads
    slice_dict = json.loads(args.slice_dict)
    task_config.update({"slice_dict": slice_dict})
    task_config["attention"] = None
    tasks_slice, payloads_slice = create_glue_tasks_payloads(task_names, **task_config)

    eval_payload = copy.deepcopy(payloads_slice[1])

    # NOTE: we need to retarget slices to the original RTE head
    for label_name in [t.name for t in tasks_slice[1:]]:
        eval_payload.retarget_labelset(label_name, task_name)

    if args.model_type.upper() == "METAL":
        baseline_scores = get_baseline_scores(tasks, args.seed, eval_payload)
    else:
        raise NotImplementedError

    model_scores = get_model_scores(
        tasks_slice, payloads_slice, args.seed, eval_payload
    )
    print(baseline_scores)
    print(model_scores)
