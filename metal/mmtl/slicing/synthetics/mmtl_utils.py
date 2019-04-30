import torch
import torch.nn as nn

from metal.mmtl.data import MmtlDataLoader, MmtlDataset
from metal.mmtl.payload import Payload
from metal.mmtl.slicing.synthetics.data_generator import (
    generate_data,
    generate_slice_labels,
)
from metal.mmtl.slicing.tasks import BinaryClassificationTask, create_slice_task
from metal.mmtl.trainer import MultitaskTrainer


def create_tasks(
    task_name,
    slice_names=[],
    slice_weights={},
    create_ind=True,
    create_base=True,
    verbose=False,
):
    input_module = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
    head_module = nn.Linear(5, 1)  # NOTE: slice_model requires 1dim output head
    task = BinaryClassificationTask(
        name=task_name, input_module=input_module, head_module=head_module
    )
    tasks = [task] if create_base else []

    # for each slice create an 'ind' task: predicts whether we are in the slice
    # and a 'pred' task: "expert" on predicting labels of slice
    for slice_name in slice_names:
        # TODO: make loss_multiplier a parameter
        # loss_multiplier = 1.0 / (2 * len(slice_names))
        if slice_name in slice_weights:
            loss_multiplier = slice_weights[slice_name]
        else:
            loss_multiplier = 1.0
        slice_pred_task = create_slice_task(
            task,
            f"{task_name}:{slice_name}:pred",
            "pred",
            loss_multiplier=loss_multiplier,
        )
        tasks.append(slice_pred_task)

        if create_ind:
            slice_ind_task = create_slice_task(
                task,
                f"{task_name}:{slice_name}:ind",
                "ind",
                loss_multiplier=loss_multiplier,
            )
            tasks.append(slice_ind_task)

    if verbose:
        print(f"Creating {len(tasks)} tasks...")
        for t in tasks:
            print(t)
    return tasks


def create_payloads(
    task_name,
    uid_lists,
    Xs,
    Ys,
    batch_size=1,
    slice_funcs={},
    SPLITS=["train", "valid", "test"],
    create_ind=True,
    create_base=True,
    verbose=False,
):

    payloads = []
    labels_to_tasks = {"labelset_gold": task_name} if create_base else {}
    for i, split in enumerate(SPLITS):
        payload_name = f"payload_{split}"

        # convert to torch tensors
        X_dict = {"data": torch.Tensor(Xs[i]), "uids": torch.Tensor(uid_lists[i])}
        Y_dict = {"labelset_gold": torch.Tensor(Ys[i])} if create_base else {}

        if slice_funcs:
            slice_labels = generate_slice_labels(Xs[i], Ys[i], slice_funcs, create_ind)

            # labelset_name -> {"ind": [1,2,1,2,2], "pred": [0,1,0,2,2]}
            for slice_name, slice_label in slice_labels.items():
                # slice_type \in {"ind", "pred"}
                for slice_type, label in slice_label.items():
                    slice_task_name = f"{task_name}:{slice_name}:{slice_type}"
                    slice_labelset_name = f"labelset:{slice_name}:{slice_type}"
                    Y_dict[slice_labelset_name] = torch.tensor(label)
                    labels_to_tasks[slice_labelset_name] = slice_task_name

        dataset = MmtlDataset(X_dict, Y_dict)
        data_loader = MmtlDataLoader(dataset, batch_size=batch_size)
        payload = Payload(payload_name, data_loader, labels_to_tasks, split)
        payloads.append(payload)

    if verbose:
        print(f"Creating {len(payloads)} payloads...")
        for p in payloads:
            print(p)
    return payloads


def train_slice_experts(
    uid_lists, Xs, Ys, model_class, slice_funcs, batch_size=16, **trainer_kwargs
):
    """ For each slice_func,
        1) initializes a model using model_class(tasks),
        2) trains on payloads where data is masked by slice.
    """

    experts = {}
    task_name = "expert"
    for slice_name, slice_fn in slice_funcs.items():
        tasks = create_tasks(
            task_name,
            slice_names=[slice_name],
            create_ind=False,
            create_base=False,
            verbose=True,
        )
        payloads = create_payloads(
            task_name,
            uid_lists,
            Xs,
            Ys,
            batch_size=batch_size,
            slice_funcs={slice_name: slice_fn},
            create_ind=False,
            create_base=False,
            verbose=True,
        )
        print(tasks)
        print(payloads)
        model = model_class(tasks)
        seed = trainer_kwargs.get("seed", None)
        trainer = MultitaskTrainer(seed=seed)
        metrics_dict = trainer.train_model(model, payloads, **trainer_kwargs)
        print(metrics_dict)
        experts[slice_name] = model

    return experts
