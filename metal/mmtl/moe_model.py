import copy
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from metal.mmtl.metal_model import MetalModel
from metal.mmtl.slicing.slice_model import SliceModel
from metal.mmtl.slicing.tasks import BinaryClassificationTask, convert_to_slicing_tasks
from metal.mmtl.task import ClassificationTask
from metal.utils import move_to_device


def convert_to_expert_tasks(slice_tasks):
    expert_tasks = []
    for t in slice_tasks:
        if isinstance(t, ClassificationTask):
            is_slice = ":" in t.name
            if is_slice:
                t.input_module = copy.deepcopy(t.input_module)
                t.middle_module = copy.deepcopy(t.middle_module)
                t.attention_module = copy.deepcopy(t.attention_module)
            expert_tasks.append(t)
        else:
            raise ValueError(f"{t.__class__.__name__} is not supported!")

    return expert_tasks


class MOEModel(MetalModel):
    """Mixture Of Experts Model"""

    def __init__(self, tasks, cache_dir="/.cache/", **kwargs):
        super().__init__(tasks, **kwargs)
        self.verbose = self.config["verbose"]
        self.cache_dir = cache_dir

        for t in tasks:
            self.cache_input_module(t.name)

        self.tasks = tasks
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.base_task = [t for t in tasks if not t.slice_head_type][0]

        # use out_features of the encoder (== in_features of the pooler)
        middle_modules = self.middle_modules[self.base_task.name].module
        out_features = middle_modules.pooler._modules["dense"].in_features
        n_slice_tasks = len(tasks)
        self.G = nn.DataParallel(
            GatedNetwork(n_slice_tasks + out_features, n_slice_tasks)
        )

    def _get_task_cache_path(self, task_name):
        return os.path.join(self.cache_dir, "{}_bert".format(task_name))

    def cache_input_module(self, task_name):
        cache_path = self._get_task_cache_path(task_name)
        torch.save(self.input_modules[task_name], cache_path)
        if self.verbose:
            print("Cached input module for {}.".format(task_name))

    def load_input_module(self, task_name):
        cache_path = self._get_task_cache_path(task_name)
        self.input_modules[task_name] = torch.load(cache_path)
        self.input_modules[task_name].eval()
        if self.verbose:
            print("Loaded input module for {}.".format(task_name))

    def forward_task(self, input, task_name):
        input = move_to_device(input, self.config["device"])
        self.load_input_module(task_name)
        input_module = self.input_modules[task_name].module
        middle_module = self.middle_modules[task_name].module
        attention_module = self.attention_modules[task_name].module
        body_out = attention_module(middle_module(input_module(input)))
        out = self.head_modules[task_name].module(body_out)
        self.cache_input_module(task_name)  # TODO: wrong. maybe do this in trainer.py?
        return out

    def forward_body(self, input):
        input = move_to_device(input, self.config["device"])
        base_task_name = self.base_task.name
        self.load_input_module(base_task_name)
        input_module = self.input_modules[base_task_name].module
        middle_module = self.middle_modules[base_task_name].module
        attention_module = self.attention_modules[base_task_name].module
        pooled = middle_module(input_module(input))
        out = attention_module(pooled)

        self.cache_input_module(base_task_name)
        return out, pooled

    def forward(self, X, task_names):
        slice_heads = {}
        for t in self.tasks:
            task_name = t.name
            slice_heads[task_name] = self.forward_task(X, task_name)

        body, base_pooled = self.forward_body(X)
        slice_preds = torch.cat(
            [slice_head["data"] for slice_head in slice_heads.values()], dim=1
        )

        slice_weights = self.G(
            F.softmax(torch.cat([slice_preds, base_pooled["data"]], dim=1))
        )

        output = {
            self.base_task.name: {
                "data": (slice_weights * slice_preds).sum(1).reshape(-1, 1)
            }
        }
        slice_heads.pop(self.base_task.name)
        output.update(slice_heads)
        return output


class GatedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x)
        return x
