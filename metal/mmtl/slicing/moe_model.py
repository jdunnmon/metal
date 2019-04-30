import torch
import torch.nn as nn
from torch.nn import functional as F

from metal.mmtl.metal_model import MetalModel
from metal.utils import move_to_device


def validate_moe_tasks(tasks, experts):
    if len(tasks) > 1:
        raise ValueError(
            "MoE only supports 1 base task. All slice tasks should be trained "
            "as experts."
        )

    for expert_name, expert in experts.items():
        if len(expert.task_map) > 1:
            raise ValueError(
                f"Expert '{expert_name}' should only be trained on 1 task."
            )


class MoEModel(MetalModel):
    """Mixture Of Experts Model"""

    def __init__(self, tasks, experts, **kwargs):
        """
        Args:
            experts: dict_mapping expert_name to pretrained model
        """
        validate_moe_tasks(tasks, experts)
        super().__init__(tasks, **kwargs)

        self.experts = experts

        # expert_name -> task_name
        self.expert_task_map = {}
        for expert_name, expert in experts.items():
            self.expert_task_map[expert_name] = list(expert.task_map.keys())[0]

        # freeze weights in experts
        for expert_name, expert in self.experts.items():
            for param in expert.parameters():
                param.requires_grad = False

        self.base_task = [
            t for t in self.task_map.values() if t.slice_head_type is None
        ][0]

        # initialize Gating Network
        neck_dim = self.base_task.head_module.module.in_features
        num_experts = len(experts)
        self.G = GatingNetwork(num_experts + neck_dim, num_experts)

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

    def forward(self, X, task_names):

        # compute and collect all expert predictions
        expert_names = sorted(self.expert_task_map.keys())
        preds = []
        for expert_name in expert_names:
            expert_task_name = self.expert_task_map[expert_name]
            expert_pred = self.experts[expert_name](X, [expert_task_name])
            pred_data = expert_pred[expert_task_name]["data"]
            preds.append(pred_data)

        # comptue weights for expert preds
        expert_preds = torch.cat(preds, dim=1)
        body = self.forward_body(X)

        gating_input = torch.cat((expert_preds, body["data"]), dim=1)
        expert_weights = self.G(gating_input)

        # compute weighted average of expert preds
        weighted_preds = (expert_weights * expert_preds).sum(1).reshape(-1, 1)
        output = {self.base_task.name: {"data": weighted_preds}}
        return output


class GatingNetwork(nn.Module):
    """ Given expert predictions and input representation (num_experts + neck_dim)
    outputs Softmaxed weights for each of the expert  models.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc(x)
        return F.softmax(out, dim=1)
