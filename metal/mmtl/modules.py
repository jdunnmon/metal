import torch.nn as nn


def unwrap_module(module):
    if isinstance(module, MetalModuleWrapper):
        return module.module
    else:
        return module


class MetalModule(nn.Module):
    """An abstract class of a module that accepts and returns a dict"""

    def __init__(self):
        super().__init__()


class MetalModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        X_out = {k: v for k, v in X.items()}
        X_out["data"] = self.module(X_out["data"])
        return X_out
