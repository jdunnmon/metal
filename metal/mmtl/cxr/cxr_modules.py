import torch
from torchvision import models
import torch.nn.init as init
import torch.nn as nn
from metal.end_model import IdentityModule

class TorchVisionEncoder(nn.Module):
    def __init__(
        self, net_name, freeze_cnn=False, pretrained=False, drop_rate=0.2):
        super().__init__()
        self.model = self.get_tv_encoder(net_name, pretrained,
            drop_rate)
        if freeze_cnn:
            for param in self.parameters():
                param.requires_grad = False

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0,1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size =  self.forward(x).size()[1]
        return frm_output_size

    def get_tv_encoder(self, net_name, pretrained, drop_rate):
        # HACK: replace linear with identity -- ideally remove this
        net = getattr(models, net_name, None)
        if net is None:
            raise ValueError(f'Unknown torchvision network {net_name}')
        if 'densenet' in net_name.lower():
            model = net(pretrained=pretrained, drop_rate=drop_rate)
            self.encode_dim=int(model.classifier.weight.size()[1])
            model.classifier=IdentityModule()
            #model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif 'resnet' in net_name.lower():
            model = net(pretrained=pretrained)
            self.encode_dim=int(model.fc.weight.size()[1])
            model.fc=IdentityModule()
            #model = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            raise ValueError('Network {net_name} not supported')
        return model

    def forward(self, X):
        out = self.model(X)
        return out

def MulticlassHead(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def BinaryHead(input_dim):
    return MulticlassHead(input_dim, 2)

def RegressionHead(input_dim):
    return MulticlassHead(input_dim, 1)


class SoftAttentionModule(nn.Module):
    def __init__(self, input_dim, nonlinearity=nn.Tanh()):
        super(SoftAttentionModule, self).__init__()
        self.nonlinearity = nonlinearity
        # Initializing as ones to maintain structure
        self.W = torch.nn.Parameter(torch.ones(input_dim))
        self.W.requires_grad = True

    def forward(self, data):
        elementwise_multiply = torch.mul(self.W, data)
        nl = self.nonlinearity(elementwise_multiply)
        scaled_data = torch.mul(nl, data)
        return scaled_data


