"""
modules.model
=============

The neural networks models and supporting functions.
"""

import math
import os
from typing import List, Tuple
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import networks.domain_adaptation_schedules


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None:
                continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + "_bn_" + k
            else:
                name = prefix + "_" + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: {:s}".format(name))


def make_optimizer(
    model, model_Adnet, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005
):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({"params": [p], "lr": lr})

    if model_Adnet is not None:
        for name, par in model_Adnet.named_parameters():
            if par.requires_grad:
                param_list.append({"params": [par], "lr": 0.001})

    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Sequential(
                            nn.Conv2d(3, 96, kernel_size=7, stride=2),
                            nn.ReLU(inplace=True),
                            nn.LocalResponseNorm(2),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv2",
                        nn.Sequential(
                            nn.Conv2d(96, 256, kernel_size=5, stride=2),
                            nn.ReLU(inplace=True),
                            nn.LocalResponseNorm(2),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        ),
                    ),
                    (
                        "conv3",
                        nn.Sequential(
                            nn.Conv2d(256, 512, kernel_size=3, stride=1),
                            nn.ReLU(inplace=True),
                        ),
                    ),
                    (
                        "fc4",
                        nn.Sequential(
                            nn.Linear(512 * 3 * 3, 512), nn.ReLU(inplace=True)
                        ),
                    ),
                    (
                        "fc5",
                        nn.Sequential(
                            nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(inplace=True)
                        ),
                    ),
                ]
            )
        )

        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(K)]
        )

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == ".pth":
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == ".mat":
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: {:s}".format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, "fc6_{:d}".format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer="conv1", out_layer="fc6"):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == "conv3":
                    x = x.reshape(x.size(0), -1)
                    # features = x
                if name == out_layer:
                    return x.reshape(x.size(0), -1)  # , features

        x = self.branches[k](x)
        if out_layer == "fc6":
            return x  # , features
        elif out_layer == "fc6_softmax":
            return F.softmax(x, dim=1)  # , features

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states["shared_layers"]
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat["layers"])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]["weights"].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(
                np.transpose(weight, (3, 2, 0, 1))
            )
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        # pos_loss_p = (torch.ones(pos_loss.size()).cuda() - F.softmax(pos_score, dim=1)[:,1]) * pos_loss
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
        # neg_loss_p = (torch.ones(neg_loss.size()).cuda() - F.softmax(neg_score, dim=1)[:,0]) * neg_loss

        # loss = pos_loss_p.sum() + neg_loss_p.sum()
        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= pos_loss.size(0) + neg_loss.size(0)
            # loss /= (pos_loss_p.size(0) + neg_loss_p.size(0))
        return loss


class Accuracy:
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (
            pos_score.size(0) + neg_score.size(0) + 1e-8
        )
        return acc.item()


class Precision:
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()


class DomainAdaptationNetwork(nn.Module):
    def __init__(self, schedule: str, direction: str, **schedule_parameters):
        super().__init__()
        self.ad_layer1 = nn.Linear(4608, 256)
        self.ad_layer2 = nn.Linear(256, 9)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        networks.domain_adaptation_schedules.set_schedule(
            schedule, direction, **schedule_parameters
        )

    def forward(self, x):
        x = self.dropout(x)
        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ad_layer2(x)
        x = self.sigmoid(x)
        x = networks.domain_adaptation_schedules.grl(x)
        return x

    # def train(self, mode=True):
    #     super().train(mode)
    #     if mode:
    #         networks.domain_adaptation_schedules.GradientReverseLayer.reset()


# ==================================================================================================
#                                                                          New code, already clean.
# ==================================================================================================
def make_networks(
    model_path: str,
    use_gpu: bool,
    cnn_learnable_parameters: List[str],
    schedule: str,
    **schedule_parameters
) -> Tuple[MDNet, DomainAdaptationNetwork]:
    """
    Construct the CNN and domain adaptation networks.

    :param str model_path: The path to the trained CNN weights on disk.
    :param bool use_gpu: ``True`` indicates to transfer the networks to GPU memory. ``False``
        indicates to leave the networks in main memory.
    :param List[str] cnn_learnable_parameters: The list of layers which can be updated when training
        the CNN.
    :return: The initialized CNN and domain adaptation network.
    :rtype: Tuple[MDNet, DomainAdaptationNetwork]
    """
    cnn = MDNet(model_path)
    domain_network = DomainAdaptationNetwork(schedule, **schedule_parameters)
    if use_gpu:
        cnn = cnn.cuda()
        domain_network = domain_network.cuda()
    cnn.set_learnable_params(cnn_learnable_parameters)
    return cnn, domain_network
