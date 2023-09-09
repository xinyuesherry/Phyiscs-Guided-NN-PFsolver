import os
import json
import pickle
from comet_ml import Experiment
import torch
import torch.nn.functional as F
import time
import numpy as np
from os.path import join
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# MLPNN to solve PF equations
# [PF input] -> [u, w]
class MLPNN(nn.Module):
    def __init__(self, n_x, n_y, n_bus, device):
        super(MLPNN, self).__init__()
        self.n_bus = n_bus
        # the hidden size
        n_rbf = 2*n_x
        # the shared hidden layers
        self.linear_h = nn.Linear(n_x, n_rbf, bias=True)  # x->h1, the hidden layer of multi-layer perceptron
        self.linear_h2 = nn.Linear(n_rbf, n_rbf, bias=True)  # h1->h2
        self.linear_y = nn.Linear(n_rbf, n_bus * 2, bias=True)  # h2->code, the output layer

    def mlp(self, x):
        z1 = self.linear_h(x)
        C1 = nn.ReLU(z1)
        # C1 = F.dropout(C1, p=0.2, training=self.training)
        z2 = self.linear_h2(C1)
        C2 = nn.ReLU(z2)
        # C2 = F.dropout(C2, p=0.2, training=self.training)
        uw = self.linear_y(C2)

        return uw

    def forward(self, x):
        uw = self.mlp(x)
        return uw
