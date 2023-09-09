import os
import json
import pickle
import torch
import torch.nn.functional as F
import time
import numpy as np
from os.path import join
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from power_flow_equations import compute_admittance_mask
from dataprepoc import load_data, PowerFlowDataset

# neural networks for forward PF computing
# MLP forward PF computing
# [u],[w] --> [p],[q]
class Forward_MLP(nn.Module):
    def __init__(self, n_x, n_bus, device):
        super(Forward_MLP, self).__init__()
        self.n_bus = n_bus
        self.device = device
        # the hidden size
        n_rbf = 2*n_x
        # the MLP hidden layer
        self.linear_h1_p = nn.Linear(n_x, n_rbf, bias=True)  # code->h2' the output layer
        self.linear_h2_p = nn.Linear(n_rbf, n_rbf, bias=True)  # h2'->h' the hidden layer of multi-layer perceptron
        self.linear_h3_p = nn.Linear(n_rbf, n_bus, bias=True)  # h'->x', the hidden layer of multi-layer perceptron
        self.linear_h1_q = nn.Linear(n_x, n_rbf, bias=True)  # code->h2' the output layer
        self.linear_h2_q = nn.Linear(n_rbf, n_rbf, bias=True)  # h2'->h' the hidden layer of multi-layer perceptron
        self.linear_h3_q = nn.Linear(n_rbf, n_bus, bias=True)  # h'->x', the hidden layer of multi-layer perceptron


    def mlp(self, x):
        z = self.linear_h1_p(x)
        C1 = nn.ReLU(z)
        # C1 = F.dropout(C1, p=0.2, training=self.training)
        z2 = self.linear_h2_p(C1)
        C2 = nn.ReLU(z2)
        # C2 = F.dropout(C2, p=0.2, training=self.training)
        p = self.linear_h3_p(C2)

        z3 = self.linear_h1_q(x)
        C3 = nn.ReLU(z3)
        # C3 = F.dropout(C3, p=0.2, training=self.training)
        z4 = self.linear_h2_q(C3)
        C4 = nn.ReLU(z4)
        # C4 = F.dropout(C4, p=0.2, training=self.training)
        q = self.linear_h3_q(C4)
        return p, q

    def forward(self, u, w):
        p, q = self.mlp(torch.cat((u, w), 1))
        return p, q

# BNN forward PF computing
# Bilinear layer xAX.
# [u],[w] --> [p],[q]
class Forward_xTx(nn.Module):
    def __init__(self, n_bus, device):
        super(Forward_xTx, self).__init__()
        self.bus_num = n_bus
        self.device = device

        # self.G = nn.Parameter(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float())
        # self.B = nn.Parameter(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float())
        self.G = nn.Parameter(torch.rand(n_bus, n_bus).float())
        self.B = nn.Parameter(torch.rand(n_bus, n_bus).float())
        # self.bias_p = nn.Parameter(torch.empty(1, n_bus).uniform_(-np.sqrt(1/n_bus), np.sqrt(1/n_bus)))
        # self.bias_q = nn.Parameter(torch.empty(1, n_bus).uniform_(-np.sqrt(1/n_bus), np.sqrt(1/n_bus)))
        self.bias_p = nn.Parameter(torch.rand(1, self.bus_num))
        self.bias_q = nn.Parameter(torch.rand(1, self.bus_num))
        self.linear_y_p_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->p, the output layer
        self.linear_y_q_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->q, the output layer

    def decoder(self, u, w):
        n_input = u.size(0)
        u1 = u.view(n_input,1,-1)
        ut = u.view(n_input,-1,1)
        w1 = w.view(n_input, 1, -1)
        wt = w.view(n_input, -1, 1)

        wiwk = torch.mul(wt, w1)
        uiuk = torch.mul(ut, u1)
        wiuk = torch.mul(wt, u1)
        uiwk = torch.mul(ut, w1)
        p = torch.mul(uiuk + wiwk, self.G) + torch.mul(wiuk - uiwk, self.B)
        p = torch.sum(p, 2)
        q = torch.mul(wiuk - uiwk, self.G) - torch.mul(uiuk + wiwk, self.B)
        q = torch.sum(q, 2)
        p = p + self.bias_p
        q = q + self.bias_q
        p = self.linear_y_p_sup(p)
        q = self.linear_y_q_sup(q)
        return p,q

    def forward(self, u, w):
        p, q = self.decoder(u, w)
        return p, q


# TPBNN forward PF computing
# Topology-pruned Bilinear layer xAX.
# [u],[w] --> [p],[q]
class Forward_xTx_topo(nn.Module):
    def __init__(self, n_bus, device, Auw=None):
        super(Forward_xTx_topo, self).__init__()
        self.bus_num = n_bus
        self.device = device

        self.G = nn.Parameter(torch.mul(torch.rand(n_bus, n_bus).float(), torch.from_numpy(Auw).float()))
        self.B = nn.Parameter(torch.mul(torch.rand(n_bus, n_bus).float(), torch.from_numpy(Auw).float()))
        self.Auw = torch.from_numpy(Auw).to(device=self.device, dtype=torch.float)  # adjacent matrix
        self.bias_p = nn.Parameter(torch.rand(1, self.bus_num))
        self.bias_q = nn.Parameter(torch.rand(1, self.bus_num))
        self.linear_y_p_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->p, the output layer
        self.linear_y_q_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->q, the output layer

    def decoder(self, u, w):
        n_input = u.size(0)
        u1 = u.view(n_input,1,-1)
        ut = u.view(n_input,-1,1)
        w1 = w.view(n_input, 1, -1)
        wt = w.view(n_input, -1, 1)

        wiwk = torch.mul(wt, w1)
        uiuk = torch.mul(ut, u1)
        wiuk = torch.mul(wt, u1)
        uiwk = torch.mul(ut, w1)
        p = torch.mul(uiuk + wiwk, self.G) + torch.mul(wiuk - uiwk, self.B)
        p = torch.mul(p, self.Auw)
        p = torch.sum(p, 2)
        q = torch.mul(wiuk - uiwk, self.G) - torch.mul(uiuk + wiwk, self.B)
        q = torch.mul(q, self.Auw)
        q = torch.sum(q, 2)
        p = p + self.bias_p
        q = q + self.bias_q
        p = self.linear_y_p_sup(p)
        q = self.linear_y_q_sup(q)
        return p,q

    def forward(self, u, w):
        p, q = self.decoder(u, w)
        return p, q
