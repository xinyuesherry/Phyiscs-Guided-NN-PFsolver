import os

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

if not os.path.exists('./AE_out'):
    os.mkdir('./AE_out')

# MLP+MLP
# [PF input] --> [u,w] --> [p],[q]
class AutoEncoder(nn.Module):
    def __init__(self, n_x, n_rbf, n_bus, device, supervised):
        super(AutoEncoder, self).__init__()
        self.bus_num = n_bus
        self.device = device
        self.supervise_signal = supervised

        # the MLP hidden layer
        self.linear_h = nn.Linear(n_x, n_rbf, bias=True)  # x->h1, the hidden layer of multi-layer perceptron
        self.linear_h2 = nn.Linear(n_rbf, n_rbf, bias=True)  # h1->h2
        self.linear_y = nn.Linear(n_rbf, n_bus * 2, bias=True)  # h2->code, the output layer
        # supervised AE in NIPS 2018
        # add additional layer upon code layer
        # self.linear_y_u_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->u, the output layer
        # self.linear_y_w_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->w, the output layer

        # the MLP hidden layer
        self.linear_h1_p = nn.Linear(n_bus * 2, n_bus * 4, bias=True)  # code->h2' the output layer
        self.linear_h2_p = nn.Linear(n_bus * 4, n_bus * 4, bias=True)  # h2'->h' the hidden layer of multi-layer perceptron
        self.linear_h3_p = nn.Linear(n_bus * 4, n_bus, bias=True)  # h'->x', the hidden layer of multi-layer perceptron
        self.linear_h1_q = nn.Linear(n_bus * 2, n_bus * 4, bias=True)  # code->h2' the output layer
        self.linear_h2_q = nn.Linear(n_bus * 4, n_bus * 4, bias=True)  # h2'->h' the hidden layer of multi-layer perceptron
        self.linear_h3_q = nn.Linear(n_bus * 4, n_bus, bias=True)  # h'->x', the hidden layer of multi-layer perceptron

    def encoder(self, x):
        z1 = self.linear_h(x)
        C1 = nn.ReLU(z1)
        # C1 = F.dropout(C1, p=0.2, training=self.training)
        z2 = self.linear_h2(C1)
        C2 = nn.ReLU(z2)
        # C2 = F.dropout(C2, p=0.2, training=self.training)
        uw = self.linear_y(C2)

        # y_u = self.linear_y_u_sup(code_u)
        # y_w = self.linear_y_w_sup(code_w)
        return uw

    def decoder(self, x):
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

    def forward(self, x):
        uw = self.encoder(x)
        '''if self.supervise_signal:
            u = uw[:, 0:self.bus_num]
            w = uw[:, self.bus_num:]
        else:
            u = uw[:, 0:self.bus_num]
            w = uw[:, self.bus_num:]'''
        p, q = self.decoder(uw)
        return uw, p, q

# MLP+BNN/TPBNN
# Bilinear layer xAX.
# [PF input] --> [u,w] --> [p],[q]
class AutoEncoder_BNN(nn.Module):
    def __init__(self, n_x, n_rbf, n_bus, device, sup_signal, topology, Auw=None):
        super(AutoEncoder_BNN, self).__init__()
        self.bus_num = n_bus
        self.device = device
        self.sup_signal = sup_signal
        self.topology_flag = topology

        # the MLP hidden layer
        self.linear_h = nn.Linear(n_x, n_rbf, bias=True)  # x->h1, the hidden layer of multi-layer perceptron
        self.linear_h2 = nn.Linear(n_rbf, n_rbf, bias=True)  # h1->h2
        self.linear_y = nn.Linear(n_rbf, n_bus * 2, bias=True)  # h2->code, the output layer
        # supervised AE in NIPS 2018
        # add additional layer upon code layer
        #self.linear_y_u_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->u, the output layer
        #self.linear_y_w_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->w, the output layer

        if not topology:
            # self.G = nn.Parameter(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float())
            # self.B = nn.Parameter(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float())
            self.G = nn.Parameter(torch.rand(n_bus, n_bus).float())
            self.B = nn.Parameter(torch.rand(n_bus, n_bus).float())
        else:
            # self.G = nn.Parameter(
            #    torch.mul(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float(),
            #              torch.from_numpy(Auw).float()))
            # self.B = nn.Parameter(
            #    torch.mul(torch.empty(n_bus, n_bus).uniform_(-np.sqrt(1 / n_bus), np.sqrt(1 / n_bus)).float(),
            #              torch.from_numpy(Auw).float()))
            self.G = nn.Parameter(torch.mul(torch.rand(n_bus, n_bus).float(), torch.from_numpy(Auw).float()))
            self.B = nn.Parameter(torch.mul(torch.rand(n_bus, n_bus).float(), torch.from_numpy(Auw).float()))
            self.Auw = torch.from_numpy(Auw).to(device=self.device, dtype=torch.float)  # adjacent matrix

        self.bias_p = nn.Parameter(torch.zeros(1, self.bus_num))
        self.bias_q = nn.Parameter(torch.zeros(1, self.bus_num))
        self.linear_y_p_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->p, the output layer
        self.linear_y_q_sup = nn.Linear(n_bus, n_bus, bias=True)  # code->q, the output layer

    def encoder(self, x):
        z1 = self.linear_h(x)
        C1 = nn.ReLU(z1)
        # C1 = F.dropout(C1, p=0.2, training=self.training)
        z2 = self.linear_h2(C1)
        C2 = nn.ReLU(z2)
        # C2 = F.dropout(C2, p=0.2, training=self.training)
        uw = self.linear_y(C2)
        # y_u = self.linear_y_u_sup(code_u)
        # y_w = self.linear_y_w_sup(code_w)

        return uw

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
        if self.topology_flag:
            p = torch.mul(p, self.Auw)
        p = torch.sum(p, 2)
        q = torch.mul(wiuk - uiwk, self.G) - torch.mul(uiuk + wiwk, self.B)
        if self.topology_flag:
            q = torch.mul(q, self.Auw)
        q = torch.sum(q, 2)
        p = p + self.bias_p
        q = q + self.bias_q
        p = self.linear_y_p_sup(p)
        q = self.linear_y_q_sup(q)
        return p,q

    def forward(self, x):
        uw = self.encoder(x)
        p, q = self.decoder(uw[:, 0:self.bus_num], uw[:, self.bus_num:])
        return uw, p, q

