import numpy as np
from numpy.linalg import norm
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn
import torch.nn.functional as F

def compute_admittance_mask(bus_num):
    Ybus_filename = '../data/IEEE'+str(bus_num)+'/case'+str(bus_num)+'_Ybus.csv'
    Ybus = np.loadtxt(Ybus_filename, delimiter=',', dtype=np.complex)
    g = Ybus.real
    b = Ybus.imag
    adjacent_matrix = np.zeros(g.shape)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i, j] != 0 or b[i, j] != 0:
                adjacent_matrix[i, j] = 1
    return adjacent_matrix, g, b
