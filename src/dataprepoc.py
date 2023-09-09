from __future__ import print_function

import os
import json
import random
import numpy as np
from os.path import join
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from itertools import product, permutations, combinations
from torch.utils.data import Dataset, DataLoader
import copy

class PowerFlowDataset(Dataset):
    def __init__(self, datasets):
        self.pq_known = datasets['pq_known']
        self.pq = datasets['pq']
        self.va_known, self.va = datasets['va_known'], datasets['va']
        self.all_known = datasets['all_known']
        self.uw = datasets['uw']
        self.p, self.q = datasets['p'], datasets['q']
        self.u, self.w = datasets['u'], datasets['w']
        self.X_dim = self.all_known.shape[1]
        self.Y_dim = self.uw.shape[1]

    def __len__(self):
        return self.pq.shape[0]

    def __getitem__(self, idx):
        # (pq, uw, all_known)
        sample = (self.pq[idx, :], self.uw[idx, :], self.all_known[idx, :])
        return sample


def generate_data(raw_data, bustype):
    """
    :param raw_data: dataset dict, 'P' -> [P], 'Q' -> [Q], etc.
    :param bustype: dict, 'pq'->[pq bus index], 'pv'->[pv bus index], 'ref'->ref bus index

    :return: data_set: dictionary
    """
    P, Q, V, A = raw_data['P'], raw_data['Q'], raw_data['V'], raw_data['A']
    U, W = raw_data['U'], raw_data['W']
    pq = np.concatenate((P, Q), axis=1)
    va = np.concatenate((V, A), axis=1)
    uw = np.concatenate((U, W), axis=1)

    pq_index = bustype['pq']
    pv_index = bustype['pv']
    ref_index = bustype['ref']  # int
    print('generating pq_known, va_known:')

    # the known PQ according to bus type
    P_known = np.delete(P, [ref_index], 1)
    Q_known = np.delete(Q, pv_index + [ref_index], 1)
    pq_known = np.concatenate((P_known, Q_known), axis=1)  # naaray
    # the known VA according to bus type
    V_known = np.delete(V, pq_index, 1)
    A_known = np.delete(A, pq_index + pv_index, 1)
    va_known = np.concatenate((V_known, A_known), axis=1)  # naaray

    data_set = {}
    data_set['p'], data_set['q'] = P, Q
    data_set['u'], data_set['w'] = U, W
    data_set['pq_known'], data_set['pq'] = pq_known, pq
    data_set['va_known'], data_set['va'] = va_known, va
    data_set['uw'] = uw
    data_set['all_known'] = np.concatenate((pq_known, va_known), axis=1)
    return data_set
