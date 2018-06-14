# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:46:39 2018

@author: zhangjiaqi
"""

import numpy as np
import scipy.io as sio 
from logistic_regression import LogisticRegression

filepath = './dataset_covtype/'
samples = []
labels = []
for i in range(1):
    data = sio.loadmat(filepath+str(i)+'.mat')
    samples = np.append(samples, data['samples'].ravel(order = 'F'))
    labels = np.append(labels, data['labels'].ravel(order = 'F'))


samples = samples.reshape((data['samples'].shape[0],-1), order = 'F')
labels = labels.reshape((data['labels'].shape[0],-1), order = 'F')
lr = LogisticRegression(samples = samples, labels = labels)

filepath = './data/higher_connectivity_nonexact_8_undirected/'
data_of_all_agents = {}

# evenly select m elements from a list of length n
evenly_select = lambda m, n: np.rint( np.linspace( 1, n, min(m,n) ) - 1 ).astype(int)
for i in range(7):
    try:
        data = sio.loadmat(filepath+str(i)+'.mat')
    except IOError:
        print(filepath+str(i)+'.mat','does not exist.', flush = True)
        continue
    est_nonexact = data['estimate']
    indices = evenly_select(200, est_nonexact.shape[0])
    est_nonexact = est_nonexact[indices]
    t = data['time'][0][indices]
    f_value = [lr.obj_func(j) for j in est_nonexact]
    grad_value = [lr.gradient(j) for j in est_nonexact]
#    grad_value = []
    data_of_all_agents['Agent '+str(i)] = {'est':est_nonexact, 'time':t, 
                                    'f_value': f_value, 'grad_value': grad_value}

sio.savemat(filepath+'plotdata.mat', data_of_all_agents)