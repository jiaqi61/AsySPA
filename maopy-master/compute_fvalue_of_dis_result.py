# -*- coding: utf-8 -*-
"""
Compute the value of global objective function of each agent.

A single agent has no access to the global objective function, and thus we can only
compute the objective function value and errors after finishing the algorithm.

Created on Fri Jun  8 15:46:39 2018

@author: zhangjiaqi
"""
import numpy as np
import scipy.io as sio 
from logistic_regression import LogisticRegression

# get the dataset
filepath = './dataset_covtype/'
data = sio.loadmat(filepath+'covtype.mat')
samples = data['samples']
labels = data['labels']

# initialize the logistic regression problem
lr = LogisticRegression(samples = samples, labels = labels)

# get the data of distributed algorithms
filepath = './data/3_dis_result_exact_conv_constant_step_0.1_same_freq_nodelay/'
data_of_all_agents = {}

# evenly select m elements from a list of length n
evenly_select = lambda m, n: np.rint( np.linspace( 1, n, min(m,n) ) - 1 ).astype(int)
for i in range(3): # data of 5 agents
    try:
        data = sio.loadmat(filepath+str(i)+'.mat')
    except IOError:
        print(filepath+str(i)+'.mat','does not exist.', flush = True) 
        continue
    est_nonexact = data['estimate'] # historical states of an agent 
    indices = evenly_select(200, est_nonexact.shape[0]) # only choose 200 data points to save time
    est_nonexact = est_nonexact[indices]
    t = data['time'][0][indices]    # time of the data
    f_value = [lr.obj_func(j) for j in est_nonexact] # compute the objective value
    grad_value = [lr.gradient(j) for j in est_nonexact] # compute the gradient
    data_of_all_agents['Agent '+str(i)] = {'est':est_nonexact, 'time':t, 
                                    'f_value': f_value, 'grad_value': grad_value}
# save the data
sio.savemat(filepath+'plotdata.mat', data_of_all_agents)