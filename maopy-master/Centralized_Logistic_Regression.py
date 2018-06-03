# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:38:01 2018

@author: zhangjiaqi
"""
import numpy as np
import scipy.io as sio 
from LogReg import LogisticRegression

filepath = './data/'
samples = []
labels = []
for i in range(5):
    data = sio.loadmat(filepath+str(i)+'.mat')
    samples = np.append(samples, data['samples'].ravel(order = 'F'))
    labels = np.append(labels, data['labels'].ravel(order = 'F'))
    
samples = samples.reshape((data['samples'].shape[0],-1), order = 'F')
labels = labels.reshape((data['labels'].shape[0],-1), order = 'F')

lr = LogisticRegression(samples = samples, labels = labels)
est = lr.minimizer(x_start = None, step_size = 2, max_ite = 1000)
print(str(est))



#est_asy = data['estimate'][-1,:,:]
#grad_asy = lr.gradient(est_asy)
#grad_asy_norm = np.linalg.norm(grad_asy)
