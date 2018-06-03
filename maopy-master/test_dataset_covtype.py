# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:43:27 2018

@author: zhangjiaqi
"""

import numpy as np
import scipy.io as sio
#from LogReg import LogisticRegression
#from push_sum_gossip_asy_gradient import Push_Sum_Asy_Gradient

class dataset_covtype:
    
    def __init__(self, filepath = './dataset_covtype/covtype.csv'):
        """ Initialize the dataset """
        
        if 'data_cov' not in globals():
            data_cov = np.genfromtxt(filepath, delimiter=',')
        
        self.samples = data_cov.T[:-1,:] 
        labels_vec = data_cov.T[-1,:]
        self.labels = self.vec2onehot(labels_vec)   
        
        self.n_s = self.samples.shape[1]
        self.n_f = self.samples.shape[0]
        self.n_c = self.labels[0]

    def vec2onehot(self, vec):
        """ Transform the integer class vector into one-hot vector """
        
        vec = np.asarray(vec,dtype = np.int) - int(min(vec))
        num_classes = max(vec) + 1
        num_instances = len(vec)
        onehot = np.zeros((num_classes, num_instances))
        onehot[vec, range(num_instances)] = 1
        return onehot
        
    def data_preprocess(self):
        """ Pre-process the covtype dataset """
        
        # Compute the mean and standard deviation of non-categorical features
        mean_num = np.mean(self.samples[:10,:], axis = 1).reshape(-1,1)
        std_num = np.std(self.samples[:10,:], axis = 1).reshape(-1,1)
        
        # Nomalization the data and append a constant to each instance
        self.samples[:10,:] = (self.samples[:10,:] - mean_num) / std_num
        self.samples = np.vstack((self.samples, np.ones((1,self.n_s))))
    
    def data_distribute(self, n = 5, folderpath = 'dataset_covtype/', filename = ''):
        """ Divide the data into n subset """
        
        per = np.random.permutation(self.n_s)
        samples = self.samples[:,per]
        labels = self.labels[:,per]
        samples_subset = np.array_split(samples, n, axis = 1)
        labels_subset = np.array_split(labels, n, axis = 1)
        for i in range(n):
            filepath = folderpath + filename + str(i) + '.mat'
            mdict = {'samples': samples_subset[i], 'labels': labels_subset[i]}
            sio.savemat(filepath, mdict)
    
if __name__ == '__main__':
    
    covtype = dataset_covtype()
    covtype.data_preprocess()
    covtype.data_distribute()

    