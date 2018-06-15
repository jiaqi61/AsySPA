# -*- coding: utf-8 -*-
"""
Proprocess the covetype dataset, including transform the original .csv file to a .mat file,
whiten the features, and divide the dataset into several chunks. The covtype data is available
from 'https://archive.ics.uci.edu/ml/datasets/covertype'

Created on Fri Jun  1 16:43:27 2018

@author: zhangjiaqi
"""

import numpy as np
import scipy.io as sio


class dataset_covtype:
    
    def __init__(self, filepath = './dataset_covtype/covtype.csv'):
        """ Initialize the dataset """
        
        data_cov = np.genfromtxt(filepath, delimiter=',')
        
        self.samples = data_cov.T[:-1,:] 
        self.labels_vec = data_cov.T[-1,:]
        self.labels = self.vec2onehot(self.labels_vec)   
        
        self.n_s = self.samples.shape[1] # number of samples
        self.n_f = self.samples.shape[0] # number of features
        self.n_c = self.labels[0] # number of classer

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
        
        # Compute the mean and standard deviation of non-categorical features (the first 10 features)
        mean_num = np.mean(self.samples[:10,:], axis = 1).reshape(-1,1)
        std_num = np.std(self.samples[:10,:], axis = 1).reshape(-1,1)
        
        # Nomalization the data by subtracting the mean and dividing by the standard deviation
        # of non-categorical features, and append a constant to each instance
        self.samples[:10,:] = (self.samples[:10,:] - mean_num) / std_num
        self.samples = np.vstack((self.samples, np.ones((1,self.n_s))))
    
    def data_distribute(self, n, folderpath = 'dataset_covtype/', filename = ''):
        """ Divide the data into n chunks """
        
        indices = self.labels_vec.argsort()[::-1]  # sort the labels in descending order    
#        indices = np.random.permutation(self.n_s) # random permute the data
        samples = self.samples[:,indices]
        labels = self.labels[:,indices]
        samples_subset = np.array_split(samples, n, axis = 1)
        labels_subset = np.array_split(labels, n, axis = 1)
        for i in range(n):
            filepath = folderpath + filename + str(i) + '.mat'
            mdict = {'samples': samples_subset[i], 'labels': labels_subset[i]}
            sio.savemat(filepath, mdict) # save data in a .mat file
    
if __name__ == '__main__':
    
    covtype = dataset_covtype(filepath = './dataset_covtype/covtype.csv')
    covtype.data_preprocess()
    # divide the dataset into 5 parts
    covtype.data_distribute(n = 5, folderpath = 'dataset_covtype/')

    