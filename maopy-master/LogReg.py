# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:24:25 2018

@author: zhangjiaqi
"""

import numpy as np

class LogisticRegression:
    """ Objective function and gradient of multi-classes logistic regression problem. """
    
    def __init__(self, samples, labels, reg = 1 ,dtype = np.float64):
        """ 
        Initialize the problem with provided data. 
        
        Parameters
        ----------
        samples: 2-D array. Each column is an instance. The number of columns
                 is the number of samples, and the number of rows is the number
                 of features.
        labels: 2-D array. Each column is an 1-hot vector of each sample. The
                number of columns is the number of samples, and the number of 
                rows is the number of classes.
        reg: Positive scalar. Regularization factor
        dtype: dtype, optional. The type of the data. Default: np.float32 
        """
        assert samples.ndim == 2 and labels.ndim ==2, "Samples and labels should be 2D arrays"
        self.dtype = dtype
        self.samples = np.asarray(samples, dtype = dtype)
        self.labels = np.asarray(labels, dtype = np.int)
        (self.n_f, self.n_s) = self.samples.shape # numbers of features and samples
        self.n_c = self.labels.shape[0] # numbers of classes
        assert self.n_s == self.labels.shape[1], "The number of samples doesn't match"
        if reg < 0:
            reg = 1
        self.reg = reg
        
    
    def obj_func(self, x):
        """ 
        The objective function of logistic regression. 
        
        Parameters
        ----------
        x: 2-D array. The weights matrix to be estimated with 
           size (num_features, num_classes)
        """
        lr = 0 # logistic regression term
        for i in range(self.n_s):
            exp_l = [np.exp(x[:,j].dot(self.samples[:,i])) for j in range(self.n_c)]
            temp  = np.log((np.dot(exp_l, self.labels[:,i]) / np.sum(exp_l))) # temp variable
            lr += temp
        
        r = 0.5 * self.reg * np.linalg.norm(x, ord='fro') ** 2 # regularization term
        
        return (-lr + r)
    
    def gradient(self, x):
        """
        The gradient of the objective function w.r.t x and y.
        
        Parameters
        ----------
        x: 2-D array. The weights matrix to be estimated with 
           size (num_features, num_classes)
        """
        grad_lr = 0 # the gradient of the logistic regression term
        for i in range(self.n_s):
            # the denominator
            
#            x_s = [x[:,j].dot(self.samples[:,i]) for j in range(self.n_c)]
#            x_s = x_s - max(x_s) + 1 # To avoid overflow
#            exp_l = np.exp(x_s) + 1e-6 # To avoid underflow
            
            exp_l = [np.exp(x[:,j].dot(self.samples[:,i])) for j in range(self.n_c)]
            
            grad_l = np.outer(self.samples[:,i], self.labels[:,i]) - \
                     np.outer(self.samples[:,i], exp_l / np.sum(exp_l))
            grad_lr += grad_l
        
        grad_r = self.reg * x # the gradient of the regularization term
        
        return (-grad_lr + grad_r)
    
    def minimizer(self, x_start = None, step_size = -1, max_ite = 1000, epi = 1e-3):
        """
        Minimize the logistic regression problem using a decaying stepsize
        
        Parameters
        ----------
        x_start: 2D array with size (num_features, num_classes). Initial point of x.
        step_size: Positive scalar. The initial stepsize.
        max_ite: Positive integer. The max number of iterations.
        epi: Positive scalar. Algorithm stop is the norm of gradient is smaller then epi
        """
        if x_start is None:
            x_start = np.random.randn(self.n_f, self.n_c)
        if step_size == -1:
            step_size = 1 / self.n_s # more instances, lager gradient, and thus smaller stepsize
        else:
            step_size = step_size / self.n_s
        x = x_start
        for k in range(max_ite):
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            x = x - step_size / np.sqrt(k+1) * grad
            
            obj = self.obj_func(x)
            print('k='+str(k),'func='+str(obj),'grad='+str(grad_norm))
            
            if grad_norm < epi * self.n_s:
                break
        return x
    
if __name__ == "__main__":
    
    def demo(num_instances, num_features, num_classes):
        """
        Run a demo
        """
        # Create datasets
        labels = np.zeros((num_classes, num_instances))
        label_vec = np.random.randint(low = 0, high = num_classes, size = num_instances)
        labels[label_vec, range(num_instances)] = 1
        samples = np.random.randn(num_features, num_instances)
        
        # Initialize the problem
        lr = LogisticRegression(samples = samples, labels = labels)
        
        # Solve the problem
        x = lr.minimizer()

        print(str(x))
        return x
        
    x = demo(4000, 10, 3)
        