# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:24:25 2018

@author: zhangjiaqi
"""
import time
import numpy as np

class LogisticRegression:
    """ Multi-classes logistic regression problem. """
    
    def __init__(self, samples, labels, reg = 1 ,dtype = np.float64):
        """ 
        Initialize the problem with given data. 
        
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
        The objective function to be minimized. 
        
        Parameters
        ----------
        x: 2-D array. The weights matrix to be estimated with 
           size (num_features, num_classes)
        """
        """
        Version 1: Very slow
        lr = 0 # logistic regression term
        for i in range(self.n_s):
            exp_l = [np.exp(x[:,j].dot(self.samples[:,i])) for j in range(self.n_c)]
            temp  = np.log((np.dot(exp_l, self.labels[:,i]) / np.sum(exp_l))) # temp variable
            lr += temp
        """
        num = np.exp(x.T.dot(self.samples)) # numerator
        den = np.sum(num, axis = 0) # denominator
        lr = np.sum(np.log(num.T[self.labels.astype(bool).T] / den))
        
        r = 0.5 * self.reg * np.linalg.norm(x, ord='fro') ** 2 # regularization term
        
        return (-lr + r)
    
    def gradient(self, x):
        """
        The gradient of the objective function w.r.t x.
        
        Parameters
        ----------
        x: 2-D array. The weights matrix to be estimated with 
           size (num_features, num_classes)
        """
        """
        Version 1: Very slow
        grad_lr = 0 # the gradient of the logistic regression term
        for i in range(self.n_s):
            # the denominator   
            exp_l = [np.exp(x[:,j].dot(self.samples[:,i])) for j in range(self.n_c)]            
            grad_l = np.outer(self.samples[:,i], self.labels[:,i]) - \
                     np.outer(self.samples[:,i], exp_l / np.sum(exp_l))
            grad_lr += grad_l
        """
        """
        Version 2: Encounter underflow in division
        temp = np.exp(x.T.dot(self.samples)).T # avoid underflow
        grad_lr = self.samples.dot(self.labels.T - temp / np.sum(temp, axis = 1).reshape(-1,1))
        """
        """
        Version 3
        temp = np.exp(x.T.dot(self.samples)).T 
        temp2 = temp * np.exp(-np.log(np.sum(temp, axis = 1).reshape(-1,1)))
        grad_lr = self.samples.dot(self.labels.T - temp2)
        """
        temp2  = x.T.dot(self.samples)
        temp2 = temp2 - np.max(temp2)
        temp = np.exp(temp2).T  # avoid underflow
        grad_lr = self.samples.dot(self.labels.T - temp / np.sum(temp, axis = 1).reshape(-1,1))
      
        grad_r = self.reg * x # the gradient of the regularization term
        
        return (-grad_lr + grad_r)
    
    def minimizer(self, x_start = None, 
                  step_size = -1, 
                  terminate_by_time = False, 
                  termination_condition=1000,
                  epi = 1e-3,
                  log = False, 
                  constant_stepsize = False):
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
            step_size = 1 / self.n_s # more instances, smaller stepsize
        else:
            step_size = step_size / self.n_s
        x = x_start
        x_history = np.asarray([x])
        t = 0
        itr = 0
        t_start = time.time()
        condition = True
        while(condition):
            itr += 1
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad) / self.n_s
            if constant_stepsize is False:
                x = x - step_size / (itr ** 0.5) * grad # Use the 1/sqrt(k) stepsize
            else:
                x = x - step_size * grad
            # log the estimates
            if log is True:
                x_history = np.concatenate((x_history, [x]))
                t = np.append(t, time.time() - t_start)
            
            # print the averaged value of objective function and gradient
            if itr % 20 == 0:
                obj = self.obj_func(x)
                print('k='+str(itr),'func='+str(obj / self.n_s),'grad='+str(grad_norm),
                      flush = True)
            
            if terminate_by_time is True:
                condition = t[-1] < termination_condition
            else:
                condition = itr < termination_condition   
            condition = condition and grad_norm > epi
            
        return (x, x_history, t)
    
if __name__ == "__main__":
    
    def demo(num_instances, num_features, num_classes):
        """
        Run a demo
        """
        # Create datasets
        labels = np.zeros((num_classes, num_instances))
        label_vec = np.random.randint(low = 0, high = num_classes, size = num_instances)
        labels[label_vec, range(num_instances)] = 1 # one-hot labels

        samples = np.random.randn(num_features, num_instances)
        
        # Initialize the problem
        lr = LogisticRegression(samples = samples, labels = labels)
        
        # Solve the problem
        x = lr.minimizer()

        print(str(x))
        return x
        
    x = demo(4000, 10, 3)
        