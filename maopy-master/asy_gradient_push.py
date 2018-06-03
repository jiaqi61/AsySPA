# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:18:54 2018

Asynchronous gradient-push class for distributed optimization using column stochastic matrix.

@author: zhangjiaqi
"""

"""

:author: Mido Assran
:description: Distributed otpimization using column stochastic mixing and greedy gradient descent.
              Based on the paper (nedich2015distributed)
"""

import time

import numpy as np
import scipy.io as sio

from gossip_comm import GossipComm
from push_sum_optimization import PushSumOptimizer

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class AsyGradientPush(PushSumOptimizer):
    """ Distributed optimization using column stochastic mixing and greedy gradient descent. """

    # Inherit docstring
    __doc__ += PushSumOptimizer.__doc__

    def __init__(self, objective,
                 sub_gradient,
                 arg_start,
                 synch=True,
                 peers=None,
                 step_size=None,
                 terminate_by_time=False,
                 termination_condition=None,
                 log=False,
                 out_degree=None,
                 in_degree=SIZE,
                 num_averaging_itr=1,
                 constant_step_size=False,
                 all_reduce=False,
                 delay = None):
        """ Initialize the gossip optimization settings. """

        self.constant_step_size = constant_step_size
        
        # Set the artificial delays (seconds)
        if synch is False:
            if delay is None:
                # Bound below the time between two update to prevent ps_w from vanishing
                self.delay = 5e-5 
            else:
                self.delay = 5e-5 + delay
        else:
            self.delay = 0

        super(AsyGradientPush, self).__init__(objective=objective,
                                                        sub_gradient=sub_gradient,
                                                        arg_start=arg_start,
                                                        synch=synch,
                                                        peers=peers,
                                                        step_size=step_size,
                                                        terminate_by_time=terminate_by_time,
                                                        termination_condition=termination_condition,
                                                        log=log,
                                                        out_degree=out_degree,
                                                        in_degree=in_degree,
                                                        num_averaging_itr=num_averaging_itr,
                                                        all_reduce=all_reduce)

    def _gradient_descent_step(self, ps_n, ps_l, ps_l_max, argmin_est):
        """ Take step in direction of negative gradient, and return the new domain point. """

        # Diminshing step-size: 1 / sqrt(k). The total step-size is determined by ps_l and ps_l_max.
        if self.constant_step_size is True:
            step_size = [self.step_size for _ in range(int(ps_l), int(ps_l_max+1))]
        else:
            step_size = [self.step_size / ((i+1) ** 0.5) for i in range(int(ps_l), int(ps_l_max+1))]

        stepsize = np.sum(step_size)
        grad = self.sub_gradient(argmin_est)
        
        return ps_n - stepsize * grad


    def minimize(self, print_info=False):
        """
        Minimize the objective specified in settings using the Subgradient-Push procedure

        Procedure:
        1) Gossip: push_sum_gossip([ps_n, ps_w])
        2) Update: ps_result = push_sum_gossip([ps_n, ps_w, ps_l])
            2.a) ps_n = ps_result[ps_n]
            2.b) ps_w = ps_result[ps_w]
            2.c) ps_l_max = ps_result['ps_l']
            2.d) argmin_est = ps_n / ps_w
            2.e) ps_n = ps_n - step_size * sub_gradient(argmin_est). step_size is related to ps_l and ps_l_max
            2.f) ps_l = ps_l_max + 1
        3) Repeat until completed $(termination_condition) itr. or time (depending on settings)

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger,
                              "ps_n": GossipLogger,
                              "ps_l": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """
        super(AsyGradientPush, self).minimize()

        # Initialize sub-gradient descent push sum gossip
        ps_n = self.argmin_est
        ps_w = 1 
        ps_l = 1
        argmin_est = ps_n / ps_w

        itr = 0
        
        log = self.log
        psga = self.ps_averager
        objective = self.objective
        gradient = self.sub_gradient

        if log:
            from gossip_log import GossipLog
            l_argmin_est = GossipLog() # Log the argmin estimate
            l_ps_w = GossipLog() # Log the push sum weight
            l_ps_n = GossipLog() # Log the push sum value
            l_ps_l = GossipLog()
            l_argmin_est.log(argmin_est, itr)
            l_ps_w.log(ps_w, itr)
            l_ps_n.log(ps_n, itr)
            l_ps_l.log(ps_l, itr)


        if self.terminate_by_time is False:
            num_gossip_itr = self.termination_condition
            condition = itr < num_gossip_itr
        else:
            gossip_time = self.termination_condition
            end_time = time.time() + gossip_time # End time of optimization
            condition = time.time() < end_time

        # Goes high if a message was not received in the last gossip round
        just_probe = False

        # Start optimization at the same time
        COMM.Barrier()

        start_time = time.time()

        # Optimization loop
        while condition:

            if self.synch is True:
                COMM.Barrier()

            # Random artificial delays
            time.sleep(self.delay*(1+np.random.rand()))
            
            itr += 1
            
            if print_info is True:
                print('UID='+str(UID),'itr='+str(itr),'ps_l='+str(ps_l), flush = True)

            # -- START Subgradient-Push update -- #

            # Gossip
            ps_result = psga.gossip(gossip_value=ps_n, ps_weight=ps_w, ps_loop=ps_l)
            ps_n = ps_result['ps_n']
            ps_w = ps_result['ps_w']
            ps_l_max = ps_result['ps_l']

            # Update argmin estimate and take a step
            argmin_est = ps_result['avg']
            ps_n = self._gradient_descent_step(ps_n=ps_n,
                                               ps_l = ps_l, 
                                               ps_l_max = ps_l_max,
                                               argmin_est=argmin_est)
            ps_l = ps_l_max + 1
            # -- END Subgradient-Push update -- #

            # Log the varaibles
            if log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w, itr)
                l_ps_n.log(ps_n, itr)
                l_ps_l.log(ps_l, itr)


            # Update the termination flag
            if self.terminate_by_time is False:
                condition = itr < num_gossip_itr
            else:
                condition = time.time() < end_time

        self.argmin_est = argmin_est

        if log is True:
            return {"argmin_est": l_argmin_est,
                    "ps_w": l_ps_w,
                    "ps_n": l_ps_n,
                    "ps_l": l_ps_l}
        else:
            return {"argmin_est": argmin_est,
                    "objective": objective(argmin_est),
                    "sub_gradient": gradient(argmin_est)}

if __name__ == "__main__":

    def demo_ls(num_instances_per_node, num_features):
        """
        Demo of the use of the AsyGradientPush class to a least square problem.

        To run the demo, run the following from the multi_agent_optimization directory CLI:
            mpiexec -n $(num_nodes) python -m do4py.push_sum_gossip_gradient_descent
        """
        # Create objective function and its gradient
        np.random.seed(seed=UID)
        x_start = np.random.randn(num_features)
        a_m = np.random.randn(num_instances_per_node, num_features)
        b_v = np.random.randn(num_instances_per_node)
        objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2
        gradient = lambda x: a_m.T.dot(a_m.dot(x)-b_v)
        
        # Set the artificial delay used in asynchronous mode (seconds)
        delay = 1e-3*(UID+1)
#        delay = None

        pssgd = AsyGradientPush(objective=objective,
                                          sub_gradient=gradient,
                                          arg_start=x_start,
                                          synch=False,
                                          peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                                          step_size=1e-2,
                                          terminate_by_time=True,
                                          termination_condition=10,
                                          log=True,
                                          in_degree=2,
                                          num_averaging_itr=1,
                                          constant_step_size=False,
                                          delay = delay)

        loggers = pssgd.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=False)
        
        # Save the data into a .mat file
        itr = np.fromiter(l_argmin_est.history.keys(), dtype=float)
        t = np.array([i[0] for i in l_argmin_est.history.values()])
        est = np.array([i[1] for i in l_argmin_est.history.values()])
        l_ps_w = loggers['ps_w']
        l_ps_n = loggers['ps_n']
        l_ps_l = loggers['ps_l']
        ps_w = np.array([i[1] for i in l_ps_w.history.values()])
        ps_n = np.array([i[1] for i in l_ps_n.history.values()])
        ps_l = np.array([i[1] for i in l_ps_l.history.values()])
        filepath = 'data/'+str(UID)+'.mat'
        sio.savemat(filepath, mdict={'A': a_m, 'b': b_v, 'itr': itr, 'time': t,
                                     'ps_w': ps_w, 'ps_n': ps_n, 'ps_l': ps_l,
                                     'estimate': est})


    def demo_lr(num_instances_per_node, num_features, num_classes):
        """
        Demo of the use of the AsyGradientPush class to a logistic regression problem.

        To run the demo, run the following from the multi_agent_optimization directory CLI:
            mpiexec -n $(num_nodes) python -m do4py.push_sum_gossip_gradient_descent
        """
        from logistic_regression import LogisticRegression
        # Create dataset
        np.random.seed(seed=UID)
        x_start = np.random.randn(num_features, num_classes)
        # Create one-hot labels
        labels = np.zeros((num_classes, num_instances_per_node))
        label_vec = np.random.randint(low = 0, high = num_classes, size = num_instances_per_node)
        labels[label_vec, range(num_instances_per_node)] = 1
        # Create instances
        samples = np.random.randn(num_features - 1, num_instances_per_node) + 2*label_vec
        samples = np.vstack((samples, np.ones((1,num_instances_per_node))))
        # Create objective function and gradient
        lr = LogisticRegression(samples = samples, labels = labels)
        objective = lr.obj_func
        gradient = lr.gradient
        
        pd = AsyGradientPush(objective=objective,
                        sub_gradient=gradient,
                        arg_start=x_start,
                        synch=False,
                        peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                        step_size= 5 / num_instances_per_node,
                        terminate_by_time=True,
                        termination_condition=300,
                        log=True,
                        in_degree=2,
                        num_averaging_itr=1)

        loggers = pd.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=False)
        
        # Save the data into a mat file
        data_save(samples, labels, loggers, filepath = 'data/')

    def demo_lr_covtype(filepath = 'dataset_covtype/'):
        """
        Demo to apply the AsyGradientPush algorithm to a multiclass logistic regression problem 
        on covtype dataset

        To run the demo, run the following from the multi_agent_optimization directory CLI:
            mpiexec -n $(num_nodes) python -m do4py.push_sum_gossip_gradient_descent
        """
        from logistic_regression import LogisticRegression
        
        # Load the dataset
        filename = filepath + str(UID) + '.mat'
        data = sio.loadmat(filename)
        samples = data['samples']
        labels = data['labels']
        
        lr = LogisticRegression(samples = samples, labels = labels)
        objective = lr.obj_func
        gradient = lr.gradient
        
        x_start = np.random.randn(lr.n_f, lr.n_c)
        
        pd = AsyGradientPush(objective=objective,
                        sub_gradient=gradient,
                        arg_start= x_start,
                        synch=False,
                        peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                        step_size= 10 / lr.n_s,
                        terminate_by_time=True,
                        termination_condition=6,
                        log=True,
                        in_degree=2,
                        num_averaging_itr=1)

        loggers = pd.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=False)
        
        # Save the data into a mat file
        data_save(samples, labels, loggers, filepath = 'data/')
                
    def data_save(samples, labels, loggers, filepath = 'data/'):
        """ Save the data into a .mat file. """
        l_argmin_est = loggers['argmin_est']
        itr = np.fromiter(l_argmin_est.history.keys(), dtype=float)
        t = np.array([i[0] for i in l_argmin_est.history.values()])
        est = np.array([i[1] for i in l_argmin_est.history.values()])
        l_ps_w = loggers['ps_w']
        l_ps_n = loggers['ps_n']
        l_ps_l = loggers['ps_l']
        ps_w = np.array([i[1] for i in l_ps_w.history.values()])
        ps_n = np.array([i[1] for i in l_ps_n.history.values()])
        ps_l = np.array([i[1] for i in l_ps_l.history.values()])
        filename = filepath + str(UID) + '.mat'
        sio.savemat(filename, mdict={'samples': samples, 'labels': labels, 'itr': itr, 'time': t,
                                     'ps_w': ps_w, 'ps_n': ps_n, 'ps_l': ps_l, 'estimate': est})
 
    # Run a demo
    demo_lr_covtype()
