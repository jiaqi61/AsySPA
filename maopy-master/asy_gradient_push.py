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
                 delay=None,
                 exact_convergence=True):
        """ Initialize the gossip optimization settings. """

        self.constant_step_size = constant_step_size
        self.exact_convergence = exact_convergence # exact convergence uses the algorithm in the paper
        
        # Set the artificial delays (seconds)
        if delay is None:
            # Bound below the time between two updates to prevent ps_w from vanishing
            # and increase numerical stabality
            self.delay = 5e-5 
        else:
            self.delay = 5e-5 + delay

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

    def _gradient_descent_step(self, ps_n, ps_l, ps_l_max, argmin_est, itr):
        """ Take step in direction of negative gradient, and return the new domain point. """

        # Diminshing step-size: 1 / sqrt(k). 
        # The total step-size in exact convergence case is determined by ps_l and ps_l_max.
        # The total step-size in nonexact convergence case is determined by itr
        if self.constant_step_size is True:
            if self.exact_convergence is True:
                step_size = [self.step_size for _ in range(int(ps_l), int(ps_l_max+1))]
            else:
                step_size = self.step_size
        else:
            if self.exact_convergence is True:
                step_size = [self.step_size / ((i+1) ** 0.5) for i in range(int(ps_l), int(ps_l_max+1))]
            else:
                step_size = self.step_size / (itr ** 0.5)

        stepsize = np.sum(step_size)
        grad = self.sub_gradient(argmin_est)
        
        return ps_n - stepsize * grad


    def minimize(self, print_info=False):
        """
        Minimize the objective specified in settings using the Subgradient-Push procedure

        Procedure:
        1) Gossip: push_sum_gossip([ps_n, ps_w])
        2) Update (exact convergence): ps_result = push_sum_gossip([ps_n, ps_w, ps_l])
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
            start_time = time.time()
            end_time = start_time + gossip_time # End time of optimization
            condition = True

        # Start optimization at the same time
        COMM.Barrier()

        # Optimization loop
        while condition:

            if self.synch is True:
                COMM.Barrier()
                
            # Add random artificial delays
            time.sleep(self.delay*(1+np.random.rand())) 
            
            itr += 1
            
            if (print_info is True) and (itr % 10 == 0):
                print('UID=' + str(UID),
                      '\ttime=' + str(int(time.time()-start_time))+'s',
                      '\titr=' + str(itr),
                      '\tps_l=' + str(int(ps_l)),
                      '\texact_conv=' + str(self.exact_convergence), flush = True)

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
                                               argmin_est=argmin_est,
                                               itr = itr)
            if self.exact_convergence is True:
                ps_l = ps_l_max + 1
            else:
                ps_l = -1
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
        Demo of the use of the AsyGradientPush class on a least square problem.

        To run the demo, run the following:
            mpiexec -n $(num_nodes) python asy_gradient_push.py
        """
        # Create objective function and its gradient
        np.random.seed(seed=UID)
        x_start = np.random.randn(num_features)
        a_m = np.random.randn(num_instances_per_node, num_features)
        b_v = np.random.randn(num_instances_per_node)
        objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2
        gradient = lambda x: a_m.T.dot(a_m.dot(x)-b_v)
        
        # Set the artificial delay used in asynchronous mode (seconds)
        delay = 0.1*(UID+1)

        pssgd = AsyGradientPush(objective=objective,
                                          sub_gradient=gradient,
                                          arg_start=x_start,
                                          synch=False,
                                          peers=[(UID + 1) % SIZE],
                                          step_size= 1e-2/ num_instances_per_node,
                                          constant_step_size=True,
                                          terminate_by_time=True,
                                          termination_condition=300,
                                          log=True,
                                          in_degree=1,
                                          num_averaging_itr=1,
                                          delay = delay,
                                          exact_convergence=False)
        
        # start the optimization
        loggers = pssgd.minimize(print_info=True)
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=False)
        
        # Save the result data into a .mat file
        itr = np.fromiter(l_argmin_est.history.keys(), dtype=float)
        t = np.array([i[0] for i in l_argmin_est.history.values()])
        est = np.array([i[1] for i in l_argmin_est.history.values()])
        l_ps_w = loggers['ps_w']
        l_ps_n = loggers['ps_n']
        l_ps_l = loggers['ps_l']
        ps_w = np.array([i[1] for i in l_ps_w.history.values()])
        ps_n = np.array([i[1] for i in l_ps_n.history.values()])
        ps_l = np.array([i[1] for i in l_ps_l.history.values()])
        filepath = 'data/least_square_nonexact/'+str(UID)+'.mat'
        sio.savemat(filepath, mdict={'A': a_m, 'b': b_v, 'itr': itr, 'time': t,
                                     'ps_w': ps_w, 'ps_n': ps_n, 'ps_l': ps_l,
                                     'estimate': est}) 
    # Run a demo
    demo_ls(num_instances_per_node = 5000, num_features = 30)
