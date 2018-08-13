# -*- coding: utf-8 -*-
"""
Demo to apply the AsyGradientPush algorithm to a multiclass logistic regression problem 
on covtype dataset

To run the demo, run the following:
    mpiexec -n $(num_nodes) -bind-to core -allow-run-as-root python distributed_asy_logistic_regression.py

Created on Sun Jun  3 17:00:07 2018

@author: Jiaqi Zhang
"""

import numpy as np
import scipy.io as sio 
from logistic_regression import LogisticRegression
from asy_gradient_push import AsyGradientPush
import asy_gradient_push
import fnmatch, os
import click

UID = asy_gradient_push.UID
SIZE = asy_gradient_push.SIZE

def demo_lr_covtype(data_dir, save_dir, run_time , delay_power, step_size, 
                    exact_convergence, constant_step_size, num_outdegrees,
                    synch):
    """
    Demo to apply the AsyGradientPush algorithm to a multiclass logistic regression problem 
    on covtype dataset

    To run the demo, run the following:
        mpiexec -n $(num_nodes) python distributed_asy_logistic_regression.py
    """
    paras = locals()
    from logistic_regression import LogisticRegression
    
    # add slashes to path if needed
    data_dir = os.path.join(data_dir,'')
    save_dir = os.path.join(save_dir,'')
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    # n_file = len(fnmatch.filter(os.listdir(data_dir), '*.mat')) # number of sub-datasets

    # used for simulate uneven updates
    if delay_power is None:
        delay = 0
    else:
        delay = 3e-2 * ((SIZE-UID-1)**delay_power)

    # Load the dataset   
    filename = data_dir + str(UID) + '.mat'
    data = sio.loadmat(filename)
    samples = data['samples']
    labels = data['labels']
    
    # initialize the problem
    lr = LogisticRegression(samples = samples, labels = labels, reg = 1/SIZE)
    objective = lr.obj_func
    gradient = lr.gradient
    
    x_start = np.random.randn(lr.n_f, lr.n_c)

    if num_outdegrees > 0:
        peers = [(UID + 1 + i) % SIZE for i in range(num_outdegrees)]
    else:
        peers = [(UID + 1 + i) % SIZE for i in range(int(SIZE / 3))]


    pd = AsyGradientPush(objective=objective,
                    sub_gradient=gradient,
                    arg_start= x_start,
                    synch=synch,
                    peers=peers,
                    # peers=[(UID + 1) % SIZE],
                    step_size= step_size / (lr.n_s),
                    constant_step_size = constant_step_size,
                    terminate_by_time=True,
                    termination_condition=run_time,
                    log=True,
                    in_degree=None,
                    num_averaging_itr=1,
                    delay = delay,
                    exact_convergence=exact_convergence)
    
    # start the optimization
    loggers = pd.minimize(print_info = True)
#    l_argmin_est = loggers['argmin_est']
#    l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=False)
    
    # Save the data into a mat file
    data_save(samples, labels, loggers, filepath = save_dir)

    if UID == 0:
        # paras = {'num_nodes':SIZE, 'exact_convergence':exact_convergence, 'constant_step_size':constant_step_size,
                #  'step_size':step_size, 'delay_power':delay_power}
        with open(os.path.join(save_dir,'paras.txt'),'w+') as f:
            for key, value in paras.items():
                f.write(key+', '+str(value)+'\n')
            
def data_save(samples, labels, loggers, filepath = 'data/'):
    """ Save the data into a .mat file. """
    try:
        l_argmin_est = loggers['argmin_est']
        itr = np.fromiter(l_argmin_est.history.keys(), dtype=float)
        t = np.array([i[0] for i in l_argmin_est.history.values()])
        est = np.array([i[1] for i in l_argmin_est.history.values()])
        # l_ps_w = loggers['ps_w']
        # l_ps_n = loggers['ps_n']
        # l_ps_l = loggers['ps_l']
        # ps_w = np.array([i[1] for i in l_ps_w.history.values()])
        # ps_n = np.array([i[1] for i in l_ps_n.history.values()])
        # ps_l = np.array([i[1] for i in l_ps_l.history.values()])
        filename = filepath + str(UID) + '.mat'
        # sio.savemat(filename, mdict={'samples': samples, 'labels': labels, 'itr': itr,
        #                              'time': t, 'estimate': est})
        # evenly select m elements from a list of length n
        evenly_select = lambda m, n: np.rint( np.linspace( 1, n, min(m,n) ) - 1 ).astype(int)
        indices = evenly_select(5000, t.shape[0]) # only choose 5000 data points to save space
        sio.savemat(filename, mdict={'itr': itr[indices], 'time': t[indices], 'estimate': est[indices]})
    except Exception as e:
        print(e)

# Solve the logistic regression problem on covtype dataset
@click.command()
@click.option('--data_dir', type=str, help='path to dataset')
@click.option('--save_dir', type=str, help='path to save the result')
@click.option('--run_time', type=float, default=60, help='path to save the result')
@click.option('--delay_power', type=float, default=None, help='time between two activation is 0.03*UID^delay_power')
@click.option('--step_size', type=float, default=0.1,help='stepsize used in gradient descent')
@click.option('--num_outdegrees',type=int, default=-1, help='number of out-neighbors of one core')
@click.option('--exact_convergence/--nonexact_convergence', default=True, help='whether to converge exactly')
@click.option('--constant_step_size/--diminishing_step_size', default=True, help='whether to use constant stepsize')
@click.option('--synch/--asynch',default=False,help='wheter to use synchronous algorithm')
def main(**kwargs):
    demo_lr_covtype(**kwargs)

main()
