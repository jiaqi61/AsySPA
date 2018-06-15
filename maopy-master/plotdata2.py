# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:28:45 2018

@author: zhangjiaqi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:49:47 2018

@author: zhangjiaqi
"""

import numpy as np
import scipy.io as sio 
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt

filepath = './dataset_covtype/data_partition_5/'
samples = []
labels = []
for i in range(5):
    data = sio.loadmat(filepath+str(i)+'.mat')
    samples = np.append(samples, data['samples'].ravel(order = 'F'))
    labels = np.append(labels, data['labels'].ravel(order = 'F'))


samples = samples.reshape((data['samples'].shape[0],-1), order = 'F')
labels = labels.reshape((data['labels'].shape[0],-1), order = 'F')
lr = LogisticRegression(samples = samples, labels = labels)

centralized_data = sio.loadmat('./data./centralized_result_1.mat')
f_best = lr.obj_func(centralized_data['estimate'][-1]) / lr.n_s + 0.012
colormap = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot1():

    # exact convergence
    f_value = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    plt.semilogy(t, f_value - f_best, label = 'Algorithm 2 (baseline)')    
    # non-exact convergence
    plotdata = sio.loadmat('./data/dis_result_nonexact_conv_8/plotdata.mat')
    ax = plt.gca()
    #ax.set_xlim([1,3000])
    #ax.set_ylim([5e-2,1e1])
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    plt.semilogy(t, f_value - f_best, label = 'Algorithm in [3]', linestyle = '--')    
    plt.xlabel('Time (s)')
    plt.ylabel(r'$f(\bar X(t))-\hat{f}^\star$')
    plt.title('Training error decay with time')
    plt.legend()
#    leg = ax.get_legend()
#    for leghandle in leg.legendHandles:
#        leghandle.set_color('black')
    plt.savefig("./figs/fig6.pdf", bbox_inches='tight')
    plt.close()

def plot6():

    # exact convergence
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8_same_freq/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value = plotdata[name][0]['f_value'][0][0] / lr.n_s
        label = 'Algorithm 2 (baseline)' if i == 0 else '_nolegend_'
        plt.semilogy(t, f_value - f_best, label = label,
                     color = colormap[0], marker = '^', markevery = 15, ms = 3)    
    # non-exact convergence
    plotdata = sio.loadmat('./data/dis_result_nonexact_conv_8_same_freq/plotdata.mat')
    ax = plt.gca()
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value = plotdata[name][0]['f_value'][0][0] / lr.n_s
        label = 'Algorithm in [3]' if i == 0 else '_nolegend_'
        plt.semilogy(t, f_value - f_best, label = label, color = colormap[1],
                     linestyle=':', marker = 'o', markevery = 20, ms = 3)    
    plt.xlabel('Time (s)')
    plt.ylabel('$f(X)-\hat{f}^\star$')
    plt.title('Training error decay with time (same frequency)')
    plt.legend()
#    leg = ax.get_legend()
#    for leghandle in leg.legendHandles:
#        leghandle.set_color('black')
    plt.savefig("./figs/fig11.pdf", bbox_inches='tight')
    plt.close()

def plot2():   
    # error of consensus term, stepsize = 1
    plotdata = sio.loadmat('./data/dis_result_exact_conv_1/plotdata.mat')
    
    est_value = [plotdata['Agent '+ str(i)][0]['est'][0] \
                          for i in range(len(plotdata.keys()) - 3)]
    est_value_mean = np.mean(est_value, axis = 0)
    ax = plt.gca()
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        name = 'Agent '+ str(i)
        t = plotdata[name][0]['time'][0].T
        est_value = plotdata[name][0]['est'][0]
        est_error = [np.linalg.norm(i) / 5 for i in est_value - est_value_mean]
        label = 'Agent '+ str(i+1)
        plt.semilogy(t, est_error, label = label,
                     color = colormap[i])    
    plt.xlabel('Time (s)')
    plt.ylabel(r'$||X(t)-\bar X(t)||_F$')
    plt.title('Error of consensus term')
    plt.legend()
    leg = ax.get_legend()
    plt.savefig("./figs/fig10.pdf", bbox_inches='tight')
    plt.close()

def plot8():
    # difference between each agent's objective function value and their average 
    f_ave = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_ave += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_ave /= 5

    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f = plotdata[name][0]['f_value'][0][0] / lr.n_s
        plt.semilogy(t, np.abs(f - f_ave), label = 'Agent '+ str(i+1))
    plt.xlabel('Time (s)')
    plt.ylabel(r'$|f(\bar X(t))-f(X_i(t))|$')
#    plt.title('Training error decay with time')
    plt.legend()
    plt.savefig("./figs/fig13.pdf", bbox_inches='tight')
    plt.close()
    
def plot3():
    # asynchronous
    f_value = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    plt.semilogy(t, f_value - f_best, label = 'Algorithm 2 (baseline)')    
    # synchronous with the same delays
#    plotdata = sio.loadmat('./data./syn_dis_delay_same/plotdata.mat')
#    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
#        t = plotdata['Agent '+ str(i)][0]['time'][0].T
#        name = 'Agent '+ str(i)
#        f_value = plotdata[name][0]['f_value'][0][0] / lr.n_s
#        label = 'Syn-SPA (same delays)' if i == 0 else '_nolegend_'
#        plt.semilogy(t, f_value - f_best, label = label,
#                     color = colormap[i], linestyle = '--')    
    # synchronous with different delays
    f_value = 0
    plotdata = sio.loadmat('./data./syn_dis_delay_different/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    plt.semilogy(t, f_value - f_best, label = 'Synchronous SPA', linestyle = '--')    
    ax = plt.gca()
    plt.xlabel('Time (s)')
    plt.ylabel(r'$f(\bar X(t))-f^\star$')
#    plt.title('Synchronous and asynchronous SPA')
    plt.legend()
#    leg = ax.get_legend()
#    for leghandle in leg.legendHandles:
#        leghandle.set_color('black')
    plt.savefig("./figs/fig7.pdf", bbox_inches='tight')
    plt.close()
    
def plot4():
    
    # stepsize = 10/n_s
    f_value = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    label = r'$\rho(k)={8}n_s^{-1}/\sqrt{k}}$ (baseline)'
    f_value /=5
    plt.semilogy(t, f_value - f_best, label = label)    
    # stepsize = 1/n_s
    f_value = 0
    plotdata = sio.loadmat('./data./dis_result_exact_conv_1/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /=5
    label = r'$\rho(k)=n_s^{-1}/\sqrt{k}}$'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--')    
    # constant stepsize 0.1
    f_value = 0
    plotdata = sio.loadmat('./data./dis_result_exact_conv_constant_step_0.1/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /=5
    label = r'$\rho(k)=0.1n_s^{-1}$'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-.')    
    # constant stepsize 0.3
    f_value = 0
    plotdata = sio.loadmat('./data./dis_result_exact_conv_constant_step_0.3/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /=5
    label = r'$\rho(k)=0.3n_s^{-1}$'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = ':', zorder = 0)    
    ax = plt.gca()
    plt.xlabel('Time (s)')
    plt.ylabel(r'$f(\bar X(t))-f^\star$')
#    plt.title('Synchronous and asynchronous SPA')
    plt.legend()
#    leg = ax.get_legend()
#    for leghandle in leg.legendHandles[:-1]:
#        leghandle.set_color('black')
    plt.savefig("./figs/fig8.pdf", bbox_inches='tight')   
    plt.close()

def plot5():
    
    # 7 agents
    f_value = 0
    plotdata = sio.loadmat('./data./7agents/plotdata.mat')
    for i in range(7):       # 3 irrevalent terms
        try:
            t = plotdata['Agent '+ str(i)][0]['time'][0].T
        except KeyError:
            print('Agent '+ str(i),'does not exist.')
            continue
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    label = r'7 agents'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-.')       
    # 5 agents
    f_value = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    label = r'5 agents (baseline)'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-')        
    # 3 agents
    f_value = 0
    plotdata = sio.loadmat('./data./3agents/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 3
    label = r'3 agents'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--') 
    
    plt.xlabel('Time (s)')
    plt.ylabel('$f(X)-f^\star$')
#    plt.title('Synchronous and asynchronous SPA')
    plt.legend()
    plt.savefig("./figs/fig9.pdf", bbox_inches='tight')  
    plt.close()

def plot7():

    # baseline
    f_value = 0
    plotdata = sio.loadmat('./data/dis_result_exact_conv_8/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    label = 'Ring graph (baseline)'
    plt.semilogy(t, f_value - f_best, label = label)    
    # higer connectivity
    f_value = 0
    plotdata = sio.loadmat('./data/higher_connectivity_exact_8/plotdata.mat')
    ax = plt.gca()
    #ax.set_xlim([1,3000])
    #ax.set_ylim([5e-2,1e1])
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /=5
    label = 'Higher connectivity'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--')    
    # undirected graphs
    f_value = 0
    plotdata = sio.loadmat('./data/higher_connectivity_exact_8_undirected/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    label = 'Undirected graph'
    plt.semilogy(t, f_value - f_best, label = label,
                     color = colormap[2], linestyle = '-.', zorder = 0)    
    plt.xlabel('Time (s)')
    plt.ylabel('$f(X)-\hat{f}^\star$')
    plt.title('Training error decay with time')
    plt.legend()
#    leg = ax.get_legend()
#    for leghandle in leg.legendHandles:
#        leghandle.set_color('black')
    plt.savefig("./figs/fig12.pdf", bbox_inches='tight')
    plt.close()
    
def plot9():
    evenly_select = lambda m, n: np.rint( np.linspace( 1, n, min(m,n) ) - 1 ).astype(int)
    # centralized 1/5 CPU
    plotdata = sio.loadmat('./data/centralized_5agents_8/centralized_0.2_result_constant_step_0.1.mat')
    t = plotdata['time'][0]
    indices = evenly_select(50,t.size)
    t = t[indices]
    f_value = np.asarray([lr.obj_func(j) for j in plotdata['estimate'][indices]]) / lr.n_s
    label = r'cen:1/5'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--', marker = None, markevery = 8)       
    # centralized 1/3 CPU
    plotdata = sio.loadmat('./data/centralized_5agents_8/centralized_0.33_result_constant_step_0.1.mat')
    t = plotdata['time'][0]
    indices = evenly_select(50,t.size)
    t = t[indices]
    f_value = np.asarray([lr.obj_func(j) for j in plotdata['estimate'][indices]]) / lr.n_s
    label = r'cen:1/3'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--')        
    # centralized 1/2 CPU
    plotdata = sio.loadmat('./data/centralized_5agents_8/centralized_0.5_result_constant_step_0.1.mat')
    t = plotdata['time'][0]
    indices = evenly_select(50,t.size)
    t = t[indices]
    f_value = np.asarray([lr.obj_func(j) for j in plotdata['estimate'][indices]]) / lr.n_s
    label = r'cen:1/2'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '--')   
    # distributed over 5 agents
    f_value = 0
    plotdata = sio.loadmat('./data./5_dis_result_exact_conv_constant_step_0.1_same_freq_nodelay/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 5
    label = r'dis:5'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-') 
    # distributed over 3 agents
    f_value = 0
    plotdata = sio.loadmat('./data./3_dis_result_exact_conv_constant_step_0.1_same_freq_nodelay/plotdata.mat')
    for i in range(len(plotdata.keys()) - 3):       # 3 irrevalent terms
        t = plotdata['Agent '+ str(i)][0]['time'][0].T
        name = 'Agent '+ str(i)
        f_value += plotdata[name][0]['f_value'][0][0] / lr.n_s
    f_value /= 3
    label = r'dis:3'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-')     
    # distributed over 7 agents
    f_value = 0
    plotdata = sio.loadmat('./data./7_dis_result_exact_conv_constant_step_0.1_same_freq_nodelay/plotdata.mat')
    i = 0
    for key in plotdata.keys():       # 3 irrevalent terms
        try:
            t = plotdata[key][0]['time'][0].T
            i += 1
        except:
            continue
        f_value += plotdata[key][0]['f_value'][0][0] / lr.n_s
    f_value /= i
    label = r'dis:7'
    plt.semilogy(t, f_value - f_best, label = label, linestyle = '-')
    
    plt.xlabel('Time (s)')
    plt.ylabel(r'$f(\bar X(t))-\hat{f}^\star$')
    plt.title('Training error decay with time')
    plt.legend()
    plt.savefig("./figs/fig14.pdf", bbox_inches='tight')
    plt.close()

for i in range(9):
    eval('plot'+str(i+1)+'()')
