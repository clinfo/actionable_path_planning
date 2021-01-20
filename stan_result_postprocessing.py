import numpy  as np
import pandas as pd

import os
import pickle
import time
import matplotlib.pyplot as plt


from utils import timeit
import utils

from multiprocessing import Pool, Manager


def load_stan_models(mcmc_dir, max_num_class):
    '''
    Args:
        mcmc_dir ():
        max_num_class ():
    Return:
        dict_stan_models ():
    '''
    dict_stan_models = {}
    for k in range(1, max_num_class+1):
        if os.path.exists(mcmc_dir + '/stan_model_{}.pkl'.format(k)):
            dict_stan_models[k] = pickle.load(open(mcmc_dir + '/stan_model_{}.pkl'.format(k),'rb'))
    
    return dict_stan_models


def load_fitted_models(mcmc_dir, max_num_class):
    '''
    Args:
        mcmc_dir ():
        max_num_class ():
    Return:
        dict_fitted_models ():
    '''
    dict_fitted_models = {}
    for k in range(1, max_num_class+1):
        if os.path.exists(mcmc_dir + '/fitted_model_{}.pkl'.format(k)):
            dict_fitted_models[k] = pickle.load(open(mcmc_dir + '/fitted_model_{}.pkl'.format(k),'rb'))
    
    return dict_fitted_models


def load_ex(result_ex_dir, max_num_class):
    '''
    Args:
        result_ex_dir ():
        max_num_class ():
    Returns:
        dict_emp_bayes ():
        dict_log_liks ():
    '''
    dict_emp_bayes = {}
    dict_log_liks = {}
    for k_class in range(1, max_num_class+1):
        if os.path.exists(result_ex_dir + '/dict_emp_bayes_{}.pkl'.format(k_class)):
            dict_emp_bayes[k_class] = pickle.load(open(result_ex_dir + '/dict_emp_bayes_{}.pkl'.format(k_class),'rb'))
            dict_log_liks[k_class] = pickle.load(open(result_ex_dir + '/dict_log_liks_{}.pkl'.format(k_class),'rb'))
            
    return dict_emp_bayes, dict_log_liks

@timeit
def multi_cont_extract_fitted_model(dict_stan_models,
                                    dict_fitted_models,
                                    size,
                                    result_ex_dir,
                                    max_num_class,
                                    multiprocess_mode_flag):
    '''
    Args: 
        dict_stan_models ():
        dict_fitted_models ():
        size ():
        result_ex_dir ():
        max_num_class ():
        multiprocess_mode_flag ():
    Return:
    '''    
    if multiprocess_mode_flag:
        k_classes = list(range(1, max_num_class+1))
        n_proc = max_num_class
        pool = Pool(processes=n_proc)
        list_args_ex = []
        for k_class in k_classes:
            list_args_ex.append((dict_stan_models, dict_fitted_models, size, result_ex_dir, k_class))
        pool.map(wrap_extract_fitted_model, list_args_ex)
        
    else:
        for k_class in range(1, max_num_class+1):
            extract_fitted_model(dict_stan_models, dict_fitted_models, size, result_ex_dir, k_class)
    return
            
def wrap_extract_fitted_model(args_ex):
    '''
    Wrapper of fitted model extraction for multiprocessing mode.
    Args:
        args_ex
    Return:
    '''
    extract_fitted_model(*args_ex)
    return

def extract_fitted_model(dict_stan_models, dict_fitted_models, size, result_ex_dir, k_class):
    '''
    Entity of fitted model extraction
    Args:
        dict_stan_models ():
        dict_fitted_models ():
        size ():
        result_ex_dir ():
        k_class ():
    Return:
    '''
    ex_flag = os.path.exists(result_ex_dir + '/dict_log_liks_{}.pkl'.format(k_class))
    if (k_class in dict_fitted_models.keys())&(ex_flag==False):
        stan_model = dict_stan_models[k_class]
        fitted_model = dict_fitted_models[k_class]
        fitted_model_ex = fitted_model.extract()
        
        dict_emp_bayes = {}
        for param in fitted_model_ex.keys():
            if param not in ['class_lp', 'log_lik', 'lp__']: 
                dict_emp_bayes[param] = fitted_model_ex[param].mean(axis=0)
        
        dict_log_liks = {}
        dict_log_liks['log_lik'] = fitted_model_ex['log_lik']
        dict_log_liks['class_lp'] = fitted_model_ex['class_lp']
        
        np_gamma = np.zeros((size, k_class))
        for k in range(k_class):
            if k_class==1:
                np_gamma += 1
            else:
                epsilon = 1e-6
                np_gamma[:,k] = ((np.exp(dict_log_liks['class_lp'][:,:,k]) + epsilon / k_class) /\
                                 (np.exp(dict_log_liks['log_lik']) + epsilon)).mean(axis=0)
            
        dict_log_liks['gamma'] = np_gamma
        
        pickle.dump(dict_emp_bayes, open(result_ex_dir + '/dict_emp_bayes_{}.pkl'.format(k_class),'wb'))
        pickle.dump(dict_log_liks, open(result_ex_dir + '/dict_log_liks_{}.pkl'.format(k_class),'wb'))
    return

@timeit    
def wbic(dict_log_liks, max_num_class):
    '''
    wbic用のMCMCが必要
    Args:
        dict_log_liks ():
        max_num_class ():
    Return:
        list_wbic ():
    '''
    list_wbic = []
    for k in range(1, max_num_class+1):        
        if k in dict_log_liks.keys():
            log_lik = dict_log_liks[k]['log_lik']
            wbic = - np.mean(log_lik.sum(axis=1))
            list_wbic.append(round(wbic, 3))
        else:
            list_wbic.append(np.nan)    
    return list_wbic

@timeit
def param_num_count(dict_emp_bayes, max_num_class):
    '''
    Args:
        dict_emp_bayes ():
        max_num_class ():
    Return:
        list_num_param ():
    '''
    list_num_param = []
    for k in range(1, max_num_class+1):
        if k in dict_emp_bayes.keys():
            num_count = 0
            for param in dict_emp_bayes[k].keys():
                if param not in ['phi_sigma_X_cont', 'beta1_0']:
                    if param == 'pi':
                        num_count += dict_emp_bayes[k][param].size - 1
                    elif 'phi_X_disc' in param:
                        num_count += dict_emp_bayes[k][param].size - k
                    else:
                        num_count += dict_emp_bayes[k][param].size
            list_num_param.append(num_count)
        else:
            list_num_param.append(np.nan)
    return list_num_param


def plot_by_class(list_y, y_name, max_num_class, result_summary_dir, fontsize=36):
    xdata = list(range(1,max_num_class+1,1))
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(xdata, list_y, linewidth=5)
#     plt.title('{} vs num_class'.format(y_name), fontsize=fontsize)
    plt.xlabel('Number of mixture components', fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    plt.xticks(fontsize=fontsize*0.75)
    plt.yticks(fontsize=fontsize*0.75)
    plt.grid(True)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(4)
    
    plt.savefig(result_summary_dir + '/num_class_vs_{}.png'.format(y_name))
#     plt.show()        
    return