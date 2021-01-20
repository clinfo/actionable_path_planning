#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd

import os
import time
import pickle

import pystan
import warnings
warnings.simplefilter('ignore')

def make_default_stan_data(xy_for_stan, alpha, sigma_y, selected_i_idx, model_type):
    print('[INFO] Make stan_data')
    n_cont = len(xy_for_stan.x_cont.clus.name)
    alpha_lasso = alpha
    mu_mu_X_cont = np.zeros(n_cont)
    sigma_mu_X_cont = np.eye(n_cont)*5
    alpha_bern = np.repeat(1., 2)    
    
    if model_type=='regressor':
        sigma_y = sigma_y
        y_mean = xy_for_stan.y_pred[selected_i_idx].mean()
        y_std = xy_for_stan.y_pred[selected_i_idx].std()

        stan_data = {
            'N'           : len(selected_i_idx),
            'n_reg'       : xy_for_stan.x_reg.array.shape[1], 
            'n_cont'      : n_cont, 
            'X_reg'       : xy_for_stan.x_reg.array[selected_i_idx], 
            'X_cont'      : xy_for_stan.x_cont.clus.array[selected_i_idx], 
            'y'           : xy_for_stan.y_modeling[selected_i_idx], 
            'alpha_lasso' : alpha_lasso, 
            'mu_mu_X_cont': mu_mu_X_cont, 
            'sigma_mu_X_cont': sigma_mu_X_cont,
            'alpha_bern'  : alpha_bern,
            'sigma_y'     : sigma_y,
            'y_mean'      : y_mean,
            'y_std'       : y_std
        }
    elif model_type=='classifier':
        stan_data = {
            'N'           : len(selected_i_idx),
            'n_reg'       : xy_for_stan.x_reg.array.shape[1], 
            'n_cont'      : n_cont,
            'X_reg'       : xy_for_stan.x_reg.array[selected_i_idx],
            'X_cont'      : xy_for_stan.x_cont.clus.array[selected_i_idx],
            'y'           : xy_for_stan.y_modeling[selected_i_idx].astype(np.int), 
            'alpha_lasso' : alpha_lasso, 
            'mu_mu_X_cont': mu_mu_X_cont, 
            'sigma_mu_X_cont': sigma_mu_X_cont,
            'alpha_bern'  : alpha_bern
        }    
    else:
        raise NotImplementedError()
    
    print('[INFO] Done')
    return stan_data


def make_disc_stan_data(x_disc, selected_i_idx):
    stan_data = {}
    num_disc = len(x_disc.index)
    
    for i in range(num_disc): 
        stan_data['X_disc_{}'.format(x_disc.index[i])] =\
                                x_disc.clus.array[selected_i_idx,i].astype(int)
        stan_data['n_cat_{}'.format(x_disc.index[i])] = x_disc.clus.n_cat[i]
        stan_data['alpha_disc_{}'.format(x_disc.index[i])] = x_disc.clus.alpha[i]
    return stan_data

def make_zero_poi_stan_data(x_zero_poi, selected_i_idx):
    stan_data = {}
    num_zero_poi = len(x_zero_poi.index)

    for i in range(num_zero_poi):
        stan_data['X_zero_poi_{}'.format(x_zero_poi.index[i])] =\
                            (-(-x_zero_poi.clus.array[selected_i_idx,i] //1)).astype(int)
    return stan_data


def wrap_stan_process(args_stan):
    _stan_process(*args_stan)        
    
def _stan_process(stan_model, stan_data_var, mcmc_dir, args, K):
    stan_data = {
        'K'          : K,
        'alpha_class': np.repeat(1., K)
    }
    stan_data.update(stan_data_var)
    start = time.time()
    
    fitted_model = stan_model.sampling(data=stan_data, iter=args.iter, chains=1, 
                                       warmup=500, n_jobs = 1, verbose=False, seed=1, check_hmc_diagnostics=False)
    
    elapsed = time.time() - start
    print('[INFO] Done in {:.2f} hours (mixture_components: {})'.format(elapsed/60/60, K))
    
    # Save model
    pickle.dump(elapsed, open(mcmc_dir + '/elapsed_{}.pkl'.format(K),'wb'))
    pickle.dump(stan_model, open(mcmc_dir + '/stan_model_{}.pkl'.format(K),'wb'))
    pickle.dump(fitted_model, open(mcmc_dir + '/fitted_model_{}.pkl'.format(K),'wb'))