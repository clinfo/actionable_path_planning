#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd
import argparse

import os
import pickle
import time

from multiprocessing import Pool, Manager

import stan_code_insertion as stanc
import stan_make_stan_data_and_modeling as stanm
import utils

import pystan

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--load_dir', default='example',
                        help='load directory')
    parser.add_argument('-y','--sigma_y', default=2, type=float,
                        help='Preferably changed to rmse/2')
    parser.add_argument('-k','--max_num_class', default=8, type=int,
                        help='number of max mixture components')
    parser.add_argument('-a','--alpha', default=1, type=float,
                        help='coef alpha')
    parser.add_argument('-s','--stan_template', default='template_for_wbic.stan',
                        help='template stan file')
    parser.add_argument('-i','--iter', default=1500, type=int,
                        help='MCMCのiter．default=1500')
    parser.add_argument('-d','--use_data', default='test', choices=['test', 'all'],
                        help='data used for modeling')    
    return parser.parse_args()

def main():
    args = get_parser()
    data_dir = args.load_dir + '/data'
    mcmc_dir = args.load_dir + '/mcmc'
    args_dir = args.load_dir + '/args'
    utils.make_dir(args_dir)
    utils.make_dir(mcmc_dir)
    
    # Load
    print('[INFO] Loading')
    # Stan data
    xy_for_stan = pickle.load(open(data_dir + '/xy_for_stan.pkl', 'rb'))
    # template file   
    with open(args.stan_template, encoding='utf-8') as fr:
        template_stan_text = fr.read()
    model_type = pickle.load(open(data_dir + '/model_type.pkl', 'rb'))
    print('[INFO] Done')    
    
    # select data index for modeling
    if args.use_data=='test':
        selected_i_idx = xy_for_stan.val_i_idx
    elif args.use_data=='all':
        selected_i_idx = sorted(xy_for_stan.train_i_idx + xy_for_stan.val_i_idx)
    else:
        raise NotImplementedError()
        
    # make basic stan_data
    stan_data_var = stanm.make_default_stan_data(xy_for_stan, args.alpha, args.sigma_y, selected_i_idx, model_type)  
    # add discrete setting
    if xy_for_stan.x_disc.index:
        template_stan_text = stanc.insert_disc_to_stan_file(template_stan_text, xy_for_stan.x_disc)
        stan_data_var.update(stanm.make_disc_stan_data(xy_for_stan.x_disc, selected_i_idx))   
    # add zero_poi setting [Not Supported]
    if xy_for_stan.x_zero_poi.index:
        template_stan_text = stanc.insert_zero_poi_to_stan_file(template_stan_text, xy_for_stan.x_zero_poi)
        stan_data_var.update(stanm.make_zero_poi_stan_data(xy_for_stan.x_zero_poi, selected_i_idx))
    output_stan_text = template_stan_text
    
    # Output stan_model
    with open(mcmc_dir + '/stan_model.stan', 'w', encoding='utf-8') as fw:
        fw.write(output_stan_text)      
    # Save setting
    pickle.dump(args, open(args_dir + '/args_02.pkl','wb'))
    pickle.dump(selected_i_idx, open(data_dir + '/selected_i_idx.pkl', 'wb'))    
    
    # Compile
    stan_model = pystan.StanModel(file=mcmc_dir+'/stan_model.stan', charset='utf-8',
                                  model_name='stan_model', verbose=False)    
    
    # multiprocessing
    n_proc = args.max_num_class
    pool = Pool(processes=n_proc)
    print('[INFO] Start MCMC')
    list_args_stan = []
    for K in range(1,args.max_num_class+1):
        list_args_stan.append((stan_model, stan_data_var, mcmc_dir, args, K))
    pool.map(stanm.wrap_stan_process, list_args_stan)    

if __name__ == '__main__':
    main()

