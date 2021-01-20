#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd
import argparse

import os
import pickle
import time

from multiprocessing import Pool, Manager

import stan_result_postprocessing as stanr
import utils


def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--load_dir', default='example',
                        help='load directory')
    parser.add_argument("--multiprocessing", default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='If True, multiprocessing')    
    return parser.parse_args()

def main():
    args = get_parser()
    
    load_dir = './' + args.load_dir
    data_dir = load_dir + '/data'
    mcmc_dir = load_dir + '/mcmc'
    args_dir = load_dir + '/args'
    result_dir = load_dir + '/result'
    result_ex_dir = result_dir + '/ex'
    result_summary_dir = result_dir + '/summary'
    utils.make_dir(result_dir)
    utils.make_dir(result_ex_dir)
    utils.make_dir(result_summary_dir)    
    

    # Load
    print('[INFO] Loading')
    args_02 = pickle.load(open(args_dir + '/args_02.pkl', 'rb'))
    dict_stan_models = stanr.load_stan_models(mcmc_dir, args_02.max_num_class)
    dict_fitted_models = stanr.load_fitted_models(mcmc_dir, args_02.max_num_class)
    model_type = pickle.load(open(data_dir + '/model_type.pkl', 'rb'))  
    xy_for_stan = pickle.load(open(data_dir + '/xy_for_stan.pkl', 'rb'))
    selected_i_idx = pickle.load(open(data_dir + '/selected_i_idx.pkl', 'rb'))
    print('[INFO] Done')

    dict_fitted_model_extract = stanr.multi_cont_extract_fitted_model(dict_stan_models,
                                                                      dict_fitted_models,
                                                                      len(selected_i_idx),
                                                                      result_ex_dir,
                                                                      args_02.max_num_class,
                                                                      args.multiprocessing)

    dict_emp_bayes, dict_log_liks = stanr.load_ex(result_ex_dir, args_02.max_num_class)
    list_num_param = stanr.param_num_count(dict_emp_bayes, args_02.max_num_class)
    pickle.dump(list_num_param, open(result_ex_dir + '/list_num_param.pkl','wb'))

    list_wbic = stanr.wbic(dict_log_liks, args_02.max_num_class)
    pickle.dump(list_wbic, open(result_ex_dir + '/list_wbic.pkl','wb')) 

    stanr.plot_by_class(list_wbic, 'WBIC', args_02.max_num_class, result_summary_dir)
    print('[INFO] Lowest WBIC was obtaind at mixture components of {}'.format(list_wbic.index(min(list_wbic)) + 1))


if __name__ == '__main__':
    main()

