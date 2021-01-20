#!/usr/bin/env python
# coding: utf-8
import numpy  as np
import pandas as pd
import argparse

import os
import pickle
import time
import datetime

import utils
def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--load_dir', default='example',
                        help='load directory')
    parser.add_argument('--path_planning_index', default='all',
                        help='instance index for path planning')
    parser.add_argument('--intervention_variables', default='all',
                        help='intervention variables for path planning')
    parser.add_argument('--mixture_components', default=2, type=int,
                        help='number of mixture components of surrogate model used for path planning') 
    parser.add_argument('--y_type', default='pred', choices=['pred', 'glmm'],
                        help='y used in path planning') 
    parser.add_argument('--destination_state', default='count', choices=['count', 'criteria'],
                        help='destination_state selection')
    parser.add_argument('--destination', default=20000, type=int,
                        help='destination. If count, iteration count, elif criteria, y_value to finish searching')    
    parser.add_argument('--step', default=0.5, type=float,
                        help='unit of change in intevention')
    parser.add_argument('--upper_is_better', default=False, 
                        type=lambda x: (str(x).lower() == 'true'),
                        help='If True, increased y state would be selected')  
    return parser.parse_args()

def main():
    args = get_parser()

    load_dir = './' + args.load_dir
    data_dir = load_dir + '/data'
    result_dir = load_dir + '/result'
    result_ex_dir = load_dir + '/result/ex'
    args_dir = load_dir + '/args'
    result_class_dir = result_dir + '/{}_classes'.format(args.mixture_components)
    utils.make_dir(result_class_dir)

    # Load
    print('[INFO] Loading')
    xy_for_stan = pickle.load(open(data_dir + '/xy_for_stan.pkl', 'rb'))
    var_info = pickle.load(open(data_dir + '/var_info.pkl', 'rb'))
    train_i_idx = pickle.load(open(data_dir + '/train_i_idx.pkl', 'rb'))
    selected_i_idx = pickle.load(open(data_dir + '/selected_i_idx.pkl', 'rb'))
    model_type = pickle.load(open(data_dir + '/model_type.pkl', 'rb'))
    args_02 = pickle.load(open(args_dir + '/args_02.pkl', 'rb'))
    dict_emp_bayes = pickle.load(open(result_ex_dir + '/dict_emp_bayes_{}.pkl'.format(args.mixture_components),'rb'))
    dict_log_liks = pickle.load(open(result_ex_dir + '/dict_log_liks_{}.pkl'.format(args.mixture_components),'rb'))
    sigma_y = args_02.sigma_y
    df_X = pd.read_csv(data_dir+'/df_X.csv', index_col=0)
    print('[INFO] Done')

    # Select instance index for path planning
    if args.path_planning_index=='all':
        list_pp_index = selected_i_idx
    else:
        '''
        list_pp_index = Index selected in some way
        '''
        raise NotImplementedError()
    print(f'[INFO] number of instances for path planning is {len(list_pp_index)}')

    # Select intervention variables for path planning
    if args.intervention_variables=='all':
        list_intervention = var_info.list_select
    else:
        '''
        list_intervention = Explanatory variables selected in some way
        '''
        raise NotImplementedError()
    print(f'[INFO] Intervention variables are {list_intervention}')

    # 1/True for fixed. 0/False for intervene
    fixed_cont = ~np.isin(xy_for_stan.x_cont.reg.name, list_intervention)
    fixed_disc_reg = np.ones(len(xy_for_stan.x_disc.reg.name))==1 # [Not Supported] Unfix
    fixed_disc = np.ones(len(xy_for_stan.x_disc.clus.name))==1 # [Not Supported] Unfix
    fixed_zero_poi = ~np.isin(xy_for_stan.x_zero_poi.reg.name, list_intervention) # [Not Supported]
    list_pp_index = np.array(list_pp_index)[~df_X.loc[list_pp_index, list_intervention].isnull().any(axis=1).values].tolist()
    print(f'[INFO] Instances with missing values in intervention variables were ejected')
    print(f'[INFO] number of instances for path planning is {len(list_pp_index)}')

    list_destination = []
    for i, pp_i_idx in enumerate(list_pp_index):
        if args.destination_state=='criteria':
            list_destination.append(args.destination)
        elif args.destination_state=='count':
            list_destination.append(args.destination)
        else:
            raise NotImplementedError()

    params = {'list_pp_i_idx': list_pp_index,
              'destination_state': args.destination_state,
              'step': args.step,
              'list_destination': list_destination,
              'fixed_cont': fixed_cont,
              'fixed_disc_reg': fixed_disc_reg,
              'fixed_disc': fixed_disc,
              'fixed_zero_poi': fixed_zero_poi,
              'y_type': args.y_type,
              'k_class': args.mixture_components,
              'sigma_y': sigma_y,
              'upper_is_better': args.upper_is_better}

    num_movable = len(list_intervention)
    day_stamp = datetime.datetime.today().strftime('%y%m%d%H%M')
    save_params_dir = load_dir + f'/k{args.mixture_components}m{num_movable}_y_{args.y_type}_ds_{args.destination_state}_{day_stamp}/'
    utils.make_dir(save_params_dir)
    pickle.dump(params, open(save_params_dir + 'params.pkl', 'wb'))

    # destination_state, destination, intervention variables.
    with open(save_params_dir + 'params.txt', 'w') as f:
        f.write(f'destination_state: {args.destination_state}\n')
        f.write(f'destination: {list_destination[0]}\n')
        f.write(f'upper_is_better: {args.upper_is_better}\n')
        f.write(f'list_intervention:\n')
        for intervention in list_intervention:
            f.write(f'    {intervention}\n')
        f.close()

if __name__ == "__main__":
    main()
