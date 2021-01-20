#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import os
os.environ["OMP_NUM_THREADS"] = "8"

import pickle
import random

from math import inf as infinity
import numpy as np
import pandas as pd

import path_search_algorithm as p
import utils

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--load_dir', default='example', 
                        help='load directory')
    parser.add_argument('--mixture_components', default=2, type=int,
                        help='number of mixture components of surrogate model used for path planning')
    parser.add_argument('-m', '--num_movable', default=5, type=int,
                        help='number of intervention variables')    
    parser.add_argument('-y', '--y_type', default='pred', choices=['pred', 'glmm'],
                        help='y used in path planning')
    parser.add_argument('-ds', '--destination_state', default='count',
                        choices=['criteria', 'count'],
                        help='Destination state selection')
    parser.add_argument('-d', '--daystamp', default='',
                        help='If blank, latest one would be selected')     
    parser.add_argument('-u', '--update_flag', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='If True, p_class would be updated') 
    parser.add_argument('-i', '--num_iter', default=10, type=int,
                        help='number of baseline paths used for the calculation of actionability score')
    parser.add_argument('--random_seed', default=0, type=int,
                        help='baseline path selection seed')
    parser.add_argument('--n_bins', default=20, type=int,
                        help='number of bins for plot')    
    return parser.parse_args()

def select_load_dir(load_dir, k, num_movable, y_type, destination_state, daystamp):
    '''
    Args:
        load_dir (str): 
        k (str): num of hb_class
        num_movable (int): number of movable values in tree search
        y_type (str): choices are [pred, glmm]
        destination_state (str): choices are [node, criteria, count]
        daystamp (str): daystamp of load_dir (and save_dir)
    Returns:
        load_dir (str): load_params dir (and save_dir)
    '''
    model_dir = f'./{load_dir}/'    
    if daystamp=='':
        all_files = os.listdir(model_dir)       
        folders = sorted([f for f in all_files if f.startswith(f'k{k}m{num_movable}_y_{y_type}_ds_{destination_state}')])
        dir_name = folders[-1]
    else:
        dir_name = f'k{k}m{num_movable}_y_{y_type}_ds_{destination_state}_{daystamp}'
    load_dir = model_dir + dir_name + '/'
    return load_dir

def main():
    args = get_parser()
    random.seed(args.random_seed)
    load_dir = './' + args.load_dir + '/'
    data_dir = load_dir + 'data/'
    result_dir = load_dir + 'result/ex/'
    ts_dir = select_load_dir(args.load_dir, args.mixture_components, args.num_movable, args.y_type,
                             args.destination_state, args.daystamp)

    print('[INFO] Loading')
    xy_for_exp = pickle.load(open(data_dir + 'xy_for_stan.pkl', 'rb'))
    dict_emp_bayes = pickle.load(open(result_dir + f'dict_emp_bayes_{args.mixture_components}.pkl', 'rb'))
    params = pickle.load(open(ts_dir + 'params.pkl', 'rb'))
    model_type = pickle.load(open(data_dir + 'model_type.pkl', 'rb'))
    print('[INFO] Done')

    df_result = pd.DataFrame(columns=['min_distance'] + [f'rand_distance_{i}' for i in range(args.num_iter)],
                             index=params['list_pp_i_idx'])  
    list_detour = []
    list_line = []
    list_noresult = []

    for pp_i in params['list_pp_i_idx']:
        pp_dir = ts_dir + f'i_{pp_i}/'
        my_check = not os.path.isdir(pp_dir)
        if my_check:
            print(f'[INFO] pp_i: {pp_i} has no results.')
            list_noresult.append(pp_i)
            continue   

        destination_node = pickle.load(open(pp_dir+'destination_node.pkl', 'rb'))
        summary = open(pp_dir+'summary.txt').read()
        num_step = len([line for line in summary.splitlines() if 'step' in line]) -1
        df_result.loc[pp_i, 'min_distance'] = destination_node.tentative_distance    

        if num_step==abs(np.array(destination_node.r_x)).max():
            list_line.append(pp_i)
        elif num_step==abs(np.array(destination_node.r_x)).sum():
            pass
        else:
            list_detour.append(pp_i)    

        destination_abs = abs(np.array(destination_node.r_x))
        list_move = []   
        for j in range(len(destination_abs)):
            for _ in range(destination_abs[j]):
                np_move = np.zeros(len(destination_abs), dtype=int)
                if destination_node.r_x[j] > 0:
                    np_move[j] = 1
                else:
                    np_move[j] = -1
                list_move.append(tuple(np_move))
        for k in range(args.num_iter):
            random.shuffle(list_move)
            initial_x_coords = p.InitialXCoords(xy_for_exp, pp_i)
            initial_r_x = tuple(np.zeros(len(xy_for_exp.x_reg.name), dtype=int))
            initial_node = p.Node(initial_x_coords, initial_r_x)
            initial_node.x_fixed.set_fixed(initial_node,
                                           fixed_cont=params['fixed_cont'],
                                           fixed_disc_reg=params['fixed_disc_reg'],
                                           fixed_disc=params['fixed_disc'],
                                           fixed_zero_poi=params['fixed_zero_poi'])
            p_class = initial_node.calc_p_class(args.update_flag, dict_emp_bayes)
            initial_node.set_y_and_class_lp(xy_for_exp.model, params['y_type'],
                                            params['k_class'], xy_for_exp, 
                                            dict_emp_bayes, params['sigma_y'],
                                            p_class, model_type=model_type)
            initial_node.tentative_distance = 0
            current_node = initial_node
            for i in range(len(list_move)):
                n_r_x = tuple(np.array(current_node.r_x) + np.array(list_move[i]))
                n_node = p.Node(initial_node.x, n_r_x, params['step'])
                n_node.set_y_and_class_lp(xy_for_exp.model, params['y_type'], params['k_class'],
                                          xy_for_exp, dict_emp_bayes, params['sigma_y'], p_class, model_type=model_type)
                n_node.set_neg_logprob()
                new_tentative_distance = current_node.tentative_distance + current_node.distance_to(n_node)
                n_node.tentative_distance = new_tentative_distance
                current_node = n_node
            # brief check
            if destination_node.r_x!=current_node.r_x:
                print('[ERROR] arrived r_x is different from destination.')
            df_result.loc[pp_i, f'rand_distance_{k}'] = current_node.tentative_distance

    # calc actionability score
    dif_distance = utils.calc_dif_distance(df_result)
    # plot actionability score
    utils.plot_dif_distance(dif_distance, ts_dir, args.n_bins, 0)
    df_actionability_score = pd.DataFrame(dif_distance, columns=['Actionability score'], index=df_result.index)
    # output actionability score
    df_actionability_score.to_csv(ts_dir + 'df_actionability_score.csv') 

    # output_summary
    with open(ts_dir+'distance_summary.txt', mode='w') as f:
        f.write('Path-planned instances: {}\n'.format(len(params['list_pp_i_idx'])))
        f.write('Detour instances: {}\n'.format(len(list_detour)))
        f.write('Straight instances: {}\n'.format(len(list_line)))
        f.write('Initial and destination consistent instances: {}\n'.format(len(list_noresult)))
        f.close()

if __name__ == '__main__':
    main()    

