#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import os
os.environ["OMP_NUM_THREADS"] = "4"
import pickle

from math import inf as infinity
import numpy as np
from multiprocessing import Pool, Manager

import path_search_algorithm as p
import utils

def write_txt(file_name, message):
    my_check = os.path.isfile(file_name)
    if not my_check:
        with open(file_name, 'w') as f:
            f.write(message + '\n')
    else:
        with open(file_name, 'a') as f:
            f.write(message + '\n')

def select_load_dir(model_id, k, num_movable, y_type, destination_state, daystamp):
    '''
    Args:
        model_id (str): 
        k (str): num of hb_class
        num_movable (int): number of movable values in tree search
        y_type (str): choices are [pred, glmm]
        destination_state (str): choices are [node, criteria, count]
        daystamp (str): daystamp of load_dir (and save_dir)
    Returns:
        load_dir (str): load_params dir (and save_dir)
    '''
    model_dir = f'./{model_id}/'    
    if daystamp=='':
        all_files = os.listdir(model_dir)       
        folders = sorted([f for f in all_files if f.startswith(f'k{k}m{num_movable}_y_{y_type}_ds_{destination_state}')])
        dir_name = folders[-1]
    else:
        dir_name = f'k{k}m{num_movable}_y_{y_type}_ds_{destination_state}_{daystamp}'
    # concat    
    load_dir = model_dir + dir_name + '/'
    return load_dir

def wrap_path_planning(list_args):
    path_planning(*list_args)

def path_planning(xy_for_exp, params, dict_emp_bayes, args, ts_dir, cypher_i, destination, model_type):
    '''
    Args:
        xy_for_exp ():
        params (dict):
        dict_emp_bayes (dict):
        args (Args):
        ts_dir (str):
        cypher_i (int):
        destination ()
    '''
    print(f'[INFO] Start pp_index: {cypher_i}')
    initial_x_coords = p.InitialXCoords(xy_for_exp, cypher_i)
    initial_r_x = tuple(np.zeros(len(xy_for_exp.x_reg.name), dtype=int))
    initial_node = p.Node(initial_x_coords, initial_r_x)
    # ------------------------------
    initial_node.x_fixed.set_fixed(initial_node,
                                   fixed_cont=params['fixed_cont'],
                                   fixed_disc_reg=params['fixed_disc_reg'],
                                   fixed_disc=params['fixed_disc'],
                                   fixed_zero_poi=params['fixed_zero_poi'])        
    p_class = initial_node.calc_p_class(args.update_flag, dict_emp_bayes)
    initial_node.neg_logprob = 0
    graph = p.Graph(initial_node, params['destination_state'], destination,
                    dict_emp_bayes, p_class, xy_for_exp, 
                    params['k_class'], params['sigma_y'], params['upper_is_better'],
                    params['step'], args.max_count, params['y_type'])

    # main process
    destination_node = graph.breadth_first_calc_distance(model_type)

    # check destination_node
    if destination_node is None:
        write_txt(ts_dir+'error_cypher.txt', 
                  f'{cypher_i}: Destination_node was not found')
        return
    if destination_node == initial_node:
        write_txt(ts_dir+'error_cypher.txt', 
                  f'{cypher_i}: Initial_node and destination_node were same')
        return 

    # find path
    nodes_on_path = graph.breadth_first_find_path(destination_node)

    # Save
    cypher_dir = ts_dir + f'i_{cypher_i}/'
    utils.make_dir(cypher_dir)
    file_name = cypher_dir + 'summary.txt'
    utils.show_steps(nodes_on_path, xy_for_exp, file_name, model_type=model_type)
    utils.x_coords_summary(initial_node, destination_node, xy_for_exp, file_name)
    pickle.dump(graph, open(cypher_dir + 'graph.pkl', 'wb'))
    pickle.dump(destination_node, open(cypher_dir + 'destination_node.pkl', 'wb'))
    print(f'[INFO] Done pp_index: {cypher_i}')
    return

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--load_dir', default='example',
                        help='load directory')
    parser.add_argument('--mixture_components', default=2, type=int,
                        help='Number of mixture components of surrogate model used for path planning')
    parser.add_argument('-m', '--num_movable', default=5, type=int,
                        help='Number of intervention variables')    
    parser.add_argument('-y', '--y_type', default='pred', choices=['pred', 'glmm'],
                        help='y used for path planning')
    parser.add_argument('-ds', '--destination_state', default='count',
                        choices=['criteria', 'count'],
                        help='destination_state selection')
    parser.add_argument('-d', '--daystamp', default='',
                        help='If blank, latest one would be selected')    
    parser.add_argument('-u', '--update_flag', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='If True, p_class would be updated')       
    parser.add_argument('-c', '--max_count', default=50000, type=int,
                        help='timeout iteration count')
    parser.add_argument('-mp', '--multi_processing', default=True,
                        help='If True, path planning would be performedn in parallel')   
    parser.add_argument('--n_proc', default=8, type=int,
                        help='number of parallel run') 
    return parser.parse_args()


def main():
    args = get_parser()

    load_dir = './' + args.load_dir + '/'
    data_dir = load_dir + 'data/'
    result_dir = load_dir + 'result/ex/'
    ts_dir = select_load_dir(args.load_dir, args.mixture_components, args.num_movable, args.y_type,
                             args.destination_state, args.daystamp)

    # Load
    print('[INFO] Loading')
    model_type = pickle.load(open(data_dir + 'model_type.pkl', 'rb'))
    dict_emp_bayes = pickle.load(open(result_dir + f'dict_emp_bayes_{args.mixture_components}.pkl', 'rb'))
    xy_for_exp = pickle.load(open(data_dir + '/xy_for_stan.pkl', 'rb'))
    params = pickle.load(open(ts_dir + 'params.pkl', 'rb'))
    print('[INFO] Done')

    # Path planning
    if args.multi_processing:
        pool = Pool(processes=args.n_proc)
        list_args = []
        for i in range(len(params['list_pp_i_idx'])):
            list_args.append((xy_for_exp, params, dict_emp_bayes, args, ts_dir, 
                              params['list_pp_i_idx'][i], params['list_destination'][i], model_type))
        pool.map(wrap_path_planning, list_args)
    else:
        for pp_i, destination in zip(params['list_pp_i_idx'], params['list_destination']):
            path_planning(xy_for_exp, params, dict_emp_bayes, args, ts_dir, pp_i, destination, model_type)

if __name__ == '__main__':
    main()
