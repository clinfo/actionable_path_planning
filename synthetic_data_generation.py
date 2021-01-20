#!/usr/bin/env python
# coding: utf-8

import pickle
import argparse

import numpy as np
import pandas as pd

import stan_base_class as stanb
import utils


def synthetic_dataset_3d(random_seed, size=200):
    '''
    Args:
        random_seed (int): 
        size (int): number of points generated per distribution
    Returns:
        df (DataFrame): explanatory variables and response variable (target)
        target (str): response variable name
    '''    
    # set random seed
    np.random.seed(random_seed)
    # distribution
    list_mu = [(0, -5, -5), (5, 0, -5), (5, 5, 0)]
    list_cov = [([5, 0, 0], [0, 1, 0], [0, 0, 1]),
                ([1, 0, 0], [0, 5, 0], [0, 0, 1]),
                ([1, 0, 0], [0, 1, 0], [0, 0, 5])]
    # Initialize
    target = 'target'
    np_x = np.array([])
    np_y = np.array([])
    np_z = np.array([]) 
    # generation
    for mu, cov in zip(list_mu, list_cov):
        values = np.random.multivariate_normal(mu, cov, size) 
        np_x = np.hstack([np_x, values[:, 0]])
        np_y = np.hstack([np_y, values[:, 1]])
        np_z = np.hstack([np_z, values[:, 2]])        
    df_dataset = pd.DataFrame(np.dstack([np_x, np_y, np_z])[0], columns=['x1', 'x2', 'x3'])
    np.random.normal(loc=0, scale=2, size=np_x.size)
    df_dataset[target] = df_dataset.sum(axis=1) + np.random.normal(loc=0, scale=2, size=np_x.size)    
    return df_dataset, target

def synthetic_dataset_5d(random_seed, size=200):
    '''
    Args:
        random_seed (int): 
        size (int): number of points generated per distribution
    Returns:
        df_dataset (DataFrame): explanatory variables and response variable (target)
        target (str): response variable name
    '''  
    # set random seed
    np.random.seed(random_seed)
    # distribution
    list_mu = [(0, -5, -5, -5, -5), 
               (5,  0, -5, -5, -5),
               (5,  5,  0, -5, -5),
               (5,  5,  5,  0, -5),
               (5,  5,  5,  5,  0)]
    list_cov = np.zeros((5, 5, 5))
    for c in range(5):
        for i in range(5):
            for j in range(5):
                if i==j:
                    if i==c:
                        list_cov[c, i, j] = 5
                    else:
                        list_cov[c, i, j] = 1
                elif i < j:
                    list_cov[c, i, j] = np.random.randn()
                elif i > j:
                    list_cov[c, i, j] = list_cov[c, j, i]
                else:
                    print('[ERROR]')   
    # Initialize
    target = 'target'
    np_x1 = np.array([])
    np_x2 = np.array([])
    np_x3 = np.array([])
    np_x4 = np.array([])
    np_x5 = np.array([])
    # generation
    for mu, cov in zip(list_mu, list_cov):
        values = np.random.multivariate_normal(mu, cov, size) 
        np_x1 = np.hstack([np_x1, values[:, 0]])
        np_x2 = np.hstack([np_x2, values[:, 1]])
        np_x3 = np.hstack([np_x3, values[:, 2]])
        np_x4 = np.hstack([np_x4, values[:, 3]])
        np_x5 = np.hstack([np_x5, values[:, 4]])      
    df_dataset = pd.DataFrame(np.dstack([np_x1, np_x2, np_x3, np_x4, np_x5])[0], columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    np.random.normal(loc=0, scale=2, size=np_x1.size)
    df_dataset[target] = df_dataset.sum(axis=1) + np.random.normal(loc=0, scale=2, size=np_x1.size)
    return df_dataset, target

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('-d', '--dataset', default='5d', choices=['3d', '5d'],
                        help='types of generated dataset')
    parser.add_argument('-s', '--save_dir', default='example',
                        help='save directory name')
    parser.add_argument('-r', '--random_seed', default=0,
                        help='random seed for dataset generation')
    return parser.parse_args()

def main():
    args = get_parser()
    print(f'[INFO] Save directory is {args.save_dir}')

    # make directories
    utils.make_dir(args.save_dir)
    data_dir = args.save_dir + '/data/'
    utils.make_dir(data_dir)

    # dataset generation
    print(f'[INFO] Start generating {args.dataset} dataset')
    if args.dataset=='3d':
        df_dataset, target = synthetic_dataset_3d(args.random_seed)
    elif args.dataset=='5d':
        df_dataset, target = synthetic_dataset_5d(args.random_seed)
    else:
        raise NotImplementedError()
    # generate variable information
    df_var = pd.DataFrame(columns=['item_name_other', 'item_type'])
    df_var['item_name_other'] = df_dataset.columns
    df_var['item_type'] = 'continuous'
    print('[INFO] Done')

    # display(df_dataset)

    # Save
    print('[INFO] Save dataset')
    df_X = df_dataset.drop(target, axis=1)
    df_y = pd.DataFrame(df_dataset[target])
    pickle.dump(df_X, open(data_dir + 'df_X.pkl','wb'))
    pickle.dump(df_y, open(data_dir + 'df_y.pkl','wb'))     
    df_X.to_csv(data_dir+'df_X.csv')
    df_y.to_csv(data_dir+'df_y.csv')
    df_var.to_csv(data_dir+'df_var.csv')
    print('[INFO] Done')

if __name__ == "__main__":
    main()

