#!/usr/bin/env python
# coding: utf-8
import numpy  as np
import pandas as pd
import argparse
import pickle

from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, confusion_matrix

import stan_base_class as stanb
import stan_preprocessing as stanp

import utils

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('-l', '--load_dir', default='example',
                        help='load directory name')
    return parser.parse_args()

def main():
    args = get_parser()   
    data_dir = './{}/data/'.format(args.load_dir)
    
    # Load
    print('[INFO] Loading')
    model = pickle.load(open(data_dir+'model.pkl', 'rb'))
    cv = pickle.load(open(data_dir+'cv.pkl', 'rb'))
    train_i_idx = pickle.load(open(data_dir+'train_i_idx.pkl', 'rb'))
    val_i_idx = pickle.load(open(data_dir+'val_i_idx.pkl', 'rb'))
    xy_for_stan = pickle.load(open(data_dir+'xy_for_stan.pkl', 'rb'))
    model_name = pickle.load(open(data_dir+'model_name.pkl', 'rb'))
    model_type = pickle.load(open(data_dir+'model_type.pkl', 'rb')) 
    print('[INFO] Done')
    
    print('[INFO] gridsearch and train model')
    # gscv and update model
    X_tv = xy_for_stan.x_reg.array[train_i_idx]
    y_tv = xy_for_stan.y[train_i_idx]
    model = stanp.set_gscv_best_param(X_tv, y_tv, model, model_name, model_type, cv, data_dir)        
    # Original model fitting using all training data
    xy_for_stan.make_model_and_pred(model_name, model, train_i_idx, val_i_idx, model_type=model_type)
    xy_for_stan.set_y_modeling() 
    print('[INFO] Done')
    
    # Score
    if model_type=='regressor':
        rmse_test = np.sqrt(mean_squared_error(xy_for_stan.y[val_i_idx],
                                               xy_for_stan.y_pred[val_i_idx]))
        print(f'[INFO] rmse_{model_name} (test): {rmse_test}')
    elif model_type=='classifier':
        y_pred_prob = xy_for_stan.model.predict_proba(xy_for_stan.x_reg.array[val_i_idx])[:,1]
        auc_test = roc_auc_score(xy_for_stan.y[val_i_idx], y_pred_prob)
        print(f'[INFO] auc_{model_name} (test): {auc_test}')
        cm = confusion_matrix(xy_for_stan.y[val_i_idx], xy_for_stan.y_pred[val_i_idx], labels=[0, 1])
    else:
        raise NotImplementedError()
     
    print('[INFO] Save')
    pickle.dump(xy_for_stan, open(data_dir + 'xy_for_stan.pkl', 'wb'))
    print('[INFO] Done')

if __name__ == '__main__':
    main()


