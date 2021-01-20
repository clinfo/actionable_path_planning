#!/usr/bin/env python
# coding: utf-8
import pickle
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import stan_base_class as stanb
import stan_preprocessing as stanp
import utils

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

def get_model(model_name, model_type):
    '''
    Hyperparameter setting will be updated based on gridsearch.
    Args:
        model_name (str): choices=[XG: XGBoost, RF: RandomForest, SVM: SupportVectorMachine]
        model_type (str): choices=[regressor, classifier]
    Returns:
        model: In classification setting, predict_proba function is needed.
    '''
    if model_type=='regressor':
        if model_name=='XG':
            model = XGBRegressor(random_state=0)
        elif model_name=='RF':
            model = RandomForestRegressor(criterion='mse', n_jobs=4, random_state=0)
        elif model_name=='SVM':
            model = SVR(C=1, kernel='rbf')
        else:
            raise NotImplementedError()   
    elif model_type=='classifier':
        if model_name=='XG':
            model = XGBClassifier(max_depth=5, n_estimators=80, learning_rate=0.1, objective='binary:logistic',
                                  n_jobs=4, importance_type='total_gain', random_state=0)
        elif model_name=='RF':
            model = RandomForestClassifier(max_depth=7, n_estimators=200, class_weight=None, n_jobs=4, random_state=0)
        elif model_name=='SVM':
            model = SVC(C=1, kernel='rbf', probability=True, random_state=0)   
        else:
            raise NotImplementedError() 
    else:
        raise NotImplementedError() 
    return model

def get_parser():
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('-l', '--load_dir', default='example',
                        help='load directory name')
    parser.add_argument('-r', '--random_seed', default=0,
                        help='random seed for tran_test_split')
    parser.add_argument('-mn', '--model_name', default='XG',
                        choices=['XG', 'RF', 'SVM'],
                        help='original model')    
    parser.add_argument('-mt', '--model_type', default='regressor',
                        choices=['regressor', 'classifier'],
                        help='model task type')     
    parser.add_argument('-np', '--nan_prep', default='imp_reg_class',
                        choices=['drop', 'imp_reg_class', 'imp_reg_round'],
                        help='fill nan method')     
    parser.add_argument('-c', '--categorical_coding', default='dummy_not_drop',
                        choices=['dummy_not_drop', 'dummy_drop', 'effect'],
                        help='categorical coding method')
    parser.add_argument('--cv', default=5, 
                        help='cv setting')    
    parser.add_argument("--rm_outlier", default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='If True, remove outlier for surrogate modeling')
    return parser.parse_args()

def main():
    args = get_parser()
    
    # load
    print('[INFO] Start loading')
    data_dir = args.load_dir + '/data/'
    if os.path.exists(data_dir+'df_X.pkl'):
        df_X = pickle.load(open(data_dir+'df_X.pkl', 'rb'))
    else:
        df_X = pd.read_csv(data_dir+'df_X.csv', index_col=0)
    if os.path.exists(data_dir+'df_y.pkl'):
        df_y = pickle.load(open(data_dir+'df_y.pkl', 'rb'))
    else:
        df_y = pd.read_csv(data_dir+'df_y.csv', index_col=0)
    df_var = pd.read_csv(data_dir+'df_var.csv', index_col=0)
    print('[INFO] Done')
    
    df_Xy = pd.concat([df_X, df_y], axis=1)
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, shuffle=True, random_state=args.random_seed)
    
    # select items
    list_select = df_X.columns.tolist()
    target = df_y.columns[0]
    # [Not Implemented] Zero Inflated Poisson
    list_zero_poi = []
    var_info = stanb.VarInfo(df_var, data_dir)
    var_info.set_each_list_var(list_select, list_zero_poi, target)
    # get (outlier) index
    train_i_idx = X_train.index
    val_i_idx = X_test.index
    train_i_idx_rm_outlier, val_i_idx_rm_outlier =                    stanp.get_outlier_idx(df_X, args.rm_outlier, var_info, train_i_idx, val_i_idx)
    
    # get mean and std values for standardization
    cont_mean, cont_std = stanp.get_mean_std(df_X, train_i_idx, var_info.list_cont)
    zero_poi_mean, zero_poi_std = stanp.get_mean_std(df_X, train_i_idx, var_info.list_zero_poi) # [Not Supported] 
    
    # nan preprocessing
    df_Xy_fna, train_i_idx, val_i_idx, train_i_idx_rm_outlier, val_i_idx_rm_outlier=                    stanp.fillna(df_Xy, var_info, args.nan_prep,
                                 train_i_idx, val_i_idx, train_i_idx_rm_outlier, val_i_idx_rm_outlier, 
                                 model_type=args.model_type)    
    
    # make data for stan
    xy_for_stan = stanp.XyForStan(df_Xy_fna)
    # set rm_outlier index for surrogate modeling
    xy_for_stan.set_train_val_idx(train_i_idx_rm_outlier, val_i_idx_rm_outlier)
    xy_for_stan.set_categorical_coding(args.categorical_coding)
    # xyのset
    xy_for_stan.set_xy(var_info, df_Xy_fna, cont_mean, cont_std, zero_poi_mean, zero_poi_std)    
    
    # model setting
    # [Future Implementation] External model loading
    model = get_model(args.model_name, args.model_type)  
    # Save
    print('[INFO] Save')
    pickle.dump(model, open(data_dir + 'model.pkl', 'wb'))
    pickle.dump(xy_for_stan, open(data_dir + 'xy_for_stan.pkl', 'wb'))
    pickle.dump(args.cv, open(data_dir + 'cv.pkl', 'wb'))
    df_Xy_fna.to_csv(data_dir + 'df_Xy_fna.csv', index=False) 
    pickle.dump(train_i_idx, open(data_dir + 'train_i_idx.pkl', 'wb'))
    pickle.dump(val_i_idx, open(data_dir + 'val_i_idx.pkl', 'wb'))
    pickle.dump(var_info, open(data_dir + 'var_info.pkl', 'wb'))
    pickle.dump(args.model_name, open(data_dir + 'model_name.pkl', 'wb'))
    pickle.dump(args.model_type, open(data_dir + 'model_type.pkl', 'wb'))
    print('[INFO] Done')

if __name__ == "__main__":
    main()

