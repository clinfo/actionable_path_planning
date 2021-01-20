#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd

from sklearn.linear_model import BayesianRidge
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import copy
import stan_base_class as stanb
from utils import timeit
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV


def impute_regressor_and_transform(X_train, X_test):
    print('[INFO] Impute by regressor')
    model = BayesianRidge()
    imp = IterativeImputer(estimator=model, max_iter=10, sample_posterior=False,
                           initial_strategy='mean', random_state=0)
    imp.fit(X_train)
    X_train_imp = pd.DataFrame(imp.transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test),
                              columns=X_test.columns,
                              index=X_test.index)
    return X_train_imp, X_test_imp

def impute_classifier_and_transform(X_train, X_test):
    print('[INFO] Impute by classifier')
    rf = RandomForestClassifier(n_estimators=200, max_depth=3, n_jobs=4, random_state=0)
    imp = IterativeImputer(estimator=rf, max_iter=10, sample_posterior=False,
                      initial_strategy='mean', random_state=0)
    imp.fit(X_train)
    X_train_imp = pd.DataFrame(imp.transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test),
                              columns=X_test.columns,
                              index=X_test.index)
    return X_train_imp, X_test_imp

def round_nominal(X, X_original, list_nominal):
    ''' round nominal feature
    Args:
        X (pd.DataFrame): DataFrame to round nominal values
        X_original (pd.DataFrame): original DataFrame to get min/max values
        list_nominal (list[str]): list of nominal features
    Returns:
        X (pd.DataFrame): nominal values are rounded
    '''
    for feature in list_nominal:
        X_nominal_tmp = X[feature]
        X_nominal_tmp_max = X_original[feature].max()
        X_nominal_tmp_min = X_original[feature].min()
        X_nominal_tmp_round = np.floor(X_nominal_tmp + 0.5)
        X_nominal_tmp_round[X_nominal_tmp_round>X_nominal_tmp_max] = X_nominal_tmp_max
        X_nominal_tmp_round[X_nominal_tmp_round<X_nominal_tmp_min] = X_nominal_tmp_min
        # update
        X[feature] = X_nominal_tmp_round
    return X

def fillna(df_Xy_select, var_info, nan_prep, 
           train_i_idx, val_i_idx, train_i_idx_rm_outlier, val_i_idx_rm_outlier, model_type):
    '''
    Args:
        df_Xy_select (pd.DataFrame): 
        var_info (VarInfo): 
        nan_prep (str): choices=[drop, imp_reg_round, imp_reg_class]
        train_i_idx (list[int]): 
        val_i_idx (list[int]): 
        train_i_idx_rm_outlier (list[int]):
        val_i_idx_rm_outlier (list[int]):
        model_type (str): choices = ['regressor', 'classifier']
    Returns:
        df_Xy_select (pd.DataFrame):
        train_i_idx (list[int]): updated if nan_prep is drop
        val_i_idx (list[int]): updated if nan_prep is drop
        train_i_idx_rm_outlier (list[int]): updated if nan_prep is drop
        val_i_idx_rm_outlier (list[int]): updated if nan_prep is drop
    '''
    print('[INFO] Preprocessing nan values')
    if nan_prep=='drop':
        print('[INFO] remove instances with any nan values')
        df_Xy_select.dropna(inplace=True)
        np_idx = np.array(range(0, df_Xy_select.shape[0]))
        train_i_idx = np_idx[np.isin(df_Xy_select.index, train_i_idx)]
        val_i_idx = np_idx[np.isin(df_Xy_select.index, val_i_idx)]
        train_i_idx_rm_outlier = np_idx[np.isin(df_Xy_select.index, train_i_idx_rm_outlier)]
        val_i_idx_rm_outlier = np_idx[np.isin(df_Xy_select.index, val_i_idx_rm_outlier)]
    else:
        df_Xy_select_train = df_Xy_select.iloc[train_i_idx]
        df_Xy_select_val = df_Xy_select.iloc[val_i_idx]
        y = df_Xy_select[var_info.target]
        cont_features = var_info.df_var.query('item_type == "continuous"')['item_name_other'].to_list()
        nominal_features = var_info.df_var.query('item_type != "continuous"')['item_name_other'].to_list()
        if (var_info.target not in cont_features)&(var_info.target not in nominal_features):
            if model_type=='regressor':
                cont_features.append(var_info.target)
            elif model_type=='classifier':
                nominal_features.append(var_info.target)
            else:
                raise NotImplementedError()
        list_cont = [f for f in df_Xy_select.columns if f in cont_features]
        list_nominal = [f for f in df_Xy_select.columns if f in nominal_features]
        if nan_prep=='imp_reg_round':
            print('[INFO] Impute continuous values, then round for nominal_values')
            X_train_imp, X_test_imp = impute_regressor_and_transform(df_Xy_select_train, df_Xy_select_val)
            X_train_imp = round_nominal(X_train_imp, df_Xy_select, list_nominal)
            X_test_imp = round_nominal(X_test_imp, df_Xy_select, list_nominal)
            df_Xy_select = pd.concat([X_train_imp, X_test_imp], axis=0).sort_index()
        elif nan_prep=='imp_reg_class':
            print('[INFO] Impute continuous values by regressor only using continuous part of data,')
            print('[INFO] then impute nominal values by classifier using all data.')
            # split cont/nominal
            X_train_cont = df_Xy_select_train[list_cont]
            X_train_nominal = df_Xy_select_train[list_nominal]
            X_test_cont = df_Xy_select_val[list_cont]
            X_test_nominal = df_Xy_select_val[list_nominal]
            X_train_cont_imp, X_test_cont_imp = impute_regressor_and_transform(X_train_cont, X_test_cont)
            X_train_concat = pd.concat([X_train_cont_imp, X_train_nominal], axis=1)
            X_test_concat = pd.concat([X_test_cont_imp, X_test_nominal], axis=1)
            X_train_imp, X_test_imp = impute_classifier_and_transform(X_train_concat, X_test_concat)
            df_Xy_select = pd.concat([X_train_imp, X_test_imp], axis=0).sort_index() # 一応sort
        else:
            print('[ERROR] Unknown nan preprocessing method.')  
        if var_info.target not in df_Xy_select.columns:
            df_Xy_select[var_info.target] = y
            
    return df_Xy_select, train_i_idx, val_i_idx, train_i_idx_rm_outlier, val_i_idx_rm_outlier

def get_mean_std(df_Xy, train_i_idx, cols):
    '''
    Args:
        df_Xy (pd.DataFrame):
        train_i_idx (list[int]):
        cols (list[str]): 
    Return:
        ar_mean (np.array[float]): 
        ar_std (np.array[float]): 
    '''
    if cols:
        ar_mean = np.array(df_Xy.loc[train_i_idx, cols].mean())
        ar_std = np.array(df_Xy.loc[train_i_idx, cols].std())
    else:
        ar_mean = None
        ar_std = None
    return ar_mean, ar_std

def get_outlier_idx(df_Xy, rm_outlier, var_info, train_i_idx, val_i_idx):
    '''
    Args:
        df_Xy (pd.DataFrame): 
        rm_outlier (int): 0 or 1 for flag
        var_info (VarInfo):
        train_i_idx (list[int]):
        val_i_idx (list[int]):
    Return:
        train_i_idx_rm_outlier (list[int]):
        val_i_idx_rm_outlier (list[int]):
    '''
    if rm_outlier:
        print('[INFO] remove outlier index')
        df_X_tmp = df_Xy[list(set(var_info.list_select) - set(var_info.list_disc))]
        df_X_tr = df_X_tmp.loc[train_i_idx]
        df_X_te = df_X_tmp.loc[val_i_idx]
        x_mean = df_X_tr.mean()
        x_std = df_X_tr.std()
        df_X_tr_norm = (df_X_tr - x_mean) / x_std
        df_X_te_norm = (df_X_te - x_mean) / x_std
        sr_bool_tr = (df_X_tr_norm.abs().max(axis=1) < 3)
        sr_bool_te = (df_X_te_norm.abs().max(axis=1) < 3)
        train_i_idx_rm_outlier = sr_bool_tr.loc[train_i_idx][sr_bool_tr.loc[train_i_idx]].index.tolist()
        val_i_idx_rm_outlier = sr_bool_te.loc[val_i_idx][sr_bool_te.loc[val_i_idx]].index.tolist()
    else:
        train_i_idx_rm_outlier = train_i_idx
        val_i_idx_rm_outlier = val_i_idx
    return train_i_idx_rm_outlier, val_i_idx_rm_outlier

class XyForStan(stanb.XyData):
    def __init__(self, df_Xy):
        ''' Constructor
        Args: 
            df_Xy (pd.DataFrame): to get index and size
        '''
        super().__init__()
        self.original_index = df_Xy.index.tolist() 
        self.size         = df_Xy.shape[0]
        self.y_pred       = None
        self.y_modeling   = None
        self.x_cont       = XFeaturesForStan()
        self.x_disc       = XFeaturesDiscForStan()
        self.x_zero_poi   = XFeaturesForStan()
        self.x_reg        = ArrayName()
        self.train_i_idx  = []
        self.val_i_idx    = []
        self.model        = None        
        self.categorical_coding = None
        self.model_age_mean = None
    
    def set_xy(self, var_info, df_Xy, cont_mean, cont_std, zero_poi_mean, zero_poi_std):
        ''' set xy data to instance
        Args:
            var_info (VarInfo): 
            df_Xy (pd.DataFrame): 
            cont_mean (np.array[float]):
            cont_std (np.array[float]):
            zero_poi_mean (np.array[float]):
            zero_poi_std (np.array[float]): 
        '''
        self.y = np.array(df_Xy[var_info.target])
        self.y_name = var_info.target
        self.x_cont._set_x_cont(var_info, df_Xy, cont_mean, cont_std)
        self.x_disc._set_x_disc(var_info, df_Xy, self.categorical_coding)
        self.x_zero_poi._set_x_zero_poi(var_info, df_Xy, zero_poi_mean, zero_poi_std)
        df_X_reg = pd.concat([pd.DataFrame(self.x_cont.reg.array, columns=self.x_cont.reg.name),
                              pd.DataFrame(self.x_disc.reg.array, columns=self.x_disc.reg.name),
                              pd.DataFrame(self.x_zero_poi.reg.array, columns=self.x_zero_poi.reg.name)],
                             axis=1)
        self.x_reg.array = np.array(df_X_reg)
        self.x_reg.name = df_X_reg.columns.tolist()        
        
    def set_train_val_idx(self, train_i_idx_rm_outlier, val_i_idx_rm_outlier):
        ''' set train and val index to instance
        Args:
            train_i_idx_rm_outlier (list[int]):
            val_i_idx_rm_outlier (list[int]):
        '''
        self.train_i_idx = train_i_idx_rm_outlier
        self.val_i_idx = val_i_idx_rm_outlier

    def make_model_and_pred(self, model_name, model, train_i_idx, val_i_idx, model_type='regressor'):
        ''' set model based on train data and predicted result.
        Args:
            model_name (str): 
            model (Model): 
        '''
        x_train = self.x_reg.array[train_i_idx]
        x_val   = self.x_reg.array[val_i_idx]
        y_train = self.y[train_i_idx]
        y_val   = self.y[val_i_idx]
        self.model = _make_model(model_name, model, x_train, x_val, y_train, y_val, model_type)
        self.y_pred = self.model.predict(self.x_reg.array)
    
    def set_y_modeling(self):
        '''
        set y_modeling to instance
        '''
        self.y_modeling = self.y_pred                             
            
    def set_categorical_coding(self, categorical_coding):
        ''' set categorical_coding to instance
        Args:
            categorical_coding (str): 
        '''
        self.categorical_coding = categorical_coding
                
class XFeaturesForStan():
    def __init__(self):
        self.index    = []
        self.raw      = ArrayName()
        self.reg      = ArrayName()
        self.clus     = ArrayName()
        self.standard = XStandardForStan()

    def _set_x_cont(self, var_info, df_Xy, cont_mean, cont_std):
        if var_info.list_cont:
            print('[INFO] Preprocessing continuous variables')
            df_X_cont = df_Xy[var_info.list_cont]
            self.raw.name = df_X_cont.columns.tolist()
            for var_name in self.raw.name:
                self.index.append(var_info.df_var.query('item_name_other == @var_name').index[0])
            self.raw.array  = np.array(df_X_cont)
            self.standard.mean = cont_mean
            self.standard.std = cont_std
            df_X_cont_norm = (df_X_cont - cont_mean) / cont_std
            self.reg.array = np.array(df_X_cont_norm)
            self.reg.name = df_X_cont_norm.columns.tolist()
            self.clus.array = np.array(df_X_cont_norm)
            self.clus.name = df_X_cont_norm.columns.tolist()
            print('[INFO] Done')

    def _set_x_zero_poi(self, var_info, df_Xy, zero_poi_mean, zero_poi_std):
        if var_info.list_zero_poi:
            print('[INFO] Preprocessing zero_poi variables')
            df_X_zero_poi = df_Xy[var_info.list_zero_poi]
            self.raw.name = df_X_zero_poi.columns.tolist()
            for var_name in self.raw.name:
                self.index.append(var_info.df_var.query('item_name_other == @var_name').index[0])
            self.raw.array  = np.array(df_X_zero_poi)
            self.standard.mean = zero_poi_mean
            self.standard.std = zero_poi_std
            df_X_zero_poi_norm = (df_X_zero_poi - zero_poi_mean) / zero_poi_std
            self.reg.array = np.array(df_X_zero_poi_norm)
            self.reg.name = df_X_zero_poi_norm.columns.tolist()            
            self.clus.array = np.array(df_X_zero_poi)
            self.clus.name = df_X_zero_poi.columns.tolist()
            print('[INFO] Done.')
            
class XStandardForStan():
    def __init__(self):
        self.mean = None # np.array
        self.std  = None # np.array
        
class ArrayName():
    def __init__(self):
        self.array = None # np.array
        self.name = []
        
class XFeaturesDiscForStan():
    def __init__(self):
        self.index    = []
        self.raw      = ArrayName()
        self.reg      = ArrayName()
        self.clus     = XClusDiscForStan()        
        
    def _set_x_disc(self, var_info, df_Xy, categorical_coding):
        if var_info.list_disc:
            print('[INFO] Preprocessing discrete variables')
            df_X_disc = df_Xy[var_info.list_disc].astype(int)
            self.raw.name = df_X_disc.columns.tolist()
            for var_name in self.raw.name:
                self.index.append(var_info.df_var.query('item_name_other == @var_name').index[0])
            self.raw.array = np.array(df_X_disc)
            self.clus.n_cat = df_X_disc.nunique().tolist()
            for n_cat in self.clus.n_cat:
                self.clus.alpha.append(np.ones(n_cat))
            df_X_disc_clus = copy.deepcopy(df_X_disc)
            df_X_disc_clus.loc[:,df_X_disc_clus.min(axis=0).values==0] += 1
            self.clus.array = np.array(df_X_disc_clus)
            self.clus.name = df_X_disc_clus.columns.tolist()
            if categorical_coding=='dummy_not_drop': # dummy_coding without drop_first
                df_X_disc_dummy = pd.get_dummies(df_X_disc, drop_first=False, columns=df_X_disc.columns)
                self.reg.array = np.array(df_X_disc_dummy)
                self.reg.name = df_X_disc_dummy.columns.tolist()
            elif categorical_coding=='dummy_drop': # dummy_coding with drop_first [Not Supported]
                df_X_disc_dummy = pd.get_dummies(df_X_disc, drop_first=True, columns=df_X_disc.columns)
                self.reg.array = np.array(df_X_disc_dummy)
                self.reg.name = df_X_disc_dummy.columns.tolist()
            elif categorical_coding=='effect': # effect_coding [Not Supported]
                df_X_disc_effect = pd.DataFrame()
                for x_disc in df_X_disc.columns:
                    df_X_disc_tmp = df_X_disc[[x_disc]]
                    df_X_disc_dummy_tmp = pd.get_dummies(df_X_disc_tmp, drop_first=False,
                                                         columns=[x_disc]).astype(int)
                    df_X_disc_dummy_tmp.loc[df_X_disc_dummy_tmp.iloc[:,0]==1] -= 1
                    df_X_disc_effect = pd.concat([df_X_disc_effect, df_X_disc_dummy_tmp.iloc[:,1:]], axis=1)
                
                self.reg.array = np.array(df_X_disc_effect)
                self.reg.name = df_X_disc_effect.columns.tolist()                   
            
            print('[INFO] Done')

class XClusDiscForStan():
    ''' 
    Attributes:
        n_cat (list[int]): 
        alpha (list[np.array]): 
        array (np.array): 
        name (list): 
    '''
    def __init__(self):
        self.n_cat = []
        self.alpha = []
        self.array = None
        self.name = []        
        
def _make_model(model_name, model, x_train, x_val, y_train, y_val, model_type='regressor'):
    '''
    Args:
        model_name (str): 
        model (Model): untrained, defined model
        x_train (np.array):
        x_val (np.array):
        y_train (np.array):
        y_val (np.array):
        model_type (str): choices=['regressor', 'classifier']
    Returns:
        model (Model): trained model
    '''
    def _modeling_RF(X_train_, y_train, model):
        ''' Sub function for modeling RF
        Args:
            X_train_ (pd.DataFrame): feature-selected X_train
            y_train (pd.DataFrame): 
            model (Model): 
        Return: 
            model (Model): trained model
        ''' 
        model.fit(X_train_, np.array(y_train).ravel())
        return model   
    
    def _modeling_XG(X_train_, y_train, X_val_, y_val, model, model_type, n_class):
        ''' Sub function for modeling XG
        Args:
            X_train_ (pd.DataFrame): feature-selected X_train
            y_train (pd.DataFrame): 
            X_val_ (pd.DataFrame): feature-selected X_val
            y_val (pd.DataFrame): 
            model (Model): 
            model_type (str): 
            n_class (int): number of class on classifier            
        Return: 
            model (Model): trained model
        '''
        if model_type=='regressor':
            eval_metric='rmse'
        elif (model_type=='classifier')&(n_class==2):
            eval_metric='logloss'
        elif model_type=='classifier':
            raise NotImplementedError()       
        model.fit(X_train_, np.array(y_train).ravel(),
                  eval_set=[(X_val_, np.array(y_val).ravel())],
                  eval_metric=eval_metric,
                  verbose=False)
        return model
    
    def _modeling_SVM(X_train_, y_train, X_val_, y_val, model, model_type, n_class):
        ''' Sub function for modeling SVM
        Args:
            X_train_ (pd.DataFrame): feature-selected X_train
            y_train (pd.DataFrame): 
            X_val_ (pd.DataFrame): feature-selected X_val
            y_val (pd.DataFrame): 
            model (Model): 
            model_type (str): 
            n_class (int):
        Return: 
            model (Model): trained model
        ''' 
        if model_type=='regressor': 
            model.fit(X_train_, np.array(y_train).ravel())
        elif (model_type=='classifier')&(n_class!=2): # multiclass
            raise NotImplementedError()
        else: # 2class
            model.fit(X_train_, np.array(y_train).ravel())    
        return model            
    
    if model_type=='classifier':
        n_class = 2 # [TODO] multiclass is not supported
    else:
        n_class = None
    if model_name=='RF':
        model = _modeling_RF(x_train, y_train, model)
    elif model_name=='XG':
        model = _modeling_XG(x_train, y_train, x_val, y_val, model, model_type=model_type, n_class=n_class)
    elif model_name=='SVM':
        model = _modeling_SVM(x_train, y_train, x_val, y_val, model, model_type=model_type, n_class=n_class)
    else:
        print('[ERROR] Not implemented')
    return model     

@timeit
def set_gscv_best_param(X_tv, y_tv, model, model_name, model_type, cv, save_dir):
    '''
    Args:
        X_tv (DataFrame):
        y_tv (DataFrame):
        model (Model): model
        model_name (str): 
        model_type (str): choices = ['classifier', 'regressor']
        cv (StratifiedKFold/KFold/int/list[tuple[1d-array[int]]]): 
        save_dir (str): 
    Return:
        model (Model): model w/ updated params    
    '''
    param_area = set_param_area(model_name, model_type)
    if model_type=='regressor':
        gscv = GridSearchCV(model, param_grid=param_area, cv=cv, verbose=2, n_jobs=4)
    elif model_type=='classifier':
        gscv = GridSearchCV(model, param_grid=param_area, cv=cv, verbose=2, n_jobs=4, scoring='roc_auc')
    else:
        raise NotImplementedError()
    gscv.fit(X_tv, np.array(y_tv).ravel())
    print(f'[INFO] Best parameters: {gscv.best_params_}')
    print(f'[INFO] Best score:      {gscv.best_score_}')
    # Save cv results
#     df_score = pd.DataFrame(gscv.cv_results_)
#     df_score = df_score[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
#     df_score_sorted = df_score.sort_values(by=['rank_test_score'])    
#     df_score_sorted.to_csv(save_dir + f'gscv_score_{model_name}.csv')
    model = set_param_to_model(model, model_name, model_type, gscv.best_params_)                        
    return model

def set_param_area(model_name, model_type):
    '''
    Args:
        model_name (str): 
        model_type (str):
    Return:
        param_area (dict[list]):
    '''
    if model_name=='RF':
        param_area = {'max_depth': [3, 5, 7], 'n_estimators': [30, 50, 70, 90]}        
    elif model_name=='XG':
        param_area = {'max_depth': [3, 5, 7], 'n_estimators': [30, 50, 70, 90]}        
    elif model_name=='SVM': 
        param_area = {'C': [0.001, 0.01, 0.1, 1, 10, 100]} 
    return param_area  

def set_param_to_model(model, model_name, model_type, best_params):
    '''
    Args:
        model (Model):
        model_name (str):
        model_type (str):
        best_params (dict[int/float]): 
    Return:
        model (Model): 
    '''
    if model_name=='RF':
        model.set_params(max_depth=best_params['max_depth'], 
                         n_estimators=best_params['n_estimators'])
    elif model_name=='XG':
        model.set_params(max_depth=best_params['max_depth'], 
                         n_estimators=best_params['n_estimators'])  
    elif model_name=='SVM':
        model.set_params(C=best_params['C'])
    else:
        raise NotImplementedError()
    return model
