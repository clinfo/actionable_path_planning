#!/usr/bin/env python
# coding: utf-8

import numpy  as np
import pandas as pd

class VarInfo():
    def __init__(self, df_var, raw_data_path):
        self.raw_data_path = raw_data_path 
        self.df_var        = df_var 
        self.list_select   = [] 
        self.target        = None 
        self.list_cont     = []
        self.list_disc     = [] 
        self.list_zero_poi = [] 
        
    def set_each_list_var(self, list_select, list_zero_poi, target):
        self.list_select = list_select
        self.target = target
        self.list_zero_poi = list_zero_poi
        
        list_select_rm_other = [v for v in self.list_select if v not in self.list_zero_poi + [self.target]]
        df_var_select = self.df_var.query('item_name_other in @list_select_rm_other')
        self.list_cont = df_var_select[(df_var_select['item_type']=='continuous')]['item_name_other'].values.tolist()
        self.list_disc = df_var_select[(df_var_select['item_type']=='nominal')|\
                                       (df_var_select['item_type']=='boolean')|\
                                       (df_var_select['item_type']=='ordinal')]['item_name_other'].values.tolist()
        
    def delete_columns(self, delete_columns):
        '''
        Args:
            delete_columns (list[str]):
        '''
        for d_column in delete_columns:
            if d_column in self.list_select:
                self.list_select.remove(d_column)
            if d_column in self.list_cont:
                self.list_cont.remove(d_column)
            if d_column in self.list_disc:
                self.list_disc.remove(d_column)
            if d_column in self.list_zero_poi:
                self.list_zero_poi.remove(d_column)
        
class XyData():
    def __init__(self):
        self.y          = None # np.array
        self.y_name     = ''
        self.y_modeling = None # np.array
        self.y_pred     = None # np.array
        self.y_age_mean = None # np.array
        self.x_reg      = ArrayName()
        self.x_cont     = XFeatures()
        self.x_disc     = XFeaturesDisc()
        self.x_zero_poi = XFeatures()
        self.size       = 0 
        self.gamma      = None
        self.intercept  = None
        self.coef       = None
        self.coef_param = None
        self.y_GLMM     = None

    def set_gamma(self, gamma):
        self.gamma = gamma
        
    def set_coef_intercept(self, dict_emp_bayes_k, k_class):
        if k_class==1: 
            beta1_mixed = np.dot(self.gamma, dict_emp_bayes_k['beta1']).T[0]
        else:
            beta1_mixed = np.dot(self.gamma, dict_emp_bayes_k['beta1'])
        
        beta2_mixed = np.dot(self.gamma, dict_emp_bayes_k['beta2'])
        y_GLMM = beta1_mixed + np.sum((beta2_mixed * self.x_reg.array), axis=1)        
        self.intercept  = beta1_mixed
        self.coef       = beta2_mixed
        self.coef_param = beta2_mixed * self.x_reg.array
        self.y_GLMM     = y_GLMM
        
    def set_categorical_coding(self, categorical_coding):
        self.categorical_coding = categorical_coding        
        
class XFeatures():
    def __init__(self):
        self.index    = []
        self.raw      = ArrayName()
        self.reg      = ArrayName()
        self.clus     = ArrayName()
        
class XFeaturesDisc():
    def __init__(self):
        self.index    = []
        self.raw      = ArrayName()
        self.reg      = ArrayName()
        self.clus     = XClusDiscForStan()         

class ArrayName():
    def __init__(self):
        self.array = None # np.array
        self.name = []
        
        
class XClusDiscForStan():
    def __init__(self):
        self.n_cat = []
        self.alpha = []
        self.array = None
        self.name = []
