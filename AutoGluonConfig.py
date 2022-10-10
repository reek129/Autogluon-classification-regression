# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:18:52 2022

@author: reekm
"""

from config import save_path_classify,CSV, save_path_reg
from autogluon.tabular import TabularPredictor
from SRNLDatasetPreprocess import SRNLDataset
import pickle
import pandas as pd

class AutoGluonConfig:
    def __init__(self):
        super()
        self.srnl_dataset = SRNLDataset()
        # self.mm = MinMaxScaler(feature_range=(0,1))
        
    def save_sc_model_reg_all(self):
        sc = self.srnl_dataset.get_sc_reg()
        pickle.dump(sc, open( save_path_reg+'/'+'std_scaler_for_reg_all.pkl','wb'))
        
    def save_sc_model_reg_target(self):
        sc = self.srnl_dataset.get_sc_reg_target()
        pickle.dump(sc, open( save_path_reg+'/'+'std_scaler_for_reg_target.pkl','wb'))
        
    def load_sc_model_reg(self):
        path = save_path_reg+'/'+'std_scaler_for_reg_all.pkl'
        sc_all = pickle.load(open(path,'rb'))
        
        path = save_path_reg+'/'+'std_scaler_for_reg_target.pkl'
        sc_target = pickle.load(open(path,'rb'))
        
        return sc_all,sc_target
    
    def save_scaler_models_reg(self):
        self.save_sc_model_reg_all()
        self.save_sc_model_reg_target()
    
    def load_classify_predictor(self):
        predictor_classify = TabularPredictor.load(save_path_classify)
        return predictor_classify
    
    def load_regression_predictor(self):
        predictor_reg = TabularPredictor.load(save_path_reg)
        return predictor_reg
        
    

        
    def get_CSV_EXTENSION(self):
        return CSV
    
    def get_classification_model_path(self):
        return save_path_classify
    
    def get_regression_model_path(self):
        return save_path_reg

# Setting Target Variables
    def set_classify_target_variable(self,df_columns):
        self.srnl_dataset.set_classify_target_variable(df_columns)
        
        
    def get_classify_target_variable(self):
        return self.srnl_dataset.get_classify_target_variable()
        
    
# Setting Target Variables Regression
    def set_reg_target_variable(self,df_columns):
        self.srnl_dataset.set_reg_target_variable(df_columns)
        
        
    def get_reg_target_variable(self):
        return self.srnl_dataset.get_reg_target_variable()
    
# Setting Training and testing data for classification analysis
    def set_train_df_classify(self,train_df):
        self.srnl_dataset.set_train_df_classify(train_df)
        
        
    def get_train_df_classify(self):
        return self.srnl_dataset.get_train_df_classify()
    
    
    def set_test_df_classify(self,test_df):
        self.srnl_dataset.set_test_df_classify(test_df)
        
        
    def get_test_df_classify(self):
        return self.srnl_dataset.get_test_df_classify()
    
# Setting Training and testing data for Regression analysis
    def set_train_df_reg(self,train_df):
        self.srnl_dataset.set_train_df_reg(train_df)
        
    def get_train_df_reg(self):
        return self.srnl_dataset.get_train_df_reg()
    
    def set_test_df_reg(self,test_df):
        self.srnl_dataset.set_test_df_reg(test_df)
    
    def get_test_df_reg(self):
        return self.srnl_dataset.get_test_df_reg()


    
    
    
    
    
    