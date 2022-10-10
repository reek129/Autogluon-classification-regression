# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 23:17:33 2022

@author: reekm
"""

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
from config import RUS_dataset_name

class SRNLDataset:
    def __init__(self):
        super()
        self.sc_reg = StandardScaler()
        
    def get_sc_reg(self):
        return self.sc_reg
    
    def get_sc_reg_target(self):
        df = pd.read_csv(RUS_dataset_name)
        
        df = df[df[df.columns[-1]] > 0]
        
        df_r = df.iloc[:,-1:]
        sc_target = StandardScaler()
        sc_target.fit(df_r)
        return sc_target
    
    def convert_to_df(self,data,columns):
        return pd.DataFrame(data,columns=columns)
    
# Setting Training and testing data for classification analysis
    def set_train_df_classify(self,train_df):
        self.train_df_classify = train_df
        
    def get_train_df_classify(self):
        return self.train_df_classify
    
    
    def set_test_df_classify(self,test_df):
        self.test_df_classify = test_df
        
    def get_test_df_classify(self):
        return self.test_df_classify

# Setting Training and testing data for Regression analysis
    def set_train_df_reg(self,train_df):
        train_df_reg = train_df[train_df[train_df.columns[-1]] > 0]
        print(f"Regression Train Datasset Shape: {train_df_reg.shape}")
        
        reg_training_set = train_df_reg.iloc[:,:].values
        
        # change sc to mm for MinMax scaler
        reg_training_set_scaled = self.sc_reg.fit_transform(reg_training_set)
        
        self.train_df_reg2 = self.convert_to_df(reg_training_set_scaled, train_df.columns)

        
    def get_train_df_reg(self):
        return self.train_df_reg2
    
    
    def set_test_df_reg(self,test_df):
        test_df_reg = test_df[test_df[test_df.columns[-1]] > 0]
        print(f"Regression Test Datasset Shape: {test_df_reg.shape}")
        
        reg_testing_set = test_df_reg.iloc[:,:].values
        
        # change sc to mm for MinMax scaler
        reg_test_set_scaled = self.sc_reg.transform(reg_testing_set)
        
        self.test_df_reg2 = self.convert_to_df(reg_test_set_scaled, test_df.columns)
        
    def get_test_df_reg(self):
            return self.test_df_reg2
        
    # Setting Target Variables
    def set_classify_target_variable(self,df_columns):
        self.target_label_classification = df_columns[-1]
        
    def get_classify_target_variable(self):
        return self.target_label_classification
        
    
# Setting Target Variables Regression
    def set_reg_target_variable(self,df_columns):
        self.target_label_reg = df_columns[-1]
        
    def get_reg_target_variable(self):
        return self.target_label_reg    


        