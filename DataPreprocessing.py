# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 00:26:43 2022

@author: reekm
"""

from config import file_name1,file_name2,test_size,RUS_dataset_name,full_dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler



class DataPreprocessingMethaneLeakageSRNL:
    def __init__(self):
        self.random_state = 42
        self.test_size =0.2


    def label_leakage2(self,row,column):
        if row[column] > 0 :
            return 1
        else:
            return 0
        
    def load_dataset_from_config(self):
        for i in range(len(full_dataset)):
            if i == 0 :
                df = pd.read_excel(full_dataset[i])
            else:
                df2 = pd.read_excel(full_dataset[i])
                df = pd.concat([df,df2])
                
        return df
    
    def get_dataset(self):
        
        df = self.load_dataset_from_config()
        
        df['leakage'] = df.apply (lambda row: self.label_leakage2(row,df.columns[-1]), axis=1)
        
        return df 
    
    def get_train_test_split(self,df):
        
        train_df,test_df = train_test_split(df,test_size=test_size,random_state=self.random_state)
        
        return train_df,test_df
    
    def get_RUS_data(self):
        df = self.get_dataset()
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]
        
        rus = RandomUnderSampler(random_state=self.random_state)
        X_rus, y_rus = rus.fit_resample(X, Y)
        
        print('Normal dataset shape %s' % Counter(Y))
        print('Resampled dataset shape %s' % Counter(y_rus))
        
        X_rus.to_csv(RUS_dataset_name,index=False)
        
        return df,X_rus
    
    def get_RUS_data_only(self):
        df = pd.read_csv(RUS_dataset_name)
        return df
    
    def get_subset_dataset_based_on_colnames(self, dataframe, colnames):
        return dataframe[colnames]
    
    def split_data_train_test(self,dataframe):
        train_df, test_df =  train_test_split(dataframe,test_size=self.test_size,random_state=self.random_state)
        return train_df, test_df
    
    def get_data_point_at_time_t(self,time):
        data= self.get_RUS_data_only()
        print(type(data[data.columns[0]]))
    #     print(data.head())
        dp = data[data[data.columns[0]] == time]
        print(dp.head())
        
        return dp
        
        
    
    
    