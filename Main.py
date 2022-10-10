# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 08:12:32 2022

@author: reekm
"""

from config import Train
from graphs import Graphs
from DataPreprocessing import DataPreprocessingMethaneLeakageSRNL
from AutoGluonTabularClassification import AutoGluonTabularClassification
from AutoGluonTabularRegression import AutoGluonTabularRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler


from AutoGluonConfig import AutoGluonConfig
import numpy as np
import pandas as pd

class Main:
    def __init__(self):
        self.classification_cols_numbers = [5,6,7,8,9,10,11,12,14,15,17]
        self.regression_cols_numbers = [5,6,7,8,9,10,11,12,14,15,16]
        
        self.data_prep = DataPreprocessingMethaneLeakageSRNL()
        
        self.reg_pred_col = "regression_pred"
        self.classify_pred_col ='classify_pred'
        # only firstTime :- or creating Graphs
        # self.df,self.df_rus = data_prep.get_RUS_data()
        
        # During Analysis Directly load Balanced data
        self.df_rus = self.data_prep.get_RUS_data_only()
        
        self.df_rus['leakage'] = self.df_rus.apply (lambda row: self.data_prep.label_leakage2(row,self.df_rus.columns[-1]), axis=1)
        
        self.classification_cols = self.df_rus.columns[self.classification_cols_numbers].tolist()
        self.regression_cols = self.df_rus.columns[self.regression_cols_numbers].tolist()
        
        self.auto_gluon_config = AutoGluonConfig()
        
        # setting leakage label and deleting tracer concentration
        # Specific to SRNL
        # print(self.df_rus.columns)
        # print(self.df_classification.columns)
        # print(self.df_regression.columns)
        
    def for_training_prep(self):
        # self.df = pd.read_csv(RUS_dataset_name)
        self.df_classification = self.data_prep.get_subset_dataset_based_on_colnames(self.df_rus, self.classification_cols)
        self.df_regression = self.data_prep.get_subset_dataset_based_on_colnames(self.df_rus, colnames=self.regression_cols)
        
        
        
    def for_testing_prep_srnl(self,datapoints):
        self.df_classification = self.data_prep.get_subset_dataset_based_on_colnames(datapoints, self.classification_cols)
        self.df_regression = self.data_prep.get_subset_dataset_based_on_colnames(datapoints, colnames=self.regression_cols)
        
    def create_graphs(self):
        # self.X_rus = data_prep.get_RUS_data_only()        
        graphs = Graphs()
        
        filename = "corr_mat_after_feature_removal.png"
        graphs.graph_corr_matrix(self.df_rus,filename)
        
        filename = "env_var_corr_matrix.png"
        graphs.graph_corr_matrix(self.df_classification,filename)
        
        filename = "data_box_plot.png"
        graphs.data_boxplot_srnl(self.df_classification,filename)
        
        filename = "box_plot.png"
        graphs.data_each_boxplot_srnl(self.df_classification, filename)
        
    def auto_gluon_classification_model(self):
        print("Classification")
        # splitting data
        train_df_classify,test_df_classify = self.data_prep.split_data_train_test(self.df_classification)
        
        self.auto_gluon_config.set_classify_target_variable(self.df_classification.columns)
        self.auto_gluon_config.set_train_df_classify(train_df_classify)
        self.auto_gluon_config.set_test_df_classify(test_df_classify)
        
        
        agtc = AutoGluonTabularClassification( ag_config = self.auto_gluon_config)
        
        # for training
        predictor_classify,leader_board_classify, classification_summary, classify_performance  = agtc.ag_classification_training()
        
    def auto_gluon_regression_model(self):
        print("Regression")
        
        train_df_reg,test_df_reg = self.data_prep.split_data_train_test(self.df_regression)
        
        self.auto_gluon_config.set_reg_target_variable(self.df_regression.columns)
        
        self.auto_gluon_config.set_train_df_reg(train_df_reg)
        self.auto_gluon_config.set_test_df_reg(test_df_reg)
       
        agtr = AutoGluonTabularRegression(ag_config = self.auto_gluon_config)
        predictor_reg,leader_board_reg, regression_summary, regression_performance  = agtr.ag_regression_training()
        
    
        
            
    def get_regression_pred_sc(self,datapoint,predictor,target_label_regression,classify_col,sc_all,sc_target):
        # print(datapoint)
        if datapoint[self.classify_pred_col] == 0.0:
            return 0
            
        else:
            dp = datapoint[self.regression_cols_numbers]
            # print(dp)
            dp_scaled = sc_all.transform(np.array(dp).reshape(1,11))
            dp_scaled_df = pd.DataFrame(dp_scaled,columns=self.regression_cols)
            pred = predictor.predict(dp_scaled_df.drop(columns=[target_label_regression]))
            reg_value = sc_target.inverse_transform(np.array(pred).reshape(1,-1))
            print(f"prediction transform value: {reg_value.ravel()[0]}")
            return reg_value.ravel()[0]    
        
        
    def get_location_for_highest_concentration(self,data):
        for idx,row in data.iterrows():
            if row[self.regression_cols[-1]] == np.max(data[self.regression_cols[-1]]):
                print('Actual highest concentration location (lat,long)')
                act_lat,act_long = row[3],row[4]
                print(row[3],row[4])
            if row[data.columns[-1]] == np.max(data[data.columns[-1]]):
                print('Predicted highest concentration  location (lat,long)')
                pred_lat,pred_long = row[3],row[4]
                print(row[3],row[4])
                
        return act_lat,act_long,pred_lat,pred_long 
        
    def testing(self,datapoints):
        self.auto_gluon_config.set_classify_target_variable(self.df_classification.columns)
        target_label_classification= self.auto_gluon_config.get_classify_target_variable()
        predictor_classify = self.auto_gluon_config.load_classify_predictor()
        datapoints[self.classify_pred_col] = predictor_classify.predict(self.df_classification.drop(columns=[target_label_classification]))
        
        # extracting points with classification as leakage
        # temp = datapoints[datapoints[self.classify_pred_col] > 0]
        
        real_values = self.df_regression[self.df_regression.columns[-1]]
        print(f"Real values : {real_values}")
        sc_all,sc_target= self.auto_gluon_config.load_sc_model_reg()
        predictor_reg = self.auto_gluon_config.load_regression_predictor()
        target_label_regression = self.auto_gluon_config.set_reg_target_variable(self.df_regression.columns)
        target_label_regression = self.auto_gluon_config.get_reg_target_variable()
        
        predictions = [self.get_regression_pred_sc(row,predictor_reg,target_label_regression,self.classify_pred_col,sc_all,sc_target) for idx,row in datapoints.iterrows()  ]
        
        print(len(predictions),real_values.shape)
        
        datapoints[self.reg_pred_col] = predictions
        
        datapoints.to_csv("final_result.csv",index=False)
        self.get_location_for_highest_concentration(datapoints)
        print(datapoints.head())
        
        return datapoints
        
        
            

    # TODO
    # 1) save standardized scaler to pickle file to reload it in testing
    # 2) Create a testing code
    # 3) expand details of parameters during training for classification and regression models
        
    def main_method(self,train=False):
        
        if train == True:
            self.for_training_prep()
            # classification
            self.auto_gluon_classification_model()
            # regression
            self.auto_gluon_regression_model()
            self.auto_gluon_config.save_scaler_models_reg()
            
        else:
            
            print("Hello, Training for SRNL dataset was done on \n"+
                  " environmental variables for random datapoints \n"+
                  "...............\n"+
                  "For testing we will select datapoints at time T from original dataset\n"+
                  "Note:- We can use new datapoints too in this step")
            
            time = input("Input value from 12.0 to 18.0 with an interval of 0.1 like 16.1,16.2 : \n-----------------------\n")
            time = round(float(time),1)
            
            if time >= 12.0 or time <= 18.0:
                datapoints = self.data_prep.get_data_point_at_time_t(time)
                datapoints['leakage'] = datapoints.apply (lambda row: self.data_prep.label_leakage2(row,datapoints.columns[-1]), axis=1)
                self.for_testing_prep_srnl(datapoints)
            
            
            
            final_dataset = self.testing(datapoints)
            
            
            graphs = Graphs()
            subset_classify_df = final_dataset[final_dataset[self.classify_pred_col] > 0]
            graphs.results_datapoints_graphs(subset_classify_df, time)
            
        
        

        
        
    
        
    # def 
        
if __name__ == "__main__":
    main_class = Main()
    
    # Creating graphs based on preprocess data
    main_class.create_graphs()
    
    # Analysis
    main_class.main_method(train=Train)
    
    
    # testing
    main_class.main_method()
    
    
    


