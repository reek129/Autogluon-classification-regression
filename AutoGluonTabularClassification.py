# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:58:55 2022

@author: reekm
"""

# from AutoGluonConfig import AutoGluonConfig
from autogluon.tabular import  TabularPredictor
# import pandas as pd

class AutoGluonTabularClassification:
    def __init__(self,ag_config):
        super()
        
        # Setting target variable based on SRNL project (Last Column)
        # Need to change it based on new projects
        self.ag_config =  ag_config
        self.target_label = self.ag_config.get_classify_target_variable()
        
        self.train_df_classify = self.ag_config.get_train_df_classify()
        self.test_df_classify = self.ag_config.get_test_df_classify()
        self.save_path = self.ag_config.get_classification_model_path()
        
        # SETTING PARAMETERS FOR TRAINING
        self.num_bag_folds=10
        self.num_bag_sets = 2
        self.num_stack_levels=1
        self.problem_type = 'binary'
        self.presets = "best_quality"
        # excluding some models for faster results (Based on Analysis)
        # will add them for further testing
        self.exclude_model = ['NN_TORCH','NN_MXNET','FASTAI','GBM']
        self.refit_var = False
        
        # setting parameters for performance study
        self.extra_metric_classification =  ['accuracy', 'balanced_accuracy', 'f1', 
                                             'f1_macro', 'f1_micro', 'f1_weighted',
                                             'roc_auc', 'roc_auc_ovo_macro', 'average_precision',
                                             'precision', 'precision_macro', 'precision_micro',
                                             'precision_weighted', 'recall', 'recall_macro',
                                             'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
        self.leaderboard_name = "leaderboard_result" + self.ag_config.get_CSV_EXTENSION()
        
        
        
        
        # print(type(self.ag_config))
        # print(self.target_label)
        # print(self.ag_config.get_classification_model_path())
    
        
    def ag_classification_training(self):
        self.predictor_classify = TabularPredictor(label=self.target_label,
                             path=self.save_path,
                             problem_type = self.problem_type).fit(
                                        train_data = self.train_df_classify,
                                        # num_bag_folds=self.num_bag_folds, 
                                        # num_bag_sets=self.num_bag_sets,
                                        # num_stack_levels=self.num_stack_levels,
                                        excluded_model_types = self.exclude_model,
                                        # refit_full = self.refit_var,
                                        presets=self.presets)
         
                                 
        leader_board, summary, performance =  self.performance_classify(save_itr=True)
        
        return self.predictor_classify,leader_board, summary, performance 
    
    def change_leaderboard_result_filename(self):
        add_text = input("input unique key to avoid overiting of result: ")
        self.leaderboard_name = add_text + self.leaderboard_name 
                                 
    def set_predictor_classify_by_path(self):
        self.predictor_classify = TabularPredictor.load(self.save_path,require_version_match=False)
          
    def leaderboard(self,save=False):
        leader_board = self.predictor_classify.leaderboard(self.test_df_classify,
                                                           extra_metrics=self.extra_metric_classification,
                                                           extra_info=True,
                                                           silent=True)
        if save == True:
            leader_board.to_csv(self.save_path+"/"+self.leaderboard_name)
        print("leaderboard")
        print(leader_board)
        return leader_board
        
    def model_summary(self):
        result = self.predictor_classify.fit_summary(show_plot=False)
        return result
    
    def model_performance(self):
        performance = self.predictor_classify.evaluate(self.test_df_classify)
        print(f"performance{performance}")
        return performance
                       
    def performance_classify(self,save_itr=False):
        leader_board, summary, performance = 0,0,0
        if type(self.predictor_classify) == None:
            print("Please set the classifier either by training (for first instance) \n or \n setting up save_path_classify to trained model repository \n")
            
        else:
            leader_board, summary, performance = self.leaderboard(save=save_itr), self.model_summary(), self.model_performance()
        
        return leader_board, summary, performance 
        