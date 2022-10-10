# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 21:21:05 2022

@author: reekm
"""
from autogluon.tabular import  TabularPredictor

class AutoGluonTabularRegression:
    def __init__(self,ag_config):
        super()
        # Setting target variable based on SRNL project (Last Column)
        # Need to change it based on new projects
        self.ag_config =  ag_config
        self.target_label_regression = self.ag_config.get_reg_target_variable()
        
        self.train_df_reg = self.ag_config.get_train_df_reg()
        self.test_df_reg = self.ag_config.get_test_df_reg()
        self.save_path = self.ag_config.get_regression_model_path()
        
        # training variables
        self.num_bag_folds=10 
        self.num_bag_sets=1
        self.num_stack_levels=3
        self.problem_type = "regression"
        self.eval_metric = "r2"
        self.presets = "best_quality"
        
        self.excluded_model_types_reg = [ 'NN_TORCH','NN_MXNET','FASTAI','CAT' ]
        self.refit_var = False
        
        # setting parameters for performance study
        self.extra_metric_regression =    ['root_mean_squared_error', 'mean_squared_error', 
                                               'mean_absolute_error', 'median_absolute_error', 
                                               'mean_absolute_percentage_error', 'r2']
        self.leaderboard_name = "leaderboard_result" + self.ag_config.get_CSV_EXTENSION()
        
    def ag_regression_training(self):
        self.predictor_regression = TabularPredictor(
                        label=self.target_label_regression,
                        problem_type=self.problem_type,
                        path=self.save_path,
                        eval_metric=self.eval_metric)

        self.predictor_regression.fit(train_data=self.train_df_reg,
                                 num_bag_folds= self.num_bag_folds, 
                                 num_bag_sets= self.num_bag_sets,
                                 num_stack_levels=self.num_stack_levels,
        #                         time_limit = 1200,
                                 excluded_model_types = self.excluded_model_types_reg,
        #                          hyperparameters=hyperparameters,
        #                          hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        #                          infer_limit_batch_size=infer_limit_batch_size,
        #                          refit_full = True,
                                 presets=self.presets)
        
        leader_board, summary, performance =  self.performance_reg(save_itr=True)
        
        return self.predictor_regression, leader_board, summary, performance
        
        
    def change_leaderboard_result_filename(self):
        add_text = input("input unique key to avoid overiting of result: ")
        self.leaderboard_name = add_text + self.leaderboard_name 
        
    def set_predictor_regression_by_path(self):
        self.predictor_regression = TabularPredictor.load(self.save_path,require_version_match=False)
        
    def leaderboard(self,save=False):
        leader_board = self.predictor_regression.leaderboard(self.test_df_reg,
                                                           extra_metrics=self.extra_metric_regression,
                                                           extra_info=True,
                                                           silent=True)
        if save == True:
            leader_board.to_csv(self.save_path+"/"+self.leaderboard_name)
        print("Regression leaderboard")
        print(leader_board)
        return leader_board
    
    def model_summary(self):
        result = self.predictor_regression.fit_summary(show_plot=False)
        return result
    
    def model_performance(self):
        performance = self.predictor_regression.evaluate(self.test_df_reg)
        print(f"Regression performance : {performance}")
        return performance
                       
    def performance_reg(self,save_itr=False):
        leader_board, summary, performance = 0,0,0
        if type(self.predictor_regression) == None:
            print("Please set the regression either by training (for first instance) \n or \n setting up save_path_reg to trained model repository \n")
            
        else:
            leader_board, summary, performance = self.leaderboard(save=save_itr), self.model_summary(), self.model_performance()
        
        return leader_board, summary, performance 
    
    