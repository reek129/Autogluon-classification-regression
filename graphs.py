# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 00:54:36 2022

@author: reekm
"""

from config import RUS_dataset_name

import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt
# %matplotlib inline

class Graphs:
    def __init__(self):
        super()
        self.graph_folder_name = 'graphs4'
        if os.path.isdir(self.graph_folder_name) == False:
            try:
                os.mkdir(self.graph_folder_name)
            except OSError as error:
                print(error)
            
        # print(f"check for graphs2: {os.path.isdir('graphs2')}")
    
        # font_size = decimal(range 1.0-2.0), fig_size = Tuple of length 2(Width, height)
    def graph_corr_matrix(self,dataframe,save_file_name,font_scale = 1.75,fig_size = (20,20)):
        sns.set(font_scale = font_scale)
        fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(dataframe.corr(),annot=True)
        plt.savefig(self.graph_folder_name+'/'+save_file_name)
        
    def data_boxplot_srnl(self,dataframe,save_file_name):
        sns.set(font_scale = 1.75)
        fig, ax = plt.subplots(nrows = 4,ncols=3,figsize=(15, 10))
        # fig = plt.figure()
        row,col = 0,0
        sns.set(rc={'figure.figsize':(1,0.25)})
        
        for i in range(dataframe.shape[1]):
            sns.boxplot(x=dataframe[dataframe.columns[i]],ax=ax[row,col],width=0.15)
        #     .set_title(df3.columns[i])
        #     .tight_layout()
            col = col+1
            if col % 3 == 0:
                col=0
                row = row+1
                
        plt.tight_layout()
        plt.savefig(self.graph_folder_name+'/'+save_file_name)
        
    def data_each_boxplot_srnl(self,dataframe,save_file_postfix):
        row,col = 0,0
        sns.set_style("whitegrid")
        for i in range(dataframe.shape[1]):
            fig, ax = plt.subplots(figsize=(3, 0.75))
            sns.boxplot(x=dataframe[dataframe.columns[i]]).set_title(dataframe.columns[i])
            plt.savefig(f'{self.graph_folder_name}/column_{i}_{save_file_postfix}.png')
            
    def results_datapoints_graphs(self,datapoints,time):
        temp_time = str(time).replace(".","_")
        print(f"time in string {temp_time}")
        
        datapoints_a,datapoints_p = datapoints,datapoints
        sns.set(font_scale = 2.5)
        fig, ax = plt.subplots(figsize=(15, 10))
        datapoints_a = datapoints_a.pivot(datapoints_a.columns[4], datapoints_a.columns[3], datapoints_a.columns[16])
        sns.heatmap(datapoints_a,linewidths=.3,xticklabels = 5, yticklabels = 3 )
        ax.set_title('Actual Concentration')
        
        plt.savefig(self.graph_folder_name+'/'+'heatmap2_time_'+temp_time+'_sec_actual_concentration.png')
        
        
        fig, ax = plt.subplots(figsize=(15, 10))
        datapoints_p = datapoints_p.pivot(datapoints_p.columns[4], datapoints_p.columns[3], datapoints_p.columns[19])
        sns.heatmap(datapoints_p,linewidths=.3,xticklabels = 5, yticklabels = 3 )
        ax.set_title('Predicted Concentration')
        plt.savefig(self.graph_folder_name+'/'+'heatmap2_time_'+temp_time+'_sec_predicted_concentration.png')
        
        
        

