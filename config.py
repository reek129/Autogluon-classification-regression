# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 00:23:39 2022

@author: reekm
"""

CSV = ".csv"

file_name1 = 'initial_dataset\\part1_1.xlsx'
file_name2 = 'initial_dataset\\part2.xlsx'

full_dataset = [file_name1,file_name2]

RUS_dataset_name = "full_data_rus.csv"



save_path_classify = 'AutoML_classify_v2_29July_rus3'
save_path_reg = 'AutoML_reg_v2_29July_rus_sc3'
Train = True

flag_rus = 1     # 1 for RandomUnderSampling 0 for RandomOversampling
test_size =0.2
