# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 16:14:58 2018

@author: gmx
"""

import tensorflow as tf
import numpy as np

deep_input = []
embedding_input = []
y_label = []

with open('dataset') as train_set:
    for line in train_set.readlines():
        batch_xs = line.strip().split(",")[2:]
        batch_ys = line.strip().split(",")[1:2]
        print(batch_ys)
        int_feature = batch_xs[0:12]
        cate_feature = batch_xs[13:]
        
        i = 0
        while i < len(int_feature):
            if(int_feature[i] == ''):
                int_feature[i] = '0'
            
            int_feature[i] = int(int_feature[i])
            i += 1
        i = 0
        while i < len(cate_feature):
            if cate_feature[i] == '':
                cate_feature[i] = '00000000'
            i += 1
            
            
        deep_input.append(int_feature)
        embedding_input.append(cate_feature)
        y_label.append(batch_ys)
        #print(int_feature)
        #print(cate_feature)