# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:04:33 2020

@author: 陈彦虎
"""
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def ctr_eval(kgcn, data, batch_size):
    start = 0
    end = start+ batch_size
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        
        user_indices = data[start:end, 0]
        item_indices = data[start:end, 1]
        labels = data[start:end, 2]
        
        
        auc, f1 = evaluation(kgcn, user_indices, item_indices, labels)
        auc_list.append(auc)
        f1_list.append(f1)
        
        start += batch_size
        end = start+ batch_size
        
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def evaluation(kgcn, user_indices, item_indices, labels):
    scores = kgcn._build_model(user_indices, item_indices, labels)
    scores = scores.data.numpy()
    auc = roc_auc_score(y_true = labels, y_score = scores)
    scores[scores>=0.5] = 1
    scores[scores< 0.5] = 0
    f1 = f1_score(y_true=labels, y_pred = scores)
    
    return auc,f1
