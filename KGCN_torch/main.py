# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:37:39 2020

@author: 陈彦虎
"""

import loaddata as Load
from KGCN import KGCN
from evaluate import ctr_eval 

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#导入数据
kg_np, kg, n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data,adj_entity, adj_relation = Load.load_data()

batch_size = 128
kgcn = KGCN(n_user,n_entity,n_relation,adj_entity,adj_relation,batch_size)

lr = 0.002
optimizer = torch.optim.Adam(kgcn.parameters(),lr = lr, weight_decay=1e-4)

n_epochs = 20


for step in range(n_epochs):
    np.random.shuffle(train_data)
    start = 0
    end = start + batch_size
    L = 0
    while start + batch_size <= train_data.shape[0]:
        user_indices = train_data[start:end, 0]
        item_indices = train_data[start:end, 1]
        labels = train_data[start:end, 2]
        
        loss = kgcn(user_indices, item_indices, labels)
        
        start += batch_size
        end = start + batch_size

        '''如果我们把每个batch_size的loss加起来再做反向传播会好些吗？'''
        optimizer.zero_grad() #把该网络中所有参数的梯度降为0
        loss.backward()#反向传递
        optimizer.step()#以学习效率优化，将梯度带进去优化参数
        '''如果不执行以上三步，那么训练参数就不会更新？'''
        L = L+loss
    print("损失函数：", L)
    
    # CTR evaluation
    train_auc, train_f1 = ctr_eval(kgcn, train_data, batch_size)
    eval_auc, eval_f1 = ctr_eval(kgcn, eval_data, batch_size)
    test_auc, test_f1 = ctr_eval(kgcn, test_data, batch_size)

    print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
    #print('epoch %d    train auc: %.4f f1: %.4f' %(step, train_auc, train_f1))

