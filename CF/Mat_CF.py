# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:53:21 2021

@author: 陈彦虎
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data #将数据分批次需要用到它
class MatCF(nn.Module):
    def __init__(self,m,n,k):
        super(MatCF,self).__init__()
        self.train_set = pd.read_csv('movielens10k/ua_base.txt',sep='\t',engine='python')
        self.test_set = pd.read_csv('movielens10k/ua_test.txt',sep='\t',engine='python')
        
        self.m = m #用户数
        self.n = n #项目数
        self.k = k #隐向量数
        self.user_emb = torch.nn.Parameter(torch.randn(m,k),requires_grad=True)
        self.item_emb = torch.nn.Parameter(torch.randn(k,n),requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,user,item):
        pre = torch.zeros(user.shape[0])
        for i in range(user.shape[0]):
            h = user[i]
            l = item[i]
            pre[i] = self.user_emb[h,:].unsqueeze(0).mm(self.item_emb[:,l].unsqueeze(1))
        pre = self.relu(4-pre)#把大于4的去掉
        pre = self.relu(4-pre) + 1#把小于0的去掉，评分区间为[1,5]

        return pre
    
    def get_batch(self,data_set,batch):
        torch.manual_seed(1)    # 种子，可复用
        torch_dataset = Data.TensorDataset(torch.tensor(data_set['user_id']),torch.tensor(data_set['item_id']),torch.tensor(data_set['rating']))
        loader = Data.DataLoader(
                dataset=torch_dataset,      # torch TensorDataset format
                batch_size=batch,           # 最新批数据
                shuffle=True)              # 是否随机打乱数据  
                #num_workers=5)              # 用于加载数据的子进程     
        
        return loader
        
    def test(self):
        test_pre = torch.zeros(len(self.test_set))
        for i in range(len(self.test_set)):
            h = self.test_set['user_id'][i]
            l = self.test_set['item_id'][i]
            #test_pre[i] = self.combine[h,l]
            test_pre[i] = self.user_emb[h,:].unsqueeze(0).mm(self.item_emb[:,l].unsqueeze(1))
        test_pre = self.relu(4-test_pre)#把大于4的去掉
        test_pre = self.relu(4-test_pre) + 1#把小于0的去掉，评分区间为[1,5]
        
        return torch.round(test_pre) 
        
        