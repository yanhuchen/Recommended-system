# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:24:50 2021

@author: 陈彦虎
"""

import torch
import torch.nn as nn
import numpy as np
from Mat_CF import MatCF


batch = 1000
epochs  = 50
m,n,k = 944,1683,10

cf = MatCF(m,n,k)
loader = cf.get_batch(cf.train_set,batch)

optim = torch.optim.Adam(cf.parameters(),lr = 0.05,weight_decay=0.0)
mse = nn.MSELoss()
train_acc = torch.zeros(epochs)
test_acc = torch.zeros(epochs)

for epoch in range(epochs):
    for step,(batch_u,batch_i,batch_r) in enumerate(loader):
        pre = cf(batch_u, batch_i)
        
        loss = mse(pre, batch_r.float())
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        #print(loss.data)
        train_acc[epoch] += (torch.abs(batch_r.float() - pre.data)).sum()
    train_acc[epoch] /= cf.train_set.shape[0]
        
    with torch.no_grad():
        test_pre = cf.test()
        test_real = torch.tensor(cf.test_set['rating'])
        test_acc[epoch] = (torch.abs(test_real - test_pre)).sum() / len(test_pre)
        
        
        print(epoch,': train_acc',train_acc[epoch],'test_acc:',test_acc[epoch])

train_acc = np.array(train_acc)
test_acc = np.array(test_acc)

np.save('train_loss_k10.npy',train_acc)
np.save('test_acc_k10.npy',test_acc)


