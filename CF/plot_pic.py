# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:58:10 2021

@author: 陈彦虎
"""

import numpy as np
import matplotlib.pyplot as plt

train = np.load('train_acc_k10.npy')
test = np.load('test_acc_k10.npy')

plt.plot(np.linspace(1,len(train),len(train)),train, np.linspace(1,len(test),len(test)),test)
plt.legend(['train_acc','test_acc'],loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('the gap of scoring')
plt.savefig('loss_k10.png',dpi=720)