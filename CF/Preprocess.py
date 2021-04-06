# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:16:23 2021

@author: 陈彦虎
"""
import pandas as pd
import numpy as np
import random
class Preprocess:
    def __init__(self):
        df = pd.read_table('last_fm/user_artists.dat', sep="\t")
        userID = self.mapping(np.array(df['userID']))
        artistID = self.mapping(np.array(df['artistID']))
        data = np.vstack([userID,artistID]).T #获得完全数据
        self.train_set,self.test_set = self.spilt_data(data)#把数据分为训练集和测试集
        self.save()
        
    def mapping(self,X):
        #把中间有空缺的数据对应到连续的自然数上，以便节省存储空间
        a = np.array(list(set(X)))
        new_X = np.zeros_like(X)
        for i in range(len(X)):
            new_X[i] = np.where(a == X[i])[0][0]#把元组变为数字
        return new_X
    
    def spilt_data(self,data):
        #按行洗牌
        np.random.shuffle(data)#shuffle直接在原本数据上打乱，没有返回值
        train_set = data[len(data)//5:len(data)]
        test_set = data[0:len(data)//5]
        return train_set, test_set
    def save(self):
        np.save('train_set.npy',self.train_set)
        np.save('test_set.npy',self.test_set)
        