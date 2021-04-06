# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:57:19 2020

@author: 陈彦虎
"""
import torch.nn.functional as F
from abc import abstractmethod
import torch
import torch.nn as nn

LAYER_IDS = {}
def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]
    

class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
    
    '''如果类中有__call__方法，那么可以允许该类在创建实例后，实例像方法一样调用，例如：
        e = Entity(1, 2, 3) // 创建实例
        e(4, 5) //实例可以象函数那样执行，并传入x y值，修改对象的x y
    '''
    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,agg):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, agg)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,agg):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass
    
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # user_embeddings: shape = [batch_size, 1, 1, dim]
            user_embeddings = torch.Tensor.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
            
            #相当于把dim那一维做平均，降维
            #这一步等于论文中的公式(1)，实际意义是某用户对某种关系的重视程度
            #neighbor_relations: shape = [batch_size,1,n_neighbor,dim]
            #计算两者内积，即只计算最后一维相乘求和取平均
            #user_relation_scores: shape = [batch_size,1,n_neighbor]
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, axis=-1)
            #把分数区间映射到0,1之间，没有负数是否意味着所有关系对用户不会产生反效果？比如我很反感cxk？
            user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)*2-1
            
            # user_relation_scores_normalized: shape = [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, dim = -1) #在数据的末尾增加一维
            
            #neighbor_vectors: shape = [batch_size,1,neighbor_vectors,dim]
            # neighbors_aggregated: shape = [batch_size, -1, dim]
            #这一步相当于论文的公式(2) 
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=2)
            #而这一步并没有做归一化
        else:
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)
            
        return neighbors_aggregated
    
    
class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.relu, name=None): #agg为聚合层声明变量
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)
        
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,agg):
        # neighbors_agg: shape = [batch_size, -1, dim]，论文中用v_{N(v)}^{u}
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
       
        # output: shape = [batch_size, dim]
        output = torch.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        
        output = F.dropout(output,p=self.dropout)
        output = agg(output)
        
        return self.act(output)
    
class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)
        pass
    
class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)
