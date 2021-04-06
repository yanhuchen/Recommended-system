# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:48:16 2020

@author: 陈彦虎
"""
from Aggregator import SumAggregator
from Aggregator import ConcatAggregator
from Aggregator import NeighborAggregator

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class KGCN(nn.Module):
    def __init__(self,n_user,n_entity,n_relation,adj_entity,adj_relation,batch_size):
        super(KGCN,self).__init__()
        
        #参数设置的函数
        self._parse_args(n_user,n_entity,n_relation,adj_entity,adj_relation,batch_size)
        
        #shape=[用户数量，表示维度]
        self.user_emb_matrix = nn.Parameter(torch.rand(self.n_user, self.dim)*2-1)#随机生成[-1,1]的可训练
        
        #shape=[实体数量，表示维度]
        self.entity_emb_matrix = nn.Parameter(torch.rand(self.n_entity, self.dim)*2-1)
        
        #shape=[关系数量，表示维度]
        self.relation_emb_matrix = nn.Parameter(torch.rand(self.n_relation, self.dim)*2-1)
        # #声明的可训练权重矩阵，矩阵大小是embedding维度*embedding维度（self.dim*self.dim）
        self.agg1 = nn.Linear(self.dim, self.dim)
        #先做一层看看情况
        
        pass
    def forward(self, user_indices, item_indices, labels):
        #这三个参数分别是训练数据中的用户索引，item索引和标签
        
        _ = self._build_model(user_indices,item_indices,labels) #这里还包含两个方法get_neighbors，aggregate
        
        loss = self._build_train()#这一步倒像是计算损失函数的过程
        
        return loss
    
    def _parse_args(self,n_user,n_entity,n_relation,adj_entity,adj_relation,batch_size):
        """KGCN的各种参数"""
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        
        
        
        self.n_iter = 1
        self.batch_size = batch_size
        self.n_neighbor = 8 #采样邻居数
        self.dim = 16
        self.l2_weight = 1e-4
        self.lr = 5e-4
        aggregator = 'sum'
        
        #三种做汇聚的方法
        if aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: ")
            
    
    def _build_model(self,user_indices,item_indices,labels):
        self.user_indices = torch.from_numpy(user_indices)
        self.item_indices = torch.from_numpy(item_indices)
        self.labels = torch.from_numpy(labels)
        
        #在embedding矩阵中根据用户索引抽取相对应的用户embedding shape=[batch_size ,dim]
        self.user_embeddings = self.user_emb_matrix[self.user_indices]
        
        #根据item就是用户能够访问的那部分实体，找到它的采样邻居和关系
        entities, relations = self.get_neighbors()
       
        # [batch_size, dim]聚合后的item_embeddings，其实item_embeddings也是从entity_emb_matrix中抽取出来做聚合的
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)
        
        # [batch_size]计算内积
        self.scores = torch.sum(self.user_embeddings * self.item_embeddings, dim=1)
        self.scores_normalized = torch.sigmoid(self.scores)
        
        return self.scores_normalized
        
    
    def get_neighbors(self):
        seeds = torch.unsqueeze(self.item_indices, dim = 1) #扩展维度，使其成为张量
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):#self.n_iter=1 表示只找一层邻居？
             #如果i=0表示找中心点的邻居，如果i=1，表示找中心点邻居的邻居，以此类推
            
            #np.sequeeze去掉多余维度
            neighbor_entities = self.adj_entity[entities[i]]#数据类型numpy.array，行为batch_size列为采样邻居数
            neighbor_relations = self.adj_relation[entities[i]]#同上
            
            #追加邻居节点，方便多次迭代
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        
        return entities, relations
   
    def aggregate(self, entities, relations):
        #传入参数：entities的数据结构为一个长度2的list，entities[0]为batch_size的列向量,表示用户直接点击的中心节点，
        #entities[1]为shape = [batch_size,1,n_neighbor]的对应的张量，表示中心节点采样的邻居节点
        #relations的数据结构为一个长度1的list，和entities[0]对应的关系，shape = [batch_size,1,n_neighbor]
        aggregators = []  # store all aggregators
        
        #在事先生成的的对entity、relation的embedding矩阵中找到对应标号的embedding
        #entity_vectors: shape = list([batch_size,1,dim],[batch_size,1,n_neighbor,dim])
        #entity_vectors: shape = list([batch_size,1,n_neighbor,dim])
        entity_vectors = [self.entity_emb_matrix[i] for i in entities]
        relation_vectors = [self.relation_emb_matrix[i] for i in relations]
        
        for i in range(self.n_iter):
            #最外层的聚合的激活函数是tanh，中间的激活函数是relu
            if i == self.n_iter -1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act = torch.tanh, )
            else:
                aggregator = self.aggregator_class(self.batch_size,self.dim)
            aggregators.append(aggregator)
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop], #用户直接点击的中心节点embedding
                                    neighbor_vectors=torch.reshape(entity_vectors[hop + 1], shape),#中心节点的相邻节点的embedding
                                    neighbor_relations=torch.reshape(relation_vectors[hop], shape),#中心节点和相邻节点之间的关系embedding
                                    user_embeddings=self.user_embeddings,agg = self.agg1)
                #这返回的vector是经过聚合过后的，dim长度的向量
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = torch.reshape(entity_vectors[0], [self.batch_size, self.dim])
        #最后返回的最终层的聚合结果而忽略了中间层的聚合

        return res, aggregators
    
    
    """真正train模型在这个函数"""
    def _build_train(self):
        
        #tf.nn.sigmoid_cross_entropy_with_logits直接写成loss函数形式
        #计算公式为：output = labels * (-log(sigmoid(logits))) + (1-labels) * (-log(1-sigmoid(logits)))
        labels, logits = self.labels, self.scores
        output =  -labels * torch.log(torch.sigmoid(logits)) - (1-labels) * torch.log(1-torch.sigmoid(logits)) 
        self.base_loss = torch.mean(output)
        self.l2_loss = torch.norm(self.user_emb_matrix) ** 2/2 + torch.norm(self.entity_emb_matrix) **2/2 + torch.norm(self.relation_emb_matrix) ** 2/2
        '''
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + torch.norm(aggregator.weights) **2/2
        '''
        loss = self.base_loss + self.l2_weight * self.l2_loss#损失函数由三部分构成
        
        return loss
