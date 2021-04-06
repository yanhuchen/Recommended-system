import numpy as np
import os

def load_data():
    #print('reading rating file ...')

    # reading rating file
    rating_file = 'ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    #将数据分为 train:eval:test = 6:2:2
    train_data, eval_data, test_data = dataset_split(rating_np)
    
    #print('reading KG file ...')
    print('reading KG file ...')

    # reading kg file
    kg_file = 'kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    """
    set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    example:
        x = set('runoob')
        y = set('google')
        x, y
        (set(['b', 'r', 'u', 'o', 'n']), set(['e', 'o', 'g', 'l']))   # 重复的被删除
        x | y         # 并集
        set(['b', 'e', 'g', 'l', 'o', 'n', 'r', 'u'])
    """
    

    kg = construct_kg(kg_np)#构造知识图谱
    adj_entity, adj_relation = construct_adj(kg, n_entity)#构造用户和物品的邻接矩阵
 
    print('data loaded.')
    
    
    
    return kg_np, kg, n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation

#将数据分割为train:eval:test
def dataset_split(rating_np):
    ratio =1;
    #print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
   
    left = set(range(n_ratings)) - set(eval_indices)
    
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    print(type(kg_np))
    kg = dict()
    """ dict() 函数用于创建一个字典。 """
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        #如果头实体不在kg中则在kg的字典中添加这个字段
        if head not in kg:
            kg[head] = []
        kg[head].append([tail, relation])
        #如果尾实体不在kg中则在kg的字典中添加这个字段
        if tail not in kg:
            kg[tail] = []
        kg[tail].append([head, relation])
    """貌似在添加过程中没有考虑关系的方向性"""
    #print(kg)
    return kg


def construct_adj(kg, entity_num):
    neighbor_sample_size = 8
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity] #找在字典中找entity的相关实体，字段是字符串
        n_neighbors = len(neighbors)
        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation