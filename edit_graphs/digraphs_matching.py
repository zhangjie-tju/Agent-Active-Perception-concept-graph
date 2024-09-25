import networkx as nx
import pandas as pd
import six.moves.cPickle as cPickle 
import gzip
from gensim.models import KeyedVectors

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = '/home/user/Documents/ZJ/concept-graphs/edit_graphs/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)


import json, copy
file = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/rebuild_object.json"
file_1 = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/rebuild_object_1.json"
with open(file, 'r') as f:
    data_raw = json.load(f)

with open(file_1, 'r') as f1:
    data_raw_1 = json.load(f1)

def sentence_vector(sentence, model, num_features=300):
    words = sentence.split()
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in words:
        if word in model:
            num_words += 1
            feature_vector = np.add(feature_vector, model[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

##使用json文件中的信息建图（分别为场景生成的图和描述生成的图）
G = nx.DiGraph()
G1 = nx.DiGraph()
data = copy.deepcopy(data_raw)
data_1 = copy.deepcopy(data_raw_1)
for item in data:
    relation = item['object_relation']
    if relation == "none of these":
        continue
    G.add_node(item['object1'] ['id'],label = item['object1']["object_tag"])
    G.add_node(item['object2'] ['id'],label = item['object2']["object_tag"])
    if relation[0] == "a":
        weight_relation = {"a on b":1,"a in b":2}.get(item['object_relation'])
        G.add_edge(item['object1'] ['id'], item['object2'] ['id'],weight = weight_relation)
    elif relation[0] == "b":
        weight_relation = {"b on a":1,"b in a":2}.get(item['object_relation'])
        G.add_edge(item['object2'] ['id'], item['object1'] ['id']  ,weight = weight_relation)
    else:
        print(item['object_relation'])
        
for item in data_1:
    relation = item['object_relation']
    if relation == "none of these":
        continue
    G1.add_node(item['object1'] ['id'],label = item['object1']["object_tag"])
    G1.add_node(item['object2'] ['id'],label = item['object2']["object_tag"])
    if relation[0] == "a":
        weight_relation = {"a on b":1,"a in b":2}.get(item['object_relation'])
        G1.add_edge(item['object1'] ['id'], item['object2'] ['id'],weight = weight_relation)
    elif relation[0] == "b":
        weight_relation = {"b on a":1,"b in a":2}.get(item['object_relation'])
        G1.add_edge(item['object2'] ['id'], item['object1'] ['id']  ,weight = weight_relation)
    else:
        print(item['object_relation'])


def node_match(n1, n2):
    """比较节点数据，确保标签相对应"""
    vector1 = sentence_vector(n1['label'],word2vec_model)
    vector2 = sentence_vector(n2['label'],word2vec_model)
    sim = cosine_similarity([vector1], [vector2])[0, 0]
    return sim>0.6

def edge_match(e1, e2):
    """比较边数据，确保权重相同"""
    return e1['weight'] == e2['weight']

from itertools import combinations

# isomorphic = nx.is_isomorphic(G, G1, node_match=node_match, edge_match=edge_match)
# print(isomorphic)
def most_common_graphs(G,G1):
    """生成并返回图G的所有可能的连通子图"""
    l = min(len(G),len(G1))
    for n in range(l + 1, 1, -1):
        for nodes in combinations(G, n):
            subG = G.subgraph(nodes)
            for nodes1 in combinations(G1, n):
                subG1 = G1.subgraph(nodes1)
                isomorphic = nx.is_isomorphic(subG, subG1, node_match=node_match, edge_match=edge_match)
                print("两个图同构：" if isomorphic else "两个图不同构")
                if isomorphic:
                    return subG,subG1

common_graph,common_graph_1 = most_common_graphs(G, G1)

print(f"公共子图的边: {common_graph.edges()}")
print(f"公共子图的结点: {common_graph.nodes()}")
