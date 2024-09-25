import networkx as nx
from scipy.sparse import csr_matrix

import pandas as pd
import six.moves.cPickle as cPickle 
import gzip
from gensim.models import KeyedVectors

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path_0 = '/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/map/scene_map_cfslam_pruned.pkl.gz'
f = gzip.open(path_0,'rb')
df = pd.DataFrame()
df = cPickle.load(f)
f.close()

model_path = '/home/user/Documents/ZJ/concept-graphs/edit_graphs/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

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

import networkx as nx
G = nx.DiGraph()

for i in range(len(df)):
    node = df[i]['caption_dict']['id']
    tag = df[i]['caption_dict']['response']['object_tag']
    G.add_node(node,label= tag)

import json, copy
file = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/cfslam_object_relations_new.json"
try:
    with open(file, 'r') as f:
        data_raw = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")
except json.JSONDecodeError:
    print("文件内容不是有效的JSON格式。")
object_dict = {}
data = copy.deepcopy(data_raw)

for item in data:
    relation = item['object_relation']
    if relation == "none of these":
        continue
    
    if relation[0] == "a":
        weight_relation = {"a on b":1,"a in b":2}.get(item['object_relation'])
        G.add_edge(item['object1'] ['id'], item['object2'] ['id'],weight = weight_relation)
    elif relation[0] == "b":
        weight_relation = {"b on a":1,"b in a":2}.get(item['object_relation'])
        G.add_edge(item['object2'] ['id'], item['object1'] ['id']  ,weight = weight_relation)
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


print(G)
