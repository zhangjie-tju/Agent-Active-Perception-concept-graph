import json
import copy
from gensim.models import KeyedVectors
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

model_path = '/home/user/Documents/ZJ/concept-graphs/edit_graphs/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)



#读取用户输入
# user_input = input("Please enter your query: ")
text = "In the scene, there is an open book, a ceramic vase, and a bottle of water placed on a wooden counter, with a packet of cookies in the drawer of this counter."

import time
from os import path
import copy
import json
from openai import OpenAI

API_KEY = 'sk-1F3yr67LlKJUTaWnAa7a6063F7094eFc9cFc4b57Ad93073e'
API_HOST = 'https://xiaoai.one/v1'
client = OpenAI(api_key=API_KEY, base_url=API_HOST)

root = "/home/user/Documents/ZJ/concept-graphs/edit_graphs"
sys_f="system_message.txt"
with open(path.join(root,sys_f)) as f:
    sys_=f.readlines()
messages = [ {"role":"system", "content": '\n'.join(sys_)} ]

with open(path.join(root,"relation.txt")) as f:
    msg_user=f.readlines()
messages.append({"role":"user", "content":"In the scene, there is an apple and a bottle of water placed on cabinet, with a bunch of keys in the cabinet."})
messages.append({"role":"assistant", "content":'\n'.join(msg_user)} )

def ask_gpt_4(question=None, messages=None):
    for i in range(35):
        try:
            if messages is None:
                messages = [
                        {"role": "user", "content": question},
                    ]
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            if i == 34:
                raise ValueError("GPT failed")
            time.sleep(2)
            continue

    raise ValueError("GPT failed")
import re


messages_temp=copy.deepcopy(messages)

messages_temp.append({"role":"user", "content":text})
answer=ask_gpt_4("",messages_temp)
answer = eval(answer)

G1 = nx.DiGraph()

for item in answer:
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

file_1 = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/rebuild_object_1.json"
with open(file_1, 'r') as f1:
    data_raw_1 = json.load(f1)
    
G = nx.DiGraph()
data = copy.deepcopy(data_raw_1)
for item in data:
    relation = item['object_relation']
    if relation == "none of these":
        continue
    G.add_node(item['object1'] ['id'],label = item['object1']["object_tag"], position_center =item['object1']["bbox_center"], position_extent = item['object1']["bbox_extent"])
    G.add_node(item['object2'] ['id'],label = item['object2']["object_tag"], position_center =item['object2']["bbox_center"], position_extent = item['object2']["bbox_extent"])
    if relation[0] == "a":
        weight_relation = {"a on b":1,"a in b":2}.get(item['object_relation'])
        G.add_edge(item['object1'] ['id'], item['object2'] ['id'],weight = weight_relation)
    elif relation[0] == "b":
        weight_relation = {"b on a":1,"b in a":2}.get(item['object_relation'])
        G.add_edge(item['object2'] ['id'], item['object1'] ['id']  ,weight = weight_relation)
    else:
        print(item['object_relation'])

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
                GM = nx.algorithms.isomorphism.DiGraphMatcher(subG1, subG, node_match=node_match, edge_match=edge_match)
                print("两个图同构：" if GM.is_isomorphic() else "两个图不同构")
                if GM.is_isomorphic():
                    return GM,subG,subG1

common_graph,subG,subG1= most_common_graphs(G, G1)

node_mapping=common_graph.mapping
edge_mapping = {(u, v): (node_mapping[u], node_mapping[v]) for u, v in subG1.edges()}
max_nodes = max(G.nodes)
for edge in G1.edges:
    if edge in edge_mapping:
        continue
    u,v =edge
    n = u
    m = v
    if u not in node_mapping:
        n = max_nodes + u
        G.add_node(n,label = G1.nodes[u]) 
    else:
        n = node_mapping[n]
    if v not in node_mapping:
        m = max_nodes + v
        G.add_node(m,label = G1.nodes[v]) 
    else:
        m = node_mapping[m]
    G.add_edge(n,m,weight = G1[u][v]['weight'])


# def dumps_graphs(G):
#     Graph = []
#     for edge in G.edges():
        