import networkx as nx
import json
import copy
from edit_graphs.util import ask_gpt_4,build_gpt_message

def agent_excute(G,Node):
    
    #寻找Node的邻跳节点
    # neighbors = set(G.successors(Node)) | set(G.predecessors(Node))
    all_edges = list(G.in_edges(Node,data = True)) + list(G.out_edges(Node,data = True))
    all_relations = ""
    objects = ""
    for u,v,w in all_edges:
        if u == Node:
            objects = objects + G.nodes[v]['label']+","
        else:
            objects = objects + G.nodes[u]['label']+","
        
        if w['weight'] == 1:
            relation = G.nodes[u]['label']+' is on '+ G.nodes[v]['label']
        elif w['weight'] == 2:
            relation = G.nodes[u]['label']+' is in '+ G.nodes[v]['label']
        all_relations = all_relations+relation+";"
    query = f"The positional relationship of items in the scene is as follows:{all_relations} Given the locations of the following items:{objects} I want to know the location of the apple and which item to look for nearby?"
    system_message = "/home/user/Documents/ZJ/concept-graphs/edit_graphs/prompt/agent_system.txt"
    example = "/home/user/Documents/ZJ/concept-graphs/edit_graphs/prompt/agent_example.txt"
    prompt = build_gpt_message(system_message,example,query)
    res = ask_gpt_4(messages=prompt)
    print(prompt)   


file_1 = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/rebuild_object_1.json"
with open(file_1, 'r') as f1:
    data_raw_1 = json.load(f1)
    
G = nx.DiGraph()
data = copy.deepcopy(data_raw_1)
for item in data:
    relation = item['object_relation']
    if relation == "none of these":
        continue
    if 'bbox_center' in item['object1']:
        G.add_node(item['object1'] ['id'],label = item['object1']["object_tag"], bbox_center =item['object1']["bbox_center"], bbox_extent = item['object1']["bbox_extent"])
    else:
        G.add_node(item['object1'] ['id'],label = item['object1']["object_tag"])
    if 'bbox_center' in item['object2']:    
        G.add_node(item['object2'] ['id'],label = item['object2']["object_tag"], bbox_center =item['object2']["bbox_center"], bbox_extent = item['object2']["bbox_extent"])
    else:
        G.add_node(item['object2'] ['id'],label = item['object2']["object_tag"])
    if relation[0] == "a":
        weight_relation = {"a on b":1,"a in b":2}.get(item['object_relation'])
        G.add_edge(item['object1'] ['id'], item['object2'] ['id'],weight = weight_relation)
    elif relation[0] == "b":
        weight_relation = {"b on a":1,"b in a":2}.get(item['object_relation'])
        G.add_edge(item['object2'] ['id'], item['object1'] ['id']  ,weight = weight_relation)
    else:
        print(item['object_relation'])

Node = 50

agent_excute(G,Node)

