import json
import networkx as nx 
import copy

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

Graph_reslt_arr = []

for u, v, attr in G.edges(data = True):
    
    ob_relation = {}
    if 'bbox_center' in G.nodes[u]:
        object_1 = {"id": u,
                    "bbox_center":G.nodes[u]['bbox_center'],
                    "bbox_extent":G.nodes[u]['bbox_extent'],
                    "object_tag":G.nodes[u]['label']}
    else:
        object_1 = {"id": u,
                    "object_tag":G.nodes[u]['label']}
    
    if 'bbox_center' in G.nodes[v]:
        object_2 = {"id": v,
                    "bbox_center":G.nodes[v]['bbox_center'],
                    "bbox_extent":G.nodes[v]['bbox_extent'],
                    "object_tag":G.nodes[v]['label']}
    else:
        object_2 = {"id": v,
                    "object_tag":G.nodes[v]['label']}
    
    ob_relation["object1"] = object_1
    ob_relation["object2"] = object_2
    
    ob_relation['object_relation'] = {1:"a on b", 2:"a in b"}.get(attr['weight'])
    
    Graph_reslt_arr.append(ob_relation)

from pathlib import Path
import json
with open("test.json", "w")as f:
    json.dump(Graph_reslt_arr, f, indent=4)
