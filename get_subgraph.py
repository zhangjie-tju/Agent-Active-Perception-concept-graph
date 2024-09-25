import networkx as nx
from itertools import combinations

def generate_all_subgraphs(G):
    subgraphs = []
    nodes = list(G.nodes())
    # 生成所有可能的节点组合
    for r in range(1, len(nodes) + 1):
        for node_comb in combinations(nodes, r):
            # 为每个节点组合生成诱导子图
            subgraph = G.subgraph(node_comb).copy()
            subgraphs.append(subgraph)
    return subgraphs

# 示例图
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

# 生成所有子图
all_subgraphs = generate_all_subgraphs(G)

# 输出子图信息
for i, sg in enumerate(all_subgraphs):
    print(f"Subgraph {i+1}: Nodes = {sg.nodes()}, Edges = {sg.edges()}")
