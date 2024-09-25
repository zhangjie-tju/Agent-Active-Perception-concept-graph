import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

# 自定义节点相似性度量
def node_match(n1, n2):
    return n1['label'] == n2['label']

# 自定义边相似性度量
def edge_match(e1, e2):
    return e1['weight'] == e2['weight']

# 查找最大公共子图
def find_mcs(G1, G2):
    # 创建图同构匹配器，使用节点和边的相似性度量
    gm = GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
    
    # 获取所有同构的子图
    mcs_subgraph = None
    max_size = 0
    
    for subgraph in gm.subgraph_isomorphisms_iter():
        subgraph_size = len(subgraph)
        if subgraph_size > max_size:
            max_size = subgraph_size
            mcs_subgraph = subgraph

    # 根据子图生成最大公共子图
    if mcs_subgraph is not None:
        common_nodes = set(mcs_subgraph.keys())
        common_edges = [(u, v) for u, v in G1.edges() if u in common_nodes and v in common_nodes and (mcs_subgraph[u], mcs_subgraph[v]) in G2.edges()]
        mcs = nx.Graph()
        mcs.add_nodes_from(common_nodes)
        mcs.add_edges_from(common_edges)
        return mcs
    else:
        return None

# 示例图
G1 = nx.Graph()
G2 = nx.Graph()

# 为节点添加属性
G1.add_node(1, label='A')
G1.add_node(2, label='B')
G1.add_node(3, label='C')
G1.add_edge(1, 2, weight=1.0)
G1.add_edge(2, 3, weight=2.0)

G2.add_node(4, label='A')
G2.add_node(5, label='B')
G2.add_node(6, label='C')
G2.add_edge(4, 5, weight=1.0)
G2.add_edge(5, 6, weight=2.0)
# G2.add_edge(6, 4, weight=3.0)

# 查找最大公共子图
mcs = find_mcs(G1, G2)

if mcs:
    print("Maximum Common Subgraph (MCS):")
    print("Nodes:", mcs.nodes())
    print("Edges:", mcs.edges())
else:
    print("No common subgraph found.")
