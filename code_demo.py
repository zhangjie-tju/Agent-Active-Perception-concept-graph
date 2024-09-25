# import networkx as nx
# from networkx.algorithms.isomorphism import GraphMatcher

# # 创建两个图 G1 和 G2
# G1 = nx.Graph()
# G2 = nx.Graph()

# # 为 G1 添加节点和边
# G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# # 为 G2 添加节点和边
# G2.add_edges_from([(5, 6), (6, 8), (7, 8), (7, 5)])

# # 使用 GraphMatcher 查找 G1 在 G2 中的所有子图同构
# GM = GraphMatcher(G1, G2)

# # 迭代生成所有的子图同构
# for mapping in GM.subgraph_isomorphisms_iter():
#     print("Subgraph isomorphism found:")
#     print(mapping)

import networkx as nx
from scipy.sparse import csr_matrix

# 创建一个scipy的稀疏矩阵
sparse_matrix = csr_matrix([[0, 1, 0],
                            [1, 0, 2],
                            [0, 2, 0]])
G = nx.from_scipy_sparse_matrix(sparse_matrix)
print(G.edges(data=True))