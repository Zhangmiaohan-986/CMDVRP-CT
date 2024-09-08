import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 示例二维坐标数据
coordinates = [(1, 2), (3, 4), (6, 7), (8, 5)]

# 创建无向图
G = nx.Graph()

# 添加节点
for i, coord in enumerate(coordinates):
    G.add_node(i, pos=coord)

# 计算节点之间的距离并添加边
for i in range(len(coordinates)):
    for j in range(i + 1, len(coordinates)):
        # 使用欧氏距离作为权重
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        G.add_edge(i, j, weight=distance)

# 获取邻接矩阵
adj_matrix = nx.adjacency_matrix(G).todense()

# 打印邻接矩阵
print("Adjacency Matrix:")
print(adj_matrix)

# 可视化无向图
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()