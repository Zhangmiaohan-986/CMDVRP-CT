import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 读取已有节点坐标，假设为 Solomon 数据集中的 xy 坐标
# 示例: xy 是一个 n x 2 的数组，每一行是一个节点的 xy 坐标
xy = np.array([
    [10, 15], [20, 35], [50, 25], [60, 45], [70, 10], [15, 60]
])

# 创建空的无向图
G = nx.Graph()

# 添加节点，使用 xy 坐标作为节点的属性
for i, (x, y) in enumerate(xy):
    G.add_node(i, pos=(x, y))

# 设置距离阈值，决定哪些节点会有连接
distance_threshold = 30

# 添加边，计算节点之间的欧几里得距离，若距离小于阈值，则连边
for i in range(len(xy)):
    for j in range(i + 1, len(xy)):
        dist = np.linalg.norm(xy[i] - xy[j])
        if dist < distance_threshold:
            G.add_edge(i, j)

# 获取节点位置
pos = nx.get_node_attributes(G, 'pos')

# 绘制图形
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.show()