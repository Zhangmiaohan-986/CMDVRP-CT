import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


# 定义节点连接的边列表 s 和 t
def generate_graph_from_edges():
    # MATLAB 的 s 和 t 定义
    s = [1, 3, 5, 7, 7] + list(range(10, 101))  # 起点节点
    t = [2, 4, 6, 8, 9] + list(np.random.randint(10, 101, 91))  # 终点节点

    # 创建无向图
    G = nx.Graph()

    # 添加边到图中
    for i in range(len(s)):
        G.add_edge(s[i], t[i])

    return G


# 可视化力导向布局的图
def visualize_graph(G):
    plt.figure(figsize=(10, 10))

    # 使用 spring_layout 模拟力导向布局
    pos = nx.spring_layout(G, k=0.7, iterations=200,seed=42)

    # 绘制节点和边
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=100, edge_color='gray')

    # 调整显示的紧凑性
    plt.axis('equal')  # MATLAB 中 axis tight equal 类似的效果
    plt.title("Force-Directed Layout Graph with Gravity")
    plt.show()


if __name__ == '__main__':
    # 生成图
    G = generate_graph_from_edges()

    # 可视化力导向布局
    visualize_graph(G)