import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math

# 基准测试函数数据选择
def generate_points(num_points,seed):
    file_path_R201 = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/R201.txt'
    file_path_RC101 = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/RC101.txt'
    np.random.seed(seed)
    with open(file_path_R201, 'r') as file:
        data0_R201 = file.read()
    # 将数据转换为DataFrame
    data_R201 = []
    for line in data0_R201.strip().split('\n'):
        data_R201.append(line.split())
    columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R201 = pd.DataFrame(data_R201[1:], columns=columns)
    # 将字符型列转换为数字
    numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R201[numeric_cols] = df_R201[numeric_cols].apply(pd.to_numeric, errors='coerce')
    if num_points <= 100:
        # position_points = df_R201[0:num_points+1]
        position_points_sample = df_R201.sample(n=num_points,random_state=seed)
        position_points_sample = position_points_sample.sort_index()
        return position_points_sample
    else:
        with open(file_path_RC101, 'r') as file:
            data0_RC101 = file.read()
        data_RC101 = []
        for line in data0_RC101.strip().split('\n'):
            data_RC101.append(line.split())
        df_RC101 = pd.DataFrame(data_RC101[1:], columns=columns)
        df_RC101[numeric_cols] = df_RC101[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df_combine = pd.concat([df_R201, df_RC101], ignore_index=True)
        position_points_sample = df_combine.sample(n=num_points,random_state=seed)
        position_points_sample = position_points_sample.sort_index()
        return position_points_sample # 输出pd格式的

# 基于所得到的测试坐标图例，设计得到无向图
def generate_graph(position_points, seed, prob = 0.75, uav_distance=None):
    if uav_distance is None:
        uav_distance = 50
    random.seed(seed)

    coordinates = list(zip(position_points['XCOORD.'], position_points['YCOORD.']))
    # 空地节点数目
    num_air = math.floor(len(coordinates) * 3/5)
    num_ground = int(len(coordinates) - num_air)
    # 空地节点坐标/随机生成
    coordinates_air = random.sample(coordinates,num_air)
    coordinates_air_matrix = compute_distance_matrix(coordinates_air)
    # 获得地面节点坐标
    coordinates_ground = list(set(coordinates).difference(coordinates_air))
    coordinates_ground_matrix = compute_distance_matrix(coordinates_ground)
    max_distance_air = np.max(coordinates_air_matrix)
    max_distance_ground = np.max(coordinates_ground_matrix)
    # 初始化节点连接次数
    connect_count_air = {i:0 for i in range(len(coordinates_air))}
    connect_count_gorund = {i:0 for i in range(len(coordinates_ground))}
    # 创建空中无向图
    G_air = nx.Graph()
    # 添加空中节点
    for i, coord in enumerate(coordinates_air):
        G_air.add_node(i, pos=coord)
    # 添加空中边的连接
    G_air = add_random_tree(G_air,coordinates_air_matrix,coordinates_air)
    return G_air

# 生成随机树的函数
def add_random_tree(G, distance_matrix, coordinates):
    num_nodes = len(distance_matrix)
    visited = np.full((num_nodes,num_nodes),False)
    coordinates_origin = (0,0)
    distance_origin = compute_distance_matrix(coordinates, coordinates_origin)
    sorted_indices_origin = np.argsort(distance_origin)[0,0]
    near_path = find_nearest_path(distance_matrix,sorted_indices_origin)
    connect_count = {i: 0 for i in range(num_nodes)}
    for i in range(len(near_path)-1):
        current = near_path[i]
        next_current = near_path[i+1]
        current_prob = connect_prob(connect_count[current])
        if random.random() < current_prob:
            G.add_edge(current,next_current,weight=distance_matrix[current,next_current])
            connect_count[current] += 1
            connect_count[next_current] += 1
            visited[current,next_current] = True
            visited[next_current,current] = True
    # 继续遍历未连接的节点，按 50% 概率连接
    for current in near_path:
        # 获取当前节点的距离排序
        sorted_neighbors = np.argsort(distance_matrix[current])
        for neighbor in sorted_neighbors:
            # 检查连接是否已经存在，且是否当前节点和邻居节点不是同一节点
            if not visited[current, neighbor] and current != neighbor:
                # 仅当两个节点的连接次数都少于 3 时才考虑连接
                if connect_count[current] < 3 and connect_count[neighbor] < 10:
                    # 50% 概率连接
                    # 连接最近的节点（确保 neighbor 也是 current 最近的）
                    # if sorted_neighbors[1] == neighbor or sorted_neighbors[0] == neighbor:
                    if random.random() < connect_prob(connect_count[current]):
                        # 添加边并更新连接计数
                        G.add_edge(current, neighbor, weight=distance_matrix[current, neighbor])
                        visited[current, neighbor] = True
                        visited[neighbor, current] = True
                        connect_count[current] += 1
                        connect_count[neighbor] += 1
    return G

def find_nearest_path(distance_matrix, start_index):
    num_nodes = len(distance_matrix)
    visited = [False]*num_nodes
    path = []
    current_node = start_index
    while len(path)<num_nodes:
        visited[current_node] = True
        path.append(current_node)
        distance = distance_matrix[current_node]
        sorted_indices = np.argsort(distance)
        found_next = False
        for neighbor in sorted_indices:
            if not visited[neighbor] and neighbor != current_node:
                current_node = neighbor
                found_next = True
                break
        if not found_next:
            break  # 没有找到未访问的邻居节点，结束循环
    return path

# 建立塔杆节点连接概率函数
def connect_prob(connections):
    # 计算连接概率
    if connections == 0:
        return 1.0
    return max(0,1.0-0.2*connections)

# 生成获得的距离矩阵
def compute_distance_matrix(coordinates, origin=None):
    num_coordinates = len(coordinates)
    if origin is None:
        distance_matrix = np.zeros((num_coordinates, num_coordinates))
        for i in range(num_coordinates):
            for j in range(i + 1, num_coordinates):
                distance_matrix[i,j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
                distance_matrix[j,i] = distance_matrix[i,j]
        return distance_matrix
    else:
        # 计算所有节点到原点的距离
        distance_matrix = np.linalg.norm(np.array(coordinates) - np.array(origin), axis=1).reshape(1, -1)
        return distance_matrix


def visualize_tower_connections(G_air, G_ground=None):
    """
    可视化塔杆节点和线路，空中和地面节点分别绘制，带有图例和节点标注。
    """
    plt.figure(figsize=(12, 12))
    # 获取空中节点的坐标
    G_air_pos = nx.get_node_attributes(G_air, 'pos')
    # 绘制空中节点：蓝色，较大，透明度alpha调整
    nx.draw_networkx_nodes(G_air, G_air_pos, node_size=130, node_color='red', alpha=1.0, label='Air Nodes')
    # 获取空中边的权重并绘制边：蓝色虚线，带宽度
    air_edges = list(G_air.edges())
    nx.draw_networkx_edges(G_air, G_air_pos, edgelist=air_edges, edge_color='blue', style='dashed', width=3, alpha=0.6, label='Air Edges')

    # 如果有地面节点，绘制地面节点和边
    if G_ground:
        G_ground_pos = nx.get_node_attributes(G_ground, 'pos')

        # 绘制地面节点：绿色，较小
        nx.draw_networkx_nodes(G_ground, G_ground_pos, node_size=80, node_color='green', alpha=0.8, label='Ground Nodes')

        # 绘制地面边：绿色实线
        ground_edges = list(G_ground.edges())
        nx.draw_networkx_edges(G_ground, G_ground_pos, edgelist=ground_edges, edge_color='green', style='solid', width=1.5, alpha=0.6, label='Ground Edges')

        # 地面节点标注：字体大小10
        ground_labels = {n: str(n) for n in G_ground.nodes()}
        nx.draw_networkx_labels(G_ground, G_ground_pos, ground_labels, font_size=10, font_color='black')

        # 绘制地面边的权重（如有）：字体大小8
        ground_edge_labels = nx.get_edge_attributes(G_ground, 'weight')
        nx.draw_networkx_edge_labels(G_ground, G_ground_pos, edge_labels=ground_edge_labels, font_size=8, font_color='green')

    x_values_air = [coord[0] for coord in G_air_pos.values()]
    y_values_air = [coord[1] for coord in G_air_pos.values()]

    # 设置X轴和Y轴的范围，确保包含所有节点
    plt.xlim(min(x_values_air) - 10, max(x_values_air) + 10)  # 给X轴增加一些边距
    plt.ylim(min(y_values_air) - 10, max(y_values_air) + 10)  # 给Y轴增加一些边距

    # 设置X轴和Y轴的坐标刻度
    plt.xticks(range(int(min(x_values_air)) - 10, int(max(x_values_air)) + 10, 10))
    plt.yticks(range(int(min(y_values_air)) - 10, int(max(y_values_air)) + 10, 10))

    # 设置坐标轴的标签
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)

    # 显示网格
    plt.grid(True)

    # 确保坐标轴显示
    ax = plt.gca()  # 获取当前的轴
    ax.spines['left'].set_position('zero')  # 显示左轴
    ax.spines['bottom'].set_position('zero')  # 显示底轴
    ax.spines['right'].set_color('none')  # 隐藏右轴
    ax.spines['top'].set_color('none')  # 隐藏上轴
    ax.xaxis.set_ticks_position('bottom')  # X轴刻度显示在下方
    ax.yaxis.set_ticks_position('left')  # Y轴刻度显示在左边

    ax.spines['left'].set_linewidth(5)  # 设置左轴线条宽度
    ax.spines['bottom'].set_linewidth(5)  # 设置底轴线条宽度
    # 设置图例，区分空中节点、地面节点和各类边
    plt.legend(loc='upper right', fontsize=15)

    # 使用 adjustable='box' 解决 axis('equal') 覆盖坐标轴比例的问题
    # plt.gca().set_aspect('equal', adjustable='box')

    # 设置标题
    plt.title("Visualized Tower and Ground Connections", fontsize=16)

    # 显示图形
    plt.show()
# 示例代码 (需要自定义 G_air 和 G_ground 的内容)
if __name__ == '__main__':
    test_points = generate_points(50, 6)  # 使用 generate_points 函数生成测试点
    G_air = generate_graph(test_points, 1)  # 使用 generate_graph 函数生成空中图

    # 可视化塔杆节点和线路
    visualize_tower_connections(G_air)
