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
        G_air.add_edge(coord[0],coord[1])

    return G_air

# 建立塔杆节点连接概率函数
def connect_prob(connections):
    # 计算连接概率
    if connections == 0:
        return 1.0
    return max(0,1.0-0.2*connections)

# 生成获得的距离矩阵
def compute_distance_matrix(coordinates):
    num_coordinates = len(coordinates)
    distance_matrix = np.zeros((num_coordinates, num_coordinates))
    for i in range(num_coordinates):
        for j in range(i + 1, num_coordinates):
            distance_matrix[i,j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
            distance_matrix[j,i] = distance_matrix[i,j]
    return distance_matrix


def visualize_tower_connections(G_air, G_groun=None):
    # 绘制空中节点
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G_air,k=2.5,iterations=220,seed=42)
    nx.draw(G_air, pos, with_labels=True, node_color='lightblue', node_size=100, font_size=10, edge_color='gray')
    # 显示图形
    plt.show()
    # G_air_edge_labels = nx.get_edge_attributes(G_air, 'weight')
    # nx.draw_networkx_edge_labels(G_air, G_air_pos, edge_labels=G_air_edge_labels)
    # # 绘制地面节点
    # G_ground_pos = nx.get_node_attributes(G_ground, 'pos')  # 获取 G_ground 节点的位置信息
    # nx.draw(G_ground, G_ground_pos, with_labels=False, node_color='lightgreen', edge_color='green', node_size=100,
    #         font_size=10, label='Ground Nodes')
    # G_ground_edge_labels = nx.get_edge_attributes(G_ground, 'weight')  # 获取 G_ground 的边权重
    # nx.draw_networkx_edge_labels(G_ground, G_ground_pos, edge_labels=G_ground_edge_labels)
    # # 设置图例，区分空中节点和地面节点
    # plt.legend(['Air Nodes', 'Ground Nodes'])
    # 显示图形
    plt.show()


if __name__ == '__main__':
    test_points = generate_points(100,1)
    G_air = generate_graph(test_points,10)
    visualize_tower_connections(G_air)
