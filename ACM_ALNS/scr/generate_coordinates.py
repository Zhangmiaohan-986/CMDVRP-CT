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
    start_pos = (float(df_R201.at[0, 'XCOORD.']), float(df_R201.at[0, 'YCOORD.']))
    if num_points <= 100:
        # position_points = df_R201[0:num_points+1]
        position_points_sample = df_R201.sample(n=num_points,random_state=seed)
        position_points_sample = position_points_sample.sort_index()
        return position_points_sample, start_pos
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
        # start_pos = (float(df_R201.at[0,'XCOORD.']), float(df_R201.at[0,'YCOORD.']))

        return position_points_sample, start_pos # 输出pd格式的
def generates_points(num_points,seed):
    file_path_R1 = '/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/homberger_400_customer_instances/R1_4_1.TXT'
    np.random.seed(seed)
    with open(file_path_R1, 'r') as file:
        data0_R1 = file.read()
    # 将数据转换为DataFrame
    data_R1 = []
    for line in data0_R1.strip().split('\n'):
        data_R1.append(line.split())
    columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R1 = pd.DataFrame(data_R1[1:], columns=columns)
    # 将字符型列转换为数字
    numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df_R1[numeric_cols] = df_R1[numeric_cols].apply(pd.to_numeric, errors='coerce')
    position_points_sample = df_R1.sample(n=num_points,random_state=seed)
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
    # 创建空中无向图
    G_air = nx.Graph()
    # 添加空中节点
    for i, coord in enumerate(coordinates_air):
        G_air.add_node(i, pos=coord)
    # 添加空中边的连接
    G_air = add_random_tree(G_air,coordinates_air_matrix,coordinates_air)
    # 创建地面节点
    G_ground = nx.Graph()
    for i, coord in enumerate(coordinates_ground):
        G_ground.add_node(i, pos=coord)
    G_ground = add_random_tree(G_ground,coordinates_ground_matrix,coordinates_ground,4)
    # 获取空地距离矩阵及邻接矩阵
    air_adj_matrix = np.array(nx.adjacency_matrix(G_air).todense())
    air_pos = nx.get_node_attributes(G_air, 'pos')
    ground_adj_matrix = np.array(nx.adjacency_matrix(G_ground).todense())
    ground_pos = nx.get_node_attributes(G_ground, 'pos')

    return G_air,G_ground,air_adj_matrix,air_pos,ground_adj_matrix,ground_pos

# 生成随机树的函数
def add_random_tree(G, distance_matrix, coordinates, max_connect_num=None):
    if max_connect_num is None:
        max_connect_num = 3
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
                if connect_count[current] < max_connect_num and connect_count[neighbor] < 10:
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

from mpl_toolkits.mplot3d import Axes3D
def visualize_tower_connections(G_air, G_ground):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    G_air_pos = nx.get_node_attributes(G_air, 'pos')
    G_air_pos_3d = {node: (x, y, 20) for node, (x, y) in G_air_pos.items()}

    air_x = [coord[0] for coord in G_air_pos_3d.values()]
    air_y = [coord[1] for coord in G_air_pos_3d.values()]
    air_z = [coord[2] for coord in G_air_pos_3d.values()]

    ax.scatter(air_x, air_y, air_z, c='red', label='Air Nodes')
    # # 添加空中节点的标签
    # for node, (x, y, z) in G_air_pos_3d.items():
    #     ax.text(x, y, z, f'{node}', fontsize=10, color='black')  # 标签显示节点编号


    for edge in G_air.edges():
        node1, node2 = edge
        x_values = [G_air_pos_3d[node1][0], G_air_pos_3d[node2][0]]
        y_values = [G_air_pos_3d[node1][1], G_air_pos_3d[node2][1]]
        z_values = [G_air_pos_3d[node1][2], G_air_pos_3d[node2][2]]
        ax.plot(x_values, y_values, z_values, color='blue', linestyle='dashed', linewidth=2, alpha=0.6)

    # 地面节点绘制
    G_ground_pos = nx.get_node_attributes(G_ground, 'pos')
    G_ground_pos_3d = {node: (x, y, 0) for node, (x, y) in G_ground_pos.items()}


    ground_x = [coord[0] for coord in G_ground_pos_3d.values()]
    ground_y = [coord[1] for coord in G_ground_pos_3d.values()]
    ground_z = [coord[2] for coord in G_ground_pos_3d.values()]

    ax.scatter(ground_x, ground_y, ground_z, c='b', marker='o', label='Ground Nodes')
    # 添加地面节点的标签
    # for node, (x, y, z) in G_ground_pos_3d.items():
    #     ax.text(x, y, z, f'{node}', fontsize=10, color='black')  # 标签显示节点编号

    for edge in G_ground.edges():
        node1, node2 = edge
        x_values_ground = [G_ground_pos_3d[node1][0], G_ground_pos_3d[node2][0]]
        y_values_ground = [G_ground_pos_3d[node1][1], G_ground_pos_3d[node2][1]]
        z_values_ground = [G_ground_pos_3d[node1][2], G_ground_pos_3d[node2][2]]
        ax.plot(x_values_ground, y_values_ground, z_values_ground, color='yellow',  linewidth=2, alpha=0.6)

    ax.set_xlim(min(air_x) - 10, max(air_x) + 10)
    ax.set_ylim(min(air_y) - 10, max(air_y) + 10)
    ax.set_zlim(0, 25)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=12)

    plt.title("Benchmark Graph of a 50-Node Dual-Layer Road Network. test_points:6. generate_graph:1", fontsize=16)
    # # 使用plt.savefig()并指定参数
    # plt.savefig('/Users/zhangmiaohan/PycharmProjects/gurobi_test/tits_CMDVRP-CT/map/env_map/100节点示意图1.1.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()  # 本地显示图形

# 示例代码
if __name__ == '__main__':
    test_points,start_pos = generate_points(50, 6)  # 使用 generate_points 函数生成测试点
    # test_points = generates_points(50, 6)  # 使用 generates_points 函数生成测试点
    G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions = generate_graph(test_points, 1)# 使用 generate_graph 函数生成空中图
    # 可视化塔杆节点和线路
    # visualize_tower_connections(G_air,G_ground)
