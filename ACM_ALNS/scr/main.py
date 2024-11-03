# main.py

import numpy as np
import networkx as nx
from generate_coordinates import generate_graph, generate_points  # 确保generate_coordinates.py中定义了这些函数
from Modul_node import Node, Graph, Vehicle, Drone
from Initial_solution_generator import InitialSolutionGenerator, ClusteringAllocator

if __name__ == '__main__':
    # 生成测试点及车辆初始点
    test_points, start_pos = generate_points(50, 6)  # 生成50个测试点，参数根据实际情况调整
    # 生成图和相关数据
    G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions = generate_graph(test_points, 1)
    # 初始地图范围为100，按比例缩为地图实际范围，单位为公里
    map_rate = 1

    # 创建空中网络的Graph实例
    air_graph = Graph()
    for node_id, position in air_positions.items():
        x, y = position
        node = Node(node_id=node_id, x=x, y=y, node_type='air')
        air_graph.add_node(node)

    for edge in G_air.edges(data=True):
        node1_id = edge[0]
        node2_id = edge[1]
        weight = edge[2].get('weight', 1.0) * map_rate  # 根据generate_graph函数确定权重字段
        air_graph.add_edge(node1_id, node2_id, weight)
    # 创建地面网络的Graph实例
    ground_graph = Graph()
    offset = max(air_positions.keys()) + 1  # 确保地面节点编号不与空中节点冲突

    for node_id, position in ground_positions.items():
        x, y = position
        adjusted_node_id = node_id + offset
        node = Node(node_id=adjusted_node_id, x=x, y=y, node_type='ground')
        ground_graph.add_node(node)

    for edge in G_ground.edges(data=True):
        node1_id = edge[0] + offset
        node2_id = edge[1] + offset
        weight = edge[2].get('weight', 1.0) * map_rate
        ground_graph.add_edge(node1_id, node2_id, weight)
    vehicle_node_start_id = next(iter(ground_graph.nodes)) # 获得车辆出发节点
    # 空地节点处理完成
    # 创建车辆列表
    vehicles_node = {}
    num_vehicles = 2  # 根据需求调整车辆数量
    vehicle_capacity = 2  # 每辆车携带的无人机数量
    start_node_id = max(air_positions.keys()) + max(ground_positions.keys()) + 2  # 车辆起始节点编号(有效识别空中节点、地面节点及无人车几点代号)

    # 设定无人机无人车信息
    for i in range(num_vehicles):
        # vehicle = Vehicle(vehicle_id=i, capacity=vehicle_capacity, start_node_id=start_node_id)
        vehicle_id = start_node_id + i
        vehicle_start_node_id = i
        vehicle_node = Node(node_id=vehicle_id, x=start_pos[0], y=start_pos[1], node_type='vehicle')

        vehicle = Vehicle(vehicle_id=vehicle_id, capacity=vehicle_capacity, start_node_id=i)

        for j in range(vehicle_capacity):
            drone_id = start_node_id + num_vehicles + i * vehicle_capacity + j
            drone_node = Node(node_id=drone_id, x=0, y=0, node_type='drone')  # 设置适当的x, y坐标

            drone = Drone(drone_id=drone_id)  # 根据需求调整参数
            vehicle.add_drone(drone)
        vehicles_node[vehicle_id] = vehicle

    # 获取所有无人机
    # drones = [drone for vehicle in vehicles for drone in vehicle.drones]
    drones_node = {}
    for vehicle in vehicles_node.values():
        for drone in vehicle.drones:
            drones_node[drone.id] = drone

    # 获得聚类结果数据
    cluster_node = ClusteringAllocator()
    air_clusters, air_adjacency_matrix = cluster_node.spectral_clustering_air_nodes(air_graph, num_vehicles)
    ground_clusters, ground_adjacency_matrix = cluster_node.cluster_ground_nodes(air_clusters, ground_graph)

    # 初始化初始解生成器
    initial_solution_generator = InitialSolutionGenerator(
        ground_graph=ground_graph,
        air_graph=air_graph,
        vehicles=vehicles_node,
        drones=drones_node,
        air_clusters=air_clusters,
        ground_clusters=ground_clusters,
        air_adj_matrix=air_adj_matrix,
        ground_adj_matrix=ground_adj_matrix,
        vehicle_node_start_id=vehicle_node_start_id
    )
    # 生成初始解
    encoded_solution, decoded_solution, fitness_initial_solution = initial_solution_generator.generate_initial_solution()

    # 执行破坏策略



