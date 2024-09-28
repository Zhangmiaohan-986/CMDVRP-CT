# main.py

import numpy as np
import networkx as nx
from generate_coordinates import generate_graph, generate_points  # 确保generate_coordinates.py中定义了这些函数
from Modul_node import Node, Graph, Vehicle, Drone
from Initial_solution_generator import InitialSolutionGenerator

if __name__ == '__main__':
    # 生成测试点
    test_points = generate_points(50, 6)  # 生成50个测试点，参数根据实际情况调整
    # 生成图和相关数据
    G_air, G_ground, air_adj_matrix, air_positions, ground_adj_matrix, ground_positions = generate_graph(test_points, 1)

    # 创建空中网络的Graph实例
    air_graph = Graph()
    for node_id, position in air_positions.items():
        x, y = position
        node = Node(node_id=node_id, x=x, y=y, node_type='air')
        air_graph.add_node(node)

    for edge in G_air.edges(data=True):
        node1_id = edge[0]
        node2_id = edge[1]
        weight = edge[2].get('weight', 1.0)  # 根据generate_graph函数确定权重字段
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
        weight = edge[2].get('weight', 1.0)
        ground_graph.add_edge(node1_id, node2_id, weight)
    # 空地节点处理完成

    # 创建车辆列表
    vehicles = []
    num_vehicles = 2  # 根据需求调整车辆数量
    vehicle_capacity = 2  # 每辆车携带的无人机数量
    start_node_id = max(air_positions.keys()) + max(ground_positions.keys()) + 1  # 车辆起始节点编号

    for i in range(num_vehicles):
        vehicle = Vehicle(vehicle_id=i, capacity=vehicle_capacity, start_node_id=start_node_id)
        for j in range(vehicle_capacity):
            drone_id = i * vehicle_capacity + j
            drone = Drone(drone_id=drone_id, endurance=100, speed=10)  # 设置适当的参数
            vehicle.add_drone(drone)
        vehicles.append(vehicle)

    # 获取所有无人机
    drones = [drone for vehicle in vehicles for drone in vehicle.drones]

    # 初始化初始解生成器
    initial_solution_generator = InitialSolutionGenerator(
        ground_graph=ground_graph,
        air_graph=air_graph,
        vehicles=vehicles,
        drones=drones
    )

    # 生成初始解
    encoded_solution, max_completion_time = initial_solution_generator.generate_initial_solution()

    # 执行破坏策略

    # 输出初始解和目标函数值
    print("初始编码解：")
    print(encoded_solution)
    print(f"\n初始最大完成时间：{max_completion_time}")

    # 验证约束条件
    for drone in drones:
        if not initial_solution_generator.constraint_handler.check_drone_endurance(drone, air_graph):
            print(f"无人机 {drone.id} 超过续航时间限制。")