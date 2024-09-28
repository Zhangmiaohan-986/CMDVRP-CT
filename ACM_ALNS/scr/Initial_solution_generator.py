import numpy as np
from sklearn.cluster import KMeans
import random

# 从Modul_node.py导入必要的类
from Modul_node import Node, Graph, Vehicle, Drone

# 1. 编码与解码模块
class EncoderDecoder:
    def encode_solution(self, vehicles, drones):
        """
        将车辆和无人机的调度方案编码为列表格式。
        """
        encoded_solution = []
        for vehicle in vehicles:
            encoded_vehicle = {
                'vehicle_id': vehicle.id,
                'route': vehicle.encode_route(),
                'drones': []
            }
            for drone in vehicle.drones:
                encoded_drone = {
                    'drone_id': drone.id,
                    'route': drone.encode_route()
                }
                encoded_vehicle['drones'].append(encoded_drone)
            encoded_solution.append(encoded_vehicle)
        return encoded_solution

    def decode_solution(self, encoded_solution, vehicles, drones):
        """
        将编码后的调度方案解码为车辆和无人机的实际路径。
        """
        for enc_vehicle in encoded_solution:
            vehicle_id = enc_vehicle['vehicle_id']
            vehicle = next(v for v in vehicles if v.id == vehicle_id)
            vehicle.decode_route(enc_vehicle['route'])
            for enc_drone in enc_vehicle['drones']:
                drone_id = enc_drone['drone_id']
                drone = next(d for d in drones if d.id == drone_id)
                drone.decode_route(enc_drone['route'])

# 2. 聚类与分配模块
class ClusteringAllocator:
    def cluster_ground_nodes(self, ground_nodes, num_vehicles):
        """
        使用K-means聚类算法将地面节点分配给车辆。
        :param ground_nodes: 地面节点列表
        :param num_vehicles: 车辆数量
        :return: 车辆聚类字典，键为车辆编号，值为分配的节点列表
        """
        coords = np.array([[node.x, node.y] for node in ground_nodes])
        kmeans = KMeans(n_clusters=num_vehicles, random_state=0).fit(coords)
        labels = kmeans.labels_
        vehicle_clusters = {i: [] for i in range(num_vehicles)}
        for idx, label in enumerate(labels):
            vehicle_clusters[label].append(ground_nodes[idx])
        return vehicle_clusters

    def cluster_air_nodes(self, air_nodes, num_drones):
        """
        使用K-means聚类算法将空中节点分配给无人机。
        :param air_nodes: 空中节点列表
        :param num_drones: 无人机数量
        :return: 无人机聚类字典，键为无人机编号，值为分配的节点列表
        """
        coords = np.array([[node.x, node.y] for node in air_nodes])
        kmeans = KMeans(n_clusters=num_drones, random_state=0).fit(coords)
        labels = kmeans.labels_
        drone_clusters = {i: [] for i in range(num_drones)}
        for idx, label in enumerate(labels):
            drone_clusters[label].append(air_nodes[idx])
        return drone_clusters

# 3. 初始路径生成模块
class InitialPathGenerator:
    def generate_vehicle_route(self, vehicle, ground_graph):
        """
        使用最近邻算法为车辆生成初始地面路径。
        :param vehicle: Vehicle对象
        :param ground_graph: 地面网络Graph对象
        :return: 生成的车辆路径列表
        """
        if not vehicle.route:
            return []
        start_node = vehicle.start_node_id
        route = [start_node]
        unvisited = set(vehicle.route)
        current_node = start_node
        if current_node in unvisited:
            unvisited.remove(current_node)
        while unvisited:
            # 找到与当前节点最近的未访问节点
            nearest_node = min(
                unvisited,
                key=lambda node: ground_graph.edges.get((current_node, node), {'weight': np.inf})['weight']
            )
            route.append(nearest_node)
            unvisited.remove(nearest_node)
            current_node = nearest_node
        vehicle.set_route(route)
        return route

    def generate_drone_route(self, drone, air_graph):
        """
        为无人机生成初始巡检路径。
        :param drone: Drone对象
        :param air_graph: 空中网络Graph对象
        :return: 生成的无人机路径列表
        """
        if not drone.route:
            return []
        # 简单地按照分配的顺序巡检
        route = drone.route.copy()
        drone.set_route(route)
        return route

# 4. 约束条件处理模块
class ConstraintHandler:
    def check_drone_endurance(self, drone, air_graph):
        """
        检查无人机巡检路径是否在续航时间内完成。
        :param drone: Drone对象
        :param air_graph: 空中网络Graph对象
        :return: True如果满足，False否则
        """
        if not drone.route:
            return True
        total_distance = 0
        for i in range(len(drone.route) - 1):
            node1 = drone.route[i]
            node2 = drone.route[i + 1]
            edge = air_graph.edges.get((node1, node2))
            if edge:
                total_distance += edge
            else:
                # 如果两个节点之间没有边，假设距离为无穷大
                total_distance += np.inf
        return total_distance <= drone.endurance

    def check_time_windows(self, vehicles, drones):
        """
        检查车辆和无人机的时间窗口约束。
        :param vehicles: 车辆列表
        :param drones: 无人机列表
        :return: True如果所有约束满足，False否则
        """
        # 实现具体的时间窗口检查逻辑
        # 需要根据您的具体时间窗口数据进行实现
        pass

    def check_launch_recover_sync(self, vehicles, drones):
        """
        确保无人机的发射和回收操作与车辆路径同步。
        :param vehicles: 车辆列表
        :param drones: 无人机列表
        :return: True如果所有同步约束满足，False否则
        """
        # 实现具体的发射与回收同步检查逻辑
        # 需要根据您的具体发射和回收点数据进行实现
        pass

# 5. 目标函数计算模块
class ObjectiveFunction:
    def calculate_max_completion_time(self, vehicles, drones, ground_graph, air_graph):
        """
        计算所有车辆和无人机任务完成后的最长完成时间。
        :param vehicles: 车辆列表
        :param drones: 无人机列表
        :param ground_graph: 地面网络Graph对象
        :param air_graph: 空中网络Graph对象
        :return: 最大完成时间
        """
        max_time = 0
        for vehicle in vehicles:
            vehicle_time = vehicle.get_completion_time(ground_graph)
            if vehicle_time > max_time:
                max_time = vehicle_time
        for drone in drones:
            drone_time = drone.get_completion_time(air_graph)
            if drone_time > max_time:
                max_time = drone_time
        return max_time

# 6. 初始解生成模块
class InitialSolutionGenerator:
    def __init__(self, ground_graph, air_graph, vehicles, drones):
        """
        初始化初始解生成器。
        :param ground_graph: 地面网络Graph对象
        :param air_graph: 空中网络Graph对象
        :param vehicles: 车辆列表
        :param drones: 无人机列表
        """
        self.ground_graph = ground_graph
        self.air_graph = air_graph
        self.vehicles = vehicles
        self.drones = drones
        self.encoder_decoder = EncoderDecoder()
        self.clustering_allocator = ClusteringAllocator()
        self.path_generator = InitialPathGenerator()
        self.constraint_handler = ConstraintHandler()
        self.objective_function = ObjectiveFunction()

    def generate_initial_solution(self):
        # 2.1 聚类分配
        vehicle_clusters = self.clustering_allocator.cluster_ground_nodes(
            ground_nodes=list(self.ground_graph.nodes.values()),
            num_vehicles=len(self.vehicles)
        )
        drone_clusters = self.clustering_allocator.cluster_air_nodes(
            air_nodes=list(self.air_graph.nodes.values()),
            num_drones=len(self.drones)
        )

        # 分配聚类结果给车辆
        for vehicle_id, cluster in vehicle_clusters.items():
            self.vehicles[vehicle_id].route = [node.id for node in cluster]

        # 分配聚类结果给无人机
        for drone_id, cluster in drone_clusters.items():
            self.drones[drone_id].route = [node.id for node in cluster]

        # 3.1 生成车辆初始路径
        for vehicle in self.vehicles:
            self.path_generator.generate_vehicle_route(vehicle, self.ground_graph)

        # 3.2 生成无人机初始路径
        for drone in self.drones:
            self.path_generator.generate_drone_route(drone, self.air_graph)

        # 1.1 编码初始解
        encoded_solution = self.encoder_decoder.encode_solution(self.vehicles, self.drones)

        # 1.2 解码并检查约束
        self.encoder_decoder.decode_solution(encoded_solution, self.vehicles, self.drones)
        for drone in self.drones:
            if not self.constraint_handler.check_drone_endurance(drone, self.air_graph):
                print(f"无人机 {drone.id} 超过续航时间限制。")

        # 5.1 计算目标函数
        max_completion_time = self.objective_function.calculate_max_completion_time(
            self.vehicles, self.drones, self.ground_graph, self.air_graph
        )

        return encoded_solution, max_completion_time