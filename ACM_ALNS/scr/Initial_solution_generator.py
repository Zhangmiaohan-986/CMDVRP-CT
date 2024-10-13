from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
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
    def spectral_clustering_air_nodes(self, air_graph, num_clusters):
        """
        对空中节点进行谱聚类
        :param air_graph: 空中网络的Graph对象
        :param num_clusters: 聚类的簇数（车辆数量）
        :return: 一个字典，键为聚类标签，值为对应的空中节点列表
        """
        # 获取空中节点的列表和节点编号列表
        air_nodes = list(air_graph.nodes.values())
        node_ids = [node.id for node in air_nodes]
        num_nodes = len(node_ids)
        # 创建节点编号到索引的映射
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        # 填充邻接矩阵
        for (node1_id, node2_id), weight in air_graph.edges.items():
            idx1 = node_id_to_index[node1_id]
            idx2 = node_id_to_index[node2_id]
            adjacency_matrix[idx1, idx2] = weight
            adjacency_matrix[idx2, idx1] = weight  # 无向图，矩阵对称
        # 使用邻接矩阵作为相似度矩阵
        spectral = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )
        labels = spectral.fit_predict(adjacency_matrix)
        # 将节点按照聚类标签分组
        air_clusters = {}
        for label, node in zip(labels, air_nodes):
            if label not in air_clusters:
                air_clusters[label] = []
            air_clusters[label].append(node)
        return air_clusters, adjacency_matrix

    def cluster_ground_nodes(self, air_clusters, ground_graph):
        """
        基于空中聚类的质心，将地面节点映射到空中节点聚类
        :param air_clusters: 一个字典，键为聚类标签，值为对应的空中节点列表
        :param ground_graph: 地面网络的 Graph 对象
        :return: 一个字典，键为聚类标签，值为对应的地面节点列表
        """
        # 计算每个空中聚类的质心
        cluster_centers = {}
        # 获取地面节点列表和节点编号列表
        ground_nodes = list(ground_graph.nodes.values())
        ground_node_ids = [node.id for node in ground_nodes]
        num_ground_nodes = len(ground_node_ids)
        # 计算每个空中聚类的质心
        for label, nodes in air_clusters.items():
            coords = np.array([[node.x, node.y, node.z] for node in nodes])
            centroid = coords.mean(axis=0)
            cluster_centers[label] = centroid
        # 获取地面节点列表
        ground_nodes = list(ground_graph.nodes.values())
        # 初始化地面聚类字典
        ground_clusters = {label: [] for label in air_clusters.keys()}
        # 将地面节点映射到最近的空中聚类质心
        for node in ground_nodes:
            point = np.array([node.x, node.y, node.z])
            min_distance = float('inf')
            nearest_label = None
            for label, centroid in cluster_centers.items():
                distance = np.linalg.norm(point - centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_label = label
            ground_clusters[nearest_label].append(node)

            # 创建节点编号到索引的映射
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(ground_node_ids)}

        # 初始化地面节点的邻接矩阵
        ground_adjacency_matrix = np.zeros((num_ground_nodes, num_ground_nodes))

        # 填充邻接矩阵
        for (node1_id, node2_id), weight in ground_graph.edges.items():
            idx1 = node_id_to_index[node1_id]
            idx2 = node_id_to_index[node2_id]
            ground_adjacency_matrix[idx1, idx2] = weight
            ground_adjacency_matrix[idx2, idx1] = weight  # 无向图，矩阵对称

        return ground_clusters, ground_adjacency_matrix

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
    def __init__(self, ground_graph, air_graph, vehicles, drones, air_clusters, ground_clusters, air_adj_matrix, ground_adj_matrix, vehicle_node_start_id=None):
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
        self.air_clusters = air_clusters
        self.ground_clusters = ground_clusters
        self.air_adj_matrix = air_adj_matrix
        self.ground_adj_matrix = ground_adj_matrix
        self.start_node = vehicle_node_start_id
        # self.encoder_decoder = EncoderDecoder()
        # self.clustering_allocator = ClusteringAllocator()
        # self.path_generator = InitialPathGenerator()
        # self.constraint_handler = ConstraintHandler()
        # self.objective_function = ObjectiveFunction()

        # self.encoder_decoder = EncoderDecoder()
        # self.clustering_allocator = ClusteringAllocator()
        # self.path_generator = InitialPathGenerator()
        # self.constraint_handler = ConstraintHandler()
        # self.objective_function = ObjectiveFunction()
    def generate_vehicle_route(self):
        from route_plan import neighbour_plan
        vehicle_id = list(self.vehicles.keys())
        id_num = 0
        for ground_vehicle in self.ground_clusters.values():
            vehicle_nodes = []
            for vehicle_node in ground_vehicle:
                vehicle_nodes.append(vehicle_node.id)
            # 使用贪心算法生成车辆路径
            vehicle_route = neighbour_plan(vehicle_nodes, self.ground_adj_matrix, self.start_node)
            self.vehicles[vehicle_id].route = vehicle_route
            id_num += 1
        return self.vehicles # 两个车辆的初始路径任务分配

    def generate_initial_solution(self):
        self.generate_vehicle_route() # 得到两个车辆初始路径




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