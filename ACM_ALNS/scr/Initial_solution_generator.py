from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import random
import math
from route_plan import dijkstra, calculate_travel_time, nodes_edge, calculate_distance, find_nearest_unvisited_edge, calculate_travel_distance

# 从Modul_node.py导入必要的类
from Modul_node import Node, Graph, Vehicle, Drone

# 1. 聚类与分配模块
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


# 2. 初始解生成模块
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
        self.global_time = 0  # 全局时间,总体目标函数时间，最后一个车辆返回原节点的时间
        self.unvisited_air_edges = set(self.air_graph.edges.keys())
        self.vehicle_list = list(self.vehicles.values())  # 车辆列表
        self.drone_list = list(self.drones.values())  # 无人机列表
        self.node_occupancy = {}  # 维护各节点车辆占据时间
        self.node_pending_times = {}
        self.launch = {}  # 发射记录，键为节点编号，值为发射的无人机列表
        self.recover = {}  # 回收记录，键为节点编号，值为回收的无人机列表

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
            self.vehicles[vehicle_id[id_num]].route = vehicle_route
            id_num += 1
        return self.vehicles # 两个车辆的初始路径任务分配

    def assign_drone_tasks(self):
        """
        实现车辆发射无人机的任务分配，直到完成所有空中线路的巡检任务。
        """
        while not all(vehicle.current_node_index >= len(vehicle.route) - 1 for vehicle in self.vehicles.values()):  # 判定条件改为直到所有车辆完成地面规划出的调度任务。
            # 遍历每辆车，按时间顺序移动
            vehicles_in_order = sorted(self.vehicles.values(), key=lambda v: v.current_time)
            for vehicle in vehicles_in_order:
                if vehicle.current_node_index >= len(vehicle.route):
                    print(f'车辆{vehicle.id}已完成路线')
                    continue  # 车辆已完成路线
                vehicle.current_node = vehicle.route[vehicle.current_node_index]
                current_node = vehicle.current_node
                next_node_index = vehicle.current_node_index + 1
                next_node = vehicle.route[next_node_index]
                # 计算前往下一个节点的行驶时间,时间为小时时间
                travel_time = calculate_travel_time(
                    current_node, next_node, self.air_graph, self.ground_graph, vehicle.speed
                )
                # 预期到达下一个节点的时间
                expected_arrival_time = vehicle.current_time + travel_time
                # 检查到达下一个节点时是否与当前时间窗产生冲突，则车辆等待占有节点的车辆离开后再进
                conflict_end_time = self.check_node_occupancy_conflict(next_node, expected_arrival_time)
                if conflict_end_time > expected_arrival_time:
                    # 调整到达时间为冲突时间窗的结束时间
                    wait_time = conflict_end_time - vehicle.current_time
                    start_time = vehicle.current_time
                    end_time = vehicle.current_time + wait_time
                    self.update_node_occupancy(current_node, start_time, end_time)
                    # vehicle.current_time += wait_time
                    expected_arrival_time = conflict_end_time + travel_time
                    # 记录在当前节点的等待时间
                    vehicle.waiting_times[current_node] = vehicle.waiting_times.get(current_node, 0) + wait_time
                    vehicle.node_exit_time[current_node] = conflict_end_time
                else:
                    vehicle.node_exit_time[current_node] = vehicle.current_time

                # 记录车辆进入/离开节点的实时时间
                vehicle.node_entry_time[next_node] = expected_arrival_time

                # 更新车辆的当前位置和时间
                vehicle.current_node_index = next_node_index
                vehicle.current_node = next_node
                vehicle.current_time = expected_arrival_time

                # 记录车辆在下一个节点的到达时间
                arrival_time = vehicle.current_time

                # 决定是否回收、发射无人机，并获取服务时间
                is_recover, recover_service_time = self.decide_recover(vehicle)
                is_launch, launch_service_time = self.decide_launch(vehicle)

                if is_recover and is_launch:
                    departure_time = arrival_time + launch_service_time + recover_service_time
                elif is_recover:
                    departure_time = arrival_time + recover_service_time
                else:
                    departure_time = arrival_time + launch_service_time

                # 更新节点的占用记录
                self.update_node_occupancy(vehicle.current_node, arrival_time, departure_time)

                # 更新车辆的当前时间为离开时间
                vehicle.current_time = departure_time

            # 更新全局时间为所有车辆的最小当前时间
            active_vehicles = [v.current_time for v in self.vehicles.values() if
                               v.current_node_index < len(v.route) - 1]
            if active_vehicles:
                self.global_time = min(active_vehicles)
            else:
                break  # 所有车辆已完成路线

    def decide_recover(self, vehicle):
        recover_service_time = 0
        current_node = vehicle.current_node
        current_time = vehicle.current_time
        remain_capacity = vehicle.available_capacity
        total_recover_drones = {}
        if remain_capacity >= vehicle.capacity:
            return False, recover_service_time
        # 遍历当前的空中无人机，选择在空中执行作业的任务无人机，然后选择当前空中备选次数最少的无人机执行回收任务
        for drone in self.drones:
            if not drone.is_available:  # 如果无人机并未在空中
                continue
            else:
                if current_node not in drone.returnable_nodes:
                    continue
                else:  # 记录当前所有可回收的空中节点无人机信息
                    total_recover_drones[drone.id] = drone.returnable_nodes
        if total_recover_drones is None:
            print(f"车辆{vehicle.id}在节点{vehicle.current_node}为找到可供回收选择的无人机")
        else:
            # 选择当前空中备选次数最少的无人机执行回收任务
            lengths = {drone_id: len(returnable_nodes) for drone_id, returnable_nodes in total_recover_drones.items()}
            sorted_drone_ids = sorted(lengths, key=lambda drone_id: lengths[drone_id])
            sorted_total_recover_drones = {drone_id: total_recover_drones[drone_id] for drone_id in sorted_drone_ids}
            for drone_id, returnable_nodes in sorted_total_recover_drones:
                vehicle.available_drones += 1
                if vehicle.available_capacity >= vehicle.capacity:
                    break
                drone_current_time = self.drones[drone_id].current_time
                drone_current_node = self.drones[drone_id].current_node
                drone_back_time = self.calculate_distance_between_nodes(current_node,drone_current_node,self.air_graph,self.ground_graph)
                drone_current_time += drone_back_time
                vehicle.add_drone(self.drones[drone_id])

                self.drones[drone_id].is_available = True
                if vehicle.current_time <= drone_current_time:
                    recover_service_time = drone_current_time - vehicle.current_time
                    vehicle.current_time = drone_current_time
        return True, recover_service_time

    def decide_launch(self, vehicle):  # 决定发射无人机巡检情况
        # available_drones = vehicle.available_drones.copy()
        available_drones = vehicle.drones.copy()

        num_available_drones = len(available_drones)
        total_launch_service_time = 0
        if num_available_drones == 0:
            print(f"车辆{vehicle.id}当前无可用车辆，无法执行配送任务")
            return total_launch_service_time
        vehicle_id = vehicle.id
        drone_ids = [drone.id for drone in available_drones]  # 获得可执行任务的无人机id ，便于更新self 无人机状态

        for drone in available_drones:
            # launched_drone_time = []
            drone.assigned_vehicle = vehicle  # 当前发射无人机的原属车辆
            if self.unvisited_air_edges:
                drone.current_node = vehicle.current_node
                drone.launch_time = vehicle.current_time + vehicle.work_time
                drone.current_time = drone.launch_time
                drone = self.assign_task_to_drone(drone)
                if drone.is_available:  # 没有发射成功
                    break
                else:
                    total_launch_service_time += vehicle.work_time
                    launch_record = {
                        'vehicle_id': vehicle_id,
                        'launch_node': vehicle.current_node,
                        'drone_id': drone.id,
                        'launch_time': drone.launch_time,
                        'inspection_nodes': drone.inspection_nodes,  # 将无人机空中巡检时填充
                        'returnable_nodes': drone.returnable_nodes
                    }
                    vehicle.launch_record.append(launch_record)
                    vehicle.available_drones -= 1
                    # 删除车辆发射的无人机信息
                    vehicle.available_drones.remove(drone)
                # # 更新全局发射记录（如果需要）
                # node_id = vehicle.current_node
                # if node_id not in self.launch:
                #     self.launch[node_id] = []
                # self.launch[node_id].append({
                #     'drone_id': drone.id,
                #     'launch_time': drone.launch_time,
                #     'inspection_nodes': inspection_nodes
                # })
                #  发射完无人机后，判断车辆是否在相同点等待发射的无人机
        return total_launch_service_time

    def assign_task_to_drone(self, drone):
        inspection_nodes = []
        returnable_nodes = []  # 待返回的备选点
        vehicle = drone.assigned_vehicle
        drone.current_time = vehicle.current_time
        drone.current_node = vehicle.current_node
        while self.unvisited_air_edges or (drone.remaining_endurance < drone.endurance*0.85) > 0:  # 通过迭代循环+判定的形式来不断生成可行方案
            # 找到距离当前节点最近的未巡检空中线路（边）
            if drone.is_available:  # 无人机未在空中，从地面节点发射出去
                nearest_edge, distance, nearest_node = find_nearest_unvisited_edge(drone.current_node, self.unvisited_air_edges, self.air_graph, self.ground_graph)
                if nearest_edge is None:
                    print(f"无人机{drone.id}没有可分配的巡检任务。/assign_task_to_drone")
                    break  # 无可分配的任务
                node_u, node_v = nearest_edge
                # 计算到node_u和node_v的距离，选择最近的节点作为目标
                target_node = nearest_node
                if nearest_edge[0] == nearest_node:
                    other_node = nearest_edge[1]
                else:
                    other_node = nearest_edge[0]
                task_nodes = (target_node, other_node)  # 按照顺序进行
                time_to_reach = (distance / drone.speed)  # 无人机从地面节点到达空中节点的时间
                # 计算任务时间
                if nodes_edge(nearest_edge, self.air_graph, self.ground_graph):  # 判断两个节点组成的边是否连接
                    # 有连接边，执行巡检
                    inspection_distance = calculate_travel_distance(task_nodes[0], target_node[1], self.air_graph, self.ground_graph)
                    inspection_time_required = (inspection_distance / drone.task_speed)  # 分钟
                    total_task_time = time_to_reach + inspection_time_required  #
                else:
                    # 没有连接边，飞往其他节点
                    distance_to_other = calculate_travel_distance(task_nodes[0], target_node[1], self.air_graph, self.ground_graph)
                    time_to_other = (distance_to_other / drone.speed)  # 分钟
                    total_task_time = time_to_reach + time_to_other
                drone.remaining_endurance -= total_task_time
                sorted_return_nodes = self.get_sorted_return_nodes(drone)  # 实时返回无人机可返回的地面节点列表
                drone.sorted_return_nodes = sorted_return_nodes
                if total_task_time >= drone.remaining_endurance or len(sorted_return_nodes) <= 1:  # 如果一条弧线都无法巡检，或者无人机没有可供返回的节点
                    # print(f"无人机{drone.id}的续航时间不足以完成任务并返回。")
                    drone.is_available = True  # 被发射至空中时。即其可以被回收时，判定为False。空中为False，地面时为True.
                    break  # 无法继续任务

                drone.current_node = other_node
                drone.current_time += total_task_time

                # 更新无人机的属性
                if self.edge_exists(target_node, other_node, self.air_graph):  # 更新被巡检完成的空中节点
                    inspection_nodes = (target_node, other_node)
                    drone.inspection_nodes.append(inspection_nodes)
                    self.unvisited_air_edges.remove((inspection_nodes))
                # drone.inspection_time = inspection_time
                drone.returnable_nodes = sorted_return_nodes
                drone.remaining_endurance -= total_task_time
                drone.is_available = False  # 记录在空中，无法在执行任务
            else:  # 无人机在空中执行任务
                nearest_node, distance = self.find_nearest_unvisited_edge(drone.current_node)
                if nearest_node is None:
                    print(f"无人机{drone.id}没有可分配的巡检任务。")
                    break
                inspection_nodes = (drone.current_node, nearest_node)
                if self.edge_exists(inspection_nodes, self.air_graph):
                    # 有连接边，执行巡检
                    inspection_distance = self.calculate_edge_distance(inspection_nodes[0], inspection_nodes[1],
                                                                      self.air_graph)
                    time_to_other = (inspection_distance / drone.task_speed) * 60  # 分钟
                else:
                    # 没有连接边，飞往其他节点
                    distance_to_other = self.calculate_distance_between_nodes(inspection_nodes[0], inspection_nodes[1], self.air_graph,
                                                                              self.ground_graph)
                    time_to_other = (distance_to_other / drone.speed) * 60  # 分钟
                sorted_return_nodes = self.get_sorted_return_nodes(drone)  # 实时返回无人机可返回的地面节点列表
                if time_to_other > drone.remaining_endurance or len(self.get_sorted_return_nodes(drone)) <=1:
                    print(f"无人机{drone.id}的续航时间不足以完成任务并返回。")
                    break
                else:
                    # 更新无人机的属性
                    if self.edge_exists(inspection_nodes[0], inspection_nodes[1]):  # 更新被巡检完成的空中节点
                        drone.inspection_nodes.append(inspection_nodes)
                        self.unvisited_air_edges.remove((inspection_nodes))
                    drone.current_time += time_to_other
                    drone.remaining_endurance -= time_to_other
                    drone.is_available = False  # 记录在空中，无法在执行任务
                    drone.returnable_nodes = sorted_return_nodes

        return drone

    def get_sorted_return_nodes(self, drone):
        """
        收集所有车辆尚未到达的地面节点，计算与无人机当前节点的距离，
        并根据距离从小到大对这些节点进行排序，将排序后的列表赋值给无人机的 returnable_nodes 属性。

        :param drone: Drone 对象，无人机实例
        :return: sorted_return_nodes 列表，按距离从小到大排序的返回节点
        """
        # Step 1: 收集所有车辆尚未到达的地面节点
        future_nodes = []
        for veh in self.vehicles.values():
            # 从车辆的路线中筛选出尚未到达的节点
            unvisited = [node for node in veh.route if node not in veh.node_exit_time]
            future_nodes.extend(unvisited)
        # print(f"所有车辆未到达的地面节点（去重前）: {future_nodes}")
        # Step 2: 移除重复节点
        future_nodes = list(set(future_nodes))
        # print(f"所有车辆未到达的地面节点（去重后）: {future_nodes}")
        # Step 3: 计算无人机当前节点到每个返回节点的距离
        distance_node_pairs = []
        for node in future_nodes:
            distance = calculate_travel_distance(
                drone.current_node,  # 无人机当前节点
                node,                # 目标返回节点
                self.air_graph,
                self.ground_graph
            )
            distance_time = distance / drone.speed
            # 将节点和距离打包成元组  整理出
            if distance_time > drone.remaining_endurance:
                continue
            else:
                distance_node_pairs.append((node, distance))

        # print(f"节点与距离对: {distance_node_pairs}")
        # Step 4: 根据距离从小到大排序节点
        sorted_distance_node_pairs = sorted(distance_node_pairs, key=lambda pair: pair[1])
        # print(f"按距离排序的节点与距离对: {sorted_distance_node_pairs}")
        # Step 5: 生成排序后的返回节点列表
        sorted_return_nodes = []
        for pair in sorted_distance_node_pairs:
            sorted_return_nodes.append(pair[0])
        # print(f"按距离排序的返回节点列表: {sorted_return_nodes}")
        # Step 6: 将排序后的列表赋值给无人机的 returnable_nodes 属性
        drone.returnable_nodes = sorted_return_nodes

        return sorted_return_nodes

    def can_vehicle_enter_node_at_time(self, node, arrival_time):
        """
        判断节点在指定的到达时间是否空闲。
        """
        if node not in self.node_occupancy:
            return True
        for (start_time, end_time) in self.node_occupancy[node]:
            if arrival_time >= start_time and arrival_time < end_time:
                return False  # 节点在该时间段被占用
        return True

    def check_node_occupancy_conflict(self, node_id, arrival_time):
        """
        检查节点在给定的到达时间是否有冲突。

        :param node_id: 节点的唯一标识符
        :param arrival_time: 车辆到达节点的时间
        :return: 如果有冲突，返回冲突的时间窗口的结束时间；如果没有冲突，返回输入的到达时间
        """
        if node_id not in self.node_occupancy or not self.node_occupancy[node_id]:
            # 节点没有任何占用记录，直接返回到达时间
            return arrival_time

        # 获取节点的最后一个占用时间段
        last_interval = self.node_occupancy[node_id][-1]
        start_time, end_time = last_interval

        # 检查是否有冲突
        if start_time <= arrival_time < end_time:
            # 有冲突，返回冲突时间段的结束时间
            return end_time
        else:
            # 没有冲突，返回到达时间
            return arrival_time

    def update_node_occupancy(self, node_id, time, is_arrival, vehicle_id):
        """
        实时更新节点占用时间。

        :param node_id: 节点ID
        :param time: 到达或离开时间
        :param is_arrival: True 表示到达，False 表示离开
        :param vehicle_id: 车辆ID
        """
        # 初始化节点占用和待定时间字典
        if not hasattr(self, 'node_occupancy'):
            self.node_occupancy = {}
        if not hasattr(self, 'node_pending_times'):
            self.node_pending_times = {}

        if node_id not in self.node_occupancy:
            self.node_occupancy[node_id] = []
        if node_id not in self.node_pending_times:
            self.node_pending_times[node_id] = {}

        if is_arrival:
            # 记录车辆的到达时间
            self.node_pending_times[node_id][vehicle_id] = time
            # print(f"记录车辆{vehicle_id}在节点{node_id}的到达时间：{time}")
        else:
            # 获取对应的到达时间
            arrival_time = self.node_pending_times[node_id].pop(vehicle_id, None)
            if arrival_time is None:
                # 如果没有找到到达时间，使用离开时间作为到达时间
                arrival_time = time
                # print(f"车辆{vehicle_id}在节点{node_id}没有记录到达时间，使用离开时间 {time} 作为到达时间。")

            if arrival_time != time:
                # 只有当到达时间不等于离开时间时，才记录占用时间段
                departure_time = time
                self.node_occupancy[node_id].append((arrival_time, departure_time))
                # print(f"更新节点{node_id}的占用时间段：({arrival_time}, {departure_time})")
            # else:
                # 如果到达时间等于离开时间，不记录占用
                # print(f"车辆{vehicle_id}在节点{node_id}的到达时间等于离开时间 {time}，节点不被视为占用。")

    def get_conflict_end_time(self, node, arrival_time):
        """
        返回在到达时间之后节点空闲的最早结束时间。
        """
        if node not in self.node_occupancy:
            return arrival_time  # 无冲突
        conflicting_end_times = [
            end_time for (start_time, end_time) in self.node_occupancy[node]
            if arrival_time >= start_time and arrival_time < end_time
        ]
        if conflicting_end_times:
            return max(conflicting_end_times)
        else:
            return arrival_time  # 无冲突


    def generate_initial_solution(self):
        self.generate_vehicle_route()  # 得到两个车辆初始路径
        self.assign_drone_tasks()  # 分配无人机任务，直到所有空中线路均被巡检完成

        # 1.1 编码初始解
        encoded_solution = self.encoder_decoder.encode_solution(self.vehicles, self.drones)

        # 1.2 解码并检查约束
        self.encoder_decoder.decode_solution(encoded_solution, self.vehicles, self.drones)

        # 5.1 计算目标函数
        max_completion_time = self.objective_function.calculate_max_completion_time(
            self.vehicles, self.drones, self.ground_graph, self.air_graph
        )

        return encoded_solution, max_completion_time