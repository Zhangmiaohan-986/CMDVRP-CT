import numpy as np
import networkx as nx
import copy

# 定义节点类
class Node:
    def __init__(self, node_id, x, y, node_type):
        """
        初始化节点
        :param node_id: 节点编号
        :param x: X坐标
        :param y: Y坐标
        :param node_type: 节点类型，'air' 或 'ground'
        """
        self.id = node_id
        self.x = x
        self.y = y
        if node_type == 'air':
            self.z = 20
        else:
            self.z = 0
        self.type = node_type  # 'air' 或 'ground'
        self.adjacent = []  # 邻接节点列表

    def add_adjacent(self, neighbor_id):
        """
        添加邻接节点
        :param neighbor_id: 邻接节点的编号
        """
        if neighbor_id not in self.adjacent:
            self.adjacent.append(neighbor_id)

# 定义图类
class Graph:
    def __init__(self):
        """
        初始化图
        """
        self.nodes = {}  # 节点字典，键为节点编号，值为Node对象
        self.edges = {}  # 边字典，键为（节点编号1, 节点编号2），值为边的权重

    def add_node(self, node):
        """
        添加节点到图中
        :param node: Node对象
        """
        self.nodes[node.id] = node

    def add_edge(self, node1_id, node2_id, weight):
        """
        添加边到图中
        :param node1_id: 节点1的编号
        :param node2_id: 节点2的编号
        :param weight: 边的权重（距离）
        """
        self.edges[(node1_id, node2_id)] = weight
        self.edges[(node2_id, node1_id)] = weight  # 无向图，双向边
        self.nodes[node1_id].add_adjacent(node2_id)
        self.nodes[node2_id].add_adjacent(node1_id)

# 定义车辆类
class Vehicle:
    def __init__(self, vehicle_id, capacity, start_node_id, speed=12.5):
        """
        初始化车辆
        :param vehicle_id: 车辆编号
        :param capacity: 车辆容量（携带无人机数量）
        :param start_node_id: 车辆起始节点编号
        """
        self.id = vehicle_id
        self.capacity = capacity
        self.route = []  # 车辆路径，存储节点编号的列表
        self.drones = []  # 车辆携带的无人机列表
        self.start_node_id = start_node_id
        self.speed = speed  # 车辆速度为12.5km/h
        self.work_time = 3  # 辅助收放电时间为3min
        self.completion_time = 0  # 完成时间
        self.type = 'vehicle'
        self.launch_records = []  # 发射记录，键为节点编号，值为发射的无人机列表
        self.recover_records = []  # 回收记录，键为节点编号，值为回收的无人机列表
        self.launch_recover = {}  # 同时起降并发射，即原地停靠策略产生的情况
        self.current_time = 0  # 记录当前时间
        self.current_node_index = 0  # 当前路径索引
        self.node_entry_time = {}  # 记录进入每个节点的时间
        self.node_exit_time = {}  # 记录离开每个节点的时间
        self.drone_capacity = capacity  # 当前可用的无人机数量
        # self.vehicle_available_drones = self.drones.copy()
        self.vehicle_available_drones = copy.deepcopy(self.drones)
        self.current_node_index = 0
        self.current_node = None  # 车辆实时全局节点
        self.current_time = 0  # 车辆的实时全局时间
        self.end_time = 0  # 车辆离开节点的时间
        self.available_drones = self.drones.copy()  # 可用的无人机列表
        self.available_capacity = self.capacity

        # 新增属性
        self.waiting_times = {}  # 键：节点ID，值：在该节点等待的总时间

    def update_entry_exit_time(self, node_id, entry_time, exit_time):  # 记录vehicle进入每个节点的时间
        self.node_entry_time[node_id] = entry_time
        self.node_exit_time[node_id] = exit_time

    def can_enter_node(self, node_id, global_time, other_vehicles):
        """
        检查在 global_time 时刻，是否可以进入节点 node_id
        :param node_id: 节点编号
        :param global_time: 全局时间
        :param other_vehicles: 其他车辆的列表
        :return: True 或 False
        """
        for vehicle in other_vehicles:
            if node_id in vehicle.node_entry_time:
                if vehicle.node_entry_time[node_id] <= global_time < vehicle.node_exit_time[node_id]:
                    return False
        return True

    def move_to_next_node(self, ground_graph, other_vehicles):
        """
        移动到下一个节点，更新当前时间和节点
        :param ground_graph: 地面网络图
        :param other_vehicles: 其他车辆的列表
        """
        if self.current_node_index < len(self.route) - 1:
            next_node = self.route[self.current_node_index + 1]
            # 检查是否可以进入下一个节点
            while not self.can_enter_node(next_node, self.current_time, other_vehicles):
                # 等待，直到可以进入
                self.current_time += 1  # 时间单位可以根据需要调整
            # 计算移动时间
            current_node = self.route[self.current_node_index]
            distance = ground_graph.edges[(current_node, next_node)]
            travel_time = distance / self.speed
            # 更新时间和节点
            entry_time = self.current_time
            self.current_time += travel_time
            exit_time = self.current_time
            self.update_entry_exit_time(next_node, entry_time, exit_time)
            self.current_node_index += 1
        else:
            # 已到达终点
            pass

    def launch_drone(self, drone, current_node):
        """
        发射无人机，更新无人机和车辆的状态
        :param drone: Drone 对象
        :param current_node: 当前节点编号
        """
        if self.drone_capacity > 0:
            # 更新无人机状态
            drone.is_available = False
            drone.current_node = current_node
            drone.start_time = self.current_time + self.work_time  # 考虑发射时间
            # 更新车辆状态
            self.drone_capacity -= 1
            if current_node not in self.launch:
                self.launch[current_node] = []
            self.launch[current_node].append(drone.id)
            # 更新车辆时间
            self.current_time += self.work_time
            return True
        else:
            return False

    def recover_drone(self, drone, current_node):
        """
        回收无人机，更新无人机和车辆的状态
        :param drone: Drone 对象
        :param current_node: 当前节点编号
        """
        # 更新无人机状态
        drone.is_available = True
        drone.current_node = current_node
        drone.end_time = self.current_time + self.work_time  # 考虑回收时间
        # 更新车辆状态
        self.drone_capacity += 1
        if current_node not in self.recover:
            self.recover[current_node] = []
        self.recover[current_node].append(drone.id)
        # 更新车辆时间
        self.current_time += self.work_time

    def add_drone(self, drone):
        """
        添加无人机到车辆
        :param drone: Drone对象
        """
        self.drones.append(drone)
        # self.drones[drone.id] = drone\

    def set_route(self, route):
        """
        设置车辆路径
        :param route: 节点编号列表
        """
        self.route = route.copy()

    def add_node_to_route(self, node_id):
        """
        向车辆路径中添加节点
        :param node_id: 节点编号
        """
        self.route.append(node_id)

    def encode_route(self):
        """
        编码车辆路径
        :return: 编码后的路径
        """
        return self.route.copy()

    def decode_route(self, encoded_route):
        """
        解码车辆路径
        :param encoded_route: 编码后的路径
        """
        self.route = encoded_route.copy()

    def get_completion_time(self, ground_graph):
        """
        计算车辆完成任务的时间
        :param ground_graph: 地面网络Graph对象
        :return: 完成时间
        """
        total_distance = 0
        for i in range(len(self.route) - 1):
            node1 = self.route[i]
            node2 = self.route[i + 1]
            edge = ground_graph.edges.get((node1, node2))
            if edge:
                total_distance += edge
            else:
                # 如果两个节点之间没有边，假设距离为无穷大
                total_distance += np.inf
        self.completion_time = total_distance
        return self.completion_time

# 定义无人机类
class Drone:
    def __init__(self, drone_id, endurance=0.77, task_speed=28.8, speed=32.4):  # 所有无人机的状态参数，按km/h换算。
        """
        初始化无人机
        :param drone_id: 无人机编号
        :param endurance: 无人机续航时间（距离）
        :param speed: 无人机速度（距离/时间）
        """
        self.id = drone_id
        self.endurance = endurance
        self.task_speed = task_speed
        self.speed = speed
        self.route = []  # 无人机路径，键为发射，降落节点编号，值为路径
        self.time = []
        self.completion_time = 0  # 完成时间
        self.type = 'drone'
        self.launch = {}
        self.recover = {}
        self.launch_recover = {}
        self.is_available = True
        self.current_node = None
        self.start_time = 0  # 发射时间
        self.end_time = 0  # 回收时间

        self.remaining_endurance = endurance
        self.current_node = None
        self.assigned_vehicle = None
        self.launch_time = None
        self.recover_time = None
        self.task_assigned = None
        self.current_time = 0
        self.need_back = False
        self.returnable_nodes = []  # 可返回的节点列表
        self.inspection_nodes = []  # 巡检节点集合
        self.inspection_time = 0  # 巡检总时间


    def assign_task(self, path):
        """
        分配巡检任务
        :param path: 巡检路径（节点编号列表）
        """
        self.route = path.copy()
        self.is_available = False

    def update_status(self, time_spent):
        """
        更新无人机状态
        :param time_spent: 花费的时间
        """
        self.remaining_time -= time_spent
        if self.remaining_time <= 0:
            self.is_available = False  # 无人机需要回收

    def set_route(self, route):
        """
        设置无人机路径
        :param route: 节点编号列表
        """
        self.route = route.copy()

    def add_node_to_route(self, node_id):
        """
        向无人机路径中添加节点
        :param node_id: 节点编号
        """
        self.route.append(node_id)

    def encode_route(self):
        """
        编码无人机路径
        :return: 编码后的路径
        """
        return self.route.copy()

    def decode_route(self, encoded_route):
        """
        解码无人机路径
        :param encoded_route: 编码后的路径
        """
        self.route = encoded_route.copy()

    def get_completion_time(self, air_graph):
        """
        计算无人机完成任务的时间
        :param air_graph: 空中网络Graph对象
        :return: 完成时间
        """
        total_distance = 0
        for i in range(len(self.route) - 1):
            node1 = self.route[i]
            node2 = self.route[i + 1]
            edge = air_graph.edges.get((node1, node2))
            if edge:
                total_distance += edge
            else:
                # 如果两个节点之间没有边，假设距离为无穷大
                total_distance += np.inf
        self.completion_time = total_distance / self.speed
        return self.completion_time

