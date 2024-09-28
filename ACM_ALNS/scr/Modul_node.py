import numpy as np
import networkx as nx

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
    def __init__(self, vehicle_id, capacity, start_node_id):
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
        self.completion_time = 0  # 完成时间

    def add_drone(self, drone):
        """
        添加无人机到车辆
        :param drone: Drone对象
        """
        self.drones.append(drone)

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
    def __init__(self, drone_id, endurance, speed):
        """
        初始化无人机
        :param drone_id: 无人机编号
        :param endurance: 无人机续航时间（距离）
        :param speed: 无人机速度（距离/时间）
        """
        self.id = drone_id
        self.endurance = endurance
        self.speed = speed
        self.route = []  # 无人机巡检路径，存储节点编号的列表
        self.completion_time = 0  # 完成时间

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

