import numpy as np
import math
import heapq

def neighbour_plan(vehicle_nodes, ground_adj_matrix, start_node):
    """
    使用基于最短路径的贪心算法生成车辆的路径，从指定的起始节点开始和结束。

    :param vehicle_nodes: 要访问的地面节点编号列表。
    :param ground_adj_matrix: 完整的邻接矩阵，ground_adj_matrix[i][j] 表示节点 i 和节点 j 之间的距离。
    :param start_node: 起始节点的编号。
    :return: 从 start_node 开始和结束的路径列表，包含所有指定的节点，每个节点只经过一次。
    """
    # 节点总数
    num_nodes = len(ground_adj_matrix)

    # 未访问的目标节点集合（指定的节点）
    vehicle_nodes_set = set(vehicle_nodes)
    vehicle_nodes_set.add(start_node)  # 确保包含起始节点

    # 已访问的目标节点集合
    visited_targets = set()
    initial_node = start_node
    # 当前节点
    current_node = start_node
    route = [current_node]
    visited_targets.add(current_node)

    # 未访问的目标节点（不包括当前节点）
    unvisited_targets = vehicle_nodes_set - visited_targets

    while unvisited_targets:
        min_distance = float('inf')
        nearest_node = None
        shortest_path = []

        # 对于每个未访问的目标节点，计算从当前节点到该节点的最短路径长度
        for target_node in unvisited_targets:
            # 使用 Dijkstra 算法计算最短路径
            path, distance = dijkstra(ground_adj_matrix, current_node, target_node, initial_node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = target_node
                shortest_path = path

        if nearest_node is None:
            print("无法找到到达剩余节点的路径")
            break

        # 将最短路径中的节点添加到路线中（跳过当前节点，以避免重复）
        for node in shortest_path[1:]:
            route.append(node)

        # 更新当前节点和已访问的目标节点集合
        current_node = nearest_node
        visited_targets.add(current_node)
        unvisited_targets = vehicle_nodes_set - visited_targets

    # 最后返回到起始节点 start_node
    path_to_start, distance_to_start = dijkstra(ground_adj_matrix, current_node, start_node, initial_node)
    if path_to_start:
        for node in path_to_start[1:]:
            route.append(node)
    else:
        print(f"无法从节点 {current_node} 返回起始节点 {start_node}")

    return route

def dijkstra(adj_matrix, start_node, end_node, initial_node):
    """
    使用 Dijkstra 算法计算从 start_node 到 end_node 的最短路径。

    :param adj_matrix: 邻接矩阵。
    :param start_node: 起始节点编号（节点 ID）。
    :param end_node: 终止节点编号（节点 ID）。
    :param initial_node: 初始节点编号（节点 ID），用于计算索引偏移。
    :return: (最短路径列表（节点 ID 列表）, 最短距离)
    """
    import heapq

    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    distance = [float('inf')] * num_nodes
    prev_node = [None] * num_nodes

    # 将节点编号转换为矩阵索引
    def node_id_to_index(node_id):
        return node_id - initial_node

    def index_to_node_id(index):
        return index + initial_node

    start_index = node_id_to_index(start_node)
    end_index = node_id_to_index(end_node)

    distance[start_index] = 0
    heap = [(0, start_index)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        if u == end_index:
            break

        for v in range(num_nodes):
            if adj_matrix[u][v] > 0 and not visited[v]:
                alt = distance[u] + adj_matrix[u][v]
                if alt < distance[v]:
                    distance[v] = alt
                    prev_node[v] = u
                    heapq.heappush(heap, (distance[v], v))

    # 构建最短路径（节点 ID 列表）
    path = []
    u = end_index
    if prev_node[u] is not None or u == start_index:
        while u is not None:
            path.insert(0, index_to_node_id(u))
            u = prev_node[u]
    else:
        # 无法到达终点
        path = []

    return path, distance[end_index]
