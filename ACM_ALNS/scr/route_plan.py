import numpy as np
import heapq

def neighbour_plan(vehicle_nodes, ground_adj_matrix, start_node):
    """
    使用基于最短路径的贪心算法生成车辆的路径，从指定的起始节点开始和结束，
    并确保来回不走重复的路径。

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

    # 用于记录已使用的边
    used_edges = set()

    while unvisited_targets:
        min_distance = float('inf')
        nearest_node = None
        shortest_path = []

        # 对于每个未访问的目标节点，计算从当前节点到该节点的最短路径，避免重复使用边
        for target_node in unvisited_targets:
            # 使用 Dijkstra 算法计算最短路径，避免已使用的边
            path, distance = dijkstra(ground_adj_matrix, current_node, target_node, initial_node, used_edges)
            if distance < min_distance:
                min_distance = distance
                nearest_node = target_node
                shortest_path = path

        if nearest_node is None:
            print("无法在不重复路径的情况下找到到达剩余节点的路径。")
            break

        # 将最短路径中的节点和边添加到路线和已使用的边集合中
        for i in range(1, len(shortest_path)):
            node_u = shortest_path[i - 1]
            node_v = shortest_path[i]
            route.append(node_v)
            # 添加边到已使用的边集合（无向图，添加 (u, v) 和 (v, u)）
            used_edges.add((node_u, node_v))
            used_edges.add((node_v, node_u))

        # 更新当前节点和已访问的目标节点集合
        current_node = nearest_node
        visited_targets.add(current_node)
        unvisited_targets = vehicle_nodes_set - visited_targets

    # 最后返回到起始节点 start_node，避免重复使用边
    path_to_start, distance_to_start = dijkstra(ground_adj_matrix, current_node, start_node, initial_node, used_edges)
    if path_to_start:
        for i in range(1, len(path_to_start)):
            node_u = path_to_start[i - 1]
            node_v = path_to_start[i]
            route.append(node_v)
            used_edges.add((node_u, node_v))
            used_edges.add((node_v, node_u))
    else:
        print(f"无法在不重复路径的情况下从节点 {current_node} 返回起始节点 {start_node}。")

    return route

def dijkstra(adj_matrix, start_node, end_node, initial_node, used_edges):
    """
    使用 Dijkstra 算法计算从 start_node 到 end_node 的最短路径，避免使用已使用的边。

    :param adj_matrix: 邻接矩阵。
    :param start_node: 起始节点编号（节点 ID）。
    :param end_node: 终止节点编号（节点 ID）。
    :param initial_node: 初始节点编号（节点 ID），用于计算索引偏移。
    :param used_edges: 已使用的边集合，包含节点编号的元组 (u, v)。
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
            # 将索引转换回节点编号
            u_node_id = index_to_node_id(u)
            v_node_id = index_to_node_id(v)

            # 检查边 (u_node_id, v_node_id) 是否已使用
            if adj_matrix[u][v] > 0 and not visited[v] and (u_node_id, v_node_id) not in used_edges:
                alt = distance[u] + adj_matrix[u][v]
                if alt < distance[v]:
                    distance[v] = alt
                    prev_node[v] = u
                    heapq.heappush(heap, (alt, v))

    # 构建最短路径（节点 ID 列表）
    path = []
    u = end_index
    if prev_node[u] is not None or u == start_index:
        while u is not None:
            path.insert(0, index_to_node_id(u))
            u = prev_node[u]
    else:
        # 无法到达终点
        return [], float('inf')

    return path, distance[end_index]

def calculate_travel_time(current_node, next_node, air_graph, ground_graph, speed):  # 计算任意情况节点的旅行时间
    nodes = (current_node,next_node)  # 输入node代号
    air_nodes = list(air_graph.nodes.keys())
    ground_nodes = list(ground_graph.nodes.keys())
    # 判断是空中节点或者是地面节点
    if all(node in air_nodes for node in nodes):
        nodes_type = 'air_node'
    elif all(node in ground_nodes for node in nodes):
        nodes_type = 'ground_node'
    else:
        nodes_type = 'mixed_node'
    if nodes_type == 'air_node':
        # 判断两个空中节点是否相邻
        if nodes_edge(nodes, air_graph, ground_graph):
            distance = air_graph.edges[nodes]  # 用已知距离索引
            travel_time = distance/speed
        else:
            distance = calculate_distance(nodes, air_graph, ground_graph)
            travel_time = distance/speed
    elif nodes_type == 'ground_node':
        if nodes_edge(nodes, air_graph, ground_graph):
            distance = ground_graph.edges[nodes]  # 用已知距离索引
            travel_time = distance/speed
        else:
            distance = calculate_distance(nodes, air_graph, ground_graph)
            travel_time = distance/speed
    else:
        distance = calculate_distance(nodes, air_graph, ground_graph)
        travel_time = distance / speed

    return travel_time

def calculate_travel_distance(current_node, next_node, air_graph, ground_graph):  # 计算任意情况节点的旅行时间
    nodes = (current_node,next_node)  # 输入node代号
    air_nodes = list(air_graph.nodes.keys())
    ground_nodes = list(ground_graph.nodes.keys())
    # 判断是空中节点或者是地面节点
    if all(node in air_nodes for node in nodes):
        nodes_type = 'air_node'
    elif all(node in ground_nodes for node in nodes):
        nodes_type = 'ground_node'
    else:
        nodes_type = 'mixed_node'
    if nodes_type == 'air_node':
        # 判断两个空中节点是否相邻
        if nodes_edge(nodes, air_graph, ground_graph):
            distance = air_graph.edges[nodes]  # 用已知距离索引
            travel_distance = distance
        else:
            distance = calculate_distance(nodes, air_graph, ground_graph)
            travel_distance = distance
    elif nodes_type == 'ground_node':
        if nodes_edge(nodes, air_graph, ground_graph):
            distance = ground_graph.edges[nodes]  # 用已知距离索引
            travel_distance = distance
        else:
            distance = calculate_distance(nodes, air_graph, ground_graph)
            travel_distance = distance
    else:
        distance = calculate_distance(nodes, air_graph, ground_graph)
        travel_distance = distance

    return travel_distance


def nodes_edge(nodes, air_graph, ground_graph):
    air_graph_keys = air_graph.edges.keys()
    ground_graph_keys = ground_graph.edges.keys()
    if nodes in air_graph_keys or nodes in ground_graph_keys:
        return True

def calculate_distance(nodes, air_graph, ground_graph):  # 仅处理空地节点情况 nodes代表从i-j的情况
    current_node_code = nodes[0]
    next_node_code = nodes[1]
    air_graph_keys = list(air_graph.nodes.keys())
    ground_graph_keys = list(ground_graph.nodes.keys())
    if current_node_code in air_graph_keys:
        air_position = np.array((air_graph_keys.nodes[current_node_code].x,air_graph_keys.nodes[current_node_code].y,air_graph_keys.nodes[current_node_code].z))
        ground_position = np.array((ground_graph_keys.nodes[next_node_code].x,ground_graph_keys.nodes[next_node_code].y,ground_graph_keys.nodes[next_node_code].z))
    else:
        air_position = np.array((air_graph_keys.nodes[next_node_code].x,air_graph_keys.nodes[next_node_code].y,air_graph_keys.nodes[next_node_code].z))
        ground_position = np.array((ground_graph_keys.nodes[current_node_code].x,ground_graph_keys.nodes[current_node_code].y,ground_graph_keys.nodes[current_node_code].z))
    distance = np.linalg.norm(air_position - ground_position)
    return distance

def find_nearest_unvisited_edge(node, unvisited_edges, air_graph, ground_graph):
    """
      找到距离当前节点最近且未被巡检的空中线路（边）。
      :param current_node: 当前节点编号
      :return: 最近的未被巡检的边及其距离，若不存在则返回 (None, None)
      """
    nearest_edge = None
    min_distance = float('inf')
    best_nearest_edge = None
    distance = float('inf')
    best_nearest_node = None

    for edge in unvisited_edges:
        node_u, node_v = edge
        # 计算距离当前节点到edge起点的距离
        if node_u == node or node_v == node:
            distance = 0  # 已在该节点
        else:
            distance_to_u = calculate_travel_distance(node, node_u, air_graph, ground_graph)
            distance_to_v = calculate_travel_distance(node, node_v, air_graph, ground_graph)
            distance = min(distance_to_u, distance_to_v)  # 取到node_u和node_v中最短的距离

        if distance < min_distance:
            min_distance = distance
            best_nearest_edge = edge
            # 找到距离最短的那个节点u或者v
            best_nearest_node = node_u if distance_to_u < distance_to_v else node_v

    return best_nearest_edge, min_distance, best_nearest_node