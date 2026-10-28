
import heapq
import time

graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('C', 20)],
    'B': [('D', 1)],
    'C': [('G', 8)],
    'D': [('G', 10)],
    'G': []
}

h = {
    'S': 10, 'A': 9, 'B': 8,
    'C': 8, 'D': 10, 'G': 0
}

def get_path(parent, goal):
    path = []
    node = goal

    while node is not None:
        path.append(node)
        node = parent[node]

    return path[::-1]

def greedy(start, goal):
    pq = [(h[start], start)]
    parent = {start: None}
    visited = set()

    while pq:
        _, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return get_path(parent, goal), visited

        for neighbor, cost in graph[node]:
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = node
                heapq.heappush(pq, (h[neighbor], neighbor))

    return [], visited

def astar(start, goal):
    pq = [(h[start], start)]
    g = {start: 0}
    parent = {start: None}
    visited = set()

    while pq:
        _, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return get_path(parent, goal), g[node], visited

        for neighbor, cost in graph[node]:
            new_cost = g[node] + cost

            if neighbor not in g or new_cost < g[neighbor]:
                g[neighbor] = new_cost
                parent[neighbor] = node
                f = new_cost + h[neighbor]
                heapq.heappush(pq, (f, neighbor))

    return [], -1, visited

start = time.perf_counter()
path1, visited1 = greedy('S', 'G')
time1 = time.perf_counter() - start

start = time.perf_counter()
path2, cost2, visited2 = astar('S', 'G')
time2 = time.perf_counter() - start

cost1 = 0
for i in range(len(path1) - 1):
    for neighbor, cost in graph[path1[i]]:
        if neighbor == path1[i + 1]:
            cost1 += cost

print("Greedy Path:", path1)
print("Greedy Cost:", cost1)
print("Greedy Nodes Explored:", visited1)
print("Greedy Time:", time1)

print("\nA* Path:", path2)
print("A* Cost:", cost2)
print("A* Nodes Explored:", visited2)
print("A* Time:", time2)
