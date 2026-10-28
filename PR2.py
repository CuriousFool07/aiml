
from collections import deque
import time

maze = [
    "S...#.",
    ".#...#",
    ".#.#..",
    "...#..",
    "#....E"
]

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def find_start():
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == "S":
                return (r, c)

def dfs():
    start = find_start()
    stack = [(start[0], start[1], 0)]
    visited = {start}

    while stack:
        r, c, steps = stack.pop()

        if maze[r][c] == "E":
            return steps

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < len(maze) and
                0 <= nc < len(maze[0]) and
                maze[nr][nc] != "#" and
                (nr, nc) not in visited):

                visited.add((nr, nc))
                stack.append((nr, nc, steps + 1))

    return -1

def bfs():
    start = find_start()
    queue = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        r, c, steps = queue.popleft()

        if maze[r][c] == "E":
            return steps

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < len(maze) and
                0 <= nc < len(maze[0]) and
                maze[nr][nc] != "#" and
                (nr, nc) not in visited):

                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))

    return -1

print("===== MAZE =====")
for row in maze:
    print(row)

start = time.perf_counter()
dfs_steps = dfs()
dfs_time = time.perf_counter() - start

start = time.perf_counter()
bfs_steps = bfs()
bfs_time = time.perf_counter() - start

print("\nDFS Steps:", dfs_steps)
print("DFS Time:", dfs_time, "seconds")

print("\nBFS Steps:", bfs_steps)
print("BFS Time:", bfs_time, "seconds")
