# Artificial Intelligence Practicals (Python Version)

## 🎓 Subject: Artificial Intelligence (BTCOL707)
### Practicals 2 to 7 Implemented in Python

---

## 🧩 **Practical 2: 8 Queens Problem (Backtracking)**
```python
N = 8

def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col or abs(board[i] - col) == abs(i - row):
            return False
    return True

def solve_queens(board, row):
    if row == N:
        print_solution(board)
        return True
    for col in range(N):
        if is_safe(board, row, col):
            board[row] = col
            if solve_queens(board, row + 1):
                return True
            board[row] = -1
    return False

def print_solution(board):
    print("\nOne possible 8-Queens arrangement:\n")
    for i in range(N):
        row = ''.join(' Q ' if board[i] == j else ' . ' for j in range(N))
        print(row)
    print("\nColumn positions:", board)

board = [-1] * N
solve_queens(board, 0)
```

---

## 🌲 **Practical 3: Depth First Search (DFS)**
```python
graph = {
    'a': ['b', 'c', 'd'],
    'b': ['e', 'f'],
    'c': ['g'],
    'd': [],
    'e': [],
    'f': [],
    'g': []
}

def dfs(graph, start, goal, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path.append(start)
    visited.add(start)

    if start == goal:
        return path

    for neighbor in graph[start]:
        if neighbor not in visited:
            result = dfs(graph, neighbor, goal, path.copy(), visited.copy())
            if result:
                return result
    return None

start_node, goal_node = 'a', 'e'
path = dfs(graph, start_node, goal_node)
print(f"Path found from {start_node} to {goal_node}: {path}")
```

---

## 🧭 **Practical 4: Best First Search**
```python
import heapq

graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('C', 2)],
    'B': [('D', 5)],
    'C': [('G', 3)],
    'D': [('G', 2)],
    'G': []
}

heuristic = {'S': 7, 'A': 6, 'B': 5, 'C': 2, 'D': 1, 'G': 0}

def best_first_search(graph, start, goal):
    pq = []
    heapq.heappush(pq, (heuristic[start], start, [start]))
    visited = set()

    while pq:
        (h, node, path) = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for (neighbor, _) in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor]))
    return None

start, goal = 'S', 'G'
path = best_first_search(graph, start, goal)
print(f"Best-First path from {start} to {goal}: {path}")
```

---

## 🧩 **Practical 5: 8-Puzzle Problem using Best First Search (A*)**
```python
import heapq

goal_state = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

class Node:
    def __init__(self, state, parent=None, g=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = self.heuristic()
        self.f = self.g + self.h

    def heuristic(self):
        h = 0
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                if val != 0:
                    goal_x, goal_y = divmod(goal_state_flat.index(val), 3)
                    h += abs(goal_x - i) + abs(goal_y - j)
        return h

    def __lt__(self, other):
        return self.f < other.f

def get_neighbors(state):
    x, y = next((i, j) for i in range(3) for j in range(3) if state[i][j] == 0)
    moves = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append(new_state)
    return moves

def a_star(start_state):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start_state))

    while open_list:
        node = heapq.heappop(open_list)
        if node.state == goal_state:
            return reconstruct_path(node)
        closed_set.add(tuple(map(tuple, node.state)))
        for neighbor in get_neighbors(node.state):
            if tuple(map(tuple, neighbor)) in closed_set:
                continue
            heapq.heappush(open_list, Node(neighbor, node, node.g + 1))
    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

goal_state_flat = sum(goal_state, [])
start_state = [[0, 8, 1], [2, 4, 3], [7, 6, 5]]
path = a_star(start_state)
for step in path:
    for row in step:
        print(row)
    print()
```

---

## 🤖 **Practical 6: Robot Traversal using Means-End Analysis**
```python
def means_end_analysis(start, goal):
    path = [start]
    while start != goal:
        if start < goal:
            start += 1
        elif start > goal:
            start -= 1
        path.append(start)
    return path

start, goal = 2, 8
print(f"Robot traversal path: {means_end_analysis(start, goal)}")
```

---

## 🚗 **Practical 7: Traveling Salesman Problem (Brute Force)**
```python
from itertools import permutations

graph = {
    'A': {'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30}
}

def traveling_salesman(graph, start):
    vertices = list(graph.keys())
    vertices.remove(start)
    min_path = None
    min_cost = float('inf')

    for perm in permutations(vertices):
        current_cost = 0
        k = start
        for j in perm:
            current_cost += graph[k][j]
            k = j
        current_cost += graph[k][start]

        if current_cost < min_cost:
            min_cost = current_cost
            min_path = (start,) + perm + (start,)

    return min_path, min_cost

start = 'A'
path, cost = traveling_salesman(graph, start)
print(f"Optimal Path: {path}\nMinimum Cost: {cost}")
```

---

✅ **End of AI Practicals (2–7) — Python Implementations**
