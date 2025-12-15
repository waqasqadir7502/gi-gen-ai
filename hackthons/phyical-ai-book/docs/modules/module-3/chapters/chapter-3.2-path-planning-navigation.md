# Chapter 3.2: Path Planning and Navigation

## Overview

Path planning and navigation are essential capabilities for humanoid robots operating in complex environments. This chapter covers fundamental path planning algorithms, navigation techniques, and integration with perception and control systems to enable humanoid robots to navigate safely and efficiently through their environments.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Implement fundamental path planning algorithms (A*, RRT, Dijkstra)
2. Design navigation systems that integrate with perception and control
3. Apply dynamic path planning for moving obstacles
4. Implement navigation safety and obstacle avoidance
5. Integrate path planning with balance and locomotion control

## Introduction to Path Planning

Path planning is the process of finding a valid route from a start position to a goal position while avoiding obstacles. For humanoid robots, path planning must consider not only geometric constraints but also dynamic and balance constraints.

### Path Planning vs. Navigation

- **Path Planning**: Algorithmic process of finding a collision-free path
- **Navigation**: Complete system including perception, localization, path planning, and execution

### Challenges in Humanoid Robot Navigation

- **Complex Geometry**: Humanoid robots have complex shapes and multiple degrees of freedom
- **Dynamic Constraints**: Balance and locomotion constraints affect possible paths
- **Real-time Requirements**: Paths must be computed and updated in real-time
- **Uncertainty**: Sensor noise and environment changes require robust planning
- **Human Safety**: Navigation must be safe around humans and delicate objects

## Fundamental Path Planning Algorithms

### Graph-Based Algorithms

#### Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path in a weighted graph by expanding from the start node and maintaining the shortest distance to each node.

```python
import heapq
import numpy as np
from collections import defaultdict

class DijkstraPlanner:
    """Dijkstra's algorithm for path planning"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v, weight):
        """Add an edge to the graph"""
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))  # Assuming undirected graph
        self.vertices.add(u)
        self.vertices.add(v)

    def dijkstra(self, start, goal):
        """Find shortest path using Dijkstra's algorithm"""
        # Initialize distances and predecessors
        distances = {vertex: float('infinity') for vertex in self.vertices}
        distances[start] = 0
        predecessors = {vertex: None for vertex in self.vertices}

        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()

        while pq:
            current_distance, current_vertex = heapq.heappop(pq)

            if current_vertex in visited:
                continue

            visited.add(current_vertex)

            # If we reached the goal, reconstruct path
            if current_vertex == goal:
                break

            # Update distances for neighbors
            for neighbor, weight in self.graph[current_vertex]:
                if neighbor not in visited:
                    new_distance = current_distance + weight

                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))

        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()

        return path if path[0] == start else []

# Example usage
def example_dijkstra():
    planner = DijkstraPlanner()

    # Add edges to create a simple grid-like graph
    # (0,0) - (0,1) - (0,2)
    #   |       |       |
    # (1,0) - (1,1) - (1,2)
    #   |       |       |
    # (2,0) - (2,1) - (2,2)

    edges = [
        ((0,0), (0,1), 1), ((0,1), (0,2), 1),
        ((1,0), (1,1), 1), ((1,1), (1,2), 1),
        ((2,0), (2,1), 1), ((2,1), (2,2), 1),
        ((0,0), (1,0), 1), ((1,0), (2,0), 1),
        ((0,1), (1,1), 1), ((1,1), (2,1), 1),
        ((0,2), (1,2), 1), ((1,2), (2,2), 1),
    ]

    for u, v, w in edges:
        planner.add_edge(u, v, w)

    path = planner.dijkstra((0,0), (2,2))
    print(f"Dijkstra path from (0,0) to (2,2): {path}")

    return path
```

#### A* Algorithm

A* improves on Dijkstra by using a heuristic function to guide the search toward the goal, making it more efficient for path planning.

```python
class AStarPlanner:
    """A* algorithm for path planning with heuristic guidance"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v, weight):
        """Add an edge to the graph"""
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
        self.vertices.add(u)
        self.vertices.add(v)

    def heuristic(self, a, b):
        """Heuristic function (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def astar(self, start, goal):
        """Find path using A* algorithm"""
        # Priority queue: (f_score, g_score, vertex)
        open_set = [(self.heuristic(start, goal), 0, start)]

        # Costs
        g_score = {vertex: float('infinity') for vertex in self.vertices}
        g_score[start] = 0

        # Predecessors for path reconstruction
        came_from = {}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Process neighbors
            for neighbor, weight in self.graph[current]:
                tentative_g = g_score[current] + weight

                if tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return []  # No path found

# Example usage
def example_astar():
    planner = AStarPlanner()

    # Add the same edges as before
    edges = [
        ((0,0), (0,1), 1), ((0,1), (0,2), 1),
        ((1,0), (1,1), 1), ((1,1), (1,2), 1),
        ((2,0), (2,1), 1), ((2,1), (2,2), 1),
        ((0,0), (1,0), 1), ((1,0), (2,0), 1),
        ((0,1), (1,1), 1), ((1,1), (2,1), 1),
        ((0,2), (1,2), 1), ((1,2), (2,2), 1),
    ]

    for u, v, w in edges:
        planner.add_edge(u, v, w)

    path = planner.astar((0,0), (2,2))
    print(f"A* path from (0,0) to (2,2): {path}")

    return path
```

### Sampling-Based Algorithms

#### Rapidly-exploring Random Trees (RRT)

RRT is particularly useful for high-dimensional spaces and complex constraints. It grows a tree from the start position toward random samples in the space.

```python
class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) for path planning"""

    def __init__(self, start, goal, bounds, step_size=0.1, max_iterations=1000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
        self.step_size = step_size
        self.max_iterations = max_iterations

        # Tree initialization
        self.vertices = [self.start]
        self.edges = {}  # child -> parent
        self.edges[self.start.tobytes()] = None

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)

    def nearest_vertex(self, point):
        """Find the nearest vertex in the tree to the given point"""
        min_dist = float('inf')
        nearest = None

        for v in self.vertices:
            dist = self.distance(v, point)
            if dist < min_dist:
                min_dist = dist
                nearest = v

        return nearest

    def steer(self, from_point, to_point):
        """Steer from from_point toward to_point by step_size"""
        direction = to_point - from_point
        norm = np.linalg.norm(direction)

        if norm <= self.step_size:
            return to_point
        else:
            return from_point + (direction / norm) * self.step_size

    def is_collision_free(self, p1, p2):
        """Check if the path between p1 and p2 is collision-free"""
        # In a real implementation, this would check against obstacle data
        # For this example, we'll assume no obstacles
        return True

    def plan(self, obstacles=None):
        """Plan path using RRT"""
        for i in range(self.max_iterations):
            # Sample random point
            rand_point = np.array([
                np.random.uniform(self.bounds[0], self.bounds[1]),
                np.random.uniform(self.bounds[2], self.bounds[3])
            ])

            # Find nearest vertex in tree
            nearest = self.nearest_vertex(rand_point)

            # Steer toward random point
            new_point = self.steer(nearest, rand_point)

            # Check for collision
            if self.is_collision_free(nearest, new_point):
                # Add new point to tree
                self.vertices.append(new_point)
                self.edges[new_point.tobytes()] = nearest

                # Check if goal is reached
                if self.distance(new_point, self.goal) < self.step_size:
                    # Goal reached, construct path
                    return self.reconstruct_path(new_point)

        # If we get here, no path was found
        return []

    def reconstruct_path(self, goal_vertex):
        """Reconstruct path from goal back to start"""
        path = [goal_vertex]
        current = goal_vertex

        while current is not None:
            parent_key = self.edges[current.tobytes()]
            if parent_key is not None:
                parent = parent_key
                path.append(parent)
                current = parent
            else:
                break

        path.reverse()
        return path

# Example usage
def example_rrt():
    planner = RRTPlanner(
        start=(0, 0),
        goal=(10, 10),
        bounds=(-1, 11, -1, 11),
        step_size=0.5,
        max_iterations=500
    )

    path = planner.plan()
    print(f"RRT path found with {len(path)} points")

    return path
```

### Grid-Based Path Planning

Grid-based methods discretize the environment into a grid and apply graph-based algorithms.

```python
class GridPathPlanner:
    """Grid-based path planning with A* algorithm"""

    def __init__(self, grid_resolution=0.1, robot_radius=0.3):
        self.grid_resolution = grid_resolution
        self.robot_radius = robot_radius
        self.grid = None

    def setup_grid(self, width, height, obstacles):
        """Set up the grid with obstacles"""
        grid_width = int(width / self.grid_resolution)
        grid_height = int(height / self.grid_resolution)

        # Create occupancy grid (0 = free, 1 = occupied)
        self.grid = np.zeros((grid_width, grid_height))

        # Mark obstacles in the grid
        for obs in obstacles:
            # Inflate obstacle by robot radius
            inflated_obs = self.inflate_obstacle(obs, self.robot_radius)
            self.mark_obstacle_in_grid(inflated_obs)

    def inflate_obstacle(self, obstacle, inflation_radius):
        """Inflate obstacle by robot radius"""
        inflated = []
        inflation_cells = int(inflation_radius / self.grid_resolution)

        for x in range(obstacle[0] - inflation_cells, obstacle[0] + inflation_cells + 1):
            for y in range(obstacle[1] - inflation_cells, obstacle[1] + inflation_cells + 1):
                inflated.append((x, y))

        return inflated

    def mark_obstacle_in_grid(self, obstacle_points):
        """Mark obstacle points in the grid"""
        for x, y in obstacle_points:
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                self.grid[x, y] = 1

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x / self.grid_resolution)
        grid_y = int(y / self.grid_resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.grid_resolution
        y = grid_y * self.grid_resolution
        return x, y

    def is_valid_cell(self, x, y):
        """Check if grid cell is valid (within bounds and not occupied)"""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
            return self.grid[x, y] == 0
        return False

    def get_neighbors(self, x, y):
        """Get valid neighboring cells (8-connected)"""
        neighbors = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        costs = [1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414]  # Diagonal vs straight costs

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if self.is_valid_cell(nx, ny):
                neighbors.append(((nx, ny), costs[i]))

        return neighbors

    def heuristic(self, a, b):
        """Manhattan distance heuristic for grid"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan_path(self, start_world, goal_world):
        """Plan path using A* on the grid"""
        start_grid = self.world_to_grid(*start_world)
        goal_grid = self.world_to_grid(*goal_world)

        # Check if start and goal are valid
        if not self.is_valid_cell(*start_grid) or not self.is_valid_cell(*goal_grid):
            return []

        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(self.grid_to_world(*start_grid))
                path.reverse()
                return path

            for neighbor, cost in self.get_neighbors(*current):
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

# Example usage
def example_grid_planner():
    planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)

    # Setup a simple grid with obstacles
    obstacles = [(5, 5), (5, 6), (6, 5), (6, 6)]  # Square obstacle
    planner.setup_grid(10, 10, obstacles)

    path = planner.plan_path((1, 1), (8, 8))
    print(f"Grid path found with {len(path)} points")

    return path
```

## Humanoid Robot Specific Considerations

### Configuration Space (C-Space)

For humanoid robots, path planning must consider the robot's full configuration space, not just its position.

```python
class HumanoidCspacePlanner:
    """Configuration space path planning for humanoid robots"""

    def __init__(self, robot_model):
        self.robot_model = robot_model  # Contains kinematic model
        self.step_size = 0.1  # Step size in configuration space
        self.rotation_step = 0.1  # Step size for orientation

    def is_collision_free(self, config1, config2):
        """Check if path between two configurations is collision-free"""
        # This would involve checking multiple points along the path
        # and verifying no collisions with environment
        num_steps = 10
        for i in range(num_steps + 1):
            t = i / num_steps
            interpolated_config = self.interpolate_configs(config1, config2, t)

            # Check collision for this configuration
            if not self.check_collision_free(interpolated_config):
                return False

        return True

    def interpolate_configs(self, config1, config2, t):
        """Interpolate between two configurations"""
        # Linear interpolation for position
        pos_interp = config1[:3] + t * (config2[:3] - config1[:3])

        # Spherical linear interpolation for orientation
        quat_interp = self.slerp(config1[3:7], config2[3:7], t)

        # Joint interpolation
        joint_interp = config1[7:] + t * (config2[7:] - config1[7:])

        return np.concatenate([pos_interp, quat_interp, joint_interp])

    def slerp(self, q1, q2, t):
        """Spherical linear interpolation for quaternions"""
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Calculate dot product
        dot = np.dot(q1, q2)

        # If dot product is negative, negate one quaternion
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        # Calculate angle between quaternions
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)

        # Calculate coefficients
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        coeff1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        coeff2 = sin_theta / sin_theta_0

        # Calculate result
        result = coeff1 * q1 + coeff2 * q2
        return result / np.linalg.norm(result)

    def check_collision_free(self, config):
        """Check if configuration is collision-free"""
        # Use robot model to calculate link positions
        link_positions = self.robot_model.calculate_link_positions(config)

        # Check each link against obstacles
        for link_pos in link_positions:
            if self.check_point_collision(link_pos):
                return False

        return True

    def check_point_collision(self, point):
        """Check if a point collides with any obstacle"""
        # In a real implementation, this would check against obstacle data
        # For this example, we'll assume a simple check
        return False  # Assume no collision for now
```

### Balance-Constrained Path Planning

Humanoid robots must maintain balance while navigating, which constrains possible paths.

```python
class BalanceConstrainedPlanner:
    """Path planning with balance constraints for humanoid robots"""

    def __init__(self, robot_height=0.8, max_slope=0.3, max_step_height=0.1):
        self.robot_height = robot_height
        self.max_slope = max_slope  # Maximum slope in radians
        self.max_step_height = max_step_height  # Maximum step height in meters
        self.omega = np.sqrt(9.81 / robot_height)

    def is_balance_feasible(self, path_segment):
        """Check if path segment is balance-feasible"""
        if len(path_segment) < 2:
            return True

        for i in range(len(path_segment) - 1):
            p1 = path_segment[i]
            p2 = path_segment[i + 1]

            # Check elevation change
            if abs(p2[2] - p1[2]) > self.max_step_height:
                return False

            # Check slope
            horizontal_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if horizontal_dist > 0:
                slope = abs(p2[2] - p1[2]) / horizontal_dist
                if slope > self.max_slope:
                    return False

        return True

    def generate_feasible_path(self, start, goal, obstacles):
        """Generate a balance-feasible path"""
        # First, plan a basic path
        grid_planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)

        # Convert to 2D for initial planning
        start_2d = (start[0], start[1])
        goal_2d = (goal[0], goal[1])

        path_2d = grid_planner.plan_path(start_2d, goal_2d)

        if not path_2d:
            return []

        # Elevate 2D path to 3D with balance constraints
        path_3d = []
        for x, y in path_2d:
            # For this example, we'll assume z=0 (flat ground)
            # In practice, this would incorporate terrain information
            path_3d.append((x, y, 0.0))

        return path_3d

    def optimize_path_for_balance(self, path):
        """Optimize path for better balance during execution"""
        if len(path) < 3:
            return path

        optimized_path = [path[0]]

        for i in range(1, len(path) - 1):
            # Check if removing this point maintains balance feasibility
            # and improves the path
            prev_point = optimized_path[-1]
            next_point = path[i + 1]

            # Create potential shortcut
            shortcut_segment = [prev_point, next_point]

            if self.is_balance_feasible(shortcut_segment):
                # Keep the shortcut if it's feasible
                continue
            else:
                # Add current point if shortcut is not feasible
                optimized_path.append(path[i])

        optimized_path.append(path[-1])

        return optimized_path
```

## Dynamic Path Planning

### Receding Horizon Planning

Dynamic environments require replanning as new information becomes available.

```python
class RecedingHorizonPlanner:
    """Receding horizon path planning for dynamic environments"""

    def __init__(self, horizon=5.0, update_frequency=10.0):
        self.horizon = horizon  # Planning horizon in meters
        self.update_frequency = update_frequency  # Hz
        self.last_update_time = 0.0

        # Local planner
        self.local_planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)

        # Global path and current position
        self.global_path = []
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.path_index = 0

    def update_plan(self, current_pos, goal_pos, obstacles, current_time):
        """Update the plan based on current information"""
        self.current_position = current_pos

        # Check if it's time to update
        if current_time - self.last_update_time < 1.0 / self.update_frequency:
            return self.follow_current_path()

        self.last_update_time = current_time

        # Plan locally within horizon
        local_goal = self.compute_local_goal(goal_pos)

        # Update local grid with new obstacle information
        self.local_planner.setup_grid(20, 20, obstacles)

        # Plan path to local goal
        local_path = self.local_planner.plan_path(
            (current_pos[0], current_pos[1]),
            (local_goal[0], local_goal[1])
        )

        if local_path:
            # Convert to 3D path
            local_path_3d = [(x, y, current_pos[2]) for x, y in local_path]
            return local_path_3d
        else:
            # No local path found, return empty
            return []

    def compute_local_goal(self, global_goal):
        """Compute local goal within planning horizon"""
        direction = global_goal - self.current_position
        distance = np.linalg.norm(direction)

        if distance <= self.horizon:
            # Global goal is within horizon
            return global_goal
        else:
            # Local goal is along the direction to global goal
            normalized_dir = direction / distance
            local_goal = self.current_position + normalized_dir * self.horizon
            return local_goal

    def follow_current_path(self):
        """Return next segment of current path"""
        # In a real implementation, this would follow the current local path
        # until it's time to replan
        return []
```

### Moving Obstacle Avoidance

Planning around moving obstacles requires prediction of future positions.

```python
class MovingObstacleAvoidance:
    """Path planning with moving obstacle prediction"""

    def __init__(self, prediction_horizon=3.0, time_step=0.1):
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step
        self.num_time_steps = int(prediction_horizon / time_step)

    def predict_obstacle_trajectories(self, moving_obstacles):
        """Predict future positions of moving obstacles"""
        predicted_paths = {}

        for obs_id, (position, velocity) in moving_obstacles.items():
            predicted_path = []
            current_pos = np.array(position)
            current_vel = np.array(velocity)

            for i in range(self.num_time_steps):
                # Simple constant velocity prediction
                future_pos = current_pos + current_vel * (i * self.time_step)
                predicted_path.append(future_pos)

            predicted_paths[obs_id] = predicted_path

        return predicted_paths

    def plan_with_dynamic_obstacles(self, start, goal, static_obstacles, moving_obstacles):
        """Plan path considering predicted moving obstacle positions"""
        # Predict obstacle positions
        predicted_obstacles = self.predict_obstacle_trajectories(moving_obstacles)

        # Create time-parameterized grid
        grid_planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)

        # Plan considering obstacle predictions
        # This is a simplified approach - in practice, you'd use 4D planning (x,y,z,t)
        path = self.temporal_astar(start, goal, static_obstacles, predicted_obstacles)

        return path

    def temporal_astar(self, start, goal, static_obstacles, predicted_moving_obstacles):
        """A* with temporal dimension for dynamic obstacles"""
        # For simplicity, we'll implement a basic version
        # that checks obstacle presence at each time step

        # This would be implemented as a 4D search (x,y,z,time)
        # For now, we'll use a simplified approach

        # Plan path in 2D first
        grid_planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)
        grid_planner.setup_grid(20, 20, static_obstacles)

        path_2d = grid_planner.plan_path(
            (start[0], start[1]),
            (goal[0], goal[1])
        )

        if not path_2d:
            return []

        # Check for conflicts with predicted obstacles
        path_3d = []
        time_estimate = 0.0
        time_step = 0.1  # 10 Hz assumption

        for i, (x, y) in enumerate(path_2d):
            # Check if this position is occupied by a moving obstacle at this time
            occupied = False

            for obs_id, obs_predictions in predicted_moving_obstacles.items():
                if i < len(obs_predictions):
                    obs_pos = obs_predictions[i]
                    # Check if robot position is too close to obstacle
                    dist = np.sqrt((x - obs_pos[0])**2 + (y - obs_pos[1])**2)
                    if dist < 0.5:  # 0.5m safety margin
                        occupied = True
                        break

            if not occupied:
                path_3d.append((x, y, start[2]))
            else:
                # Need to replan around this obstacle
                # For simplicity, we'll just add a detour
                detour_x = x + 0.5  # Move right to avoid obstacle
                detour_y = y
                path_3d.append((detour_x, detour_y, start[2]))

            time_estimate += time_step

        return path_3d
```

## Navigation System Integration

### Navigation Stack Architecture

A complete navigation system integrates multiple components:

```python
class NavigationSystem:
    """Complete navigation system for humanoid robots"""

    def __init__(self, robot_config):
        self.robot_config = robot_config

        # Components
        self.localizer = self.initialize_localizer()
        self.perceptor = self.initialize_perceptor()
        self.path_planner = self.initialize_path_planner()
        self.controller = self.initialize_controller()

        # State
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
        self.goal_pose = None
        self.current_path = []
        self.path_index = 0
        self.navigation_state = "IDLE"  # IDLE, PLANNING, EXECUTING, AVOIDING, STOPPED

    def initialize_localizer(self):
        """Initialize localization component"""
        return LocalizationSystem()

    def initialize_perceptor(self):
        """Initialize perception component"""
        return PerceptionSystem()

    def initialize_path_planner(self):
        """Initialize path planning component"""
        return PathPlanningSystem(self.robot_config)

    def initialize_controller(self):
        """Initialize motion controller"""
        return MotionController(self.robot_config)

    def set_goal(self, goal_pose):
        """Set navigation goal"""
        self.goal_pose = goal_pose
        self.navigation_state = "PLANNING"
        self.plan_path()

    def plan_path(self):
        """Plan path to goal"""
        if self.goal_pose is None:
            return False

        # Get current pose
        current_pos = self.current_pose[:3]
        goal_pos = self.goal_pose[:3]

        # Get obstacle information from perception
        obstacles = self.perceptor.get_obstacles()

        # Plan path
        self.current_path = self.path_planner.plan_path(
            current_pos, goal_pos, obstacles
        )

        if self.current_path:
            self.path_index = 0
            self.navigation_state = "EXECUTING"
            return True
        else:
            self.navigation_state = "STOPPED"
            return False

    def execute_navigation(self):
        """Execute navigation based on current state"""
        if self.navigation_state == "EXECUTING":
            return self.follow_path()
        elif self.navigation_state == "AVOIDING":
            return self.avoid_obstacles()
        else:
            return False

    def follow_path(self):
        """Follow the planned path"""
        if self.path_index >= len(self.current_path):
            # Reached goal
            self.navigation_state = "IDLE"
            return True

        # Get next waypoint
        target = self.current_path[self.path_index]

        # Check if we've reached the current waypoint
        current_pos = self.current_pose[:3]
        dist_to_waypoint = np.linalg.norm(target - current_pos)

        if dist_to_waypoint < 0.1:  # Waypoint tolerance
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                # Reached goal
                self.navigation_state = "IDLE"
                return True
            else:
                # Move to next waypoint
                target = self.current_path[self.path_index]

        # Send command to controller
        success = self.controller.move_to(target, self.current_pose)

        # Check for obstacles during execution
        obstacles = self.perceptor.get_obstacles()
        if self.detect_immediate_obstacles(obstacles):
            self.navigation_state = "AVOIDING"
            return False

        return success

    def detect_immediate_obstacles(self, obstacles):
        """Detect obstacles that require immediate action"""
        current_pos = self.current_pose[:3]

        for obs_pos, obs_size in obstacles:
            dist = np.linalg.norm(obs_pos[:3] - current_pos)
            if dist < 0.5:  # 50cm safety zone
                return True

        return False

    def avoid_obstacles(self):
        """Execute obstacle avoidance"""
        # Get current obstacle information
        obstacles = self.perceptor.get_obstacles()

        # Plan local avoidance maneuver
        avoidance_path = self.path_planner.plan_local_avoidance(
            self.current_pose[:3], obstacles
        )

        if avoidance_path:
            # Execute avoidance maneuver
            for waypoint in avoidance_path:
                success = self.controller.move_to(waypoint, self.current_pose)
                if not success:
                    break

            # Resume original path planning
            self.navigation_state = "PLANNING"
            return True
        else:
            # Cannot avoid, stop
            self.controller.stop()
            self.navigation_state = "STOPPED"
            return False

    def update(self, sensor_data, current_time):
        """Update navigation system with new sensor data"""
        # Update localization
        self.current_pose = self.localizer.update(sensor_data, self.current_pose)

        # Update perception
        self.perceptor.update(sensor_data)

        # Execute navigation
        return self.execute_navigation()

class LocalizationSystem:
    """Localization component for navigation"""

    def __init__(self):
        self.pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def update(self, sensor_data, prev_pose):
        """Update pose estimate based on sensor data"""
        # In a real implementation, this would use sensor fusion
        # (IMU, encoders, vision, etc.) to estimate pose
        return prev_pose  # Simplified - returns previous estimate

class PerceptionSystem:
    """Perception component for navigation"""

    def __init__(self):
        self.obstacles = []

    def update(self, sensor_data):
        """Update obstacle detection"""
        # In a real implementation, this would process sensor data
        # (lidar, cameras, etc.) to detect obstacles
        pass

    def get_obstacles(self):
        """Get current obstacle information"""
        # Return list of (position, size) tuples
        return self.obstacles

class PathPlanningSystem:
    """Path planning component"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.planner = GridPathPlanner(
            grid_resolution=0.1,
            robot_radius=robot_config.get('radius', 0.3)
        )

    def plan_path(self, start, goal, obstacles):
        """Plan path from start to goal avoiding obstacles"""
        # Setup grid with obstacles
        self.planner.setup_grid(20, 20, obstacles)

        # Plan path
        path_2d = self.planner.plan_path(
            (start[0], start[1]),
            (goal[0], goal[1])
        )

        # Convert to 3D
        path_3d = [(x, y, start[2]) for x, y in path_2d]

        return path_3d

    def plan_local_avoidance(self, current_pos, obstacles):
        """Plan local obstacle avoidance maneuver"""
        # Implement local avoidance planning
        # This could use RRT* or other sampling-based methods
        return []

class MotionController:
    """Motion control component"""

    def __init__(self, robot_config):
        self.robot_config = robot_config
        self.balance_controller = self.initialize_balance_controller()

    def initialize_balance_controller(self):
        """Initialize balance controller"""
        return BalanceController(robot_height=self.robot_config.get('height', 0.8))

    def move_to(self, target_pos, current_pose):
        """Move robot toward target position"""
        # In a real implementation, this would generate
        # locomotion commands while maintaining balance
        return True  # Simplified

    def stop(self):
        """Stop robot motion"""
        # In a real implementation, this would safely stop the robot
        pass

class BalanceController:
    """Balance controller for humanoid navigation"""

    def __init__(self, robot_height):
        self.robot_height = robot_height
        self.omega = np.sqrt(9.81 / robot_height)
```

## Hands-on Exercise: Implementing a Navigation System

In this exercise, you'll implement a complete navigation system that integrates path planning with balance control for humanoid robots.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of robotics concepts

### Exercise Steps
1. Implement a grid-based path planner with A* algorithm
2. Create a simple navigation system that integrates planning and control
3. Add obstacle avoidance capabilities
4. Test the system with various scenarios
5. Analyze performance and safety metrics

### Expected Outcome
You should have a working navigation system that can plan paths, avoid obstacles, and maintain balance while navigating.

### Sample Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time

class SimpleNavigationExercise:
    """Hands-on exercise for navigation system implementation"""

    def __init__(self):
        # Environment setup
        self.grid_size = (20, 20)
        self.resolution = 0.5  # meters per cell
        self.robot_radius = 0.5  # meters

        # Robot state
        self.robot_pos = np.array([1.0, 1.0])
        self.robot_yaw = 0.0
        self.goal_pos = np.array([18.0, 18.0])

        # Obstacles (positions and sizes)
        self.obstacles = [
            {'pos': np.array([5.0, 5.0]), 'size': 1.0},
            {'pos': np.array([10.0, 8.0]), 'size': 1.5},
            {'pos': np.array([15.0, 12.0]), 'size': 1.0},
            {'pos': np.array([8.0, 15.0]), 'size': 1.2}
        ]

        # Path planning
        self.path = []
        self.current_waypoint_idx = 0

        # Visualization
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))

    def world_to_grid(self, pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(pos[0] / self.resolution)
        grid_y = int(pos[1] / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.resolution
        world_y = grid_y * self.resolution
        return np.array([world_x, world_y])

    def is_collision(self, pos):
        """Check if position collides with any obstacle"""
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < (obs['size']/2 + self.robot_radius):
                return True
        return False

    def plan_path(self):
        """Plan path using A* algorithm"""
        start_grid = self.world_to_grid(self.robot_pos)
        goal_grid = self.world_to_grid(self.goal_pos)

        # Create grid
        grid = np.zeros(self.grid_size)

        # Mark obstacles in grid
        for obs in self.obstacles:
            obs_grid = self.world_to_grid(obs['pos'])
            obs_cells = int(obs['size'] / self.resolution) + 1

            for dx in range(-obs_cells, obs_cells + 1):
                for dy in range(-obs_cells, obs_cells + 1):
                    x, y = obs_grid[0] + dx, obs_grid[1] + dy
                    if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                        grid[x, y] = 1  # Occupied

        # A* implementation
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(self.grid_to_world(*start_grid))
                path.reverse()
                return path

            # Get neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < self.grid_size[0] and
                    0 <= neighbor[1] < self.grid_size[1]):

                    if grid[neighbor[0], neighbor[1]] == 1:  # Occupied
                        continue

                    # Calculate tentative g_score
                    movement_cost = np.sqrt(dx*dx + dy*dy) if (dx != 0 and dy != 0) else 1
                    tentative_g = g_score[current] + movement_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, a, b):
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def move_robot(self):
        """Move robot along the planned path"""
        if self.current_waypoint_idx >= len(self.path):
            return True  # Reached goal

        target = self.path[self.current_waypoint_idx]
        direction = target - self.robot_pos
        distance = np.linalg.norm(direction)

        if distance < 0.2:  # Close enough to waypoint
            self.current_waypoint_idx += 1
            return False  # Continue moving

        # Move toward target
        if distance > 0:
            velocity = direction / distance * 0.1  # 0.1 m/s
            new_pos = self.robot_pos + velocity

            # Check for collision before moving
            if not self.is_collision(new_pos):
                self.robot_pos = new_pos
                self.robot_yaw = np.arctan2(velocity[1], velocity[0])

        return False  # Haven't reached goal yet

    def visualize(self):
        """Visualize the navigation environment"""
        self.ax.clear()

        # Draw grid
        for i in range(self.grid_size[0] + 1):
            self.ax.axvline(i * self.resolution, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(self.grid_size[1] + 1):
            self.ax.axhline(j * self.resolution, color='gray', linewidth=0.5, alpha=0.3)

        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle(obs['pos'], obs['size']/2, color='red', alpha=0.5)
            self.ax.add_patch(circle)

        # Draw robot
        robot_circle = Circle(self.robot_pos, self.robot_radius, color='blue', alpha=0.7)
        self.ax.add_patch(robot_circle)

        # Draw robot orientation
        arrow_length = 0.8
        arrow_dx = arrow_length * np.cos(self.robot_yaw)
        arrow_dy = arrow_length * np.sin(self.robot_yaw)
        self.ax.arrow(self.robot_pos[0], self.robot_pos[1],
                     arrow_dx, arrow_dy,
                     head_width=0.2, head_length=0.2,
                     fc='black', ec='black')

        # Draw goal
        goal_circle = Circle(self.goal_pos, 0.3, color='green', alpha=0.7)
        self.ax.add_patch(goal_circle)

        # Draw path
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            self.ax.plot(path_x, path_y, 'b--', linewidth=2, alpha=0.7, label='Planned Path')

            # Highlight current waypoint
            if self.current_waypoint_idx < len(self.path):
                wp = self.path[self.current_waypoint_idx]
                self.ax.plot(wp[0], wp[1], 'ro', markersize=8, label='Current Waypoint')

        self.ax.set_xlim(0, self.grid_size[0] * self.resolution)
        self.ax.set_ylim(0, self.grid_size[1] * self.resolution)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_title('Humanoid Robot Navigation Exercise')

        plt.draw()
        plt.pause(0.01)

    def run_simulation(self, max_steps=1000):
        """Run navigation simulation"""
        print("Starting navigation simulation...")
        print(f"Robot position: {self.robot_pos}")
        print(f"Goal position: {self.goal_pos}")

        # Plan initial path
        self.path = self.plan_path()
        self.current_waypoint_idx = 0

        if not self.path:
            print("No path found to goal!")
            return False

        print(f"Planned path with {len(self.path)} waypoints")

        # Simulation loop
        for step in range(max_steps):
            # Visualize current state
            self.visualize()

            # Move robot
            reached_goal = self.move_robot()

            # Check if we need to replan (obstacle appeared or path blocked)
            if step % 50 == 0:  # Replan every 50 steps
                new_path = self.plan_path()
                if len(new_path) > 0:
                    self.path = new_path
                    # Find closest point on new path to current position
                    closest_idx = 0
                    min_dist = float('inf')
                    for i, pos in enumerate(self.path):
                        dist = np.linalg.norm(pos - self.robot_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                    self.current_waypoint_idx = closest_idx

            if reached_goal:
                print(f"Goal reached in {step} steps!")
                self.visualize()
                return True

            time.sleep(0.05)  # Slow down simulation

        print("Max steps reached, goal not achieved")
        return False

# Run the exercise
def run_navigation_exercise():
    """Run the complete navigation exercise"""
    nav_exercise = SimpleNavigationExercise()
    success = nav_exercise.run_simulation()

    if success:
        print("Navigation exercise completed successfully!")
    else:
        print("Navigation exercise did not reach the goal.")

    plt.ioff()
    plt.show()

    return nav_exercise

# Uncomment to run the exercise
# exercise_result = run_navigation_exercise()
```

## Advanced Navigation Techniques

### Multi-Layer Navigation

```python
class MultiLayerNavigation:
    """Multi-layer navigation system with global and local planners"""

    def __init__(self):
        self.global_planner = GridPathPlanner(grid_resolution=1.0, robot_radius=0.5)
        self.local_planner = GridPathPlanner(grid_resolution=0.1, robot_radius=0.3)
        self.controller = MotionController({'height': 0.8})

        self.global_path = []
        self.local_path = []
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.goal_pos = np.array([0.0, 0.0, 0.0])

    def plan_global_path(self, start, goal, large_scale_obstacles):
        """Plan coarse global path"""
        # Use lower resolution for global planning
        self.global_planner.grid_resolution = 1.0
        self.global_planner.setup_grid(100, 100, large_scale_obstacles)

        global_path = self.global_planner.plan_path(
            (start[0], start[1]),
            (goal[0], goal[1])
        )

        return global_path

    def plan_local_path(self, current_pos, local_goals, local_obstacles):
        """Plan detailed local path"""
        # Use higher resolution for local planning
        self.local_planner.grid_resolution = 0.1
        self.local_planner.setup_grid(20, 20, local_obstacles)

        local_path = []
        for goal in local_goals:
            segment = self.local_planner.plan_path(
                (current_pos[0], current_pos[1]),
                (goal[0], goal[1])
            )
            local_path.extend(segment[1:])  # Don't repeat the start point

        return local_path
```

### Human-Aware Navigation

```python
class HumanAwareNavigation:
    """Navigation that considers human safety and comfort"""

    def __init__(self, personal_space_radius=1.0, social_zones=None):
        self.personal_space_radius = personal_space_radius
        self.social_zones = social_zones or {
            'intimate': 0.45,    # 0.45m - very close
            'personal': 1.2,     # 1.2m - comfortable distance
            'social': 3.6,       # 3.6m - social distance
            'public': 7.6        # 7.6m+ - public distance
        }

    def compute_human_cost(self, position, humans):
        """Compute cost based on proximity to humans"""
        total_cost = 0.0

        for human_pos in humans:
            dist = np.linalg.norm(position - human_pos)

            # Use inverse square law for cost (closer = much higher cost)
            if dist < self.social_zones['personal']:
                # Very high cost for personal space invasion
                cost = 1000.0 / (dist + 0.1)  # +0.1 to avoid division by zero
            elif dist < self.social_zones['social']:
                # Medium cost for social zone
                cost = 10.0 / (dist + 0.5)
            else:
                # Low cost for public zone
                cost = 1.0 / (dist + 1.0)

            total_cost += cost

        return total_cost

    def plan_human_aware_path(self, start, goal, obstacles, humans):
        """Plan path considering human positions"""
        # This would be implemented as a modified A* or D* algorithm
        # that incorporates human cost into the path cost function

        # For simplicity, we'll show the concept:
        print(f"Planning path considering {len(humans)} humans nearby")

        # The actual implementation would modify the path planning algorithm
        # to consider human cost when evaluating path segments
        base_path = self.plan_base_path(start, goal, obstacles)

        # Adjust path based on human positions
        human_aware_path = self.adjust_path_for_humans(base_path, humans)

        return human_aware_path

    def adjust_path_for_humans(self, path, humans):
        """Adjust path to respect human personal space"""
        adjusted_path = []

        for point in path:
            # Check proximity to humans
            closest_human_dist = float('inf')
            for human_pos in humans:
                dist = np.linalg.norm(np.array(point)[:2] - human_pos[:2])
                closest_human_dist = min(closest_human_dist, dist)

            # If too close to humans, adjust the point
            if closest_human_dist < self.social_zones['personal']:
                # Find a safer point by moving away from humans
                safe_point = self.find_safe_point(point, humans)
                adjusted_path.append(safe_point)
            else:
                adjusted_path.append(point)

        return adjusted_path

    def find_safe_point(self, original_point, humans):
        """Find a nearby point that respects human space"""
        original_point = np.array(original_point)

        # Try different directions away from humans
        for angle in np.linspace(0, 2*np.pi, 16):  # 16 directions
            for dist in [0.2, 0.5, 1.0, 1.5]:  # Different distances
                candidate = np.array([
                    original_point[0] + dist * np.cos(angle),
                    original_point[1] + dist * np.sin(angle),
                    original_point[2]  # Keep same height
                ])

                # Check if this point is safe from all humans
                safe = True
                for human_pos in humans:
                    if np.linalg.norm(candidate[:2] - human_pos[:2]) < self.social_zones['personal']:
                        safe = False
                        break

                if safe:
                    return tuple(candidate)

        # If no safe point found, return original (shouldn't happen in practice)
        return tuple(original_point)
```

## Summary

This chapter covered path planning and navigation for humanoid robots, from fundamental algorithms like A* and RRT to advanced concepts like human-aware navigation and multi-layer planning systems.

We explored grid-based planning for discrete environments, configuration space planning for the robot's full kinematic state, and balance-constrained planning that considers the robot's stability requirements.

The chapter also covered dynamic path planning for changing environments, including receding horizon approaches and moving obstacle prediction. We discussed the integration of path planning with perception and control systems to create a complete navigation stack.

The hands-on exercise provided practical experience implementing a navigation system that integrates path planning with obstacle avoidance, demonstrating the complexity of real-world navigation for humanoid robots.

## Key Takeaways

- Path planning algorithms (A*, RRT) form the foundation of navigation
- Humanoid robots require configuration space planning considering full kinematics
- Balance constraints significantly affect feasible paths for humanoid robots
- Dynamic environments require receding horizon and predictive planning
- Human-aware navigation is essential for safe human-robot interaction
- Multi-layer navigation (global + local) provides both efficiency and accuracy
- Real-time performance is critical for reactive navigation
- Integration with perception and control systems is essential

## Next Steps

In the next chapter, we'll explore manipulation and grasping techniques for humanoid robots, building on the navigation and path planning foundations established here. We'll cover how humanoid robots can interact with objects in their environment.

## References and Further Reading

1. LaValle, S. M. (2006). Planning Algorithms. Cambridge University Press.
2. Choset, H., et al. (2005). Principles of Robot Motion: Theory, Algorithms, and Implementations. MIT Press.
3. Kuffner, J. J., & LaValle, S. M. (2000). RRT-connect: An efficient approach to single-query path planning. IEEE International Conference on Robotics and Automation.
4. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics.
5. Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance. IEEE Robotics & Automation Magazine.