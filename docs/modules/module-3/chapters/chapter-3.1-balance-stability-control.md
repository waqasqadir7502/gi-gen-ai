# Chapter 3.1: Balance and Stability Control

## Overview

Balance and stability control is fundamental to humanoid robotics, enabling robots to maintain equilibrium while performing complex tasks. This chapter covers the principles of balance control, including the Zero Moment Point (ZMP) theory, center of mass control, and practical implementation of balance algorithms.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the principles of static and dynamic balance in humanoid robots
2. Implement Zero Moment Point (ZMP) control algorithms
3. Design center of mass (CoM) control strategies for stability
4. Apply balance control techniques during dynamic movements
5. Implement disturbance recovery algorithms for balance maintenance

## Introduction to Balance Control

Balance control in humanoid robots is the ability to maintain stability while standing, walking, or performing dynamic tasks. Unlike wheeled or stationary robots, humanoid robots must constantly manage their center of mass (CoM) relative to their support base to prevent falling.

### Why Balance Control Matters

Balance control is essential for:
- **Locomotion**: Stable walking and running patterns
- **Manipulation**: Maintaining stability while interacting with objects
- **Human Interaction**: Safe and predictable behavior around people
- **Dynamic Tasks**: Performing acrobatic or athletic movements

### Types of Balance Control

#### Static Balance
- CoM projection remains within the support polygon at all times
- No net moments acting on the robot
- Generally safer but less efficient

#### Dynamic Balance
- CoM may temporarily move outside support polygon
- Uses dynamic effects (momentum, angular momentum) for stability
- More efficient but harder to control

## Zero Moment Point (ZMP) Theory

### ZMP Definition and Importance

The Zero Moment Point (ZMP) is a critical concept in humanoid balance control. It represents the point on the ground where the sum of all moments (torques) due to ground reaction forces equals zero in the horizontal plane.

Mathematically, for a robot in the x-y plane:
```
ZMP_x = x_grf - (h * F_y) / F_z
ZMP_y = y_grf - (h * F_x) / F_z
```

Where:
- (x_grf, y_grf) is the point of ground reaction force application
- h is the height of the center of mass above ground
- (F_x, F_y, F_z) are the components of ground reaction force

### ZMP Stability Criterion

For a humanoid robot to maintain balance, the ZMP must lie within the support polygon (convex hull of ground contact points). This is the fundamental stability criterion in humanoid robotics.

### ZMP Calculation Methods

#### Method 1: Direct Calculation from Forces
```python
def calculate_zmp_forces(grf_x, grf_y, grf_z, cop_x, cop_y, com_height):
    """
    Calculate ZMP from ground reaction forces

    Args:
        grf_x, grf_y, grf_z: Ground reaction force components
        cop_x, cop_y: Center of pressure coordinates
        com_height: Center of mass height above ground

    Returns:
        zmp_x, zmp_y: Zero Moment Point coordinates
    """
    if grf_z == 0:
        return cop_x, cop_y

    zmp_x = cop_x - (com_height * grf_y) / grf_z
    zmp_y = cop_y + (com_height * grf_x) / grf_z

    return zmp_x, zmp_y
```

#### Method 2: CoM-based Calculation
```python
def calculate_zmp_com(com_pos, com_acc, gravity=9.81):
    """
    Calculate ZMP from CoM position and acceleration

    Args:
        com_pos: Center of mass position [x, y, z]
        com_acc: Center of mass acceleration [x, y, z]
        gravity: Gravitational acceleration

    Returns:
        zmp_x, zmp_y: Zero Moment Point coordinates
    """
    zmp_x = com_pos[0] - (com_pos[2] * com_acc[0]) / (gravity + com_acc[2])
    zmp_y = com_pos[1] - (com_pos[2] * com_acc[1]) / (gravity + com_acc[2])

    return zmp_x, zmp_y
```

## Center of Mass (CoM) Control

### CoM Planning and Control

The Center of Mass (CoM) trajectory planning is crucial for balance control. The CoM must be positioned appropriately relative to the support base to maintain stability.

#### Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model simplifies the balance control problem by assuming:
- Constant CoM height
- No angular momentum about the CoM
- Point mass model

The LIPM equation is:
```
ẍ_com = ω²(x_com - x_zmp)
```

Where ω = √(g/h) is the natural frequency of the pendulum.

```python
import numpy as np

class LinearInvertedPendulum:
    """Linear Inverted Pendulum Model for balance control"""

    def __init__(self, robot_height, gravity=9.81):
        self.height = robot_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / robot_height)

    def compute_com_acceleration(self, com_pos, zmp_pos):
        """Compute CoM acceleration based on LIPM"""
        return self.omega**2 * (com_pos - zmp_pos)

    def compute_zmp_from_com(self, com_pos, com_acc):
        """Compute ZMP from CoM position and acceleration"""
        return com_pos - com_acc / (self.omega**2)

    def generate_trajectory(self, initial_com, final_zmp, duration, dt=0.01):
        """Generate CoM trajectory following ZMP reference"""
        steps = int(duration / dt)
        t = np.linspace(0, duration, steps)

        # For LIPM, CoM trajectory is exponential toward ZMP
        com_x = np.zeros(steps)
        com_y = np.zeros(steps)

        com_x[0] = initial_com[0]
        com_y[0] = initial_com[1]

        for i in range(1, steps):
            # Simple first-order approximation
            dx_dt = self.omega**2 * (com_x[i-1] - final_zmp[0]) * dt
            dy_dt = self.omega**2 * (com_y[i-1] - final_zmp[1]) * dt

            com_x[i] = com_x[i-1] + dx_dt * dt
            com_y[i] = com_y[i-1] + dy_dt * dt

        return t, com_x, com_y
```

### CoM Control Strategies

#### Feedforward Control
- Predicts required CoM position based on planned movements
- Anticipates disturbances from planned actions
- Requires accurate models and planning

#### Feedback Control
- Corrects CoM position based on measured errors
- Compensates for modeling errors and disturbances
- Uses ZMP error as feedback signal

```python
class ComController:
    """Center of Mass controller with feedforward and feedback"""

    def __init__(self, kp=10.0, ki=1.0, kd=2.0, dt=0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step

        # Error accumulation for integral term
        self.error_integral_x = 0
        self.error_integral_y = 0
        self.prev_error_x = 0
        self.prev_error_y = 0

    def update(self, com_current, zmp_desired, zmp_current):
        """Update controller with current state"""
        # Calculate ZMP error (difference between desired and actual)
        zmp_error_x = zmp_desired[0] - zmp_current[0]
        zmp_error_y = zmp_desired[1] - zmp_current[1]

        # Proportional term
        p_term_x = self.kp * zmp_error_x
        p_term_y = self.kp * zmp_error_y

        # Integral term (accumulate error over time)
        self.error_integral_x += zmp_error_x * self.dt
        self.error_integral_y += zmp_error_y * self.dt

        i_term_x = self.ki * self.error_integral_x
        i_term_y = self.ki * self.error_integral_y

        # Derivative term (rate of change of error)
        derivative_x = (zmp_error_x - self.prev_error_x) / self.dt
        derivative_y = (zmp_error_y - self.prev_error_y) / self.dt

        d_term_x = self.kd * derivative_x
        d_term_y = self.kd * derivative_y

        # Store current error for next iteration
        self.prev_error_x = zmp_error_x
        self.prev_error_y = zmp_error_y

        # Calculate control output
        control_x = p_term_x + i_term_x + d_term_x
        control_y = p_term_y + i_term_y + d_term_y

        return control_x, control_y
```

## Balance Control Algorithms

### Capture Point Theory

The capture point is the location where a robot should place its foot to come to a complete stop. It's calculated as:

```
Capture_Point = CoM_Position + CoM_Velocity / ω
```

```python
def compute_capture_point(com_pos, com_vel, omega):
    """
    Compute capture point for stopping

    Args:
        com_pos: Current CoM position [x, y]
        com_vel: Current CoM velocity [vx, vy]
        omega: Natural frequency of the inverted pendulum

    Returns:
        capture_point: [cx, cy] coordinates where to place foot to stop
    """
    capture_x = com_pos[0] + com_vel[0] / omega
    capture_y = com_pos[1] + com_vel[1] / omega

    return np.array([capture_x, capture_y])

def is_stoppable(com_pos, com_vel, support_polygon, omega):
    """
    Check if robot can come to stop without falling

    Args:
        com_pos: Current CoM position
        com_vel: Current CoM velocity
        support_polygon: Current support polygon vertices
        omega: Natural frequency

    Returns:
        bool: True if robot can stop, False otherwise
    """
    capture_point = compute_capture_point(com_pos, com_vel, omega)

    # Check if capture point is inside support polygon
    return point_in_polygon(capture_point, support_polygon)

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
```

### Balance Control Strategies

#### Ankle Strategy
- **Purpose**: Small balance adjustments using ankle torques
- **Range**: ±2-3 cm of CoM movement
- **Speed**: Fast response (10-50 ms)
- **Application**: Small disturbances and fine adjustments

#### Hip Strategy
- **Purpose**: Medium-range balance adjustments using hip torques
- **Range**: ±5-10 cm of CoM movement
- **Speed**: Moderate response (50-200 ms)
- **Application**: Medium disturbances

#### Stepping Strategy
- **Purpose**: Large balance recovery by taking a step
- **Range**: Unlimited (by changing support base)
- **Speed**: Slower (200-500 ms)
- **Application**: Large disturbances

```python
class BalanceController:
    """Comprehensive balance controller with multiple strategies"""

    def __init__(self, robot_height=0.8, control_freq=200):
        self.lip_model = LinearInvertedPendulum(robot_height)
        self.com_controller = ComController()
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Strategy thresholds
        self.ankle_threshold = 0.03  # 3 cm
        self.hip_threshold = 0.10    # 10 cm

        # Robot state
        self.com_pos = np.array([0.0, 0.0, robot_height])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.zmp_pos = np.array([0.0, 0.0])
        self.support_polygon = []

    def update_state(self, com_pos, com_vel, zmp_pos, support_polygon):
        """Update controller with current robot state"""
        self.com_pos = com_pos
        self.com_vel = com_vel
        self.zmp_pos = zmp_pos
        self.support_polygon = support_polygon

    def select_strategy(self, zmp_error_norm):
        """Select appropriate balance strategy based on error magnitude"""
        if zmp_error_norm < self.ankle_threshold:
            return "ankle"
        elif zmp_error_norm < self.hip_threshold:
            return "hip"
        else:
            return "stepping"

    def compute_balance_control(self, zmp_ref):
        """Compute balance control output"""
        # Calculate current ZMP error
        zmp_error = zmp_ref - self.zmp_pos
        zmp_error_norm = np.linalg.norm(zmp_error)

        # Select strategy based on error magnitude
        strategy = self.select_strategy(zmp_error_norm)

        if strategy == "ankle":
            # Ankle strategy: Use ankle joint torques
            control_output = self.ankle_control(zmp_error)
        elif strategy == "hip":
            # Hip strategy: Use hip joint adjustments
            control_output = self.hip_control(zmp_error)
        else:
            # Stepping strategy: Plan a recovery step
            control_output = self.stepping_strategy(zmp_error)

        return control_output, strategy

    def ankle_control(self, zmp_error):
        """Ankle strategy control"""
        # Simple proportional control on ankle joints
        ankle_torques = 100 * zmp_error  # Simplified - real implementation would use joint angles
        return ankle_torques

    def hip_control(self, zmp_error):
        """Hip strategy control"""
        # Use hip joint adjustments to shift CoM
        hip_adjustment = 50 * zmp_error  # Simplified control law
        return hip_adjustment

    def stepping_strategy(self, zmp_error):
        """Stepping strategy - plan recovery step"""
        # Compute capture point
        capture_pt = compute_capture_point(self.com_pos[:2], self.com_vel[:2],
                                          self.lip_model.omega)

        # Plan step to capture point
        step_target = capture_pt

        # Calculate required step timing and location
        step_info = {
            'position': step_target,
            'timing': 0.2,  # 200ms to execute step
            'height': 0.05  # 5cm step height
        }

        return step_info
```

## Practical Implementation Examples

### ZMP-Based Walking Control

```python
class ZMPWalkingController:
    """ZMP-based walking controller for humanoid robots"""

    def __init__(self, robot_height=0.8, step_length=0.3, step_width=0.2):
        self.robot_height = robot_height
        self.step_length = step_length
        self.step_width = step_width
        self.omega = np.sqrt(9.81 / robot_height)

        # Walking state
        self.current_support_foot = "left"  # Start with left support
        self.step_phase = 0.0  # 0.0 to 1.0, representing step cycle
        self.cycle_time = 1.0  # 1 second per step

        # ZMP reference trajectory
        self.zmp_reference = np.array([0.0, 0.0])

    def generate_zmp_trajectory(self, step_count=10):
        """Generate ZMP reference trajectory for walking"""
        dt = 0.01  # 100Hz control
        total_time = step_count * self.cycle_time
        steps = int(total_time / dt)

        # Initialize trajectory arrays
        t = np.linspace(0, total_time, steps)
        zmp_x = np.zeros(steps)
        zmp_y = np.zeros(steps)

        # Generate walking pattern with alternating support
        for step_idx in range(step_count):
            step_start = step_idx * self.cycle_time
            step_end = (step_idx + 1) * self.cycle_time

            # Determine support foot for this step
            support_side = "right" if step_idx % 2 == 0 else "left"

            # Calculate indices for this step
            start_idx = int(step_start / dt)
            end_idx = min(int(step_end / dt), steps)

            # Set ZMP to be under support foot
            support_y = -self.step_width/2 if support_side == "left" else self.step_width/2

            for i in range(start_idx, end_idx):
                # ZMP follows the support foot in Y
                zmp_y[i] = support_y

                # X position follows the step pattern
                phase = (t[i] - step_start) / self.cycle_time
                zmp_x[i] = step_idx * self.step_length + phase * self.step_length * 0.8  # Slightly less than step for stability

        return t, zmp_x, zmp_y

    def compute_com_trajectory(self, zmp_x, zmp_y):
        """Compute CoM trajectory from ZMP reference using LIPM"""
        dt = 0.01
        steps = len(zmp_x)

        # Initialize CoM trajectory
        com_x = np.zeros(steps)
        com_y = np.zeros(steps)

        # Set initial conditions
        com_x[0] = zmp_x[0]  # Start with CoM at ZMP
        com_y[0] = zmp_y[0]

        # Integrate LIPM equation forward
        for i in range(1, steps):
            # LIPM: ẍ = ω²(x - zmp)
            acc_x = self.omega**2 * (com_x[i-1] - zmp_x[i-1])
            acc_y = self.omega**2 * (com_y[i-1] - zmp_y[i-1])

            # Update velocities
            vel_x = (com_x[i-1] - com_x[i-2]) / dt if i > 1 else 0
            vel_y = (com_y[i-1] - com_y[i-2]) / dt if i > 1 else 0

            new_vel_x = vel_x + acc_x * dt
            new_vel_y = vel_y + acc_y * dt

            # Update positions
            com_x[i] = com_x[i-1] + new_vel_x * dt
            com_y[i] = com_y[i-1] + new_vel_y * dt

        return com_x, com_y
```

## Hands-on Exercise: Implementing a Balance Controller

In this exercise, you'll implement a complete balance controller that can respond to disturbances and maintain stability.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of control systems

### Exercise Steps
1. Implement the Linear Inverted Pendulum Model
2. Create a ZMP-based balance controller
3. Test the controller with simulated disturbances
4. Analyze stability margins and performance metrics
5. Implement capture point-based recovery

### Expected Outcome
You should have a working balance controller that can maintain stability under various conditions and recover from disturbances.

### Sample Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_balance_control():
    """Complete balance control simulation"""

    # Initialize controller
    controller = BalanceController(robot_height=0.8)

    # Simulation parameters
    dt = 0.01  # 100 Hz control
    simulation_time = 10.0  # 10 seconds
    steps = int(simulation_time / dt)

    # Initialize state arrays
    time = np.linspace(0, simulation_time, steps)
    com_x = np.zeros(steps)
    com_y = np.zeros(steps)
    zmp_x = np.zeros(steps)
    zmp_y = np.zeros(steps)
    control_output_x = np.zeros(steps)
    control_output_y = np.zeros(steps)
    strategies = []

    # Initial conditions
    com_x[0] = 0.0
    com_y[0] = 0.0
    zmp_x[0] = 0.0
    zmp_y[0] = 0.0

    # Create a simple support polygon (rectangle)
    support_polygon = np.array([
        [-0.1, -0.05],  # front-left
        [0.1, -0.05],   # front-right
        [0.1, 0.05],    # back-right
        [-0.1, 0.05]    # back-left
    ])

    # Reference ZMP trajectory (with occasional disturbances)
    zmp_ref_x = np.zeros(steps)
    zmp_ref_y = np.zeros(steps)

    # Add some reference trajectory and disturbances
    for i in range(steps):
        # Normal walking pattern
        zmp_ref_x[i] = 0.0  # Stay in middle
        zmp_ref_y[i] = 0.0

        # Add disturbances at specific times
        if 2.0 < time[i] < 2.1:  # Push in X direction
            zmp_ref_x[i] = 0.05
        elif 5.0 < time[i] < 5.1:  # Push in Y direction
            zmp_ref_y[i] = 0.03
        elif 7.0 < time[i] < 7.1:  # Push in opposite Y direction
            zmp_ref_y[i] = -0.04

    # Simulation loop
    for i in range(1, steps):
        # Update controller state (simplified)
        current_com = np.array([com_x[i-1], com_y[i-1], 0.8])

        # Approximate velocity (in real implementation, this would come from sensors)
        if i > 1:
            current_vel = np.array([(com_x[i-1] - com_x[i-2])/dt,
                                   (com_y[i-1] - com_y[i-2])/dt,
                                   0.0])
        else:
            current_vel = np.array([0.0, 0.0, 0.0])

        current_zmp = np.array([zmp_x[i-1], zmp_y[i-1]])

        # Update controller
        controller.update_state(current_com, current_vel, current_zmp, support_polygon)

        # Compute balance control
        control_out, strategy = controller.compute_balance_control(
            np.array([zmp_ref_x[i], zmp_ref_y[i]])
        )

        if isinstance(control_out, np.ndarray):
            control_output_x[i] = control_out[0]
            control_output_y[i] = control_out[1]
        else:
            # If it's a stepping strategy, handle differently
            control_output_x[i] = control_out.get('position', [0, 0])[0] if isinstance(control_out, dict) else 0
            control_output_y[i] = control_out.get('position', [0, 0])[1] if isinstance(control_out, dict) else 0

        strategies.append(strategy)

        # Update CoM based on control (simplified dynamics)
        # In real implementation, this would use actual robot dynamics
        zmp_error_x = zmp_ref_x[i] - current_zmp[0]
        zmp_error_y = zmp_ref_y[i] - current_zmp[1]

        # Simple model: CoM follows ZMP reference with some delay and dynamics
        com_acc_x = controller.lip_model.omega**2 * (zmp_ref_x[i] - com_x[i-1])
        com_acc_y = controller.lip_model.omega**2 * (zmp_ref_y[i] - com_y[i-1])

        # Update velocities
        if i > 1:
            com_vel_x = (com_x[i-1] - com_x[i-2]) / dt
            com_vel_y = (com_y[i-1] - com_y[i-2]) / dt
        else:
            com_vel_x = 0.0
            com_vel_y = 0.0

        new_vel_x = com_vel_x + com_acc_x * dt
        new_vel_y = com_vel_y + com_acc_y * dt

        # Update positions
        com_x[i] = com_x[i-1] + new_vel_x * dt
        com_y[i] = com_y[i-1] + new_vel_y * dt

        # ZMP follows CoM with some filtering (simplified)
        zmp_x[i] = zmp_x[i-1] + 0.1 * (com_x[i] - zmp_x[i-1])
        zmp_y[i] = zmp_y[i-1] + 0.1 * (com_y[i] - zmp_y[i-1])

    # Plot results
    plt.figure(figsize=(15, 12))

    # Plot 1: X trajectories
    plt.subplot(2, 3, 1)
    plt.plot(time, com_x, 'b-', label='CoM X', linewidth=2)
    plt.plot(time, zmp_x, 'r--', label='ZMP X', linewidth=2)
    plt.plot(time, zmp_ref_x, 'g:', label='Ref ZMP X', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('X-Axis Balance Control')
    plt.legend()
    plt.grid(True)

    # Plot 2: Y trajectories
    plt.subplot(2, 3, 2)
    plt.plot(time, com_y, 'b-', label='CoM Y', linewidth=2)
    plt.plot(time, zmp_y, 'r--', label='ZMP Y', linewidth=2)
    plt.plot(time, zmp_ref_y, 'g:', label='Ref ZMP Y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Y-Axis Balance Control')
    plt.legend()
    plt.grid(True)

    # Plot 3: Phase portrait (CoM vs ZMP)
    plt.subplot(2, 3, 3)
    plt.plot(com_x, com_y, 'b-', label='CoM Path', linewidth=2)
    plt.scatter(zmp_x[::50], zmp_y[::50], c='red', s=30, label='ZMP Samples', alpha=0.7)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM vs ZMP Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # Plot 4: Control effort
    plt.subplot(2, 3, 4)
    plt.plot(time, control_output_x, 'g-', label='Control X', linewidth=2)
    plt.plot(time, control_output_y, 'm-', label='Control Y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Output')
    plt.title('Balance Control Effort')
    plt.legend()
    plt.grid(True)

    # Plot 5: Stability margin (distance from support polygon)
    plt.subplot(2, 3, 5)
    # Calculate distance to boundary of support polygon
    # For simplicity, assume rectangular support polygon
    support_margin = np.zeros(steps)
    for i in range(steps):
        # Distance to support polygon boundary (simplified)
        dist_x = max(0, abs(com_x[i]) - 0.1)  # Assuming 10cm support width
        dist_y = max(0, abs(com_y[i]) - 0.05)  # Assuming 5cm support depth
        support_margin[i] = np.sqrt(dist_x**2 + dist_y**2)

    plt.plot(time, support_margin, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Support Margin (m)')
    plt.title('Distance to Support Boundary')
    plt.grid(True)

    # Plot 6: Strategy selection
    plt.subplot(2, 3, 6)
    strategy_nums = [0 if s == 'ankle' else (1 if s == 'hip' else 2) for s in strategies] + [strategy_nums[-1]]  # Extend for time array
    if len(strategy_nums) < len(time):
        strategy_nums.extend([strategy_nums[-1]] * (len(time) - len(strategy_nums)))
    elif len(strategy_nums) > len(time):
        strategy_nums = strategy_nums[:len(time)]

    plt.plot(time, strategy_nums[:len(time)], 'b-', drawstyle='steps-post', linewidth=2)
    plt.yticks([0, 1, 2], ['Ankle', 'Hip', 'Stepping'])
    plt.xlabel('Time (s)')
    plt.ylabel('Strategy')
    plt.title('Balance Strategy Selection')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate performance metrics
    zmp_error = np.sqrt((com_x - zmp_x)**2 + (com_y - zmp_y)**2)
    avg_error = np.mean(zmp_error)
    max_error = np.max(zmp_error)

    print(f"Balance Control Performance:")
    print(f"  Average ZMP tracking error: {avg_error:.4f} m")
    print(f"  Maximum ZMP tracking error: {max_error:.4f} m")
    print(f"  Final CoM position: ({com_x[-1]:.3f}, {com_y[-1]:.3f}) m")
    print(f"  Stability maintained: {np.mean(support_margin[:-1] < 0.02):.1%} of time (within 2cm)")

    return {
        'time': time,
        'com_x': com_x,
        'com_y': com_y,
        'zmp_x': zmp_x,
        'zmp_y': zmp_y,
        'zmp_ref_x': zmp_ref_x,
        'zmp_ref_y': zmp_ref_y,
        'errors': zmp_error
    }

# Run the simulation
balance_results = simulate_balance_control()
```

## Advanced Balance Control Techniques

### Model Predictive Control (MPC) for Balance

Model Predictive Control can optimize balance by predicting future states and optimizing control inputs over a finite horizon.

```python
class MPCBalanceController:
    """Model Predictive Control for humanoid balance"""

    def __init__(self, prediction_horizon=10, control_horizon=5, dt=0.01):
        self.prediction_horizon = prediction_horizon  # N steps ahead
        self.control_horizon = control_horizon        # M control moves
        self.dt = dt

        # Cost function weights
        self.Q = np.eye(2) * 10  # State cost (CoM tracking)
        self.R = np.eye(2) * 1   # Control effort cost
        self.P = np.eye(2) * 20  # Terminal cost

    def setup_prediction_matrices(self, omega):
        """Set up prediction matrices for LIPM"""
        # For LIPM: x[k+1] = A*x[k] + B*u[k]
        # where x = [com_x, com_y]^T, u = [zmp_x, zmp_y]^T
        # Continuous: ẍ = ω²(x - zmp) => ẍ = ω²x - ω²zmp
        # Discrete approximation: x[k+1] = x[k] + ẍ*dt*dt = x[k] + ω²(x-zmp)*dt*dt

        dt = self.dt
        A = np.array([
            [1 + omega**2 * dt**2, 0],
            [0, 1 + omega**2 * dt**2]
        ])

        B = np.array([
            [-omega**2 * dt**2, 0],
            [0, -omega**2 * dt**2]
        ])

        return A, B

    def predict_trajectory(self, x0, U, A, B):
        """Predict state trajectory given initial state and control sequence"""
        N = self.prediction_horizon
        X = np.zeros((N+1, 2))  # State trajectory
        X[0] = x0

        for k in range(N):
            # Apply control (last control is held constant)
            u_k = U[min(k, len(U)-1)] if len(U) > 0 else np.zeros(2)
            X[k+1] = A @ X[k] + B @ u_k

        return X
```

### Adaptive Balance Control

Adaptive control adjusts parameters based on changing conditions or robot state.

```python
class AdaptiveBalanceController:
    """Adaptive balance controller that adjusts parameters based on conditions"""

    def __init__(self, initial_params=None):
        # Initialize with default parameters
        self.params = {
            'kp': 10.0,      # Proportional gain
            'ki': 1.0,       # Integral gain
            'kd': 2.0,       # Derivative gain
            'omega': 3.5,    # Natural frequency (sqrt(g/h))
            'adaptation_rate': 0.01
        }

        if initial_params:
            self.params.update(initial_params)

    def update_parameters(self, error_history, stability_metrics):
        """Adapt control parameters based on performance"""
        # Calculate performance indicators
        recent_error = np.mean(np.abs(error_history[-10:])) if len(error_history) >= 10 else 0
        error_trend = (error_history[-1] - error_history[-10]) / 10 if len(error_history) >= 10 else 0

        # Adapt proportional gain based on error magnitude
        if recent_error > 0.05:  # High error
            self.params['kp'] *= (1 + self.params['adaptation_rate'])
        elif recent_error < 0.01:  # Low error, possibly oscillating
            self.params['kp'] *= (1 - self.params['adaptation_rate'])

        # Constrain gains to reasonable bounds
        self.params['kp'] = np.clip(self.params['kp'], 1.0, 50.0)
        self.params['ki'] = np.clip(self.params['ki'], 0.1, 10.0)
        self.params['kd'] = np.clip(self.params['kd'], 0.1, 10.0)

        return self.params
```

## Summary

This chapter covered the essential concepts of balance and stability control in humanoid robots. We explored the Zero Moment Point (ZMP) theory, which is fundamental to humanoid balance, and implemented practical control algorithms including the Linear Inverted Pendulum Model.

We implemented various balance control strategies including ankle, hip, and stepping strategies, each appropriate for different magnitudes of disturbances. The hands-on exercise provided practical experience with implementing and testing a complete balance control system.

We also covered advanced topics like Model Predictive Control and adaptive control techniques that can improve balance performance in complex scenarios.

Balance control remains one of the most challenging aspects of humanoid robotics, requiring sophisticated algorithms to maintain stability while performing complex tasks.

## Key Takeaways

- ZMP theory is fundamental to humanoid balance control
- Multiple strategies (ankle, hip, stepping) are needed for different disturbance magnitudes
- CoM control is essential for maintaining stability
- Feedback control helps compensate for disturbances and modeling errors
- Capture point theory enables recovery from large disturbances
- Advanced techniques like MPC can optimize balance performance
- Real-time performance is critical for stable balance

## Next Steps

In the next chapter, we'll explore path planning and navigation for humanoid robots, building on the balance and stability foundations established here. We'll cover algorithms for navigating complex environments while maintaining stability.

## References and Further Reading

1. Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K., & Hirukawa, H. (2003). Biped walking pattern generation by using preview control of zero-moment point. IEEE International Conference on Robotics and Automation.
2. Pratt, J., Carff, J., Drakunov, S., & Goswami, A. (2006). Capture point: A step toward humanoid push recovery. IEEE-RAS International Conference on Humanoid Robots.
3. Takenaka, T., Matsumoto, T., & Yoshiike, T. (2009). Real time motion generation and control for biped robot. IEEE-RAS International Conference on Humanoid Robots.
4. Shafii, N., Schreiber, M., Reiter, A., & Kirchner, F. (2014). A stable tracking controller for a humanoid robot using the 3D linear inverted pendulum mode. IEEE-RAS International Conference on Humanoid Robots.