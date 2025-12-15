# Chapter 2.4: Basic Locomotion Patterns

## Overview

Locomotion is one of the most challenging aspects of humanoid robotics, requiring the coordination of multiple systems including kinematics, perception, actuators, and control. This chapter covers fundamental locomotion patterns, from static balance to dynamic walking, and explores how humanoid robots achieve stable movement through various terrains and conditions.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the principles of static and dynamic balance in humanoid robots
2. Implement basic walking patterns using ZMP (Zero Moment Point) theory
3. Design stable locomotion controllers for humanoid robots
4. Apply balance control techniques to maintain stability during movement
5. Simulate basic locomotion patterns in a physics environment

## Introduction to Humanoid Locomotion

Humanoid locomotion involves the complex coordination of multiple body segments to achieve controlled movement from one location to another. Unlike wheeled or tracked robots, humanoid robots must manage their center of mass (CoM) and maintain balance while moving, making locomotion particularly challenging.

### Challenges in Humanoid Locomotion

- **Balance Maintenance**: Keeping the robot upright during movement
- **Dynamic Stability**: Managing forces during walking phases
- **Contact Transitions**: Smoothly transitioning between foot contacts
- **Terrain Adaptation**: Handling uneven surfaces and obstacles
- **Energy Efficiency**: Minimizing power consumption during movement
- **Real-time Control**: Responding quickly to disturbances

### Classification of Locomotion

#### Static vs. Dynamic Locomotion
- **Static Locomotion**: CoM remains within support polygon at all times
- **Dynamic Locomotion**: CoM may move outside support polygon temporarily

#### Periodic vs. Non-periodic
- **Periodic**: Repetitive gait patterns (walking, running)
- **Non-periodic**: Irregular movement patterns (climbing, crawling)

## Static Balance and Stability

### Support Polygon

The support polygon is the convex hull of all contact points with the ground. For static balance, the projection of the center of mass (CoM) must remain within this polygon.

#### Single Support Phase
When only one foot is in contact with the ground, the support polygon is the area of that foot.

#### Double Support Phase
When both feet are in contact, the support polygon is the quadrilateral connecting both feet.

### Center of Mass (CoM) and Center of Pressure (CoP)

- **CoM**: The point where the total mass of the robot can be considered to be concentrated
- **CoP**: The point where the ground reaction force acts

For balance, CoM projection must be within the support polygon, and CoP must be controlled to maintain balance.

### Static Balance Strategies

#### Ankle Strategy
- **Mechanism**: Adjust ankle torques to move CoM
- **Range**: Small adjustments, energy efficient
- **Speed**: Fast response to small disturbances

#### Hip Strategy
- **Mechanism**: Move hip joints to adjust CoM
- **Range**: Larger adjustments than ankle strategy
- **Speed**: Slower than ankle strategy

#### Stepping Strategy
- **Mechanism**: Take a step to expand support polygon
- **Range**: Largest range of all strategies
- **Speed**: Slowest response, requires planning

## Zero Moment Point (ZMP) Theory

The Zero Moment Point is a critical concept in dynamic balance and walking control for humanoid robots.

### ZMP Definition

ZMP is the point on the ground where the moment of the ground reaction force is zero in the horizontal plane. For stable walking, the ZMP must remain within the support polygon.

### ZMP Calculation

For a robot in the x-y plane with ground contact at (x_g, y_g):

```
ZMP_x = x_g - (I_y / F_z)
ZMP_y = y_g - (I_x / F_z)
```

Where:
- I_x, I_y: Angular momentum about x and y axes
- F_z: Vertical ground reaction force

### ZMP-Based Walking Pattern Generation

#### Preview Control Method
Uses future ZMP reference trajectory to compute stable CoM trajectory:

```
ẍ_com + 2ζω_n ẋ_com + ω_n² x_com = ω_n² x_zmp
```

Where:
- ζ: Damping ratio
- ω_n: Natural frequency
- x_com: CoM position
- x_zmp: ZMP reference

### Implementing ZMP-Based Walking

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class ZMPLocomotion:
    """ZMP-based walking pattern generator"""

    def __init__(self, robot_height=0.8, step_length=0.3, step_width=0.2,
                 step_height=0.05, control_freq=200):
        self.robot_height = robot_height  # Height of CoM above ground
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # ZMP-based parameters
        self.omega = np.sqrt(9.81 / robot_height)  # Natural frequency

        # Walking state
        self.left_support = True  # Start with left foot support
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

    def generate_zmp_trajectory(self, steps=5):
        """Generate ZMP reference trajectory for walking"""
        # Calculate step timing
        step_time = 1.0  # 1 second per step
        double_support_time = 0.1  # 10% double support
        single_support_time = step_time - double_support_time

        total_time = steps * step_time
        num_points = int(total_time / self.dt)

        # Initialize trajectories
        t = np.linspace(0, total_time, num_points)
        zmp_x = np.zeros(num_points)
        zmp_y = np.zeros(num_points)

        # Generate ZMP trajectory
        for i in range(steps):
            start_idx = int(i * step_time / self.dt)
            end_idx = min(int((i + 1) * step_time / self.dt), num_points)

            if i % 2 == 0:  # Odd steps (right foot moves)
                # Left foot support
                zmp_y[start_idx:end_idx] = -self.step_width / 2
                zmp_x[start_idx:end_idx] = i * self.step_length
            else:  # Even steps (left foot moves)
                # Right foot support
                zmp_y[start_idx:end_idx] = self.step_width / 2
                zmp_x[start_idx:end_idx] = i * self.step_length

        return t, zmp_x, zmp_y

    def compute_com_trajectory(self, zmp_x, zmp_y):
        """Compute CoM trajectory from ZMP reference using 3D Linear Inverted Pendulum Model"""
        # 3D LIPM equations
        # ẍ = ω²(x - x_zmp)
        # ÿ = ω²(y - y_zmp)

        com_x = np.zeros_like(zmp_x)
        com_y = np.zeros_like(zmp_y)
        com_z = np.full_like(zmp_x, self.robot_height)

        # Initialize with first ZMP values
        com_x[0] = zmp_x[0]
        com_y[0] = zmp_y[0]

        # Numerical integration using the LIPM equation
        for i in range(1, len(zmp_x)):
            # Acceleration based on LIPM
            ax = self.omega**2 * (com_x[i-1] - zmp_x[i-1])
            ay = self.omega**2 * (com_y[i-1] - zmp_y[i-1])

            # Update velocities
            vx = self.dt * ax
            vy = self.dt * ay

            # Update positions
            com_x[i] = com_x[i-1] + vx * self.dt
            com_y[i] = com_y[i-1] + vy * self.dt

        return com_x, com_y, com_z

    def generate_foot_trajectory(self, steps=5):
        """Generate foot trajectories for walking"""
        step_time = 1.0
        total_time = steps * step_time
        num_points = int(total_time / self.dt)

        t = np.linspace(0, total_time, num_points)

        # Initialize foot trajectories
        left_foot_x = np.zeros(num_points)
        left_foot_y = np.full(num_points, -self.step_width/2)
        left_foot_z = np.zeros(num_points)

        right_foot_x = np.zeros(num_points)
        right_foot_y = np.full(num_points, self.step_width/2)
        right_foot_z = np.zeros(num_points)

        # Generate foot trajectories
        for i in range(steps):
            step_start = i * step_time
            single_support_start = step_start + 0.05  # Start lifting at 5% of step
            apex_time = 0.5 * step_time  # Foot reaches maximum height at mid-step
            single_support_end = (i + 1) * step_time - 0.05  # Finish placing at 95% of step

            # Determine which foot moves in this step
            if i % 2 == 0:  # Right foot moves
                # Generate trajectory for right foot
                for j in range(num_points):
                    current_t = j * self.dt
                    if step_start <= current_t < single_support_start:
                        # Foot is on ground, stationary
                        right_foot_x[j] = i * self.step_length
                    elif single_support_start <= current_t <= single_support_end:
                        # Foot is swinging
                        swing_phase = (current_t - single_support_start) / (single_support_end - single_support_start)
                        right_foot_x[j] = i * self.step_length + swing_phase * self.step_length

                        # Vertical trajectory (parabolic)
                        if swing_phase < 0.5:
                            right_foot_z[j] = self.step_height * (4 * swing_phase)
                        else:
                            right_foot_z[j] = self.step_height * (4 * (1 - swing_phase))
                    else:
                        # Foot is on ground at new position
                        right_foot_x[j] = (i + 1) * self.step_length
            else:  # Left foot moves
                # Generate trajectory for left foot
                for j in range(num_points):
                    current_t = j * self.dt
                    if step_start <= current_t < single_support_start:
                        # Foot is on ground, stationary
                        left_foot_x[j] = i * self.step_length
                    elif single_support_start <= current_t <= single_support_end:
                        # Foot is swinging
                        swing_phase = (current_t - single_support_start) / (single_support_end - single_support_start)
                        left_foot_x[j] = i * self.step_length + swing_phase * self.step_length

                        # Vertical trajectory (parabolic)
                        if swing_phase < 0.5:
                            left_foot_z[j] = self.step_height * (4 * swing_phase)
                        else:
                            left_foot_z[j] = self.step_height * (4 * (1 - swing_phase))
                    else:
                        # Foot is on ground at new position
                        left_foot_x[j] = (i + 1) * self.step_length

        return t, left_foot_x, left_foot_y, left_foot_z, right_foot_x, right_foot_y, right_foot_z

    def simulate_walking(self, steps=3):
        """Simulate the complete walking pattern"""
        # Generate ZMP trajectory
        t, zmp_x, zmp_y = self.generate_zmp_trajectory(steps)

        # Compute CoM trajectory
        com_x, com_y, com_z = self.compute_com_trajectory(zmp_x, zmp_y)

        # Generate foot trajectories
        ft, lf_x, lf_y, lf_z, rf_x, rf_y, rf_z = self.generate_foot_trajectory(steps)

        return {
            'time': t,
            'zmp_x': zmp_x,
            'zmp_y': zmp_y,
            'com_x': com_x,
            'com_y': com_y,
            'com_z': com_z,
            'left_foot_x': lf_x,
            'left_foot_y': lf_y,
            'left_foot_z': lf_z,
            'right_foot_x': rf_x,
            'right_foot_y': rf_y,
            'right_foot_z': rf_z
        }

def visualize_walking_pattern(walking_data):
    """Visualize the walking pattern"""
    t = walking_data['time']
    com_x = walking_data['com_x']
    com_y = walking_data['com_y']
    zmp_x = walking_data['zmp_x']
    zmp_y = walking_data['zmp_y']

    lf_x = walking_data['left_foot_x']
    lf_y = walking_data['left_foot_y']
    rf_x = walking_data['right_foot_x']
    rf_y = walking_data['right_foot_y']

    plt.figure(figsize=(15, 10))

    # Plot 1: X trajectories
    plt.subplot(2, 3, 1)
    plt.plot(t, com_x, 'b-', label='CoM X', linewidth=2)
    plt.plot(t, zmp_x, 'r--', label='ZMP X', linewidth=2)
    plt.plot(t, lf_x, 'g:', label='Left Foot X', linewidth=2)
    plt.plot(t, rf_x, 'm:', label='Right Foot X', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('X-Axis Trajectories')
    plt.legend()
    plt.grid(True)

    # Plot 2: Y trajectories
    plt.subplot(2, 3, 2)
    plt.plot(t, com_y, 'b-', label='CoM Y', linewidth=2)
    plt.plot(t, zmp_y, 'r--', label='ZMP Y', linewidth=2)
    plt.plot(t, lf_y, 'g:', label='Left Foot Y', linewidth=2)
    plt.plot(t, rf_y, 'm:', label='Right Foot Y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Y-Axis Trajectories')
    plt.legend()
    plt.grid(True)

    # Plot 3: X-Y CoM and ZMP projection
    plt.subplot(2, 3, 3)
    plt.plot(com_x, com_y, 'b-', label='CoM Path', linewidth=2)
    plt.scatter(zmp_x[::20], zmp_y[::20], c='red', s=20, label='ZMP Samples', alpha=0.7)
    plt.scatter(lf_x[::20], lf_y[::20], c='green', s=20, label='Left Foot', alpha=0.5)
    plt.scatter(rf_x[::20], rf_y[::20], c='magenta', s=20, label='Right Foot', alpha=0.5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM and ZMP Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # Plot 4: Foot positions over time
    plt.subplot(2, 3, 4)
    plt.plot(t, lf_y, 'g-', label='Left Foot Y', linewidth=2)
    plt.plot(t, rf_y, 'm-', label='Right Foot Y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Foot Lateral Positions')
    plt.legend()
    plt.grid(True)

    # Plot 5: Support polygon visualization
    plt.subplot(2, 3, 5)
    # Determine support polygon over time
    support_x = []
    support_y = []
    for i in range(0, len(lf_x), 50):  # Sample every 50 points
        # Create approximate support polygon for this time step
        if abs(lf_y[i] - rf_y[i]) > 0.01:  # Both feet on ground
            # Draw line between feet
            plt.plot([lf_x[i], rf_x[i]], [lf_y[i], rf_y[i]], 'k-', alpha=0.3)
        else:  # Single support
            # Draw point for single foot
            if lf_y[i] < rf_y[i]:  # Left foot support
                plt.plot(lf_x[i], lf_y[i], 'go', markersize=8, alpha=0.5)
            else:  # Right foot support
                plt.plot(rf_x[i], rf_y[i], 'mo', markersize=8, alpha=0.5)

    plt.plot(com_x[::50], com_y[::50], 'b-', label='CoM Path', linewidth=2)
    plt.scatter(zmp_x[::50], zmp_y[::50], c='red', s=30, label='ZMP', alpha=0.7)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Support Polygon & Stability')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # Plot 6: ZMP error
    plt.subplot(2, 3, 6)
    zmp_error = np.sqrt((com_x - zmp_x)**2 + (com_y - zmp_y)**2)
    plt.plot(t, zmp_error, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('ZMP Error (m)')
    plt.title('ZMP Tracking Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
def example_walking_simulation():
    """Example of ZMP-based walking simulation"""
    # Create locomotion controller
    locomotion = ZMPLocomotion(robot_height=0.8, step_length=0.3, step_width=0.2)

    # Simulate walking
    walking_data = locomotion.simulate_walking(steps=3)

    # Visualize results
    visualize_walking_pattern(walking_data)

    # Print summary statistics
    zmp_error = np.sqrt((walking_data['com_x'] - walking_data['zmp_x'])**2 +
                        (walking_data['com_y'] - walking_data['zmp_y'])**2)

    print(f"Walking Simulation Results:")
    print(f"  Average ZMP tracking error: {np.mean(zmp_error):.4f} m")
    print(f"  Maximum ZMP tracking error: {np.max(zmp_error):.4f} m")
    print(f"  Final CoM position: ({walking_data['com_x'][-1]:.3f}, {walking_data['com_y'][-1]:.3f}) m")
    print(f"  Total distance traveled: {walking_data['com_x'][-1]:.3f} m")

# Run the example
example_walking_simulation()
```

## Walking Pattern Generation

### Inverted Pendulum Model

The inverted pendulum model is fundamental to understanding bipedal walking:

#### Linear Inverted Pendulum (LIP)
- **Assumption**: CoM height remains constant
- **Equation**: ẍ = ω²(x - x_zmp)
- **Solution**: CoM moves in a curved path around ZMP

#### 3D Linear Inverted Pendulum Model
Extends LIP to three dimensions:
- **X-Z plane**: Forward/backward and up/down movement
- **Y-Z plane**: Lateral movement and up/down
- **Coupling**: CoM height variations affect stability

### Foot Placement Strategy

#### Capture Point (Capture Region)
The capture point is where a robot should place its foot to stop:
```
Capture Point = CoM Position + CoM Velocity / ω
```

#### Foot Placement for Stability
- **In front of CoM**: For forward momentum
- **Behind CoM**: For stopping or backward movement
- **Lateral**: For balance recovery in side-to-side direction

### Gait Parameters

#### Step Timing
- **Double Support**: Both feet on ground (typically 10-20% of step cycle)
- **Single Support**: One foot on ground (remaining time)
- **Step Cycle**: Complete sequence of one step

#### Step Characteristics
- **Step Length**: Forward distance between consecutive foot placements
- **Step Width**: Lateral distance between feet
- **Step Height**: Maximum height of swinging foot
- **Step Frequency**: Steps per unit time

## Balance Control During Locomotion

### Feedback Control Strategies

#### ZMP Feedback Control
```
τ = K_p * (zmp_desired - zmp_actual) + K_d * (żmp_desired - żmp_actual)
```

#### CoM Feedback Control
```
F = K_p * (com_desired - com_actual) + K_d * (v_com_desired - v_com_actual)
```

### Disturbance Recovery

#### Push Recovery Strategies
1. **Ankle Strategy**: Use ankle torques for small disturbances
2. **Hip Strategy**: Use hip torques for medium disturbances
3. **Stepping Strategy**: Take evasive step for large disturbances

#### Recovery Trajectory Generation
- **Immediate response**: Adjust ZMP to counter disturbance
- **Stabilization**: Generate new stable trajectory
- **Return to gait**: Resume normal walking pattern

### Implementing Balance Control

```python
import numpy as np

class BalanceController:
    """Balance controller for humanoid robot"""

    def __init__(self, robot_height=0.8, control_freq=200):
        self.robot_height = robot_height
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.omega = np.sqrt(9.81 / robot_height)

        # Control gains
        self.kp_zmp = 50.0  # ZMP proportional gain
        self.kd_zmp = 10.0  # ZMP derivative gain
        self.kp_com = 100.0  # CoM proportional gain
        self.kd_com = 20.0   # CoM derivative gain

        # State variables
        self.com_pos = np.array([0.0, 0.0, robot_height])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.zmp_pos = np.array([0.0, 0.0])

        # Support polygon (approximate)
        self.support_polygon = {
            'center': np.array([0.0, 0.0]),
            'size_x': 0.15,  # 15cm in x direction
            'size_y': 0.10   # 10cm in y direction
        }

    def update_state(self, com_pos, com_vel, zmp_pos):
        """Update controller with current state"""
        self.com_pos = com_pos
        self.com_vel = com_vel
        self.zmp_pos = zmp_pos

    def compute_balance_control(self, desired_zmp):
        """Compute balance control corrections"""
        # Calculate errors
        zmp_error = desired_zmp - self.zmp_pos
        com_error = desired_zmp - self.com_pos[:2]  # Project CoM to ground

        # ZMP feedback control
        zmp_correction = self.kp_zmp * zmp_error + self.kd_zmp * (-self.com_vel[:2])  # Simplified velocity feedback

        # CoM feedback control
        com_correction = self.kp_com * com_error + self.kd_com * (-self.com_vel[:2])

        # Combine corrections
        total_correction = 0.6 * zmp_correction + 0.4 * com_correction

        # Limit correction to prevent excessive control effort
        max_correction = 0.2  # 20cm maximum adjustment
        total_correction = np.clip(total_correction, -max_correction, max_correction)

        return total_correction

    def check_stability(self):
        """Check if robot is in stable state"""
        # Check if ZMP is within support polygon
        zmp_in_polygon = (
            abs(self.zmp_pos[0] - self.support_polygon['center'][0]) <= self.support_polygon['size_x'] and
            abs(self.zmp_pos[1] - self.support_polygon['center'][1]) <= self.support_polygon['size_y']
        )

        # Check if CoM is reasonably stable
        com_stable = np.linalg.norm(self.com_vel) < 0.5  # Velocity < 0.5 m/s

        return zmp_in_polygon and com_stable

    def compute_capture_point(self):
        """Compute capture point for recovery"""
        # Capture point = CoM + CoM_velocity / omega
        capture_point = self.com_pos[:2] + self.com_vel[:2] / self.omega
        return capture_point

    def generate_recovery_trajectory(self, disturbance_direction):
        """Generate recovery trajectory based on disturbance"""
        # Calculate capture point
        capture_pt = self.compute_capture_point()

        # Determine recovery step location
        # Move in opposite direction of disturbance but toward stability
        recovery_offset = -0.3 * disturbance_direction  # 30cm recovery step
        recovery_pos = capture_pt + recovery_offset

        # Generate smooth trajectory to recovery position
        recovery_trajectory = {
            'step_position': recovery_pos,
            'step_timing': 0.5,  # 0.5 seconds to execute
            'foot_height': 0.05  # 5cm foot lift
        }

        return recovery_trajectory

def simulate_balance_control():
    """Simulate balance control with disturbances"""
    controller = BalanceController(robot_height=0.8)

    # Simulate time steps
    dt = 0.005  # 200Hz control
    simulation_time = 5.0  # 5 seconds
    steps = int(simulation_time / dt)

    # Initialize state arrays
    time = np.linspace(0, simulation_time, steps)
    com_x = np.zeros(steps)
    com_y = np.zeros(steps)
    zmp_x = np.zeros(steps)
    zmp_y = np.zeros(steps)
    balance_corr_x = np.zeros(steps)
    balance_corr_y = np.zeros(steps)
    stability = np.zeros(steps, dtype=bool)

    # Initial conditions
    com_x[0] = 0.0
    com_y[0] = 0.0
    zmp_x[0] = 0.0
    zmp_y[0] = 0.0

    # Simulate with periodic disturbances
    for i in range(1, steps):
        # Simulate robot dynamics (simplified)
        # Apply some forward walking motion
        if 1.0 < time[i] < 4.0:
            desired_zmp = np.array([0.1 * np.sin(2 * np.pi * 0.5 * time[i]), 0.0])
        else:
            desired_zmp = np.array([0.0, 0.0])

        # Add disturbances at specific times
        if 2.0 < time[i] < 2.1:  # Push in X direction
            desired_zmp[0] += 0.1  # Simulated external force
        elif 3.0 < time[i] < 3.1:  # Push in Y direction
            desired_zmp[1] += 0.08  # Simulated external force

        # Update controller state (simplified)
        current_com = np.array([com_x[i-1], com_y[i-1], 0.8])
        current_vel = np.array([(com_x[i-1] - com_x[i-2])/dt if i > 1 else 0,
                               (com_y[i-1] - com_y[i-2])/dt if i > 1 else 0,
                               0])
        current_zmp = np.array([zmp_x[i-1], zmp_y[i-1]])

        controller.update_state(current_com, current_vel, current_zmp)

        # Compute balance corrections
        balance_correction = controller.compute_balance_control(desired_zmp)
        balance_corr_x[i] = balance_correction[0]
        balance_corr_y[i] = balance_correction[1]

        # Apply corrections to ZMP (simplified model)
        corrected_zmp = desired_zmp + balance_correction

        # Update CoM based on ZMP (LIPM model)
        com_acc_x = controller.omega**2 * (com_x[i-1] - corrected_zmp[0])
        com_acc_y = controller.omega**2 * (com_y[i-1] - corrected_zmp[1])

        # Update velocities and positions
        com_vel_x = (com_x[i-1] - com_x[i-2])/dt if i > 1 else 0
        com_vel_y = (com_y[i-1] - com_y[i-2])/dt if i > 1 else 0

        new_vel_x = com_vel_x + com_acc_x * dt
        new_vel_y = com_vel_y + com_acc_y * dt

        com_x[i] = com_x[i-1] + new_vel_x * dt
        com_y[i] = com_y[i-1] + new_vel_y * dt

        # Update ZMP
        zmp_x[i] = corrected_zmp[0]
        zmp_y[i] = corrected_zmp[1]

        # Check stability
        stability[i] = controller.check_stability()

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(time, com_x, 'b-', label='CoM X', linewidth=2)
    plt.plot(time, zmp_x, 'r--', label='ZMP X', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('X-Axis Balance Control')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(time, com_y, 'b-', label='CoM Y', linewidth=2)
    plt.plot(time, zmp_y, 'r--', label='ZMP Y', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Y-Axis Balance Control')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(com_x, com_y, 'b-', label='CoM Path', linewidth=2)
    plt.scatter(zmp_x[::50], zmp_y[::50], c='red', s=30, label='ZMP', alpha=0.7)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM vs ZMP Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(time, balance_corr_x, 'g-', label='X Correction', linewidth=2)
    plt.plot(time, balance_corr_y, 'm-', label='Y Correction', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Correction (m)')
    plt.title('Balance Control Corrections')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(time, stability.astype(int))
    plt.xlabel('Time (s)')
    plt.ylabel('Stable (1/0)')
    plt.title('Stability Status')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(time, np.sqrt((com_x - zmp_x)**2 + (com_y - zmp_y)**2))
    plt.xlabel('Time (s)')
    plt.ylabel('ZMP Error (m)')
    plt.title('ZMP Tracking Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Balance Control Simulation Results:")
    print(f"  Final CoM position: ({com_x[-1]:.3f}, {com_y[-1]:.3f}) m")
    print(f"  Average ZMP error: {np.mean(np.sqrt((com_x - zmp_x)**2 + (com_y - zmp_y)**2)):.4f} m")
    print(f"  Stability maintained: {np.mean(stability):.1%} of the time")

# Run the simulation
simulate_balance_control()
```

## Different Locomotion Patterns

### Walking Gaits

#### Double Support Walking
- **Characteristics**: Both feet on ground during transition
- **Advantages**: More stable, easier to control
- **Disadvantages**: Slower, less human-like
- **Applications**: Early humanoid robots, safety-critical applications

#### Single Support Walking
- **Characteristics**: Alternating single foot support
- **Advantages**: More efficient, human-like
- **Disadvantages**: More complex control, less stable
- **Applications**: Advanced humanoid robots

#### Bipedal Walking Variants
- **Static Walking**: Very slow, CoM always in support polygon
- **Dynamic Walking**: Faster, CoM can be outside support polygon
- **Limit Cycle Walking**: Periodic stable walking pattern

### Alternative Locomotion Patterns

#### Crawling
- **Use Case**: Low clearance environments
- **Control**: Quadrupedal-like movement with arms and legs
- **Stability**: Very high, multiple contact points

#### Climbing
- **Use Case**: Stairs, ladders, uneven terrain
- **Control**: Sequential foot and hand placement
- **Stability**: Requires upper body support

#### Running
- **Use Case**: Faster locomotion
- **Control**: More dynamic, flight phases
- **Stability**: Momentarily airborne, requires precise control

## Simulation and Control Integration

### Physics-Based Simulation

```python
import pybullet as p
import pybullet_data
import numpy as np
import time

def create_simple_walker():
    """Create a simple 2D walker in PyBullet for testing"""
    # Connect to physics server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Create ground plane
    planeId = p.loadURDF("plane.urdf")

    # Create a simple walker (simplified 2D model)
    # This is a conceptual example - in practice, you'd load a more complex humanoid model
    walkerStartPos = [0, 0, 0.8]  # Start above ground
    walkerStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

    # Create a torso
    torso_visual = p.createVisualShape(p.GEOM_CAPSULE, radius=0.1, length=0.6, rgbaColor=[0.8, 0.2, 0.2, 1])
    torso_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.1, length=0.6)
    torso_id = p.createMultiBody(10, torso_collision, torso_visual,
                                 basePosition=walkerStartPos,
                                 baseOrientation=walkerStartOrientation)

    # Create legs (simplified as single links)
    leg_visual = p.createVisualShape(p.GEOM_CAPSULE, radius=0.05, length=0.7, rgbaColor=[0.2, 0.2, 0.8, 1])
    leg_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.05, length=0.7)

    # Create left leg
    left_leg = p.createMultiBody(2, leg_collision, leg_visual,
                                 basePosition=[0, -0.1, 0.45],
                                 baseOrientation=walkerStartOrientation)

    # Create right leg
    right_leg = p.createMultiBody(2, leg_collision, leg_visual,
                                  basePosition=[0, 0.1, 0.45],
                                  baseOrientation=walkerStartOrientation)

    # Connect legs to torso with joints
    # Left hip joint
    p.createConstraint(torso_id, -1, left_leg, -1, p.JOINT_POINT2POINT,
                       [0, 0, 0], [0, 0, 0.35], [0, -0.1, 0.8])

    # Right hip joint
    p.createConstraint(torso_id, -1, right_leg, -1, p.JOINT_POINT2POINT,
                       [0, 0, 0], [0, 0, 0.35], [0, 0.1, 0.8])

    # Create simple walker controller
    controller = ZMPLocomotion(robot_height=0.8)

    # Run simulation
    for i in range(10000):  # 10000 steps
        p.stepSimulation()

        # Simple walking controller (conceptual)
        if i % 240 == 0:  # Every 240 steps (1 second at 240Hz)
            # Apply simple walking control
            # In a real implementation, you would use the controller to compute joint torques
            pass

        time.sleep(1./240.)

    p.disconnect()
    return controller

# Note: The above is a conceptual example. A full implementation would require
# a proper humanoid model with all necessary joints and sensors.
```

## Hands-on Exercise: Implementing a Simple Walking Controller

In this exercise, you'll implement a basic walking controller that can generate stable walking patterns.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of kinematics and control

### Exercise Steps
1. Implement a ZMP-based walking pattern generator
2. Create a simple balance controller
3. Test the controller with different walking speeds
4. Add disturbance recovery capabilities
5. Visualize the walking patterns and stability metrics

### Expected Outcome
You should have a working walking controller that can generate stable walking patterns and respond to disturbances, with visualization showing the relationship between CoM, ZMP, and foot positions.

### Sample Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class SimpleWalkingController:
    """Simple walking controller implementing basic locomotion patterns"""

    def __init__(self, robot_height=0.8, step_length=0.3, step_width=0.2, step_time=1.0):
        self.robot_height = robot_height
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.omega = np.sqrt(9.81 / robot_height)

        # Walking state
        self.current_x = 0.0
        self.current_y = 0.0
        self.left_support = True
        self.step_phase = 0.0  # 0.0 to 1.0, representing step cycle

        # Control parameters
        self.kp_balance = 10.0
        self.kd_balance = 5.0

    def generate_step_trajectory(self, num_steps=5):
        """Generate complete walking trajectory for specified number of steps"""
        dt = 0.01  # 100 Hz control
        total_time = num_steps * self.step_time
        n_points = int(total_time / dt)

        time_vec = np.linspace(0, total_time, n_points)

        # Initialize trajectories
        com_x = np.zeros(n_points)
        com_y = np.zeros(n_points)
        zmp_x = np.zeros(n_points)
        zmp_y = np.zeros(n_points)
        left_foot_x = np.zeros(n_points)
        left_foot_y = np.full(n_points, -self.step_width/2)
        right_foot_x = np.zeros(n_points)
        right_foot_y = np.full(n_points, self.step_width/2)

        # Generate walking pattern
        for step_idx in range(num_steps):
            step_start_time = step_idx * self.step_time
            step_start_idx = int(step_start_time / dt)
            step_end_idx = min(int((step_idx + 1) * self.step_time / dt), n_points)

            # Determine which foot is swing foot in this step
            swing_foot = 'right' if step_idx % 2 == 0 else 'left'

            # Set stance foot ZMP
            if swing_foot == 'right':
                # Left foot is stance foot
                zmp_y[step_start_idx:step_end_idx] = -self.step_width / 2
            else:
                # Right foot is stance foot
                zmp_y[step_start_idx:step_end_idx] = self.step_width / 2

            # Fill in X positions for ZMP and CoM
            for i in range(step_start_idx, step_end_idx):
                if i < n_points:
                    # ZMP X position follows the stance foot
                    zmp_x[i] = step_idx * self.step_length

                    # CoM X position - smooth transition
                    phase = (time_vec[i] - step_start_time) / self.step_time
                    com_x[i] = step_idx * self.step_length + phase * self.step_length * 0.8  # Slightly less than step length for stability
                    com_y[i] = 0.0  # Keep CoM centered laterally

            # Generate foot trajectories
            for i in range(step_start_idx, step_end_idx):
                if i < n_points:
                    phase_in_step = (time_vec[i] - step_start_time) / self.step_time

                    if swing_foot == 'right':
                        # Right foot is swinging forward
                        right_foot_x[i] = step_idx * self.step_length + phase_in_step * self.step_length
                    else:
                        # Left foot is swinging forward
                        left_foot_x[i] = step_idx * self.step_length + phase_in_step * self.step_length

        return {
            'time': time_vec,
            'com_x': com_x,
            'com_y': com_y,
            'zmp_x': zmp_x,
            'zmp_y': zmp_y,
            'left_foot_x': left_foot_x,
            'left_foot_y': left_foot_y,
            'right_foot_x': right_foot_x,
            'right_foot_y': right_foot_y
        }

    def add_balance_control(self, trajectory_data):
        """Add simple balance feedback to trajectory"""
        com_x = trajectory_data['com_x']
        com_y = trajectory_data['com_y']
        zmp_x = trajectory_data['zmp_x']
        zmp_y = trajectory_data['zmp_y']

        # Apply simple feedback control to adjust ZMP based on CoM position
        corrected_zmp_x = np.copy(zmp_x)
        corrected_zmp_y = np.copy(zmp_y)

        for i in range(1, len(com_x)):
            # Simple PD control: adjust ZMP based on CoM error
            com_error_x = com_x[i-1] - corrected_zmp_x[i-1]
            com_error_y = com_y[i-1] - corrected_zmp_y[i-1]

            # Apply feedback (simplified)
            corrected_zmp_x[i] = corrected_zmp_x[i-1] + 0.1 * com_error_x
            corrected_zmp_y[i] = corrected_zmp_y[i-1] + 0.1 * com_error_y

        trajectory_data['zmp_x_corrected'] = corrected_zmp_x
        trajectory_data['zmp_y_corrected'] = corrected_zmp_y

        return trajectory_data

    def simulate_disturbance_recovery(self, trajectory_data, disturbance_time=2.0, disturbance_mag=0.1):
        """Simulate response to external disturbance"""
        # Add disturbance at specified time
        dt = trajectory_data['time'][1] - trajectory_data['time'][0]
        disturbance_idx = int(disturbance_time / dt)

        if disturbance_idx < len(trajectory_data['com_x']):
            # Apply lateral disturbance
            trajectory_data['com_y'][disturbance_idx:] += disturbance_mag

            # Simulate recovery by adjusting ZMP
            recovery_start = disturbance_idx
            recovery_duration = int(1.0 / dt)  # 1 second recovery

            for i in range(recovery_start, min(recovery_start + recovery_duration, len(trajectory_data['com_y']))):
                # Move ZMP back toward CoM to recover balance
                zmp_to_com = trajectory_data['com_y'][i] - trajectory_data['zmp_y_corrected'][i]
                recovery_rate = (i - recovery_start) / recovery_duration
                trajectory_data['zmp_y_corrected'][i] += recovery_rate * zmp_to_com * 0.5

        return trajectory_data

def visualize_walking(trajectory_data):
    """Visualize the walking trajectory"""
    t = trajectory_data['time']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: X trajectories
    axes[0, 0].plot(t, trajectory_data['com_x'], 'b-', label='CoM X', linewidth=2)
    axes[0, 0].plot(t, trajectory_data['zmp_x'], 'r--', label='Original ZMP X', linewidth=2)
    axes[0, 0].plot(t, trajectory_data['zmp_x_corrected'], 'g:', label='Corrected ZMP X', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('X-Axis Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Y trajectories
    axes[0, 1].plot(t, trajectory_data['com_y'], 'b-', label='CoM Y', linewidth=2)
    axes[0, 1].plot(t, trajectory_data['zmp_y'], 'r--', label='Original ZMP Y', linewidth=2)
    axes[0, 1].plot(t, trajectory_data['zmp_y_corrected'], 'g:', label='Corrected ZMP Y', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].set_title('Y-Axis Trajectories')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: X-Y projection
    axes[0, 2].plot(trajectory_data['com_x'], trajectory_data['com_y'], 'b-', label='CoM Path', linewidth=2)
    axes[0, 2].scatter(trajectory_data['zmp_x_corrected'][::50],
                      trajectory_data['zmp_y_corrected'][::50],
                      c='red', s=30, label='Corrected ZMP', alpha=0.7)
    axes[0, 2].set_xlabel('X Position (m)')
    axes[0, 2].set_ylabel('Y Position (m)')
    axes[0, 2].set_title('CoM vs ZMP Trajectory')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].axis('equal')

    # Plot 4: Foot positions
    axes[1, 0].plot(t, trajectory_data['left_foot_x'], 'g-', label='Left Foot X', linewidth=2)
    axes[1, 0].plot(t, trajectory_data['right_foot_x'], 'm-', label='Right Foot X', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].set_title('Foot X Positions')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 5: Lateral foot positions
    axes[1, 1].plot(t, trajectory_data['left_foot_y'], 'g-', label='Left Foot Y', linewidth=2)
    axes[1, 1].plot(t, trajectory_data['right_foot_y'], 'm-', label='Right Foot Y', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].set_title('Foot Lateral Positions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot 6: ZMP error
    zmp_error = np.sqrt((trajectory_data['com_x'] - trajectory_data['zmp_x_corrected'])**2 +
                        (trajectory_data['com_y'] - trajectory_data['zmp_y_corrected'])**2)
    axes[1, 2].plot(t, zmp_error, 'r-', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('ZMP Error (m)')
    axes[1, 2].set_title('ZMP Tracking Error')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

def run_walking_exercise():
    """Run the complete walking controller exercise"""
    print("Running Simple Walking Controller Exercise...")

    # Create controller
    controller = SimpleWalkingController(robot_height=0.8, step_length=0.3, step_width=0.2, step_time=1.0)

    # Generate basic walking trajectory
    print("  Generating basic walking trajectory...")
    trajectory = controller.generate_step_trajectory(num_steps=5)

    # Add balance control
    print("  Adding balance control...")
    trajectory = controller.add_balance_control(trajectory)

    # Simulate disturbance recovery
    print("  Simulating disturbance recovery...")
    trajectory = controller.simulate_disturbance_recovery(trajectory, disturbance_time=3.0, disturbance_mag=0.05)

    # Visualize results
    print("  Visualizing results...")
    visualize_walking(trajectory)

    # Calculate performance metrics
    zmp_error = np.sqrt((trajectory['com_x'] - trajectory['zmp_x_corrected'])**2 +
                        (trajectory['com_y'] - trajectory['zmp_y_corrected'])**2)

    final_x_position = trajectory['com_x'][-1]
    avg_zmp_error = np.mean(zmp_error)
    max_zmp_error = np.max(zmp_error)

    print(f"\nWalking Controller Results:")
    print(f"  Distance traveled: {final_x_position:.3f} m")
    print(f"  Average ZMP error: {avg_zmp_error:.4f} m")
    print(f"  Maximum ZMP error: {max_zmp_error:.4f} m")
    print(f"  Steps completed: 5 steps of {0.3} m each")
    print(f"  Balance maintained: {np.mean(zmp_error < 0.05):.1%} of the time (within 5cm)")

    return trajectory

# Run the exercise
walking_trajectory = run_walking_exercise()
```

## Advanced Locomotion Considerations

### Terrain Adaptation

#### Uneven Ground Walking
- **Footstep Planning**: Adjust foot placement for terrain variations
- **Ankle Control**: Adjust foot orientation for ground slope
- **Body Height Adjustment**: Modify CoM height for obstacles

#### Stair Climbing
- **Foot Placement**: Precise positioning on step edges
- **Body Trajectory**: Adjust CoM trajectory for step transitions
- **Timing Control**: Coordinate with double support phases

### Energy Efficiency

#### Walking Cost Analysis
- **Transport Cost**: Energy per unit weight per unit distance
- **Optimization**: Minimize unnecessary movements
- **Passive Dynamics**: Utilize natural pendulum motion

#### Gait Optimization
- **Step Parameters**: Optimize step length and frequency
- **Joint Coordination**: Optimize inter-joint movement
- **Control Strategy**: Optimize feedback gains

## Summary

This chapter covered the fundamental concepts of humanoid locomotion, focusing on walking patterns and balance control. We explored static and dynamic balance principles, with particular emphasis on Zero Moment Point (ZMP) theory as the foundation for stable walking control.

We implemented practical examples of ZMP-based walking pattern generation, balance control systems, and disturbance recovery mechanisms. The hands-on exercise provided experience with creating stable walking controllers that can respond to external disturbances.

We also discussed different locomotion patterns beyond walking, including considerations for terrain adaptation and energy efficiency. The integration of perception, control, and actuation systems was highlighted as essential for successful locomotion.

Understanding locomotion patterns is crucial for developing humanoid robots that can move safely and efficiently in human environments, whether for assistance, exploration, or other applications.

## Key Takeaways

- Static balance requires CoM to remain within support polygon
- ZMP theory is fundamental to dynamic walking control
- Balance control involves multiple strategies (ankle, hip, stepping)
- Walking patterns must consider timing, foot placement, and stability
- Disturbance recovery is essential for robust locomotion
- Energy efficiency is important for practical applications
- Terrain adaptation requires additional sensing and planning

## Next Steps

In the next chapter, we'll move to Module 3: Control and Intelligence, beginning with Chapter 3.1: Balance and Stability Control. We'll explore more advanced control techniques, including how to maintain balance during complex movements and how to integrate perception with control for adaptive locomotion.

## References and Further Reading

1. Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K., & Hirukawa, H. (2003). Biped walking pattern generation by using preview control of zero-moment point. IEEE International Conference on Robotics and Automation.
2. Pratt, J., Carff, J., Drakunov, S., & Goswami, A. (2006). Capture point: A step toward humanoid push recovery. IEEE-RAS International Conference on Humanoid Robots.
3. Hofmann, A., Iida, F., & Pfeifer, R. (2008). Designing a force-controllable ankle for a biped robot: Physical and numerical experiments. IEEE International Conference on Robotics and Automation.
4. Wight, D. L., Kubica, E. G., & Wang, D. W. (2008). Control of a walking biped using linearization of the zero moment point. Proceedings of the 2008 American Control Conference.
5. Englsberger, J., Ott, C., & Peer, A. (2015). Bipedal walking control based on Capture Point dynamics. IEEE/RSJ International Conference on Intelligent Robots and Systems.