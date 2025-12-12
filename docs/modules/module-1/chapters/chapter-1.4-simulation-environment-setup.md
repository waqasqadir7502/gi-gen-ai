# Chapter 1.4: Simulation Environment Setup

## Overview

Setting up a proper simulation environment is crucial for learning and experimenting with Physical AI concepts. This chapter provides a comprehensive guide to installing and configuring PyBullet, one of the most popular physics simulators for robotics and Physical AI research. We'll also cover alternative simulation environments and best practices for simulation-based development.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Install and configure PyBullet for Physical AI experiments
2. Set up a basic simulation environment with robots and objects
3. Implement basic robot control in simulation
4. Troubleshoot common simulation issues
5. Understand alternative simulation environments for Physical AI

## Why Simulation for Physical AI?

Simulation is essential for Physical AI development because:

- **Safety**: Test algorithms without risk of damaging expensive hardware
- **Cost-effectiveness**: No need for physical robots for initial development
- **Repeatability**: Controlled environments for consistent testing
- **Speed**: Faster iteration cycles compared to real hardware
- **Accessibility**: Enables learning without specialized hardware

## PyBullet: The Recommended Simulation Environment

PyBullet is a physics engine with collision detection, contact forces, and dynamics simulation. It's widely used in robotics research and is particularly suitable for Physical AI due to its:

- Free and open-source nature
- Python API for easy integration
- Support for various robot formats (URDF, SDF, MJCF)
- Realistic physics simulation
- Integration with reinforcement learning frameworks

## Prerequisites

Before installing PyBullet, ensure you have:

- Python 3.7 or higher
- pip (Python package installer)
- Basic familiarity with Python programming
- System with at least 4GB RAM (8GB recommended)

## Installation Process

### Step 1: Install Python Dependencies

First, ensure you have Python 3.7+ and pip installed:

```bash
python --version
pip --version
```

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps manage dependencies:

```bash
# Create virtual environment
python -m venv physical_ai_env

# Activate virtual environment
# On Windows:
physical_ai_env\Scripts\activate
# On macOS/Linux:
source physical_ai_env/bin/activate
```

### Step 3: Install PyBullet

Install PyBullet using pip:

```bash
pip install pybullet
```

### Step 4: Install Additional Dependencies

For a complete simulation environment, install additional packages:

```bash
pip install numpy matplotlib scipy
```

### Step 5: Verify Installation

Test the installation with a simple script:

```python
import pybullet as p
import pybullet_data
import time

# Connect to physics server in GUI mode
physicsClient = p.connect(p.GUI)

# Add plane, robot, and objects
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0,0,0]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)

# Set gravity
p.setGravity(0, 0, -9.8)

# Run simulation for a few steps
for i in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect
p.disconnect()
```

## Basic PyBullet Concepts

### Physics Client
The physics client manages the simulation. You connect to it using:
```python
client_id = p.connect(p.GUI)  # GUI mode
# or
client_id = p.connect(p.DIRECT)  # Headless mode
```

### URDF Files
URDF (Unified Robot Description Format) files describe robot models. PyBullet comes with many built-in models in `pybullet_data`.

### Coordinate Systems
PyBullet uses a right-handed coordinate system:
- X: Forward/backward
- Y: Left/right
- Z: Up/down

## Setting Up Your First Simulation

Let's create a complete simulation environment:

```python
import pybullet as p
import pybullet_data
import numpy as np
import time

class PhysicalAISimulation:
    def __init__(self, use_gui=True):
        """Initialize the simulation environment"""
        if use_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        # Set up simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)  # 240 Hz simulation

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load a simple robot (R2D2 as example)
        self.robot_start_pos = [0, 0, 0.5]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("r2d2.urdf",
                                  self.robot_start_pos,
                                  self.robot_start_orientation)

    def add_object(self, object_name, position, orientation=None):
        """Add an object to the simulation"""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])

        object_id = p.loadURDF(object_name, position, orientation)
        return object_id

    def run_simulation(self, steps=1000):
        """Run the simulation for a specified number of steps"""
        for i in range(steps):
            p.stepSimulation()
            time.sleep(1./240.)  # Slow down to real-time

    def reset_robot(self):
        """Reset robot to initial position"""
        p.resetBasePositionAndOrientation(self.robot_id,
                                        self.robot_start_pos,
                                        self.robot_start_orientation)

    def close(self):
        """Close the simulation"""
        p.disconnect(self.client_id)

# Example usage
if __name__ == "__main__":
    # Create simulation
    sim = PhysicalAISimulation(use_gui=True)

    # Add some objects
    box_id = sim.add_object("cube.urdf", [1, 0, 0.5])
    sphere_id = sim.add_object("sphere2.urdf", [2, 1, 0.5])

    # Run simulation
    sim.run_simulation(steps=500)

    # Reset and run again
    sim.reset_robot()
    sim.run_simulation(steps=500)

    # Clean up
    sim.close()
```

## Controlling Robots in Simulation

### Joint Control
You can control robot joints using different control modes:

```python
import pybullet as p
import time

# Position control
p.setJointMotorControl2(bodyUniqueId=robot_id,
                       jointIndex=joint_index,
                       controlMode=p.POSITION_CONTROL,
                       targetPosition=target_angle)

# Velocity control
p.setJointMotorControl2(bodyUniqueId=robot_id,
                       jointIndex=joint_index,
                       controlMode=p.VELOCITY_CONTROL,
                       targetVelocity=target_velocity)

# Torque control
p.setJointMotorControl2(bodyUniqueId=robot_id,
                       jointIndex=joint_index,
                       controlMode=p.TORQUE_CONTROL,
                       force=torque_value)
```

### Cartesian Control
For end-effector control, you might use inverse kinematics:

```python
# Calculate joint positions for desired end-effector position
joint_positions = p.calculateInverseKinematics(robot_id,
                                              end_effector_link_index,
                                              target_position)

# Apply joint positions
for i, joint_pos in enumerate(joint_positions):
    p.setJointMotorControl2(bodyUniqueId=robot_id,
                           jointIndex=i,
                           controlMode=p.POSITION_CONTROL,
                           targetPosition=joint_pos)
```

## Troubleshooting Common Issues

### Issue 1: Installation Problems
**Problem**: PyBullet installation fails
**Solution**:
- Ensure you're using Python 3.7+
- Try: `pip install --upgrade pip`
- Try: `pip install pybullet --no-cache-dir`

### Issue 2: GUI Not Working
**Problem**: Simulation runs but GUI doesn't appear
**Solution**:
- Check if you have proper graphics drivers
- Try running with `p.GUI` instead of other modes
- On remote systems, use VNC or X11 forwarding

### Issue 3: Performance Issues
**Problem**: Simulation runs slowly
**Solution**:
- Reduce the number of objects in the scene
- Increase the time step (less accurate but faster)
- Use `p.DIRECT` mode instead of `p.GUI` for headless operation

### Issue 4: Robot Falls Through Ground
**Problem**: Robot or objects fall through the ground plane
**Solution**:
- Ensure gravity is set: `p.setGravity(0, 0, -9.81)`
- Check that objects are loaded above the ground plane
- Verify proper collision shapes are defined

## Alternative Simulation Environments

### Gazebo
- More complex but feature-rich
- ROS integration
- Better for complex multi-robot scenarios

### Webots
- Browser-based simulation
- Good educational tools
- Built-in robot models

### Mujoco
- High-fidelity physics
- Commercial license required
- Excellent for research

### NVIDIA Isaac Gym
- GPU-accelerated
- Good for reinforcement learning
- Requires specific hardware

## Best Practices for Simulation

### 1. Start Simple
Begin with basic scenarios before moving to complex environments.

### 2. Validate Against Theory
Compare simulation results with analytical solutions when possible.

### 3. Systematic Testing
Use consistent test scenarios to compare different algorithms.

### 4. Transfer Learning
Test on simulation first, then validate on real hardware when possible.

### 5. Parameter Tuning
Keep simulation parameters realistic to ensure transferability.

## Hands-on Exercise: Building Your First Simulation

In this exercise, you'll create a complete simulation environment with a robot and simple objects, then implement basic control.

### Requirements
- PyBullet installed
- Python 3.7+
- Basic programming knowledge

### Exercise Steps
1. Install PyBullet and verify the installation
2. Create a basic simulation with a ground plane and robot
3. Add at least 3 different objects to the environment
4. Implement a simple control loop that moves the robot
5. Document your observations about the simulation

### Expected Outcome
You should have a working simulation environment that you can extend for future exercises and experiments.

### Sample Solution Template
```python
import pybullet as p
import pybullet_data
import numpy as np
import time

def create_basic_simulation():
    """Create a basic simulation environment"""
    # Connect to physics server
    physicsClient = p.connect(p.GUI)

    # Set up environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    planeId = p.loadURDF("plane.urdf")

    # Load robot
    robotStartPos = [0, 0, 1]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)

    # Add objects
    boxId = p.loadURDF("cube.urdf", [1, 0, 1])
    sphereId = p.loadURDF("sphere2.urdf", [2, 1, 1])
    cylinderId = p.loadURDF("cylinder.urdf", [0, 2, 1])

    # Simple control loop
    for i in range(1000):
        # Simple control - move forward
        if i < 500:
            # Apply forces or control joints here
            pass
        else:
            # Apply different control
            pass

        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()

if __name__ == "__main__":
    create_basic_simulation()
```

## Summary

This chapter has provided a comprehensive guide to setting up a simulation environment for Physical AI experiments using PyBullet. You've learned how to install and configure PyBullet, create basic simulations, control robots, and troubleshoot common issues. The hands-on exercise gives you practical experience with creating your own simulation environment.

## Key Takeaways

- PyBullet is an excellent free and open-source simulation environment for Physical AI
- Proper installation and configuration are essential for successful simulation
- Understanding coordinate systems and control modes is crucial for robot control
- Troubleshooting skills are important for resolving common simulation issues
- Simulation provides a safe and cost-effective way to experiment with Physical AI concepts

## Next Steps

With your simulation environment set up, you now have the foundation to explore more advanced Physical AI concepts in the upcoming modules. The mathematical foundations from Chapter 1.3 and your simulation setup will be essential tools for hands-on experimentation.

## References and Further Reading

1. Coumans, E., & Bai, Y. (2016). PyBullet, a Python module for physics simulation for games, robotics and machine learning.
2. Murai, R., et al. (2021). PyBullet Quickstart Guide. Available at: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGstqzTH4ZJN4bLGJXK6bJKdGE/
3. Robotics Stack Exchange. PyBullet tutorials and examples.
4. Reinforcement Learning with PyBullet: https://github.com/benelot/pybullet-gym