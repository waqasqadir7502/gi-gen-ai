# Chapter 2.1: Kinematics and Movement Systems

## Overview

Kinematics is the study of motion without considering the forces that cause it. In humanoid robotics, kinematics is fundamental to understanding how robots move and how to control their movements. This chapter covers both forward and inverse kinematics, which are essential for controlling robot arms, legs, and other articulated systems.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the difference between forward and inverse kinematics
2. Apply forward kinematics to determine end-effector position from joint angles
3. Solve inverse kinematics problems to find joint angles for desired positions
4. Implement basic kinematic calculations in simulation
5. Understand the challenges and limitations of kinematic solutions

## Introduction to Kinematics in Robotics

Kinematics is the branch of mechanics that deals with the motion of objects without considering the forces that cause the motion. In robotics, kinematics is concerned with the relationship between the joint angles of a robot and the position and orientation of its end-effectors (such as hands, feet, or tools).

### Why Kinematics Matters for Humanoid Robots

Humanoid robots have complex articulated structures with multiple joints, similar to human bodies. Understanding kinematics is crucial for:

- **Locomotion**: Controlling leg movements for walking, running, or other forms of locomotion
- **Manipulation**: Controlling arm movements to reach, grasp, and manipulate objects
- **Balance**: Understanding how body segment positions affect center of mass and stability
- **Motion Planning**: Planning smooth, coordinated movements of multiple body parts

## Forward Kinematics

Forward kinematics is the process of determining the position and orientation of the end-effector given the joint angles of the robot. It answers the question: "Where is the end-effector if the joints are at these angles?"

### Mathematical Foundation

For a robot with n joints, forward kinematics involves transforming coordinates from the joint space to the Cartesian space. Each joint contributes a transformation matrix that describes how it moves relative to its parent joint.

The overall transformation from the base to the end-effector is the product of all individual joint transformations:

```
T_total = T₁ × T₂ × ... × Tₙ
```

Where Tᵢ is the transformation matrix for joint i.

### Denavit-Hartenberg Convention

The Denavit-Hartenberg (D-H) convention is a standardized method for defining the coordinate frames on the links of a robot manipulator. It uses four parameters for each joint:

1. **aᵢ** (link length): Distance along the xᵢ axis from zᵢ to zᵢ₊₁
2. **αᵢ** (link twist): Angle about the xᵢ axis from zᵢ to zᵢ₊₁
3. **dᵢ** (link offset): Distance along the zᵢ axis from the origin of frame i to the intersection of xᵢ₊₁ and zᵢ
4. **θᵢ** (joint angle): Angle about the zᵢ axis from xᵢ to xᵢ₊₁

### Example: 2-Link Planar Robot

Let's consider a simple 2-link planar robot arm with revolute joints:

```
    y
    ↑
    |     θ₂
    |     /
    |    / L₂
    |   /
    |  ●
    | /  \
    |/    \ L₁
   /| θ₁   \
  /_|______\______→ x
  |  \
  |   \
  |    \
  |     \
  |      \
  |       \
  |        \
  |         \
  |          \
  |           \
  |            \
  |             \
  |              \
  |               \
  |                \
  |                 \
  |                  \
  |                   \
  |                    \
  |                     \
  |                      \
  |                       \
  |                        \
  |                         \
  |                          \
  |                           \
  |                            \
  |                             \
  |                              \
  |                               \
  |                                \
  |                                 \
  |                                  \
  |                                   \
  |                                    \
  |                                     \
  |                                      \
  |                                       \
  |                                        \
  |                                         \
  |                                          \
  |                                           \
  |                                            \
  |                                             \
  |                                              \
  |                                               \
  |                                                \
  |                                                 \
  |                                                  \
  |                                                   \
  |                                                    \
  |                                                     \
  |                                                      \
  |                                                       \
  |                                                        \
  |                                                         \
  |                                                          \
  |                                                           \
  |                                                            \
  |                                                             \
  |                                                              \
  |                                                               \
  |                                                                \
  |                                                                 \
  |                                                                  \
  |                                                                   \
  |                                                                    \
  |                                                                     \
  |                                                                      \
  |                                                                       \
  |                                                                        \
  |                                                                         \
  |                                                                          \
  |                                                                           \
  |                                                                            \
  |                                                                             \
  |                                                                              \
  |                                                                               \
  |                                                                                \
  |                                                                                 \
  |                                                                                  \
  |                                                                                   \
  |                                                                                    \
  |                                                                                     \
  |                                                                                      \
  |                                                                                       \
  |                                                                                        \
  |                                                                                         \
  |                                                                                          \
  |                                                                                           \
  |                                                                                            \
  |                                                                                             \
  |                                                                                              \
  |                                                                                               \
  |                                                                                                \
  |                                                                                                 \
  |                                                                                                  \
  |                                                                                                   \
  |                                                                                                    \
  |                                                                                                     \
  |                                                                                                      \
  |                                                                                                       \
  |                                                                                                        \
  |                                                                                                         \
  |                                                                                                          \
  |                                                                                                           \
  |                                                                                                            \
  |                                                                                                             \
  |                                                                                                              \
  |                                                                                                               \
  |                                                                                                                \
  |                                                                                                                 \
  |                                                                                                                  \
  |                                                                                                                   \
  |                                                                                                                    \
  |                                                                                                                     \
  |                                                                                                                      \
  |                                                                                                                       \
  |                                                                                                                        \
  |                                                                                                                         \
  |                                                                                                                          \
  |                                                                                                                           \
  |                                                                                                                            \
  |                                                                                                                             \
  |                                                                                                                              \
  |                                                                                                                               \
  |                                                                                                                                \
  |                                                                                                                         ● End-effector
  |                                                                                                                        /
  |                                                                                                                       /
  |                                                                                                                      /
  |                                                                                                                     /
  |                                                                                                                    /
  |                                                                                                                   /
  |                                                                                                                  /
  |                                                                                                                 /
  |                                                                                                                /
  |                                                                                                               /
  |                                                                                                              /
  |                                                                                                             /
  |                                                                                                            /
  |                                                                                                           /
  |                                                                                                          /
  |                                                                                                         /
  |                                                                                                        /
  |                                                                                                       /
  |                                                                                                      /
  |                                                                                                     /
  |                                                                                                    /
  |                                                                                                   /
  |                                                                                                  /
  |                                                                                                 /
  |                                                                                                /
  |                                                                                               /
  |                                                                                              /
  |                                                                                             /
  |                                                                                            /
  |                                                                                           /
  |                                                                                          /
  |                                                                                         /
  |                                                                                        /
  |                                                                                       /
  |                                                                                      /
  |                                                                                     /
  |                                                                                    /
  |                                                                                   /
  |                                                                                  /
  |                                                                                 /
  |                                                                                /
  |                                                                               /
  |                                                                              /
  |                                                                             /
  |                                                                            /
  |                                                                           /
  |                                                                          /
  |                                                                         /
  |                                                                        /
  |                                                                       /
  |                                                                      /
  |                                                                     /
  |                                                                    /
  |                                                                   /
  |                                                                  /
  |                                                                 /
  |                                                                /
  |                                                               /
  |                                                              /
  |                                                             /
  |                                                            /
  |                                                           /
  |                                                          /
  |                                                         /
  |                                                        /
  |                                                       /
  |                                                      /
  |                                                     /
  |                                                    /
  |                                                   /
  |                                                  /
  |                                                 /
  |                                                /
  |                                               /
  |                                              /
  |                                             /
  |                                            /
  |                                           /
  |                                          /
  |                                         /
  |                                        /
  |                                       /
  |                                      /
  |                                     /
  |                                    /
  |                                   /
  |                                  /
  |                                 /
  |                                /
  |                               /
  |                              /
  |                             /
  |                            /
  |                           /
  |                          /
  |                         /
  |                        /
  |                       /
  |                      /
  |                     /
  |                    /
  |                   /
  |                  /
  |                 /
  |                /
  |               /
  |              /
  |             /
  |            /
  |           /
  |          /
  |         /
  |        /
  |       /
  |      /
  |     /
  |    /
  |   /
  |  /
  | /
  |/
  +----------------------------------------------------------→ x
```

For this 2-link robot, the forward kinematics equations are:

```
x = L₁ * cos(θ₁) + L₂ * cos(θ₁ + θ₂)
y = L₁ * sin(θ₁) + L₂ * sin(θ₁ + θ₂)
```

Where L₁ and L₂ are the lengths of the two links.

## Inverse Kinematics

Inverse kinematics is the reverse problem: given a desired position and orientation of the end-effector, determine the joint angles needed to achieve that position. It answers the question: "What joint angles are needed to place the end-effector at this position?"

### Challenges in Inverse Kinematics

Inverse kinematics is generally more challenging than forward kinematics because:

1. **Multiple Solutions**: There may be multiple sets of joint angles that achieve the same end-effector position
2. **No Solution**: The desired position may be outside the robot's workspace
3. **Singularities**: Special configurations where the robot loses degrees of freedom
4. **Computational Complexity**: Analytical solutions may not exist for complex robots

### Analytical vs. Numerical Solutions

**Analytical Solutions**: Exact mathematical solutions that work well for simple robots with special geometric arrangements. For our 2-link robot:

```
cos(θ₂) = (x² + y² - L₁² - L₂²) / (2 * L₁ * L₂)
θ₂ = ±arccos(cos(θ₂))
θ₁ = arctan2(y, x) - arctan2(L₂ * sin(θ₂), L₁ + L₂ * cos(θ₂))
```

**Numerical Solutions**: Iterative methods that work for more complex robots. Common approaches include:
- Jacobian-based methods
- Cyclic Coordinate Descent (CCD)
- FABRIK (Forwards and Backwards Reaching Inverse Kinematics)

## Jacobian Matrix

The Jacobian matrix relates joint velocities to end-effector velocities. For a robot with n joints and an end-effector with m degrees of freedom:

```
v = J(θ) × θ̇
```

Where:
- v is the end-effector velocity vector
- J(θ) is the Jacobian matrix
- θ̇ is the joint velocity vector

The Jacobian is particularly important for:
- Velocity control
- Force control
- Singularity analysis
- Motion planning

## Movement Systems in Humanoid Robots

Humanoid robots have complex movement systems that mimic human biomechanics. Key components include:

### Degrees of Freedom (DOF)
The number of independent movements a robot can make. A typical humanoid robot has 20-30+ DOF distributed across:
- Legs (6 DOF each for full mobility)
- Arms (7 DOF each for dexterity)
- Torso (2-6 DOF for flexibility)
- Head/neck (2-3 DOF for orientation)

### Joint Types
1. **Revolute Joints**: Rotational joints (like human elbows and knees)
2. **Prismatic Joints**: Linear joints (less common in humanoid robots)
3. **Spherical Joints**: Multi-axis joints (like human shoulders and hips)

### Workspace
The workspace is the set of all positions that the end-effector can reach. It's characterized by:
- **Dexterous Workspace**: Positions where the end-effector can be oriented in any direction
- **Reachable Workspace**: Positions where the end-effector can reach, but with limited orientation

## Simulation Examples

Let's implement a simple kinematic simulation using PyBullet:

```python
import pybullet as p
import pybullet_data
import numpy as np
import math

def create_2dof_robot():
    """Create a simple 2-DOF robot arm in PyBullet"""
    # Connect to physics server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Create ground plane
    planeId = p.loadURDF("plane.urdf")

    # Create a simple 2-DOF robot arm
    # For this example, we'll use a pre-built robot or create one with basic shapes

    # Load a simple robot (using Kuka arm as example, but we'll configure it differently)
    robotStartPos = [0, 0, 0]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

    # For this example, we'll create a simple visualization
    # Create base
    base_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.5, 0.5, 0.5, 1])
    base_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    base_id = p.createMultiBody(baseVisualShapeIndex=base_visual,
                               baseCollisionShapeIndex=base_collision,
                               basePosition=[0, 0, 0.1])

    # Create first link
    link1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.05, 0.05], rgbaColor=[0.8, 0.2, 0.2, 1])
    link1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.05, 0.05])

    # Create second link
    link2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.05, 0.05], rgbaColor=[0.2, 0.2, 0.8, 1])
    link2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.05, 0.05])

    # Create joint between base and first link
    joint1_c = p.createConstraint(base_id, -1, -1, -1,
                                  p.JOINT_REVOLUTE,
                                  [0, 0, 1],  # Joint axis
                                  [0, 0, 0],  # Parent position
                                  [0.3, 0, 0]) # Child position

    # Create joint between first and second link
    joint2_c = p.createConstraint(-1, -1, -1, -1,
                                  p.JOINT_REVOLUTE,
                                  [0, 0, 1],  # Joint axis
                                  [0.6, 0, 0],  # Parent position (relative to first link)
                                  [0.25, 0, 0]) # Child position

    # Forward kinematics example
    def forward_kinematics(theta1, theta2, l1=0.6, l2=0.5):
        """Calculate end-effector position given joint angles"""
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
        return x, y

    # Inverse kinematics example
    def inverse_kinematics(x, y, l1=0.6, l2=0.5):
        """Calculate joint angles given end-effector position"""
        # Check if position is reachable
        distance = math.sqrt(x**2 + y**2)
        if distance > l1 + l2:
            print("Position is outside workspace")
            return None, None

        if distance < abs(l1 - l2):
            print("Position is inside inner workspace limit")
            return None, None

        # Calculate theta2
        cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        theta2 = math.acos(max(-1, min(1, cos_theta2)))  # Clamp to [-1, 1] to avoid numerical errors

        # Calculate theta1
        k1 = l1 + l2 * math.cos(theta2)
        k2 = l2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        return theta1, theta2

    # Test the kinematic functions
    print("Testing Forward Kinematics:")
    theta1, theta2 = math.pi/4, math.pi/6  # 45° and 30°
    x, y = forward_kinematics(theta1, theta2)
    print(f"Joint angles: θ1={theta1:.2f}, θ2={theta2:.2f}")
    print(f"End-effector position: x={x:.2f}, y={y:.2f}")

    print("\nTesting Inverse Kinematics:")
    # Use the position we just calculated to verify inverse kinematics
    inv_theta1, inv_theta2 = inverse_kinematics(x, y)
    if inv_theta1 is not None:
        print(f"Inverse joint angles: θ1={inv_theta1:.2f}, θ2={inv_theta2:.2f}")
        inv_x, inv_y = forward_kinematics(inv_theta1, inv_theta2)
        print(f"Verification: x={inv_x:.2f}, y={inv_y:.2f}")

    # Run simulation for a bit
    for i in range(1000):
        p.stepSimulation()
        # Change joint angles over time to see movement
        if i % 100 == 0:
            target_theta1 = math.pi/4 + 0.2 * math.sin(i/100)
            target_theta2 = math.pi/6 + 0.3 * math.cos(i/100)
            # In a real implementation, we would set joint positions here
        #time.sleep(1./240.)

    p.disconnect()
    return forward_kinematics, inverse_kinematics

# Example usage (uncomment to run)
# fk_func, ik_func = create_2dof_robot()
```

## Hands-on Exercise: Implementing Forward Kinematics

In this exercise, you'll implement forward kinematics for a simple 2-link robot and visualize the results.

### Requirements
- Python 3.8+
- PyBullet physics simulator
- NumPy library
- Matplotlib for visualization

### Exercise Steps
1. Implement the forward kinematics function for a 2-link planar robot
2. Create a visualization showing the robot arm and its workspace
3. Test the function with different joint angle combinations
4. Verify the results mathematically

### Expected Outcome
You should have a working implementation of forward kinematics that can calculate the end-effector position given joint angles, with visualization to help understand the relationship.

### Sample Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics_2dof(theta1, theta2, l1=1.0, l2=1.0):
    """
    Calculate end-effector position for a 2-DOF planar robot

    Args:
        theta1: Angle of first joint (in radians)
        theta2: Angle of second joint (in radians)
        l1: Length of first link
        l2: Length of second link

    Returns:
        tuple: (x, y) end-effector position
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def visualize_robot_arm(theta1, theta2, l1=1.0, l2=1.0):
    """Visualize the 2-DOF robot arm"""
    # Calculate joint positions
    joint1_x = l1 * np.cos(theta1)
    joint1_y = l1 * np.sin(theta1)

    end_effector_x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    end_effector_y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

    # Plot the robot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot links
    ax.plot([0, joint1_x], [0, joint1_y], 'b-', linewidth=5, label='Link 1')
    ax.plot([joint1_x, end_effector_x], [joint1_y, end_effector_y], 'r-', linewidth=5, label='Link 2')

    # Plot joints
    ax.plot(0, 0, 'ko', markersize=10, label='Base')
    ax.plot(joint1_x, joint1_y, 'bo', markersize=8, label='Joint 1')
    ax.plot(end_effector_x, end_effector_y, 'ro', markersize=8, label='End-effector')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(f'2-DOF Robot Arm\nθ1={theta1:.2f} rad, θ2={theta2:.2f} rad')

    plt.show()

    return end_effector_x, end_effector_y

# Test the implementation
theta1 = np.pi/3  # 60 degrees
theta2 = np.pi/4  # 45 degrees
l1, l2 = 1.0, 0.8

x, y = forward_kinematics_2dof(theta1, theta2, l1, l2)
print(f"End-effector position: ({x:.3f}, {y:.3f})")

# Visualize
end_x, end_y = visualize_robot_arm(theta1, theta2, l1, l2)
print(f"Visualized robot arm with end-effector at: ({end_x:.3f}, {end_y:.3f})")
```

## Common Challenges and Solutions

### Singularities
Singularities occur when the robot loses one or more degrees of freedom. At these configurations, small changes in end-effector position require large changes in joint angles, or the Jacobian matrix becomes singular (non-invertible).

**Solutions:**
- Avoid singular configurations in motion planning
- Use damped least squares method for inverse kinematics near singularities
- Implement singularity detection and avoidance algorithms

### Joint Limits
Real robots have physical joint limits that must be respected in kinematic solutions.

**Solutions:**
- Include joint limits in inverse kinematics solvers
- Use constrained optimization techniques
- Implement joint limit checking in control algorithms

### Redundancy
Humanoid robots often have more degrees of freedom than necessary to achieve a task (redundant robots). This provides flexibility but complicates control.

**Solutions:**
- Use null-space projection to optimize secondary objectives (e.g., joint centering, obstacle avoidance)
- Implement priority-based task control
- Apply optimization techniques to select the best solution among infinite possibilities

## Summary

This chapter introduced kinematics as a fundamental concept for controlling humanoid robots. We covered both forward and inverse kinematics, with practical examples and implementation details. Forward kinematics allows us to determine where the end-effector will be given joint angles, while inverse kinematics helps us determine the joint angles needed to achieve a desired end-effector position.

We explored the mathematical foundations, including the Denavit-Hartenberg convention for representing robot kinematics, and discussed the Jacobian matrix which relates joint velocities to end-effector velocities. The hands-on exercise provided practical experience implementing and visualizing kinematic calculations.

Understanding kinematics is essential for controlling the complex movement systems of humanoid robots, from simple arm movements to complex whole-body motions for locomotion and manipulation.

## Key Takeaways

- Forward kinematics: Calculate end-effector position from joint angles
- Inverse kinematics: Calculate joint angles from desired end-effector position
- The Jacobian matrix relates joint velocities to end-effector velocities
- Singularities are problematic configurations where the robot loses degrees of freedom
- Joint limits and redundancy add complexity to kinematic solutions
- Kinematics is fundamental to motion planning and control in humanoid robotics

## Next Steps

In the next chapter, we'll explore sensor systems and perception in humanoid robots, which work together with kinematic systems to enable robots to understand and interact with their environment. We'll cover vision systems, touch sensors, proprioception, and sensor fusion techniques.

## References and Further Reading

1. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). Robot Modeling and Control. John Wiley & Sons.
2. Craig, J. J. (2005). Introduction to Robotics: Mechanics and Control (3rd ed.). Pearson Prentice Hall.
3. Siciliano, B., & Khatib, O. (Eds.). (2016). Springer Handbook of Robotics. Springer.
4. Murray, R. M., Li, Z., & Sastry, S. S. (1994). A Mathematical Introduction to Robotic Manipulation. CRC Press.