# Chapter 1.3: Basic Mathematics for Physical AI

## Overview

Physical AI systems operate in the real world, which means they must deal with continuous spaces, uncertainty, dynamics, and complex interactions. This chapter covers the essential mathematical foundations needed to understand and develop Physical AI systems. We'll focus on practical applications rather than theoretical proofs, emphasizing concepts that you'll encounter in real Physical AI implementations.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Apply linear algebra concepts to represent physical systems and transformations
2. Use calculus to understand motion, change, and optimization in Physical AI
3. Apply probability and statistics to handle uncertainty in physical systems
4. Understand basic optimization techniques used in Physical AI
5. Use trigonometry for kinematic calculations and spatial reasoning

## Linear Algebra for Physical AI

Linear algebra is fundamental to representing physical systems, transformations, and relationships between different coordinate systems.

### Vectors and Their Applications

A vector is a mathematical object that has both magnitude and direction. In Physical AI, vectors are used to represent:

- **Position**: Location in 2D or 3D space
- **Velocity**: Rate of change of position
- **Force**: Magnitude and direction of applied force
- **Torque**: Rotational force

**Example**: A robot's position in 3D space can be represented as a vector **p** = [x, y, z]ᵀ, where x, y, and z are coordinates in meters.

### Vector Operations

#### Addition and Subtraction
Vector addition is used to combine positions, velocities, or forces:

```
[a₁, a₂] + [b₁, b₂] = [a₁ + b₁, a₂ + b₂]
```

**Application**: If a robot moves by displacement vector **d** from position **p**, its new position is **p** + **d**.

#### Dot Product
The dot product of two vectors **a** and **b** is defined as:
```
**a** · **b** = |**a**| |**b**| cos(θ)
```
where θ is the angle between the vectors.

**Application**: Determining if two vectors are perpendicular (dot product = 0) or finding the angle between directions.

#### Cross Product
The cross product of two 3D vectors produces a third vector perpendicular to both:
```
**c** = **a** × **b**
```

**Application**: Computing torque, angular momentum, and finding normal vectors to surfaces.

### Matrices and Transformations

Matrices are used extensively in Physical AI for transformations, rotations, and representing systems of equations.

#### Rotation Matrices
A 2D rotation matrix that rotates a point by angle θ is:
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

For 3D rotations, we have rotation matrices around each axis (x, y, z).

**Application**: Converting coordinates between different reference frames, such as robot base frame to end-effector frame.

#### Transformation Matrices
Homogeneous transformation matrices combine rotation and translation:
```
T = [R  p]
    [0  1]
```
where R is a rotation matrix and p is a translation vector.

**Application**: Representing the complete pose (position and orientation) of a robot or object in space.

### Systems of Linear Equations
Many Physical AI problems can be formulated as systems of linear equations:
```
Ax = b
```
where A is a coefficient matrix, x is the unknown vector, and b is the result vector.

**Application**: Solving for joint angles in inverse kinematics, force distribution in multi-contact scenarios.

## Calculus for Physical AI

Calculus is essential for understanding motion, change, and optimization in physical systems.

### Derivatives and Motion
The derivative represents the rate of change. In robotics and Physical AI:

- **Position**: x(t) - location as a function of time
- **Velocity**: v(t) = dx/dt - rate of change of position
- **Acceleration**: a(t) = dv/dt = d²x/dt² - rate of change of velocity

**Application**: Controlling robot motion, trajectory planning, and understanding dynamic systems.

### Integration
Integration is the reverse of differentiation and is used to find total quantities from rates of change.

**Application**: Computing position from velocity measurements, calculating work done by forces, determining total distance traveled.

### Optimization
Many Physical AI problems involve finding optimal solutions, such as:
- Minimizing energy consumption
- Maximizing stability
- Finding optimal trajectories

The derivative is used to find minima and maxima: if df/dx = 0, then x is a critical point.

## Probability and Statistics for Physical AI

Physical systems are inherently uncertain due to sensor noise, environmental variability, and model inaccuracies.

### Probability Distributions
- **Gaussian/Normal Distribution**: Most sensor noise follows a normal distribution
- **Uniform Distribution**: Used when all outcomes are equally likely
- **Multivariate Gaussian**: For multiple correlated variables (e.g., 2D position with correlated x,y errors)

### Bayes' Theorem
Bayes' theorem is fundamental for sensor fusion and state estimation:
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Application**: Updating beliefs about robot state based on sensor measurements, localization, and mapping.

### Statistical Measures
- **Mean**: Average value of measurements
- **Variance**: Measure of uncertainty or spread
- **Covariance**: How two variables change together

**Application**: Characterizing sensor noise, uncertainty propagation, and performance evaluation.

## Trigonometry for Physical AI

Trigonometry is essential for spatial reasoning and kinematic calculations.

### Basic Trigonometric Functions
For a right triangle with angle θ:
- **sin(θ)** = opposite/hypotenuse
- **cos(θ)** = adjacent/hypotenuse
- **tan(θ)** = opposite/adjacent

### Applications in Robotics
- **Forward Kinematics**: Calculating end-effector position from joint angles
- **Inverse Kinematics**: Finding joint angles for desired end-effector position
- **Path Planning**: Calculating turning angles and distances

### Important Trigonometric Identities
- **Pythagorean Identity**: sin²(θ) + cos²(θ) = 1
- **Angle Addition**: sin(a±b) = sin(a)cos(b) ± cos(a)sin(b)
- **Double Angle**: sin(2θ) = 2sin(θ)cos(θ)

## Optimization in Physical AI

Optimization is crucial for finding the best solutions to Physical AI problems.

### Linear Programming
Optimizing linear objective functions subject to linear constraints:
```
minimize: cᵀx
subject to: Ax ≤ b, x ≥ 0
```

**Application**: Resource allocation, trajectory optimization with linear constraints.

### Nonlinear Optimization
For more complex problems with nonlinear objective functions or constraints.

**Application**: Trajectory optimization, parameter tuning, control design.

### Gradient Descent
An iterative optimization algorithm that follows the negative gradient of the objective function:
```
x_{k+1} = x_k - α∇f(x_k)
```
where α is the learning rate and ∇f is the gradient.

**Application**: Training neural networks for control, parameter estimation.

## Hands-on Exercise: Mathematical Modeling of a Simple Robot Arm

In this exercise, you'll apply the mathematical concepts to model a simple 2D robot arm with two joints.

### Requirements
- Python 3.8+
- NumPy library
- Basic programming knowledge

### Exercise Steps
1. Define the kinematic model for a 2-link planar robot arm
2. Implement forward kinematics to calculate end-effector position from joint angles
3. Implement basic inverse kinematics to find joint angles for desired end-effector position
4. Visualize the workspace of the robot arm
5. Analyze the mathematical relationships in the system

### Expected Outcome
You should understand how linear algebra, trigonometry, and basic calculus apply to robot kinematics and gain practical experience implementing mathematical models.

### Sample Code Structure
```python
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(theta1, theta2, l1=1.0, l2=1.0):
    """Calculate end-effector position given joint angles"""
    # Calculate position using trigonometry
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def jacobian(theta1, theta2, l1=1.0, l2=1.0):
    """Calculate the Jacobian matrix for the robot arm"""
    # The Jacobian relates joint velocities to end-effector velocities
    # J = [∂x/∂θ1, ∂x/∂θ2]
    #     [∂y/∂θ1, ∂y/∂θ2]
    J = np.array([
        [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)],
        [l1*np.cos(theta1) + l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)]
    ])
    return J

# Test the functions with sample inputs
theta1, theta2 = np.pi/4, np.pi/6  # 45° and 30°
x, y = forward_kinematics(theta1, theta2)
print(f"End-effector position: ({x:.2f}, {y:.2f})")
```

## Summary

This chapter has covered the essential mathematical foundations for Physical AI, including linear algebra for representing physical systems and transformations, calculus for understanding motion and change, probability and statistics for handling uncertainty, trigonometry for spatial reasoning, and optimization techniques for finding optimal solutions. These mathematical tools are fundamental to understanding and developing Physical AI systems, from simple robot arms to complex humanoid robots.

## Key Takeaways

- Linear algebra provides the foundation for representing positions, orientations, and transformations in physical space
- Calculus is essential for understanding motion, dynamics, and change in physical systems
- Probability and statistics are crucial for handling the inherent uncertainty in physical systems
- Trigonometry is fundamental for spatial reasoning and kinematic calculations
- Optimization techniques help find the best solutions to Physical AI problems
- Mathematical modeling is essential for designing, controlling, and analyzing Physical AI systems

## Next Steps

In the next chapter, we'll cover setting up your simulation environment for hands-on experimentation with Physical AI concepts. The mathematical foundations covered in this chapter will be essential as you implement and experiment with physical systems in simulation.

## References and Further Reading

1. Strang, G. (2009). Introduction to Linear Algebra. Wellesley-Cambridge Press.
2. Stewart, J. (2015). Calculus: Early Transcendentals. Cengage Learning.
3. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
4. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). Robot Modeling and Control. John Wiley & Sons.
5. Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.