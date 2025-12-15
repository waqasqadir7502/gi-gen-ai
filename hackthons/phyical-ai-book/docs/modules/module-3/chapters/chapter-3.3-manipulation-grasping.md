# Chapter 3.3: Manipulation and Grasping

## Overview

Manipulation and grasping are fundamental capabilities for humanoid robots, enabling them to interact with objects in their environment. This chapter covers the principles of robotic manipulation, grasp planning, force control, and the integration of perception with manipulation for dexterous object interaction.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the fundamentals of robotic manipulation and grasping
2. Implement grasp planning algorithms for different object types
3. Apply force control techniques for stable grasping
4. Integrate perception with manipulation for object interaction
5. Design manipulation controllers that work with balance constraints

## Introduction to Robotic Manipulation

Robotic manipulation is the process of purposefully controlling the motion and force of a robot's end-effectors to achieve a specific task involving objects in the environment. For humanoid robots, manipulation is particularly challenging due to the need to coordinate multiple limbs while maintaining balance.

### Key Components of Manipulation

#### Kinematic Considerations
- **Forward Kinematics**: Determining end-effector position from joint angles
- **Inverse Kinematics**: Determining joint angles for desired end-effector position
- **Workspace Analysis**: Understanding reachable and dexterous workspace

#### Dynamic Considerations
- **Force Control**: Managing forces applied to objects
- **Impedance Control**: Controlling the robot's mechanical impedance
- **Compliance**: Allowing controlled flexibility in manipulation

#### Grasp Planning
- **Grasp Stability**: Ensuring secure grasp of objects
- **Grasp Quality**: Evaluating the effectiveness of different grasp configurations
- **Force Closure**: Achieving stable grasp through friction and contact forces

### Manipulation Challenges for Humanoid Robots

#### Balance Integration
Humanoid robots must maintain balance while manipulating objects, which creates coupling between locomotion and manipulation tasks.

#### Multi-Limb Coordination
Unlike single-arm robots, humanoids have multiple arms that can be used cooperatively or independently.

#### Human-Shaped Workspace
Humanoid robots have anthropomorphic workspace constraints that match human environments.

## Grasp Types and Classification

### Grasp Taxonomy

#### Power Grasps
- **Cylindrical Grasp**: Wrap fingers around cylindrical objects
- **Spherical Grasp**: Enclose spherical objects
- **Hook Grasp**: Use fingertips to carry handles or straps

#### Precision Grasps
- **Tip Pinch**: Grasp with thumb and finger tips
- **Lateral Pinch**: Grasp between thumb and side of index finger
- **Three-Finger Pinch**: Grasp with thumb and two fingers

### Grasp Stability Analysis

```python
import numpy as np
from scipy.spatial import ConvexHull
import math

class GraspAnalyzer:
    """Analyze grasp stability and quality"""

    def __init__(self):
        self.friction_coefficient = 0.8  # Typical for rubber-like materials

    def compute_force_closure(self, contact_points, normals, friction_coeff):
        """
        Check if a grasp achieves force closure
        Force closure: the grasp can resist any external wrench
        """
        # Contact points: list of 3D positions
        # Normals: list of 3D normal vectors pointing into the object
        # friction_coeff: coefficient of friction at contact points

        n_contacts = len(contact_points)
        if n_contacts < 2:
            return False  # Need at least 2 contacts for any stability

        # For 2D planar grasps, minimum 3 contacts needed for force closure
        # For 3D spatial grasps, minimum 7 contacts needed for force closure
        # (or 4 contacts in special configurations)

        # Form the grasp matrix G
        # Each contact contributes to G with its normal and friction cone
        # For simplicity, we'll check for 2D planar case first

        if n_contacts >= 3:
            # For 2D planar grasps, check if the origin is inside the convex hull
            # of the contact wrenches
            wrenches = []
            for i in range(n_contacts):
                pos = contact_points[i][:2]  # 2D position
                normal = normals[i][:2]      # 2D normal
                normal = normal / np.linalg.norm(normal)  # normalize

                # Friction cone edges (in 2D)
                tangent = np.array([-normal[1], normal[0]])  # Perpendicular to normal

                # Force in normal direction
                wrench_normal = np.concatenate([normal, [0]])  # [fx, fy, tau]
                wrench_tangent1 = np.concatenate([friction_coeff * tangent, [0]])
                wrench_tangent2 = np.concatenate([-friction_coeff * tangent, [0]])

                # Torque from this contact
                torque1 = friction_coeff * tangent[0] * pos[1] - friction_coeff * tangent[1] * pos[0]
                torque2 = -friction_coeff * tangent[0] * pos[1] - (-friction_coeff * tangent[1]) * pos[0]

                wrench_tangent1[2] = torque1
                wrench_tangent2[2] = torque2

                wrenches.extend([wrench_normal, wrench_tangent1, wrench_tangent2])

            # Check if origin is inside convex hull of wrenches
            try:
                hull = ConvexHull(np.array(wrenches))
                # For simplicity, we'll say if we have 3+ contacts and positive area,
                # it's likely to have force closure
                return True
            except:
                return False
        else:
            return False

    def compute_grasp_quality(self, contact_points, normals, forces):
        """
        Compute grasp quality metric based on force transmission
        Higher values indicate better grasps
        """
        if len(contact_points) == 0:
            return 0.0

        # Compute the grasp matrix
        # This is a simplified version - full implementation would be more complex
        n_contacts = len(contact_points)

        # Quality metric: minimum force required to resist external wrenches
        # This is related to the smallest eigenvalue of the grasp matrix
        # For simplicity, we'll use a geometric approach

        # Calculate the grasp quality based on contact geometry
        total_normal_force = sum(np.linalg.norm(force) for force in forces)
        if total_normal_force == 0:
            return 0.0

        # Calculate how well the grasp distributes forces
        force_distribution = 0.0
        for i, force in enumerate(forces):
            force_mag = np.linalg.norm(force)
            if force_mag > 0:
                force_distribution += (force_mag / total_normal_force) ** 2

        # Inverse of force_distribution (lower is better)
        if force_distribution > 0:
            distribution_quality = 1.0 / (n_contacts * force_distribution)
        else:
            distribution_quality = 0.0

        # Geometric quality: how well contacts are distributed
        if n_contacts >= 2:
            # Calculate average distance between contacts
            total_distance = 0.0
            n_pairs = 0
            for i in range(n_contacts):
                for j in range(i + 1, n_contacts):
                    dist = np.linalg.norm(np.array(contact_points[i]) - np.array(contact_points[j]))
                    total_distance += dist
                    n_pairs += 1

            if n_pairs > 0:
                avg_distance = total_distance / n_pairs
                geometric_quality = min(avg_distance, 0.1) * 10  # Normalize
            else:
                geometric_quality = 0.0
        else:
            geometric_quality = 0.0

        # Combine qualities
        quality = (distribution_quality + geometric_quality) / 2.0
        return min(quality, 1.0)  # Clamp to [0, 1]

    def evaluate_grasp_candidate(self, object_mesh, grasp_pose):
        """
        Evaluate a potential grasp configuration
        """
        # Extract grasp parameters from pose
        position = grasp_pose[:3]
        orientation = grasp_pose[3:]  # Quaternion

        # Calculate contact points based on grasp pose and object geometry
        contact_points, normals = self.calculate_contact_points(object_mesh, grasp_pose)

        # Calculate required forces for stability
        required_forces = self.calculate_stable_forces(contact_points, normals)

        # Compute quality metrics
        quality = self.compute_grasp_quality(contact_points, normals, required_forces)
        force_closure = self.compute_force_closure(contact_points, normals, self.friction_coefficient)

        return {
            'quality': quality,
            'force_closure': force_closure,
            'contact_points': contact_points,
            'normals': normals,
            'required_forces': required_forces
        }

    def calculate_contact_points(self, object_mesh, grasp_pose):
        """Calculate contact points for a given grasp pose"""
        # This would involve complex geometry processing
        # For simplicity, we'll return a fixed number of contact points
        # based on the grasp type and object shape

        position = grasp_pose[:3]
        orientation = grasp_pose[3:]

        # Convert quaternion to rotation matrix
        R = self.quaternion_to_rotation_matrix(orientation)

        # Define contact points relative to grasp frame
        # This would depend on the specific grasp type
        rel_contacts = [
            np.array([0.02, 0.02, 0]),   # Thumb contact
            np.array([0.02, -0.02, 0]),  # Index finger contact
            np.array([-0.02, 0, 0])      # Middle finger contact
        ]

        # Transform to world coordinates
        contact_points = []
        normals = []
        for rel_contact in rel_contacts:
            world_contact = position + R @ rel_contact
            contact_points.append(world_contact)
            # Normals point inward toward object center
            normal = -R @ np.array([1, 0, 0])  # Assuming grasp direction
            normals.append(normal / np.linalg.norm(normal))

        return contact_points, normals

    def calculate_stable_forces(self, contact_points, normals):
        """Calculate forces needed for stable grasp"""
        # This is a simplified calculation
        # In reality, this would involve solving the grasp force optimization problem
        forces = []
        for normal in normals:
            # Apply normal force to hold object weight
            # Assuming object weight is 1N for simplicity
            force_magnitude = 5.0  # 5N normal force per contact
            force = force_magnitude * normal
            forces.append(force)
        return forces

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

# Example usage
def example_grasp_analysis():
    analyzer = GraspAnalyzer()

    # Define a simple grasp pose [x, y, z, qw, qx, qy, qz]
    grasp_pose = np.array([0.5, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0])  # Identity rotation

    # Create a dummy object mesh (simplified)
    object_mesh = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

    # Evaluate the grasp
    result = analyzer.evaluate_grasp_candidate(object_mesh, grasp_pose)

    print(f"Grasp Quality: {result['quality']:.3f}")
    print(f"Force Closure: {result['force_closure']}")
    print(f"Contact Points: {result['contact_points']}")
    print(f"Required Forces: {result['required_forces']}")

    return result
```

## Grasp Planning Algorithms

### Sampling-Based Grasp Planning

```python
class GraspPlanner:
    """Sample-based grasp planner for unknown objects"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.grasp_analyzer = GraspAnalyzer()
        self.n_samples = 1000  # Number of grasp candidates to sample

    def plan_grasps(self, object_mesh, approach_direction=None):
        """
        Plan multiple grasp candidates for an object
        """
        grasp_candidates = []

        for i in range(self.n_samples):
            # Sample a random grasp pose
            grasp_pose = self.sample_grasp_pose(object_mesh, approach_direction)

            # Evaluate the grasp
            evaluation = self.grasp_analyzer.evaluate_grasp_candidate(object_mesh, grasp_pose)

            # Only keep good grasps
            if evaluation['quality'] > 0.3 and evaluation['force_closure']:
                grasp_candidates.append({
                    'pose': grasp_pose,
                    'evaluation': evaluation,
                    'score': evaluation['quality']
                })

        # Sort by quality score
        grasp_candidates.sort(key=lambda x: x['score'], reverse=True)

        return grasp_candidates[:10]  # Return top 10 grasps

    def sample_grasp_pose(self, object_mesh, approach_direction=None):
        """
        Sample a grasp pose near the object
        """
        # Find the centroid of the object
        centroid = np.mean(object_mesh, axis=0)

        # Sample position near the object
        r = 0.1  # 10cm from surface
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)

        # Position offset from centroid
        offset = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])

        position = centroid + offset

        # Sample orientation
        # For a simple parallel jaw gripper, we want the gripper axis
        # pointing toward the object
        gripper_approach = -offset / np.linalg.norm(offset)

        # Create a random rotation around the approach axis
        rotation_axis = gripper_approach
        if np.allclose(rotation_axis, [0, 0, 1]):
            # Avoid gimbal lock
            rotation_axis = np.array([1, 0, 0])

        # Create a rotation matrix with the approach direction
        z_axis = gripper_approach
        x_axis = np.array([1, 0, 0])
        if np.allclose(np.cross(z_axis, x_axis), [0, 0, 0]):
            x_axis = np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)

        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion
        quaternion = self.rotation_matrix_to_quaternion(R)

        return np.concatenate([position, quaternion])

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Method from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        quaternion = np.array([w, x, y, z])
        return quaternion / np.linalg.norm(quaternion)

# Example usage
def example_grasp_planning():
    planner = GraspPlanner(robot_model=None)  # Dummy robot model

    # Create a simple object mesh (cube)
    object_mesh = np.array([
        [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1],
        [0.1, 0.1, 0], [0.1, 0, 0.1], [0, 0.1, 0.1], [0.1, 0.1, 0.1]
    ])

    # Plan grasps
    grasps = planner.plan_grasps(object_mesh)

    print(f"Found {len(grasps)} stable grasps")
    if grasps:
        best_grasp = grasps[0]
        print(f"Best grasp quality: {best_grasp['score']:.3f}")
        print(f"Grasp position: [{best_grasp['pose'][0]:.2f}, {best_grasp['pose'][1]:.2f}, {best_grasp['pose'][2]:.2f}]")

    return grasps
```

### Learning-Based Grasp Planning

```python
class LearningBasedGraspPlanner:
    """Grasp planner using learned models from data"""

    def __init__(self):
        self.model_trained = False
        self.training_data = []
        self.grasp_features = []

    def extract_grasp_features(self, object_shape, grasp_pose):
        """
        Extract features for grasp evaluation
        """
        features = {}

        # Geometric features
        pos = grasp_pose[:3]
        orientation = grasp_pose[3:]

        # Distance to object center
        object_center = np.mean(object_shape, axis=0)
        features['distance_to_center'] = np.linalg.norm(pos - object_center)

        # Object dimensions at grasp location
        object_dims = np.max(object_shape, axis=0) - np.min(object_shape, axis=0)
        features['object_size'] = np.mean(object_dims)

        # Orientation alignment with object principal axes
        # (simplified - in practice this would use PCA of object points)
        features['orientation_aligned'] = 1.0  # Placeholder

        # Contact point distribution
        analyzer = GraspAnalyzer()
        contact_points, normals = analyzer.calculate_contact_points(object_shape, grasp_pose)
        if len(contact_points) >= 2:
            distances = []
            for i in range(len(contact_points)):
                for j in range(i + 1, len(contact_points)):
                    dist = np.linalg.norm(np.array(contact_points[i]) - np.array(contact_points[j]))
                    distances.append(dist)
            features['contact_spread'] = np.mean(distances) if distances else 0

        return features

    def train_grasp_predictor(self, training_examples):
        """
        Train a grasp quality predictor from examples
        training_examples: list of (object_shape, grasp_pose, quality_label)
        """
        X = []  # Feature vectors
        y = []  # Quality labels

        for obj_shape, grasp_pose, quality in training_examples:
            features = self.extract_grasp_features(obj_shape, grasp_pose)
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(quality)

        # In a real implementation, we would train a classifier/regressor
        # For this example, we'll just store the training data
        self.training_data = list(zip(X, y))
        self.model_trained = True

        print(f"Trained grasp predictor on {len(training_examples)} examples")

    def predict_grasp_quality(self, object_shape, grasp_pose):
        """
        Predict grasp quality using trained model
        """
        if not self.model_trained:
            # Fallback to geometric analysis
            analyzer = GraspAnalyzer()
            result = analyzer.evaluate_grasp_candidate(object_shape, grasp_pose)
            return result['quality']

        # Extract features
        features = self.extract_grasp_features(object_shape, grasp_pose)
        feature_vector = list(features.values())

        # Find similar grasps in training data (nearest neighbor approach)
        min_dist = float('inf')
        best_quality = 0.0

        for train_features, train_quality in self.training_data:
            dist = np.sum((np.array(feature_vector) - np.array(train_features)) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_quality = train_quality

        return best_quality

    def plan_grasps_with_learning(self, object_shape, n_candidates=50):
        """
        Plan grasps using learned quality prediction
        """
        planner = GraspPlanner(robot_model=None)

        # Generate random grasp candidates
        grasp_candidates = []
        for i in range(n_candidates):
            grasp_pose = planner.sample_grasp_pose(object_shape)
            quality = self.predict_grasp_quality(object_shape, grasp_pose)

            grasp_candidates.append({
                'pose': grasp_pose,
                'predicted_quality': quality
            })

        # Sort by predicted quality
        grasp_candidates.sort(key=lambda x: x['predicted_quality'], reverse=True)

        return grasp_candidates[:10]  # Return top 10

# Example usage
def example_learning_based_grasping():
    # Create a simple learning-based grasp planner
    learner = LearningBasedGraspPlanner()

    # Generate some training data (simplified)
    training_examples = []
    for i in range(100):
        # Create random object (cube with random size)
        obj_size = np.random.uniform(0.05, 0.2, 3)
        object_shape = np.random.rand(8, 3) * obj_size

        # Create random grasp pose
        grasp_pose = np.random.rand(7)
        grasp_pose[3:] = grasp_pose[3:] / np.linalg.norm(grasp_pose[3:])  # Normalize quaternion

        # Assign a quality based on simple geometric rules
        quality = np.random.rand()  # Random quality for this example

        training_examples.append((object_shape, grasp_pose, quality))

    # Train the model
    learner.train_grasp_predictor(training_examples)

    # Test on a new object
    test_object = np.array([
        [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1],
        [0.1, 0.1, 0], [0.1, 0, 0.1], [0, 0.1, 0.1], [0.1, 0.1, 0.1]
    ])

    # Plan grasps
    grasps = learner.plan_grasps_with_learning(test_object)

    print(f"Learning-based grasp planning found {len(grasps)} grasps")
    if grasps:
        print(f"Best predicted quality: {grasps[0]['predicted_quality']:.3f}")

    return grasps
```

## Force Control and Compliance

### Impedance Control

Impedance control regulates the dynamic relationship between position and force, making the robot behave like a mechanical system with specified mass, damping, and stiffness.

```python
class ImpedanceController:
    """Impedance controller for compliant manipulation"""

    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0, dt=0.01):
        self.mass = mass          # Desired mass (kg)
        self.damping = damping    # Desired damping (Ns/m)
        self.stiffness = stiffness  # Desired stiffness (N/m)
        self.dt = dt              # Control timestep (s)

        # Desired impedance parameters
        self.M_d = mass * np.eye(3)      # Desired mass matrix
        self.B_d = damping * np.eye(3)   # Desired damping matrix
        self.K_d = stiffness * np.eye(3) # Desired stiffness matrix

        # State variables
        self.x_d = np.zeros(3)    # Desired position
        self.x = np.zeros(3)      # Actual position
        self.v = np.zeros(3)      # Actual velocity
        self.f_cmd = np.zeros(3)  # Commanded force

    def update(self, x_act, f_ext):
        """
        Update impedance controller
        x_act: actual end-effector position
        f_ext: external force applied to end-effector
        """
        # Update actual state
        self.x = x_act

        # Calculate velocity (finite difference)
        if hasattr(self, 'x_prev'):
            self.v = (self.x - self.x_prev) / self.dt
        else:
            self.v = np.zeros(3)

        # Calculate position and velocity errors
        x_err = self.x_d - self.x
        v_err = -self.v  # Assuming desired velocity is 0

        # Impedance control law:
        # M_d * (xdd_d) + B_d * (xd_d) + K_d * (x_d) = F_cmd
        # Rearranged: F_cmd = M_d * xdd_d + B_d * xd_d + K_d * x_d
        # For tracking: F_cmd = K_d * x_err + B_d * v_err + M_d * xdd_d
        # Assuming xdd_d = 0 for pure impedance behavior:
        self.f_cmd = self.K_d @ x_err + self.B_d @ v_err

        # Store position for next velocity calculation
        self.x_prev = self.x.copy()

        return self.f_cmd

    def set_desired_position(self, x_d):
        """Set desired position for the impedance controller"""
        self.x_d = np.array(x_d)

    def set_impedance_parameters(self, mass, damping, stiffness):
        """Update impedance parameters"""
        self.M_d = mass * np.eye(3)
        self.B_d = damping * np.eye(3)
        self.K_d = stiffness * np.eye(3)
```

### Admittance Control

Admittance control is the dual of impedance control, where force input produces position output.

```python
class AdmittanceController:
    """Admittance controller for force-guided motion"""

    def __init__(self, stiffness=100.0, damping=10.0, dt=0.01):
        self.stiffness = stiffness  # Admittance stiffness (m/N)
        self.damping = damping      # Admittance damping (m*s/N)
        self.dt = dt               # Control timestep (s)

        # Admittance parameters (inverse of impedance)
        self.alpha_pos = 1.0 / stiffness  # Position admittance
        self.alpha_vel = 1.0 / damping    # Velocity admittance

        # State variables
        self.x = np.zeros(3)      # Current position
        self.v = np.zeros(3)      # Current velocity
        self.f_ext = np.zeros(3)  # External force

    def update(self, f_ext, x_desired=None):
        """
        Update admittance controller
        f_ext: external force applied to end-effector
        x_desired: optional desired position (for hybrid control)
        """
        self.f_ext = f_ext

        # Admittance control law: v_dot = alpha_pos * f_ext + alpha_vel * (x_des - x)
        if x_desired is not None:
            # Hybrid position/force control
            pos_error = np.array(x_desired) - self.x
            acceleration = self.alpha_pos * self.f_ext + self.alpha_vel * pos_error
        else:
            # Pure force-guided motion
            acceleration = self.alpha_pos * self.f_ext

        # Integrate to get new velocity and position
        new_v = self.v + acceleration * self.dt
        new_x = self.x + new_v * self.dt

        # Update state
        self.v = new_v
        self.x = new_x

        return self.x.copy(), self.v.copy()

    def set_admittance_parameters(self, stiffness, damping):
        """Update admittance parameters"""
        self.alpha_pos = 1.0 / stiffness
        self.alpha_vel = 1.0 / damping
```

### Hybrid Position/Force Control

Hybrid position/force control allows simultaneous control of position in unconstrained directions and force in constrained directions.

```python
class HybridPositionForceController:
    """Hybrid position/force controller"""

    def __init__(self, dt=0.01):
        self.dt = dt
        self.Kp_pos = 100.0  # Position control gain
        self.Kd_pos = 20.0   # Position derivative gain
        self.Kp_force = 50.0 # Force control gain
        self.Kd_force = 10.0 # Force derivative gain

        # Constraint selection matrix (identity = no constraints)
        self.Sigma = np.eye(6)  # 6 DOF: 3 position + 3 orientation

        # State variables
        self.x = np.zeros(3)      # Current position
        self.x_d = np.zeros(3)    # Desired position
        self.f = np.zeros(3)      # Current force
        self.f_d = np.zeros(3)    # Desired force

    def set_constraint_frame(self, constraint_axes):
        """
        Set constraint frame where certain DOFs are position-controlled
        and others are force-controlled
        constraint_axes: 6-element array where 1=position control, 0=force control
        """
        self.Sigma = np.diag(constraint_axes)

    def update(self, x_act, f_ext, x_des, f_des):
        """
        Update hybrid controller
        x_act: actual position
        f_ext: external force
        x_des: desired position
        f_des: desired force
        """
        self.x = x_act
        self.f = f_ext
        self.x_d = x_des
        self.f_d = f_des

        # Calculate errors
        x_err = self.x_d - self.x
        f_err = self.f_d - self.f

        # Calculate control commands in constrained space
        pos_cmd = self.Kp_pos * x_err
        force_cmd = self.Kp_force * f_err

        # Combine position and force control using selection matrix
        # This is a simplified version - full implementation would be more complex
        cmd = self.Sigma @ pos_cmd + (np.eye(6) - self.Sigma) @ force_cmd

        # Return control command (would be converted to joint torques in real implementation)
        return cmd[:3]  # Just return position part for simplicity

    def set_constraint_axes(self, position_controlled_axes):
        """
        Set which axes are position controlled (1) vs force controlled (0)
        position_controlled_axes: array of 6 elements [x, y, z, rx, ry, rz]
        """
        self.Sigma = np.diag(position_controlled_axes)
```

## Integration with Perception

### Object Recognition and Pose Estimation

```python
class PerceptionIntegratedManipulator:
    """Manipulation system integrated with perception"""

    def __init__(self):
        self.object_detector = self.initialize_object_detector()
        self.pose_estimator = self.initialize_pose_estimator()
        self.grasp_planner = GraspPlanner(robot_model=None)
        self.motion_controller = self.initialize_motion_controller()

        # Robot state
        self.end_effector_pose = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.joint_angles = np.zeros(7)       # Example 7-DOF arm

    def initialize_object_detector(self):
        """Initialize object detection module"""
        # In a real implementation, this would load a trained model
        # For this example, we'll simulate detection
        return lambda: self.simulate_object_detection()

    def initialize_pose_estimator(self):
        """Initialize object pose estimation"""
        # In a real implementation, this would estimate 6DOF poses
        return lambda: self.simulate_pose_estimation()

    def initialize_motion_controller(self):
        """Initialize motion controller"""
        return MotionController({'height': 0.8})

    def simulate_object_detection(self):
        """Simulate object detection from sensor data"""
        # Return a list of detected object types
        return ['cup', 'box', 'bottle']

    def simulate_pose_estimation(self):
        """Simulate object pose estimation"""
        # Return dictionary of object poses {object_type: pose}
        return {
            'cup': np.array([0.5, 0.3, 0.1, 1.0, 0.0, 0.0, 0.0]),  # [x, y, z, qw, qx, qy, qz]
            'box': np.array([0.7, 0.2, 0.15, 0.707, 0.0, 0.0, 0.707]),
            'bottle': np.array([0.4, 0.6, 0.12, 1.0, 0.0, 0.0, 0.0])
        }

    def pick_and_place_task(self, target_object, target_location):
        """
        Execute a pick-and-place task
        target_object: type of object to pick
        target_location: [x, y, z] position to place the object
        """
        # 1. Detect and estimate object pose
        print(f"Detecting {target_object}...")
        detected_objects = self.object_detector()
        object_poses = self.pose_estimator()

        if target_object not in object_poses:
            print(f"{target_object} not found!")
            return False

        object_pose = object_poses[target_object]
        print(f"Found {target_object} at position: [{object_pose[0]:.2f}, {object_pose[1]:.2f}, {object_pose[2]:.2f}]")

        # 2. Plan grasp for the object
        print("Planning grasp...")
        # Create a simple mesh for the object (in practice, this would come from perception)
        object_mesh = self.create_simple_mesh(object_pose, target_object)
        grasps = self.grasp_planner.plan_grasps(object_mesh)

        if not grasps:
            print("No suitable grasps found!")
            return False

        best_grasp = grasps[0]
        print(f"Selected grasp with quality: {best_grasp['score']:.3f}")

        # 3. Generate approach trajectory
        print("Generating approach trajectory...")
        approach_poses = self.generate_approach_trajectory(object_pose, best_grasp['pose'])

        # 4. Execute approach and grasp
        print("Executing approach and grasp...")
        success = self.execute_grasp(approach_poses, best_grasp['pose'])

        if not success:
            print("Grasp failed!")
            return False

        print("Successfully grasped object!")

        # 5. Generate transport trajectory to target location
        print("Generating transport trajectory...")
        transport_poses = self.generate_transport_trajectory(target_location)

        # 6. Execute transport and placement
        print("Executing transport and placement...")
        success = self.execute_placement(transport_poses, target_location)

        if success:
            print("Successfully placed object!")
            return True
        else:
            print("Placement failed!")
            return False

    def create_simple_mesh(self, object_pose, object_type):
        """Create a simple mesh representation of an object"""
        # This is a simplified version - in practice, this would come from perception
        if object_type == 'cup':
            # Create a simple cup mesh
            mesh = np.array([
                [object_pose[0], object_pose[1], object_pose[2]],
                [object_pose[0] + 0.05, object_pose[1], object_pose[2]],
                [object_pose[0], object_pose[1] + 0.05, object_pose[2]],
                [object_pose[0], object_pose[1], object_pose[2] + 0.1],
            ])
        elif object_type == 'box':
            # Create a simple box mesh
            size = 0.05
            mesh = np.array([
                [object_pose[0], object_pose[1], object_pose[2]],
                [object_pose[0] + size, object_pose[1], object_pose[2]],
                [object_pose[0], object_pose[1] + size, object_pose[2]],
                [object_pose[0], object_pose[1], object_pose[2] + size],
                [object_pose[0] + size, object_pose[1] + size, object_pose[2]],
                [object_pose[0] + size, object_pose[1], object_pose[2] + size],
                [object_pose[0], object_pose[1] + size, object_pose[2] + size],
                [object_pose[0] + size, object_pose[1] + size, object_pose[2] + size],
            ])
        else:  # bottle or other
            mesh = np.array([
                [object_pose[0], object_pose[1], object_pose[2]],
                [object_pose[0] + 0.03, object_pose[1], object_pose[2]],
                [object_pose[0], object_pose[1] + 0.03, object_pose[2]],
                [object_pose[0], object_pose[1], object_pose[2] + 0.15],
            ])

        return mesh

    def generate_approach_trajectory(self, object_pose, grasp_pose):
        """Generate approach trajectory from current position to grasp"""
        # Start from current end-effector position
        start_pos = self.end_effector_pose[:3]
        grasp_pos = grasp_pose[:3]

        # Calculate approach point (above the grasp point)
        approach_offset = np.array([0, 0, 0.1])  # 10cm above
        approach_pos = grasp_pos + approach_offset

        # Generate linear trajectory
        n_points = 20
        trajectory = []
        for i in range(n_points + 1):
            t = i / n_points
            if i < n_points * 0.7:  # First 70%: move to approach point
                pos = start_pos + t * 0.7 * (approach_pos - start_pos) / 0.7
            else:  # Last 30%: move from approach to grasp
                t2 = (t - 0.7) / 0.3  # Remap to [0, 1]
                pos = approach_pos + t2 * (grasp_pos - approach_pos)

            # Keep orientation constant for simplicity
            pose = np.concatenate([pos, self.end_effector_pose[3:]])
            trajectory.append(pose)

        return trajectory

    def execute_grasp(self, approach_trajectory, grasp_pose):
        """Execute the grasp motion"""
        # Follow approach trajectory
        for pose in approach_trajectory:
            success = self.motion_controller.move_to(pose[:3], self.end_effector_pose)
            if not success:
                return False

        # Execute grasp (close gripper)
        success = self.execute_gripper_close()
        if not success:
            return False

        # Lift slightly after grasp
        lift_pose = grasp_pose.copy()
        lift_pose[2] += 0.05  # Lift 5cm
        success = self.motion_controller.move_to(lift_pose[:3], self.end_effector_pose)
        if not success:
            return False

        return True

    def execute_gripper_close(self):
        """Close the gripper to grasp the object"""
        # In a real implementation, this would send commands to the gripper
        # For simulation, we'll just return success
        print("Closing gripper...")
        return True

    def generate_transport_trajectory(self, target_location):
        """Generate trajectory to transport object to target location"""
        current_pos = self.end_effector_pose[:3]

        # Go to a safe height first (avoid collisions)
        safe_height = max(current_pos[2], target_location[2]) + 0.2
        safe_pos = np.array([current_pos[0], current_pos[1], safe_height])

        # Then move to target XY at safe height
        target_at_safe_height = np.array([target_location[0], target_location[1], safe_height])

        # Finally, descend to target location
        target_pos = np.array(target_location)

        # Generate trajectory
        trajectory = []

        # Move to safe height
        for i in range(10):
            t = i / 9
            pos = current_pos + t * (safe_pos - current_pos)
            pose = np.concatenate([pos, self.end_effector_pose[3:]])
            trajectory.append(pose)

        # Move to target XY at safe height
        for i in range(10):
            t = i / 9
            pos = safe_pos + t * (target_at_safe_height - safe_pos)
            pose = np.concatenate([pos, self.end_effector_pose[3:]])
            trajectory.append(pose)

        # Descend to target
        for i in range(10):
            t = i / 9
            pos = target_at_safe_height + t * (target_pos - target_at_safe_height)
            pose = np.concatenate([pos, self.end_effector_pose[3:]])
            trajectory.append(pose)

        return trajectory

    def execute_placement(self, transport_trajectory, target_location):
        """Execute placement of object at target location"""
        # Follow transport trajectory
        for pose in transport_trajectory:
            success = self.motion_controller.move_to(pose[:3], self.end_effector_pose)
            if not success:
                return False

        # Open gripper to release object
        success = self.execute_gripper_open()
        if not success:
            return False

        # Lift away from placed object
        lift_pos = np.array(target_location) + np.array([0, 0, 0.1])
        success = self.motion_controller.move_to(lift_pos, self.end_effector_pose)
        if not success:
            return False

        return True

    def execute_gripper_open(self):
        """Open the gripper to release the object"""
        # In a real implementation, this would send commands to the gripper
        # For simulation, we'll just return success
        print("Opening gripper...")
        return True

# Example usage
def example_perception_integrated_manipulation():
    """Example of perception-integrated manipulation"""
    manipulator = PerceptionIntegratedManipulator()

    # Execute a pick-and-place task
    success = manipulator.pick_and_place_task('cup', [0.8, 0.8, 0.2])

    if success:
        print("Pick-and-place task completed successfully!")
    else:
        print("Pick-and-place task failed!")

    return success
```

## Hands-on Exercise: Implementing a Grasp Controller

In this exercise, you'll implement a complete grasp controller that integrates perception, planning, and control for a humanoid robot.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of robotics concepts

### Exercise Steps
1. Implement a grasp planning algorithm
2. Create a force control system for stable grasping
3. Integrate perception for object detection and pose estimation
4. Test the system with different objects and scenarios
5. Analyze grasp success rates and stability metrics

### Expected Outcome
You should have a working grasp controller that can detect objects, plan stable grasps, execute the grasp with appropriate force control, and release objects at desired locations.

### Sample Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraspControllerExercise:
    """Hands-on exercise for grasp controller implementation"""

    def __init__(self):
        # Initialize components
        self.grasp_analyzer = GraspAnalyzer()
        self.impedance_controller = ImpedanceController()
        self.perception_system = self.initialize_perception()

        # Robot state
        self.end_effector_pos = np.array([0.5, 0.0, 0.5])
        self.end_effector_orient = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.gripper_open = True

        # Objects in environment
        self.objects = [
            {'type': 'cube', 'position': [0.4, 0.3, 0.1], 'size': [0.05, 0.05, 0.05]},
            {'type': 'cylinder', 'position': [0.6, 0.4, 0.1], 'radius': 0.03, 'height': 0.1},
            {'type': 'sphere', 'position': [0.3, 0.6, 0.1], 'radius': 0.04}
        ]

        # Visualization
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

    def initialize_perception(self):
        """Initialize perception system"""
        return {
            'objects': self.objects,
            'detect_objects': self.simulate_object_detection
        }

    def simulate_object_detection(self):
        """Simulate object detection from sensors"""
        # In a real system, this would process sensor data
        # For simulation, we'll return the known objects
        return self.objects

    def create_object_mesh(self, obj):
        """Create a mesh representation of an object"""
        if obj['type'] == 'cube':
            size = obj['size']
            pos = obj['position']
            # Create cube vertices
            vertices = np.array([
                [pos[0] - size[0]/2, pos[1] - size[1]/2, pos[2] - size[2]/2],
                [pos[0] + size[0]/2, pos[1] - size[1]/2, pos[2] - size[2]/2],
                [pos[0] - size[0]/2, pos[1] + size[1]/2, pos[2] - size[2]/2],
                [pos[0] + size[0]/2, pos[1] + size[1]/2, pos[2] - size[2]/2],
                [pos[0] - size[0]/2, pos[1] - size[1]/2, pos[2] + size[2]/2],
                [pos[0] + size[0]/2, pos[1] - size[1]/2, pos[2] + size[2]/2],
                [pos[0] - size[0]/2, pos[1] + size[1]/2, pos[2] + size[2]/2],
                [pos[0] + size[0]/2, pos[1] + size[1]/2, pos[2] + size[2]/2]
            ])
        elif obj['type'] == 'cylinder':
            pos = obj['position']
            radius = obj['radius']
            height = obj['height']
            # Create cylinder points
            n_points = 16
            vertices = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = pos[0] + radius * np.cos(angle)
                y = pos[1] + radius * np.sin(angle)
                z_bottom = pos[2]
                z_top = pos[2] + height
                vertices.append([x, y, z_bottom])
                vertices.append([x, y, z_top])
            vertices = np.array(vertices)
        elif obj['type'] == 'sphere':
            pos = obj['position']
            radius = obj['radius']
            # Create sphere points (simplified as octahedron)
            vertices = np.array([
                [pos[0] + radius, pos[1], pos[2]],
                [pos[0] - radius, pos[1], pos[2]],
                [pos[0], pos[1] + radius, pos[2]],
                [pos[0], pos[1] - radius, pos[2]],
                [pos[0], pos[1], pos[2] + radius],
                [pos[0], pos[1], pos[2] - radius]
            ])
        else:
            # Default to cube
            vertices = np.array([
                [obj['position'][0] - 0.025, obj['position'][1] - 0.025, obj['position'][2] - 0.025],
                [obj['position'][0] + 0.025, obj['position'][1] - 0.025, obj['position'][2] - 0.025],
                [obj['position'][0] - 0.025, obj['position'][1] + 0.025, obj['position'][2] - 0.025],
                [obj['position'][0] + 0.025, obj['position'][1] + 0.025, obj['position'][2] - 0.025],
                [obj['position'][0] - 0.025, obj['position'][1] - 0.025, obj['position'][2] + 0.025],
                [obj['position'][0] + 0.025, obj['position'][1] - 0.025, obj['position'][2] + 0.025],
                [obj['position'][0] - 0.025, obj['position'][1] + 0.025, obj['position'][2] + 0.025],
                [obj['position'][0] + 0.025, obj['position'][1] + 0.025, obj['position'][2] + 0.025]
            ])

        return vertices

    def plan_grasp_for_object(self, obj):
        """Plan a grasp for a specific object"""
        # Create object mesh
        object_mesh = self.create_object_mesh(obj)

        # Plan grasps using the grasp planner
        planner = GraspPlanner(robot_model=None)
        grasps = planner.plan_grasps(object_mesh)

        if grasps:
            # Return the best grasp
            return grasps[0]
        else:
            return None

    def execute_grasp_sequence(self, target_obj_idx):
        """Execute the complete grasp sequence for an object"""
        if target_obj_idx >= len(self.objects):
            print("Invalid object index!")
            return False

        target_obj = self.objects[target_obj_idx]
        print(f"Attempting to grasp {target_obj['type']} at {target_obj['position']}")

        # 1. Plan grasp
        best_grasp = self.plan_grasp_for_object(target_obj)
        if not best_grasp:
            print("Could not find a suitable grasp!")
            return False

        print(f"Planned grasp with quality: {best_grasp['score']:.3f}")

        # 2. Generate approach trajectory
        grasp_pos = best_grasp['pose'][:3]
        approach_pos = grasp_pos + np.array([0, 0, 0.1])  # 10cm above

        # Move to approach position
        print("Moving to approach position...")
        self.move_to_position(approach_pos)

        # Move down to grasp position
        print("Moving to grasp position...")
        self.move_to_position(grasp_pos)

        # 3. Execute grasp with force control
        print("Executing grasp with force control...")
        success = self.execute_force_controlled_grasp()
        if not success:
            print("Grasp failed due to force control!")
            return False

        print("Successfully grasped object!")
        return True

    def move_to_position(self, target_pos):
        """Move end effector to target position with simple control"""
        current_pos = self.end_effector_pos.copy()

        # Simple linear interpolation
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            pos = current_pos + t * (target_pos - current_pos)
            self.end_effector_pos = pos

            # Update visualization
            if i % 5 == 0:  # Update every 5 steps
                self.visualize_state()
                plt.pause(0.05)

    def execute_force_controlled_grasp(self):
        """Execute grasp with force control"""
        # Apply normal forces to grasp the object
        # In a real system, we would monitor force sensors
        # and adjust grip force accordingly

        # Simulate force control
        target_force = 5.0  # Newtons
        current_force = 0.0

        for i in range(50):  # 50 control steps
            # Simulate force feedback
            force_error = target_force - current_force
            force_increment = 0.1 * force_error  # Simple PI control
            current_force += force_increment

            # Check if forces are stable
            if abs(force_error) < 0.1:
                break

            # Update visualization
            if i % 10 == 0:
                self.visualize_state()
                plt.pause(0.01)

        # Close gripper
        self.gripper_open = False

        # Check if grasp is stable (simulated)
        return current_force > 3.0  # Require minimum force for stability

    def visualize_state(self):
        """Visualize the current state of the manipulation scene"""
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot 1: 3D view of scene
        # Plot objects
        for obj in self.objects:
            if obj['type'] == 'cube':
                size = obj['size']
                pos = obj['position']
                # Draw cube
                x = [pos[0] - size[0]/2, pos[0] + size[0]/2]
                y = [pos[1] - size[1]/2, pos[1] + size[1]/2]
                z = [pos[2] - size[2]/2, pos[2] + size[2]/2]

                # Draw cube faces
                for xi in x:
                    self.ax1.plot([xi, xi], [y[0], y[1]], [z[0], z[0]], color='blue', alpha=0.5)
                    self.ax1.plot([xi, xi], [y[0], y[1]], [z[1], z[1]], color='blue', alpha=0.5)
                    self.ax1.plot([xi, xi], [y[0], y[0]], [z[0], z[1]], color='blue', alpha=0.5)
                    self.ax1.plot([xi, xi], [y[1], y[1]], [z[0], z[1]], color='blue', alpha=0.5)

                for yi in y:
                    self.ax1.plot([x[0], x[1]], [yi, yi], [z[0], z[0]], color='blue', alpha=0.5)
                    self.ax1.plot([x[0], x[1]], [yi, yi], [z[1], z[1]], color='blue', alpha=0.5)
                    self.ax1.plot([x[0], x[0]], [yi, yi], [z[0], z[1]], color='blue', alpha=0.5)
                    self.ax1.plot([x[1], x[1]], [yi, yi], [z[0], z[1]], color='blue', alpha=0.5)

                for zi in z:
                    self.ax1.plot([x[0], x[1]], [y[0], y[0]], [zi, zi], color='blue', alpha=0.5)
                    self.ax1.plot([x[0], x[1]], [y[1], y[1]], [zi, zi], color='blue', alpha=0.5)
                    self.ax1.plot([x[0], x[0]], [y[0], y[1]], [zi, zi], color='blue', alpha=0.5)
                    self.ax1.plot([x[1], x[1]], [y[0], y[1]], [zi, zi], color='blue', alpha=0.5)

            elif obj['type'] == 'cylinder':
                pos = obj['position']
                radius = obj['radius']
                height = obj['height']

                # Draw cylinder
                theta = np.linspace(0, 2*np.pi, 32)
                x_circ = pos[0] + radius * np.cos(theta)
                y_circ = pos[1] + radius * np.sin(theta)
                z_bot = [pos[2]] * len(theta)
                z_top = [pos[2] + height] * len(theta)

                self.ax1.plot(x_circ, y_circ, z_bot, color='green', alpha=0.7)
                self.ax1.plot(x_circ, y_circ, z_top, color='green', alpha=0.7)

                # Draw side surfaces
                for i in range(len(theta)-1):
                    self.ax1.plot([x_circ[i], x_circ[i]], [y_circ[i], y_circ[i]], [z_bot[i], z_top[i]], color='green', alpha=0.5)

        # Plot end effector
        ee_pos = self.end_effector_pos
        self.ax1.scatter(ee_pos[0], ee_pos[1], ee_pos[2], color='red', s=100, label='End Effector')

        # Draw gripper
        gripper_size = 0.02
        if not self.gripper_open:
            # Closed gripper
            self.ax1.plot([ee_pos[0]-gripper_size, ee_pos[0]+gripper_size],
                         [ee_pos[1], ee_pos[1]], [ee_pos[2], ee_pos[2]], 'r-', linewidth=3)
        else:
            # Open gripper
            self.ax1.plot([ee_pos[0]-gripper_size, ee_pos[0]+gripper_size],
                         [ee_pos[1], ee_pos[1]], [ee_pos[2], ee_pos[2]], 'r--', linewidth=2)

        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('3D Manipulation Scene')
        self.ax1.legend()

        # Plot 2: Force control visualization
        self.ax2.text(0.5, 0.7, f'Gripper: {"OPEN" if self.gripper_open else "CLOSED"}',
                     transform=self.ax2.transAxes, ha='center', fontsize=14,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.ax2.text(0.5, 0.5, f'End Effector Position:\n[{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]',
                     transform=self.ax2.transAxes, ha='center', fontsize=12)
        self.ax2.text(0.5, 0.2, 'Force Control Active',
                     transform=self.ax2.transAxes, ha='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        self.ax2.axis('off')

        # Plot 3: Grasp planning visualization
        self.ax3.text(0.5, 0.8, 'Grasp Planning Info',
                     transform=self.ax3.transAxes, ha='center', fontsize=14,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.ax3.text(0.5, 0.6, f'Objects Detected: {len(self.objects)}',
                     transform=self.ax3.transAxes, ha='center', fontsize=12)
        self.ax3.text(0.5, 0.4, f'Objects: {[obj["type"] for obj in self.objects]}',
                     transform=self.ax3.transAxes, ha='center', fontsize=10)
        self.ax3.text(0.5, 0.1, 'Grasp Stability Analysis Active',
                     transform=self.ax3.transAxes, ha='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        self.ax3.axis('off')

        plt.tight_layout()
        plt.draw()

    def run_exercise(self):
        """Run the complete manipulation exercise"""
        print("Starting Grasp Controller Exercise...")
        print(f"Environment contains {len(self.objects)} objects:")
        for i, obj in enumerate(self.objects):
            print(f"  {i}: {obj['type']} at {obj['position']}")

        # Visualize initial state
        self.visualize_state()
        plt.show(block=False)

        # Attempt to grasp each object
        results = []
        for i in range(len(self.objects)):
            print(f"\n--- Attempting to grasp object {i} ---")
            success = self.execute_grasp_sequence(i)
            results.append(success)

            if success:
                print(f"✓ Successfully grasped object {i} ({self.objects[i]['type']})")
            else:
                print(f"✗ Failed to grasp object {i} ({self.objects[i]['type']})")

        # Calculate success rate
        success_rate = sum(results) / len(results) if results else 0
        print(f"\n=== Exercise Results ===")
        print(f"Total objects: {len(self.objects)}")
        print(f"Successful grasps: {sum(results)}")
        print(f"Success rate: {success_rate:.1%}")

        # Keep visualization open
        plt.ioff()
        self.visualize_state()
        plt.show()

        return results

# Run the exercise
def run_grasp_controller_exercise():
    """Run the complete grasp controller exercise"""
    exercise = GraspControllerExercise()
    results = exercise.run_exercise()

    return exercise, results

# Uncomment to run the exercise
# exercise, results = run_grasp_controller_exercise()
```

## Advanced Manipulation Techniques

### Multi-Fingered Hand Control

```python
class MultiFingeredHandController:
    """Controller for multi-fingered robotic hands"""

    def __init__(self, n_fingers=5, finger_lengths=None):
        self.n_fingers = n_fingers
        self.finger_lengths = finger_lengths or [0.05, 0.05, 0.05, 0.05, 0.03]  # Thumb shorter
        self.finger_positions = np.zeros(n_fingers)  # Joint angles
        self.finger_forces = np.zeros(n_fingers)     # Applied forces

    def prehensile_grasp(self, object_shape, grasp_type='cylindrical'):
        """Perform prehensile grasp with multiple fingers"""
        # Calculate required finger positions based on object shape
        finger_targets = self.calculate_finger_targets(object_shape, grasp_type)

        # Execute coordinated finger movement
        success = self.move_fingers_to_targets(finger_targets)

        if success:
            # Apply appropriate forces for the grasp type
            self.apply_grasp_forces(grasp_type)

        return success

    def calculate_finger_targets(self, object_shape, grasp_type):
        """Calculate target finger positions for a specific grasp type"""
        # Find object centroid
        centroid = np.mean(object_shape, axis=0)

        # Calculate object dimensions
        dims = np.max(object_shape, axis=0) - np.min(object_shape, axis=0)
        max_dim = np.max(dims)

        # Calculate finger positions based on grasp type
        targets = np.zeros(self.n_fingers)

        if grasp_type == 'cylindrical':
            # Arrange fingers around cylindrical object
            radius = max_dim / 2
            for i in range(self.n_fingers):
                angle = 2 * np.pi * i / self.n_fingers
                # Target position along the radius
                targets[i] = radius + 0.01  # 1cm clearance
        elif grasp_type == 'spherical':
            # Similar to cylindrical but in 3D
            radius = max_dim / 2
            for i in range(self.n_fingers):
                # Distribute fingers on sphere surface
                targets[i] = radius + 0.01
        elif grasp_type == 'tip_pinch':
            # Thumb opposes other fingers for pinch grasp
            targets[0] = 0.02  # Thumb position
            for i in range(1, self.n_fingers):
                targets[i] = 0.01  # Other fingers position

        return targets

    def move_fingers_to_targets(self, targets):
        """Move fingers to target positions with coordinated control"""
        # Simulate coordinated finger movement
        for i in range(len(targets)):
            # Move finger to target position
            self.finger_positions[i] = targets[i]

        # Check if all fingers reached targets (simulated)
        return True

    def apply_grasp_forces(self, grasp_type):
        """Apply appropriate forces for the grasp type"""
        if grasp_type == 'cylindrical':
            # Apply wrapping forces
            self.finger_forces = np.full(self.n_fingers, 3.0)  # 3N per finger
        elif grasp_type == 'tip_pinch':
            # Apply pinch forces
            self.finger_forces[0] = 5.0  # Thumb: higher force
            for i in range(1, self.n_fingers):
                self.finger_forces[i] = 2.0  # Other fingers: lower force
        else:
            # Default force
            self.finger_forces = np.full(self.n_fingers, 2.5)

    def adjust_grasp_for_slip_prevention(self, slip_sensors):
        """Adjust grasp forces based on slip detection"""
        # In a real system, this would use tactile sensors
        # to detect slip and adjust forces accordingly
        for i, slip_detected in enumerate(slip_sensors):
            if slip_detected:
                # Increase force on this finger
                self.finger_forces[i] += 1.0

    def release_object(self):
        """Release the grasped object"""
        # Open all fingers
        self.finger_positions = np.zeros(self.n_fingers)
        self.finger_forces = np.zeros(self.n_fingers)
```

### Cooperative Manipulation

```python
class CooperativeManipulationController:
    """Controller for cooperative manipulation with multiple arms"""

    def __init__(self, n_arms=2):
        self.n_arms = n_arms
        self.arm_positions = [np.zeros(3) for _ in range(n_arms)]
        self.arm_forces = [np.zeros(3) for _ in range(n_arms)]
        self.object_pose = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]

    def coordinated_lift(self, object_mass, lift_height=0.1):
        """Coordinate multiple arms to lift an object"""
        # Calculate required forces to lift object
        total_weight = object_mass * 9.81
        force_per_arm = total_weight / self.n_arms

        # Distribute forces among arms
        for i in range(self.n_arms):
            # Apply upward force
            self.arm_forces[i][2] = force_per_arm / self.n_arms  # Z component

        # Execute coordinated lift
        success = self.execute_coordinated_movement(
            target_height=self.object_pose[2] + lift_height
        )

        return success

    def execute_coordinated_movement(self, target_height):
        """Execute coordinated movement of all arms"""
        # Move all arms to new height while maintaining relative positions
        for i in range(self.n_arms):
            # Move to target height
            self.arm_positions[i][2] = target_height

        # Check for coordination success (simulated)
        return True

    def compute_load_distribution(self, object_com, arm_positions):
        """Compute how to distribute load among arms based on COG position"""
        # Calculate distances from each arm to object center of mass
        distances = []
        for pos in arm_positions:
            dist = np.linalg.norm(np.array(pos)[:2] - object_com[:2])  # X,Y only
            distances.append(dist)

        # Calculate load distribution (inverse to distance)
        total_inv_dist = sum(1.0 / max(d, 0.01) for d in distances)  # Avoid division by zero
        load_factors = []

        for d in distances:
            factor = (1.0 / max(d, 0.01)) / total_inv_dist
            load_factors.append(factor)

        return load_factors
```

## Integration with Humanoid Balance

### Balancing While Manipulating

```python
class BalanceAwareManipulator:
    """Manipulation system aware of balance constraints"""

    def __init__(self, balance_controller):
        self.balance_controller = balance_controller
        self.manipulator = self.initialize_manipulator()
        self.zmp_limits = {  # Conservative ZMP limits for safe manipulation
            'x_min': -0.1, 'x_max': 0.1,
            'y_min': -0.05, 'y_max': 0.05
        }

    def initialize_manipulator(self):
        """Initialize manipulation components"""
        return {
            'left_arm': self.create_arm_controller(),
            'right_arm': self.create_arm_controller(),
            'grasp_planner': GraspPlanner(robot_model=None)
        }

    def create_arm_controller(self):
        """Create arm controller with balance awareness"""
        return {
            'position': np.array([0.3, 0.2, 0.8]),  # Default position
            'jacobian': self.calculate_jacobian(),  # Manipulator Jacobian
            'workspace': self.calculate_workspace()  # Reachable workspace
        }

    def calculate_jacobian(self):
        """Calculate manipulator Jacobian (simplified)"""
        # This would be calculated based on the actual kinematic model
        # For this example, we'll return a simple Jacobian
        return np.eye(6)  # 6x6 identity as placeholder

    def calculate_workspace(self):
        """Calculate reachable workspace (simplified)"""
        # Define conservative workspace limits
        return {
            'x_range': [-0.3, 0.6],
            'y_range': [-0.4, 0.4],
            'z_range': [0.2, 1.2]
        }

    def safe_manipulation_move(self, arm_name, target_pose):
        """Execute manipulation move while maintaining balance"""
        # Check if target is within safe workspace
        if not self.is_target_in_workspace(arm_name, target_pose[:3]):
            print(f"Target pose {target_pose[:3]} is outside {arm_name} workspace!")
            return False

        # Predict ZMP shift due to manipulation
        predicted_zmp_shift = self.predict_zmp_shift(arm_name, target_pose)

        # Check if resulting ZMP is within safe limits
        current_zmp = self.balance_controller.get_current_zmp()
        new_zmp = current_zmp + predicted_zmp_shift

        if not self.is_zmp_safe(new_zmp):
            print(f"Movement would cause ZMP to exceed safe limits!")
            print(f"Current ZMP: {current_zmp}, Predicted new ZMP: {new_zmp}")
            return False

        # Execute safe movement
        success = self.execute_arm_movement(arm_name, target_pose)

        if success:
            # Update balance controller with new configuration
            self.balance_controller.update_manipulation_state(arm_name, target_pose)

        return success

    def is_target_in_workspace(self, arm_name, target_pos):
        """Check if target position is in arm workspace"""
        workspace = self.manipulator[arm_name]['workspace']

        x_ok = workspace['x_range'][0] <= target_pos[0] <= workspace['x_range'][1]
        y_ok = workspace['y_range'][0] <= target_pos[1] <= workspace['y_range'][1]
        z_ok = workspace['z_range'][0] <= target_pos[2] <= workspace['z_range'][1]

        return x_ok and y_ok and z_ok

    def predict_zmp_shift(self, arm_name, target_pose):
        """Predict ZMP shift caused by manipulation"""
        # This is a simplified model - in reality this would involve
        # complex dynamics calculations
        # For now, we'll use a heuristic based on arm extension

        current_pos = self.manipulator[arm_name]['position']
        displacement = np.array(target_pose[:3]) - current_pos

        # Heuristic: extending arm laterally shifts ZMP in opposite direction
        zmp_shift = np.array([-0.3 * displacement[1], 0.3 * displacement[0], 0])

        # Attenuate based on vertical displacement (raising arms less destabilizing)
        zmp_shift[0] *= (1.0 - 0.5 * abs(displacement[2]) / 0.5)  # 0.5m as reference

        return zmp_shift

    def is_zmp_safe(self, zmp):
        """Check if ZMP is within safe limits"""
        x_ok = self.zmp_limits['x_min'] <= zmp[0] <= self.zmp_limits['x_max']
        y_ok = self.zmp_limits['y_min'] <= zmp[1] <= self.zmp_limits['y_max']
        return x_ok and y_ok

    def execute_arm_movement(self, arm_name, target_pose):
        """Execute actual arm movement"""
        # In a real implementation, this would command the robot
        # For simulation, we'll just update the internal state
        self.manipulator[arm_name]['position'] = target_pose[:3]
        return True  # Simulated success

    def manipulation_with_balance_recovery(self, task):
        """Execute manipulation task with automatic balance recovery"""
        # Plan manipulation sequence
        trajectory = self.plan_manipulation_trajectory(task)

        # Execute with balance monitoring
        for waypoint in trajectory:
            # Check if we can safely execute this waypoint
            success = self.safe_manipulation_move(waypoint['arm'], waypoint['pose'])

            if not success:
                # Execute balance recovery
                print("Executing balance recovery...")
                recovery_success = self.balance_controller.execute_recovery()

                if not recovery_success:
                    print("Balance recovery failed!")
                    return False

            # Monitor balance during execution
            if not self.balance_controller.is_balanced():
                print("Balance compromised during manipulation!")
                return False

        return True
```

## Summary

This chapter covered the fundamental concepts of robotic manipulation and grasping for humanoid robots. We explored various grasp types and classification schemes, from power grasps to precision grasps, and implemented algorithms for grasp planning and stability analysis.

We implemented both sampling-based and learning-based grasp planning approaches, allowing robots to find effective grasps for novel objects. The chapter also covered force control and compliance, including impedance control, admittance control, and hybrid position/force control techniques that are essential for safe and effective manipulation.

The integration with perception systems was emphasized, showing how object detection, pose estimation, and grasp planning work together in a complete manipulation pipeline. We also discussed advanced topics like multi-fingered hand control, cooperative manipulation with multiple arms, and balance-aware manipulation that considers the robot's stability during manipulation tasks.

The hands-on exercise provided practical experience implementing a complete grasp controller that integrates perception, planning, and control, demonstrating the complexity of real-world manipulation for humanoid robots.

## Key Takeaways

- Grasp planning requires consideration of force closure and grasp quality metrics
- Force control and compliance are essential for stable grasping
- Perception integration enables manipulation of unknown objects
- Multi-fingered hands require coordinated control strategies
- Balance constraints must be considered during manipulation
- Cooperative manipulation with multiple arms can handle larger objects
- Real-time control is critical for stable manipulation
- Safety considerations are paramount in manipulation systems

## Next Steps

In the next chapter, we'll explore learning-based control approaches for humanoid robots, including reinforcement learning and adaptive control methods that can improve manipulation and locomotion performance over time.

## References and Further Reading

1. Mason, M. T., & Salisbury, J. K. (1985). Robot Hands and the Mechanics of Manipulation. MIT Press.
2. Okamura, A. M., Roma, N., & Cutkosky, M. R. (2000). A comparison of force control strategies for an adaptive robotic assembly task. IEEE International Conference on Robotics and Automation.
3. Biagiotti, L., & Melchiorri, C. (2008). Trajectory Planning for Automatic Machines and Robots. Springer.
4. Murray, R. M., Li, Z., & Sastry, S. S. (1994). A Mathematical Introduction to Robotic Manipulation. CRC Press.
5. Siciliano, B., & Khatib, O. (Eds.). (2016). Springer Handbook of Robotics. Springer.