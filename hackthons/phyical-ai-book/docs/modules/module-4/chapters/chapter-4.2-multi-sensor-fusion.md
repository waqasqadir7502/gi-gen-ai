# Chapter 4.2: Multi-Sensor Fusion

## Overview

Multi-sensor fusion is the process of combining data from multiple sensors to create a more accurate, reliable, and comprehensive understanding of the environment than would be possible with any single sensor alone. This chapter covers fundamental fusion techniques, sensor integration strategies, and practical implementation approaches for humanoid robots that must operate in complex, dynamic environments.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the principles of sensor fusion and data integration
2. Implement Kalman filtering and particle filtering for state estimation
3. Design sensor fusion architectures for humanoid robots
4. Apply sensor fusion techniques to perception and navigation tasks
5. Evaluate fusion system performance and handle sensor failures

## Introduction to Multi-Sensor Fusion

Multi-sensor fusion is essential for humanoid robots because no single sensor can provide complete information about the environment and robot state. Different sensors have complementary strengths and weaknesses, and combining their data allows for more robust and accurate perception.

### Why Sensor Fusion is Necessary

#### Sensor Limitations
- **Limited Field of View**: Cameras and depth sensors only see what's in their field of view
- **Noise and Uncertainty**: All sensors have inherent measurement noise
- **Environmental Sensitivity**: Performance varies with lighting, weather, etc.
- **Temporal Limitations**: Some sensors have slow update rates
- **Physical Constraints**: Sensors may be occluded or damaged

#### Benefits of Fusion
- **Redundancy**: Multiple sensors provide backup when one fails
- **Complementarity**: Different sensors provide different types of information
- **Improved Accuracy**: Combined measurements can be more accurate than individual ones
- **Enhanced Reliability**: System continues functioning despite partial sensor failures
- **Extended Capabilities**: Combined sensors can perceive things individually impossible

### Types of Sensor Fusion

#### Data-Level Fusion
Combines raw sensor measurements before any interpretation. This preserves maximum information but requires synchronization and calibration.

#### Feature-Level Fusion
Extracts features from individual sensors, then combines them. This reduces data volume while maintaining relevant information.

#### Decision-Level Fusion
Makes decisions based on each sensor independently, then combines the decisions. This is computationally efficient but may lose information.

#### Hybrid Fusion
Combines approaches at multiple levels for optimal performance.

## Mathematical Foundations

### Probability Theory Review

Sensor fusion relies heavily on probability theory to represent uncertainty in measurements and states.

#### Bayes' Rule
```
P(A|B) = P(B|A) × P(A) / P(B)
```

In sensor fusion, this becomes:
```
P(state|measurement) = P(measurement|state) × P(state) / P(measurement)
```

#### Gaussian Distributions
Many sensors have Gaussian (normal) noise, making Kalman filters applicable:
```
p(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
```

For multivariate data:
```
p(x) = (1/√((2π)ⁿ|Σ|)) × exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

### Kalman Filtering

Kalman filters provide optimal state estimation for linear systems with Gaussian noise.

```python
import numpy as np
from scipy.linalg import block_diag

class KalmanFilter:
    """Basic Kalman Filter implementation"""

    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector [x, y, z, vx, vy, vz] for position and velocity
        self.x = np.zeros(state_dim)  # State estimate

        # Covariance matrix (uncertainty in state estimate)
        self.P = np.eye(state_dim) * 1000  # Initial uncertainty

        # Process noise (how uncertain our model is)
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise
        self.R = np.eye(measurement_dim) * 1.0

        # State transition model (how state evolves over time)
        self.F = np.eye(state_dim)  # Will be updated based on dt

        # Measurement model (how state maps to measurements)
        self.H = np.zeros((measurement_dim, state_dim))
        # For position measurements: map position part of state to measurements
        for i in range(min(measurement_dim, state_dim)):
            self.H[i, i] = 1.0

    def predict(self, dt):
        """Prediction step: predict state forward in time"""
        # Update state transition matrix based on time step
        # For constant velocity model:
        # x_new = x + vx*dt, y_new = y + vy*dt, etc.
        for i in range(self.state_dim // 2):  # For each position component
            self.F[i, i + self.state_dim // 2] = dt  # Position += velocity * dt

        # Predict state: x = F * x
        self.x = self.F @ self.x

        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update step: incorporate new measurement"""
        # Innovation: y = z - H * x (measurement residual)
        innovation = measurement - self.H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state: x = x + K * y
        self.x = self.x + K @ innovation

        # Update covariance: P = (I - K * H) * P
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P

    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()

    def get_covariance(self):
        """Get current covariance estimate"""
        return self.P.copy()

class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter for nonlinear systems"""

    def __init__(self, state_dim, measurement_dim):
        super().__init__(state_dim, measurement_dim)
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

    def predict(self, dt, control_input=None):
        """Nonlinear prediction step"""
        # In EKF, we linearize around current state
        # This is a simplified example - in practice, you'd implement
        # the actual nonlinear dynamics and compute Jacobians

        # Example: constant velocity model with control
        # x = f(x, u, dt)
        if control_input is not None:
            # Apply control input (e.g., acceleration commands)
            for i in range(self.state_dim // 2):
                self.x[i + self.state_dim // 2] += control_input[i] * dt  # Update velocity
                self.x[i] += self.x[i + self.state_dim // 2] * dt         # Update position

        # Linearize the process model to get F matrix
        self.F = self.compute_process_jacobian(dt)

        # Predict covariance as in regular KF
        self.P = self.F @ self.P @ self.F.T + self.Q

    def compute_process_jacobian(self, dt):
        """Compute Jacobian of process model"""
        # For constant velocity model, the Jacobian is:
        F = np.eye(self.state_dim)
        for i in range(self.state_dim // 2):
            F[i, i + self.state_dim // 2] = dt
        return F

    def update(self, measurement):
        """Nonlinear update step"""
        # Linearize measurement model to get H matrix
        H = self.compute_measurement_jacobian()

        # Innovation
        predicted_measurement = self.h_function(self.x)  # Nonlinear measurement function
        innovation = measurement - predicted_measurement

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ innovation

        # Update covariance
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P

    def h_function(self, state):
        """Nonlinear measurement function"""
        # This maps state to expected measurement
        # For position measurements, it's typically linear
        return self.H @ state

    def compute_measurement_jacobian(self):
        """Compute Jacobian of measurement model"""
        # For simple position measurements, this is just H
        return self.H

class UnscentedKalmanFilter:
    """Unscented Kalman Filter implementation"""

    def __init__(self, state_dim, measurement_dim, alpha=1e-3, beta=2, kappa=0):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1.0

        # UKF parameters
        self.alpha = alpha  # Spread of sigma points
        self.beta = beta    # Prior knowledge of distribution
        self.kappa = kappa  # Secondary scaling parameter

        # Calculate weights
        self.L = state_dim  # Dimension of state
        self.lambda_ = alpha**2 * (self.L + kappa) - self.L

        self.W_m = np.zeros(2*self.L + 1)  # Mean weights
        self.W_c = np.zeros(2*self.L + 1)  # Covariance weights

        self.W_m[0] = self.lambda_ / (self.L + self.lambda_)
        self.W_c[0] = self.lambda_ / (self.L + self.lambda_) + (1 - alpha**2 + beta)

        for i in range(1, 2*self.L + 1):
            self.W_m[i] = 1 / (2*(self.L + self.lambda_))
            self.W_c[i] = 1 / (2*(self.L + self.lambda_))

    def predict(self, dt):
        """UKF prediction step"""
        # Generate sigma points
        sigma_points = self.compute_sigma_points()

        # Propagate sigma points through process model
        propagated_points = []
        for point in sigma_points:
            propagated_point = self.process_model(point, dt)
            propagated_points.append(propagated_point)

        # Calculate predicted state and covariance
        x_pred = np.zeros(self.state_dim)
        for i in range(len(propagated_points)):
            x_pred += self.W_m[i] * propagated_points[i]

        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(len(propagated_points)):
            diff = propagated_points[i] - x_pred
            P_pred += self.W_c[i] * np.outer(diff, diff)

        P_pred += self.Q

        # Update state and covariance
        self.x = x_pred
        self.P = P_pred

    def update(self, measurement):
        """UKF update step"""
        # Generate sigma points from current state
        sigma_points = self.compute_sigma_points()

        # Transform sigma points through measurement model
        measurement_points = []
        for point in sigma_points:
            meas_point = self.measurement_model(point)
            measurement_points.append(meas_point)

        # Calculate predicted measurement
        z_pred = np.zeros(self.measurement_dim)
        for i in range(len(measurement_points)):
            z_pred += self.W_m[i] * measurement_points[i]

        # Calculate innovation covariance
        P_zz = np.zeros((self.measurement_dim, self.measurement_dim))
        for i in range(len(measurement_points)):
            diff = measurement_points[i] - z_pred
            P_zz += self.W_c[i] * np.outer(diff, diff)
        P_zz += self.R

        # Calculate cross-covariance
        P_xz = np.zeros((self.state_dim, self.measurement_dim))
        for i in range(len(sigma_points)):
            state_diff = sigma_points[i] - self.x
            meas_diff = measurement_points[i] - z_pred
            P_xz += self.W_c[i] * np.outer(state_diff, meas_diff)

        # Calculate Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)

        # Update state and covariance
        innovation = measurement - z_pred
        self.x = self.x + K @ innovation
        self.P = self.P - K @ P_zz @ K.T

    def compute_sigma_points(self):
        """Generate sigma points from current state and covariance"""
        # Calculate scaling factor
        U = np.linalg.cholesky((self.L + self.lambda_) * self.P)

        sigma_points = [self.x.copy()]  # Center point

        # Add points in positive and negative directions
        for i in range(self.state_dim):
            sigma_points.append(self.x + U[:, i])
            sigma_points.append(self.x - U[:, i])

        return np.array(sigma_points)

    def process_model(self, state, dt):
        """Nonlinear process model (example: constant velocity)"""
        # Simple constant velocity model
        new_state = state.copy()

        # Update positions based on velocities
        n_pos = len(state) // 2  # Number of position components
        for i in range(n_pos):
            new_state[i] += new_state[i + n_pos] * dt  # position += velocity * dt

        return new_state

    def measurement_model(self, state):
        """Nonlinear measurement model (example: position measurements)"""
        # Return position components of state
        n_pos = min(len(state) // 2, self.measurement_dim)
        return state[:n_pos]
```

### Particle Filtering

Particle filters can handle non-Gaussian distributions and nonlinear systems.

```python
class ParticleFilter:
    """Particle Filter for nonlinear/non-Gaussian systems"""

    def __init__(self, state_dim, measurement_dim, n_particles=1000):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.n_particles = n_particles

        # Initialize particles randomly
        self.particles = np.random.randn(n_particles, state_dim) * 10
        self.weights = np.ones(n_particles) / n_particles

        # Process and measurement noise
        self.process_noise = 0.1
        self.measurement_noise = 1.0

    def predict(self, dt, control_input=None):
        """Predict step: propagate particles forward"""
        for i in range(self.n_particles):
            # Apply process model with noise
            self.particles[i] = self.process_model(
                self.particles[i], dt, control_input
            ) + np.random.normal(0, self.process_noise, self.state_dim)

    def update(self, measurement):
        """Update step: adjust weights based on measurement"""
        for i in range(self.n_particles):
            # Calculate likelihood of measurement given particle state
            predicted_measurement = self.measurement_model(self.particles[i])
            likelihood = self.calculate_likelihood(measurement, predicted_measurement)

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1.e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = self.systematic_resample()

        # Resample particles
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.n_particles)

    def systematic_resample(self):
        """Systematic resampling algorithm"""
        n = self.n_particles
        indices = np.zeros(n, dtype=int)

        # Cumulative sum of weights
        cumulative_sum = np.cumsum(self.weights)

        # Generate random start point
        start = np.random.uniform(0, 1/n)
        offsets = (np.arange(n) + start) / n

        i, j = 0, 0
        while i < n:
            if offsets[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def estimate_state(self):
        """Estimate state from particles"""
        # Weighted average of particles
        estimated_state = np.zeros(self.state_dim)
        for i in range(self.n_particles):
            estimated_state += self.weights[i] * self.particles[i]
        return estimated_state

    def process_model(self, state, dt, control_input):
        """Nonlinear process model"""
        # Example: constant velocity model
        new_state = state.copy()
        n_pos = len(state) // 2  # Position components

        for i in range(n_pos):
            new_state[i] += new_state[i + n_pos] * dt  # Update position
            # Could also apply control input here
            if control_input is not None:
                new_state[i + n_pos] += control_input[i] * dt  # Update velocity

        return new_state

    def measurement_model(self, state):
        """Nonlinear measurement model"""
        # Example: return position components
        n_pos = min(len(state) // 2, self.measurement_dim)
        return state[:n_pos]

    def calculate_likelihood(self, measurement, predicted_measurement):
        """Calculate likelihood of measurement given prediction"""
        # Assume Gaussian noise
        innovation = measurement - predicted_measurement
        variance = self.measurement_noise**2
        likelihood = np.exp(-0.5 * np.sum(innovation**2) / variance)
        return likelihood + 1.e-300  # Avoid zero likelihood
```

## Sensor Integration Architectures

### Centralized Fusion

In centralized fusion, all sensor data is sent to a central processor that performs the fusion.

```python
class CentralizedFusionArchitecture:
    """Centralized sensor fusion architecture"""

    def __init__(self):
        self.sensors = {}
        self.fusion_engine = KalmanFilter(state_dim=12, measurement_dim=6)  # Example: 6D pose + 6D velocity
        self.sensor_data_buffer = {}
        self.time_sync_buffer = []
        self.fusion_result = None

    def register_sensor(self, sensor_id, sensor_type, data_callback):
        """Register a sensor with the fusion system"""
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'callback': data_callback,
            'last_update': 0,
            'data_queue': []
        }

    def sensor_callback(self, sensor_id, data, timestamp):
        """Callback for sensor data reception"""
        if sensor_id in self.sensors:
            # Store data with timestamp
            self.sensors[sensor_id]['data_queue'].append((data, timestamp))
            self.sensors[sensor_id]['last_update'] = timestamp

            # Trigger fusion if we have enough data
            self.process_sensor_data(sensor_id, data, timestamp)

    def process_sensor_data(self, sensor_id, data, timestamp):
        """Process incoming sensor data"""
        # Time synchronization
        synchronized_data = self.synchronize_sensor_data(timestamp)

        if synchronized_data:
            # Perform fusion
            self.fusion_result = self.perform_fusion(synchronized_data)

    def synchronize_sensor_data(self, target_time):
        """Synchronize data from different sensors to the same time"""
        synchronized = {}

        for sensor_id, sensor_info in self.sensors.items():
            # Find data closest to target_time
            if sensor_info['data_queue']:
                # Find the data point closest to target_time
                closest_data = min(
                    sensor_info['data_queue'],
                    key=lambda x: abs(x[1] - target_time)
                )

                # If the time difference is acceptable, include it
                if abs(closest_data[1] - target_time) < 0.1:  # 100ms tolerance
                    synchronized[sensor_id] = closest_data[0]

        return synchronized if len(synchronized) >= 2 else None  # Need at least 2 sensors

    def perform_fusion(self, synchronized_data):
        """Perform sensor fusion on synchronized data"""
        # Convert sensor data to measurements for fusion
        measurements = self.convert_sensor_data_to_measurements(synchronized_data)

        # Predict step
        dt = 0.01  # 10ms time step
        self.fusion_engine.predict(dt)

        # Update step with each measurement
        for measurement in measurements:
            self.fusion_engine.update(measurement)

        # Return fused state estimate
        return {
            'state': self.fusion_engine.get_state(),
            'covariance': self.fusion_engine.get_covariance(),
            'timestamp': time.time()
        }

    def convert_sensor_data_to_measurements(self, sensor_data):
        """Convert sensor-specific data to measurement vectors"""
        measurements = []

        for sensor_id, data in sensor_data.items():
            sensor_type = self.sensors[sensor_id]['type']

            if sensor_type == 'imu':
                # IMU provides acceleration, angular velocity, magnetometer
                # Could use this for attitude estimation
                measurements.append(data[:6])  # accel + gyro
            elif sensor_type == 'camera':
                # Camera provides 2D/3D position measurements
                measurements.append(data[:3])  # position
            elif sensor_type == 'lidar':
                # LiDAR provides range measurements
                measurements.append(data[:3])  # closest obstacle position
            elif sensor_type == 'encoder':
                # Encoders provide joint position measurements
                measurements.append(data)  # joint angles

        return measurements

class DistributedFusionNode:
    """A node in a distributed fusion system"""

    def __init__(self, node_id, sensor_type):
        self.node_id = node_id
        self.sensor_type = sensor_type
        self.local_filter = self.initialize_local_filter()
        self.neighbors = []
        self.communication_buffer = []
        self.local_estimate = None
        self.timestamp = 0

    def initialize_local_filter(self):
        """Initialize local filter based on sensor type"""
        if self.sensor_type == 'pose':
            # 6D pose estimation
            return ExtendedKalmanFilter(state_dim=12, measurement_dim=6)
        elif self.sensor_type == 'velocity':
            # 6D velocity estimation
            return KalmanFilter(state_dim=6, measurement_dim=3)
        elif self.sensor_type == 'imu':
            # IMU data processing
            return KalmanFilter(state_dim=9, measurement_dim=6)  # accel, gyro, mag
        else:
            return KalmanFilter(state_dim=3, measurement_dim=3)  # Generic

    def process_local_measurement(self, measurement, timestamp):
        """Process local sensor measurement"""
        # Predict based on time since last update
        dt = timestamp - self.timestamp if self.timestamp > 0 else 0.01
        self.local_filter.predict(dt)

        # Update with measurement
        self.local_filter.update(measurement)

        # Store estimate
        self.local_estimate = {
            'state': self.local_filter.get_state(),
            'covariance': self.local_filter.get_covariance(),
            'timestamp': timestamp
        }
        self.timestamp = timestamp

    def share_estimate(self):
        """Share local estimate with neighbors"""
        if self.local_estimate:
            message = {
                'node_id': self.node_id,
                'estimate': self.local_estimate,
                'timestamp': self.timestamp,
                'sensor_type': self.sensor_type
            }

            # Send to neighbors
            for neighbor in self.neighbors:
                neighbor.receive_estimate(message)

    def receive_estimate(self, message):
        """Receive estimate from another node"""
        self.communication_buffer.append(message)

    def fuse_neighbor_estimates(self):
        """Fuse estimates from neighboring nodes"""
        if not self.communication_buffer:
            return

        # Simple consensus-based fusion
        neighbor_estimates = []
        for msg in self.communication_buffer:
            if msg['timestamp'] > self.timestamp - 1.0:  # Only recent estimates
                neighbor_estimates.append(msg['estimate'])

        if neighbor_estimates:
            # Weighted average based on covariance
            total_weight = np.zeros_like(self.local_estimate['state'])
            weighted_sum = np.zeros_like(self.local_estimate['state'])

            # Local estimate contribution
            local_cov_inv = np.linalg.inv(self.local_estimate['covariance'])
            local_weight = local_cov_inv
            weighted_sum += local_weight @ self.local_estimate['state']
            total_weight += local_weight

            # Neighbor contributions
            for est in neighbor_estimates:
                cov_inv = np.linalg.inv(est['covariance'])
                weight = cov_inv
                weighted_sum += weight @ est['state']
                total_weight += weight

            # Compute fused estimate
            fused_state = np.linalg.solve(total_weight, weighted_sum)
            fused_covariance = np.linalg.inv(total_weight)

            # Update local estimate
            self.local_estimate['state'] = fused_state
            self.local_estimate['covariance'] = fused_covariance

        # Clear buffer
        self.communication_buffer.clear()
```

## Specific Sensor Integration Examples

### Vision and Inertial Integration

Combining visual odometry with IMU data for robust localization:

```python
class VisualInertialOdometry:
    """Visual-Inertial Odometry fusion system"""

    def __init__(self):
        # Visual odometry component
        self.visual_odom = self.initialize_visual_odometry()

        # IMU processing component
        self.imu_processor = self.initialize_imu_processor()

        # Fusion filter
        self.fusion_filter = ExtendedKalmanFilter(
            state_dim=15,  # [position, velocity, orientation, bias_gyro, bias_accel]
            measurement_dim=9   # [position, velocity, orientation from visual]
        )

        # State components
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1, 0, 0, 0])  # Quaternion
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)

        # Timestamps
        self.last_visual_time = 0
        self.last_imu_time = 0

    def initialize_visual_odometry(self):
        """Initialize visual odometry system"""
        return {
            'features': [],
            'matches': [],
            'camera_matrix': np.eye(3),
            'distortion_coeffs': np.zeros(5),
            'pose_history': []
        }

    def initialize_imu_processor(self):
        """Initialize IMU processing system"""
        return {
            'accel_bias': np.zeros(3),
            'gyro_bias': np.zeros(3),
            'gravity_estimate': np.array([0, 0, -9.81]),
            'integration_buffer': []
        }

    def process_visual_frame(self, image, timestamp):
        """Process visual frame for pose estimation"""
        # This would involve feature detection, tracking, and pose estimation
        # For this example, we'll simulate the output

        if self.last_visual_time > 0:
            dt = timestamp - self.last_visual_time

            # Simulate visual pose change based on current motion
            visual_translation = self.velocity * dt + 0.5 * self.get_gravity_aligned_acceleration() * dt**2
            visual_rotation = self.integrate_angular_velocity(self.bias_gyro, dt)

            visual_pose = {
                'position': self.position + visual_translation,
                'orientation': self.multiply_quaternions(self.orientation, visual_rotation),
                'velocity': self.velocity,  # Estimated from position differences
                'confidence': 0.8  # Simulated confidence
            }

            # Update fusion filter with visual measurement
            self.update_visual_measurement(visual_pose, timestamp)

        self.last_visual_time = timestamp

    def process_imu_data(self, accel, gyro, timestamp):
        """Process IMU data for state prediction"""
        if self.last_imu_time > 0:
            dt = timestamp - self.last_imu_time

            # Correct for biases
            corrected_accel = accel - self.bias_accel
            corrected_gyro = gyro - self.bias_gyro

            # Update state prediction using IMU data
            self.predict_state_from_imu(corrected_accel, corrected_gyro, dt)

        self.last_imu_time = timestamp

    def predict_state_from_imu(self, accel, gyro, dt):
        """Predict state using IMU measurements"""
        # Update orientation
        dq = self.angular_velocity_to_quaternion_derivative(self.orientation, gyro)
        self.orientation += dq * dt
        self.orientation /= np.linalg.norm(self.orientation)  # Normalize

        # Update velocity (in world frame)
        world_accel = self.rotate_vector_to_world(accel, self.orientation)
        # Remove gravity
        gravity_world = self.rotate_vector_to_world(
            self.imu_processor['gravity_estimate'], self.orientation
        )
        net_accel = world_accel - gravity_world

        self.velocity += net_accel * dt

        # Update position
        self.position += self.velocity * dt + 0.5 * net_accel * dt**2

        # Update fusion filter prediction
        self.fusion_filter.predict(dt)

    def update_visual_measurement(self, visual_pose, timestamp):
        """Update fusion filter with visual measurement"""
        # Create measurement vector [position, velocity, orientation]
        measurement = np.concatenate([
            visual_pose['position'],
            visual_pose['velocity'],
            visual_pose['orientation'][1:]  # Only imaginary parts of quaternion
        ])

        # Measurement covariance based on confidence
        R = np.eye(len(measurement)) * (1.0 - visual_pose['confidence'])

        # Update fusion filter
        self.fusion_filter.update(measurement)

    def angular_velocity_to_quaternion_derivative(self, q, omega):
        """Convert angular velocity to quaternion derivative"""
        # From quaternion derivative formula
        # dq/dt = 0.5 * Omega(w) * q
        # where Omega(w) is the skew-symmetric matrix
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return 0.5 * Omega @ q

    def rotate_vector_to_world(self, v_body, q):
        """Rotate vector from body frame to world frame using quaternion"""
        # Convert quaternion to rotation matrix
        R = self.quaternion_to_rotation_matrix(q)
        return R @ v_body

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def multiply_quaternions(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    def get_gravity_aligned_acceleration(self):
        """Get acceleration with gravity removed"""
        # Rotate gravity vector to body frame
        q_inv = np.array([self.orientation[0], -self.orientation[1], -self.orientation[2], -self.orientation[3]])
        gravity_body = self.rotate_vector_to_world(self.imu_processor['gravity_estimate'], q_inv)
        return gravity_body

    def get_pose_estimate(self):
        """Get current pose estimate"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'timestamp': max(self.last_visual_time, self.last_imu_time)
        }
```

### Force-Torque and Tactile Integration

Integrating force-torque sensors with tactile sensing for manipulation:

```python
class ForceTactileFusion:
    """Fusion of force-torque and tactile sensing for manipulation"""

    def __init__(self, robot_hand_config):
        self.robot_hand_config = robot_hand_config

        # Force-torque sensor processing
        self.ft_sensor = {
            'raw_data': np.zeros(6),  # [Fx, Fy, Fz, Tx, Ty, Tz]
            'filtered_data': np.zeros(6),
            'bias': np.zeros(6),
            'calibrated': False
        }

        # Tactile sensor processing
        self.tactile_array = {
            'sensor_grid': [],  # 2D array of tactile sensors
            'contact_points': [],
            'pressure_distribution': np.zeros((8, 4)),  # Example: 8x4 grid
            'temperature_data': np.zeros((8, 4))  # Temperature sensors
        }

        # Fusion state
        self.object_properties = {
            'slip_detected': False,
            'grasp_stability': 0.0,
            'object_stiffness': 0.0,
            'surface_texture': 0.0,
            'estimated_weight': 0.0
        }

        # Filters and buffers
        self.force_filter = self.initialize_low_pass_filter(cutoff_freq=10.0)
        self.slip_detector = SlipDetectionAlgorithm(window_size=50)
        self.contact_estimator = ContactPointEstimator()

    def initialize_low_pass_filter(self, cutoff_freq, sampling_freq=100.0):
        """Initialize low-pass filter for force data"""
        # Simple first-order low-pass filter
        dt = 1.0 / sampling_freq
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        alpha = dt / (rc + dt)
        return {'alpha': alpha, 'prev_output': np.zeros(6)}

    def process_force_torque_data(self, ft_raw, timestamp):
        """Process force-torque sensor data"""
        # Apply low-pass filter
        filtered = np.zeros(6)
        for i in range(6):
            filtered[i] = (1 - self.force_filter['alpha']) * self.force_filter['prev_output'][i] + \
                          self.force_filter['alpha'] * ft_raw[i]

        self.ft_sensor['filtered_data'] = filtered
        self.force_filter['prev_output'] = filtered

        # Update object property estimates based on force data
        self.update_weight_estimate(ft_raw)
        self.update_stability_metrics(ft_raw)

    def process_tactile_data(self, tactile_raw, timestamp):
        """Process tactile sensor array data"""
        # Update tactile grid
        self.tactile_array['pressure_distribution'] = tactile_raw['pressure']
        self.tactile_array['temperature_data'] = tactile_raw['temperature']

        # Detect contact points
        self.tactile_array['contact_points'] = self.contact_estimator.estimate_contact_points(
            tactile_raw['pressure']
        )

        # Analyze surface properties
        self.analyze_surface_properties(tactile_raw)

    def update_weight_estimate(self, force_torque_data):
        """Estimate object weight from force measurements"""
        # Extract vertical force (assuming Z is up)
        fz = force_torque_data[2]

        # Account for robot's own weight (if calibrated)
        if self.ft_sensor['calibrated']:
            net_force_z = fz - self.ft_sensor['bias'][2]
        else:
            net_force_z = fz

        # Estimate weight (assuming gravitational acceleration)
        estimated_weight = abs(net_force_z) / 9.81
        self.object_properties['estimated_weight'] = estimated_weight

    def update_stability_metrics(self, force_torque_data):
        """Update grasp stability metrics"""
        # Calculate grasp stability based on force distribution
        forces = force_torque_data[:3]  # [Fx, Fy, Fz]
        torques = force_torque_data[3:]  # [Tx, Ty, Tz]

        # Stability metric based on force magnitude and distribution
        force_magnitude = np.linalg.norm(forces)
        torque_magnitude = np.linalg.norm(torques)

        # Normalize based on expected values for stable grasp
        expected_force = 5.0  # N, typical grasp force
        expected_torque = 0.1  # Nm, typical disturbance torque

        stability_force = min(1.0, expected_force / max(force_magnitude, 0.1))
        stability_torque = min(1.0, expected_torque / max(torque_magnitude, 0.01))

        # Combined stability metric
        self.object_properties['grasp_stability'] = 0.7 * stability_force + 0.3 * stability_torque

    def analyze_surface_properties(self, tactile_data):
        """Analyze surface properties from tactile data"""
        pressure_grid = tactile_data['pressure']

        # Texture analysis: variance in pressure distribution
        texture_metric = np.var(pressure_grid)
        self.object_properties['surface_texture'] = min(1.0, texture_metric / 100.0)

        # Stiffness analysis: rate of pressure change with applied force
        total_pressure = np.sum(pressure_grid)
        if total_pressure > 0:
            avg_pressure = total_pressure / pressure_grid.size
            # Stiffness inversely related to deformation (higher pressure = stiffer)
            self.object_properties['object_stiffness'] = min(1.0, avg_pressure / 500.0)

    def detect_slip(self, tactile_data, force_data):
        """Detect slip using both tactile and force data"""
        # Tactile-based slip detection
        tactile_slip = self.slip_detector.detect_from_tactile(tactile_data['pressure'])

        # Force-based slip detection (sudden changes in tangential forces)
        fx, fy = force_data[0], force_data[1]
        tangential_force = np.sqrt(fx**2 + fy**2)

        # If tangential force increases rapidly while normal force stays constant,
        # it may indicate slip
        force_slip_indicator = self.detect_force_based_slip(fx, fy)

        # Combine both indicators
        slip_confirmed = tactile_slip or force_slip_indicator
        self.object_properties['slip_detected'] = slip_confirmed

        return slip_confirmed

    def detect_force_based_slip(self, fx, fy):
        """Detect slip based on force changes"""
        # This would involve tracking force patterns over time
        # For now, we'll use a simple threshold
        tangential_force = np.sqrt(fx**2 + fy**2)

        # If tangential force is high relative to normal force, potential slip
        normal_force = abs(self.ft_sensor['filtered_data'][2])
        if normal_force > 0:
            force_ratio = tangential_force / normal_force
            return force_ratio > 0.8  # High friction coefficient suggests slip
        else:
            return False

    def get_object_properties(self):
        """Get fused object property estimates"""
        return self.object_properties.copy()

    def calibrate_sensors(self, calibration_data):
        """Calibrate force-torque sensor"""
        # Calculate bias from calibration data (when no external forces)
        self.ft_sensor['bias'] = np.mean(calibration_data, axis=0)
        self.ft_sensor['calibrated'] = True

class SlipDetectionAlgorithm:
    """Algorithm for detecting slip from tactile sensor data"""

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.pressure_history = []
        self.slip_threshold = 0.1  # Change threshold for slip detection

    def detect_from_tactile(self, pressure_grid):
        """Detect slip from tactile pressure patterns"""
        # Add current pressure to history
        self.pressure_history.append(pressure_grid.flatten())

        # Keep only recent history
        if len(self.pressure_history) > self.window_size:
            self.pressure_history.pop(0)

        if len(self.pressure_history) < 10:  # Need some history
            return False

        # Analyze changes in pressure distribution
        recent_changes = []
        for i in range(1, len(self.pressure_history)):
            change = np.mean(np.abs(self.pressure_history[i] - self.pressure_history[i-1]))
            recent_changes.append(change)

        # If recent changes are increasing rapidly, likely slip
        if len(recent_changes) >= 5:
            recent_avg_change = np.mean(recent_changes[-5:])
            overall_avg_change = np.mean(recent_changes)

            # If recent changes are significantly higher than overall average
            slip_detected = recent_avg_change > (overall_avg_change * 1.5) and \
                           recent_avg_change > self.slip_threshold

        return slip_detected

class ContactPointEstimator:
    """Estimate contact points from tactile sensor data"""

    def estimate_contact_points(self, pressure_grid):
        """Estimate contact points from pressure distribution"""
        contact_points = []

        # Find pressure peaks (potential contact points)
        rows, cols = pressure_grid.shape
        for i in range(rows):
            for j in range(cols):
                if pressure_grid[i, j] > 0.1:  # Threshold for contact detection
                    # Convert grid coordinates to real-world coordinates
                    # This depends on the tactile sensor layout
                    x_real = i * 0.01  # 1cm spacing example
                    y_real = j * 0.01  # 1cm spacing example
                    pressure_value = pressure_grid[i, j]

                    contact_points.append({
                        'position': (x_real, y_real, 0.0),
                        'pressure': pressure_value,
                        'grid_coords': (i, j)
                    })

        return contact_points
```

## Fault Detection and Isolation

Robust sensor fusion systems must handle sensor failures gracefully.

```python
class SensorFaultDetection:
    """Sensor fault detection and isolation for fusion systems"""

    def __init__(self):
        self.sensor_health = {}
        self.fault_detection_models = {}
        self.fusion_backup_strategies = {}
        self.health_history = {}

    def register_sensor_for_monitoring(self, sensor_id, sensor_type, nominal_params):
        """Register a sensor for health monitoring"""
        self.sensor_health[sensor_id] = {
            'type': sensor_type,
            'nominal_params': nominal_params,
            'health_score': 1.0,  # 1.0 = healthy, 0.0 = failed
            'last_update': time.time(),
            'fault_indicators': [],
            'status': 'nominal'  # nominal, degraded, failed
        }

        self.health_history[sensor_id] = []

        # Initialize fault detection model based on sensor type
        self.fault_detection_models[sensor_id] = self.initialize_fault_model(sensor_type)

    def initialize_fault_model(self, sensor_type):
        """Initialize fault detection model for sensor type"""
        if sensor_type == 'imu':
            return IMUFaultDetector()
        elif sensor_type == 'camera':
            return CameraFaultDetector()
        elif sensor_type == 'lidar':
            return LIDARFaultDetector()
        elif sensor_type == 'force_torque':
            return ForceTorqueFaultDetector()
        else:
            return GenericSensorFaultDetector()

    def update_sensor_health(self, sensor_id, sensor_data, timestamp):
        """Update health assessment for a sensor"""
        if sensor_id not in self.sensor_health:
            return True  # Sensor not registered, assume healthy

        # Run fault detection
        fault_indicators = self.fault_detection_models[sensor_id].detect_faults(
            sensor_data, self.sensor_health[sensor_id]
        )

        # Update health score based on fault indicators
        health_score = self.calculate_health_score(fault_indicators)

        # Update sensor health record
        self.sensor_health[sensor_id]['health_score'] = health_score
        self.sensor_health[sensor_id]['fault_indicators'] = fault_indicators
        self.sensor_health[sensor_id]['last_update'] = timestamp

        # Determine status
        if health_score < 0.2:
            self.sensor_health[sensor_id]['status'] = 'failed'
        elif health_score < 0.7:
            self.sensor_health[sensor_id]['status'] = 'degraded'
        else:
            self.sensor_health[sensor_id]['status'] = 'nominal'

        # Log health history
        self.health_history[sensor_id].append({
            'timestamp': timestamp,
            'health_score': health_score,
            'status': self.sensor_health[sensor_id]['status'],
            'fault_indicators': fault_indicators
        })

        # Keep only recent history
        if len(self.health_history[sensor_id]) > 1000:
            self.health_history[sensor_id] = self.health_history[sensor_id][-1000:]

        return self.sensor_health[sensor_id]['status'] != 'failed'

    def calculate_health_score(self, fault_indicators):
        """Calculate health score from fault indicators"""
        if not fault_indicators:
            return 1.0

        # Weight different fault types
        fault_weights = {
            'bias_drift': 0.7,
            'noise_increase': 0.5,
            'outlier_presence': 0.3,
            'periodicity_change': 0.4,
            'magnitude_change': 0.6
        }

        total_penalty = 0.0
        for indicator in fault_indicators:
            fault_type = indicator['type']
            severity = indicator['severity']
            weight = fault_weights.get(fault_type, 0.5)
            total_penalty += severity * weight

        # Health score decreases with total penalty
        health_score = max(0.0, 1.0 - total_penalty)
        return health_score

    def get_healthy_sensors(self):
        """Get list of healthy sensors"""
        healthy_sensors = []
        for sensor_id, health_info in self.sensor_health.items():
            if health_info['status'] in ['nominal', 'degraded']:
                healthy_sensors.append(sensor_id)
        return healthy_sensors

    def adapt_fusion_for_sensor_failures(self, fusion_system):
        """Adapt fusion system when sensors fail"""
        healthy_sensors = self.get_healthy_sensors()

        # Reconfigure fusion system to use only healthy sensors
        fusion_system.update_available_sensors(healthy_sensors)

        # If critical sensors have failed, switch to backup strategy
        critical_sensors_failed = self.check_critical_sensor_failures()
        if critical_sensors_failed:
            backup_strategy = self.get_backup_strategy(critical_sensors_failed)
            fusion_system.switch_to_backup_mode(backup_strategy)

    def check_critical_sensor_failures(self):
        """Check for critical sensor failures"""
        critical_failures = []

        for sensor_id, health_info in self.sensor_health.items():
            # Define critical sensors based on system requirements
            if health_info['type'] in ['imu', 'encoders'] and health_info['status'] == 'failed':
                critical_failures.append(sensor_id)

        return critical_failures

    def get_backup_strategy(self, failed_sensors):
        """Get backup strategy for failed sensors"""
        # This would return appropriate backup fusion strategies
        # For example, if IMU fails, use vision-based estimation
        backup_strategies = {}

        for sensor_id in failed_sensors:
            sensor_type = self.sensor_health[sensor_id]['type']
            if sensor_type == 'imu':
                backup_strategies[sensor_id] = 'vision_based_orientation'
            elif sensor_type == 'encoders':
                backup_strategies[sensor_id] = 'vision_based_position'

        return backup_strategies

class IMUFaultDetector:
    """Fault detection for IMU sensors"""

    def __init__(self):
        self.accel_history = []
        self.gyro_history = []
        self.mag_history = []
        self.bias_estimate = np.zeros(6)  # [accel_bias, gyro_bias]
        self.bias_window = 100

    def detect_faults(self, imu_data, sensor_health):
        """Detect faults in IMU data"""
        faults = []

        # Separate data
        accel = np.array(imu_data[:3])
        gyro = np.array(imu_data[3:6])
        mag = np.array(imu_data[6:]) if len(imu_data) > 6 else None

        # Update histories
        self.accel_history.append(accel)
        self.gyro_history.append(gyro)
        if mag is not None:
            self.mag_history.append(mag)

        # Keep only recent history
        if len(self.accel_history) > 100:
            self.accel_history.pop(0)
            self.gyro_history.pop(0)
            if self.mag_history:
                self.mag_history.pop(0)

        # Check for various fault types
        if len(self.accel_history) >= 10:
            # Bias drift detection
            recent_bias = np.mean(self.accel_history[-10:], axis=0)
            expected_bias = np.array([0, 0, -9.81])  # Gravity in z-direction
            bias_drift = np.linalg.norm(recent_bias - expected_bias)

            if bias_drift > 1.0:  # 1m/s^2 threshold
                faults.append({
                    'type': 'bias_drift',
                    'severity': min(1.0, bias_drift / 5.0),
                    'description': f'Accelerometer bias drift detected: {bias_drift:.3f}'
                })

            # Noise level detection
            accel_var = np.var(self.accel_history[-10:], axis=0)
            if np.mean(accel_var) > 0.1:  # High noise threshold
                faults.append({
                    'type': 'noise_increase',
                    'severity': min(1.0, np.mean(accel_var) / 0.5),
                    'description': f'Increased accelerometer noise: {np.mean(accel_var):.3f}'
                })

        # Gyroscope checks
        if len(self.gyro_history) >= 10:
            gyro_mean = np.mean(self.gyro_history[-10:], axis=0)
            if np.linalg.norm(gyro_mean) > 0.1:  # Should be close to 0 when stationary
                faults.append({
                    'type': 'bias_drift',
                    'severity': min(1.0, np.linalg.norm(gyro_mean) / 0.5),
                    'description': f'Gyroscope bias detected: {np.linalg.norm(gyro_mean):.3f}'
                })

        return faults

class CameraFaultDetector:
    """Fault detection for camera sensors"""

    def __init__(self):
        self.image_quality_history = []
        self.feature_count_history = []
        self.exposure_history = []

    def detect_faults(self, camera_data, sensor_health):
        """Detect faults in camera data"""
        faults = []

        # Extract image features from camera data
        # This would typically involve image processing
        image_quality = camera_data.get('sharpness', 0.5)
        feature_count = len(camera_data.get('features', []))
        exposure = camera_data.get('exposure', 0.5)

        self.image_quality_history.append(image_quality)
        self.feature_count_history.append(feature_count)
        self.exposure_history.append(exposure)

        # Keep only recent history
        if len(self.image_quality_history) > 50:
            self.image_quality_history.pop(0)
            self.feature_count_history.pop(0)
            self.exposure_history.pop(0)

        # Quality checks
        if image_quality < 0.2:  # Poor sharpness
            faults.append({
                'type': 'image_quality',
                'severity': 1.0 - image_quality,
                'description': f'Poor image quality detected: {image_quality:.3f}'
            })

        # Feature count checks
        if len(self.feature_count_history) >= 5:
            recent_avg = np.mean(self.feature_count_history[-5:])
            if recent_avg < 10:  # Too few features for reliable processing
                faults.append({
                    'type': 'feature_availability',
                    'severity': max(0.0, (50 - recent_avg) / 50.0),
                    'description': f'Insufficient features for tracking: {recent_avg:.1f}'
                })

        # Exposure checks
        if exposure < 0.1 or exposure > 0.9:  # Too dark or too bright
            faults.append({
                'type': 'exposure_issue',
                'severity': 0.5,
                'description': f'Exposure outside optimal range: {exposure:.3f}'
            })

        return faults

class GenericSensorFaultDetector:
    """Generic fault detector for any sensor type"""

    def __init__(self):
        self.data_history = []
        self.nominal_stats = None

    def detect_faults(self, sensor_data, sensor_health):
        """Generic fault detection for arbitrary sensor data"""
        faults = []

        # Convert to numpy array for processing
        if isinstance(sensor_data, (list, tuple)):
            data_array = np.array(sensor_data)
        else:
            data_array = np.atleast_1d(sensor_data)

        self.data_history.append(data_array)

        # Keep only recent history
        if len(self.data_history) > 100:
            self.data_history.pop(0)

        if len(self.data_history) >= 10:
            # Calculate statistics
            recent_data = np.array(self.data_history[-10:])
            current_mean = np.mean(recent_data, axis=0)
            current_std = np.std(recent_data, axis=0)

            # Compare with nominal parameters if available
            nominal_params = sensor_health.get('nominal_params', {})
            nominal_mean = nominal_params.get('mean', np.zeros_like(current_mean))
            nominal_std = nominal_params.get('std', np.ones_like(current_std))

            # Check for significant deviations
            mean_deviation = np.abs(current_mean - nominal_mean) / (nominal_std + 1e-6)
            std_deviation = np.abs(current_std - nominal_std) / (nominal_std + 1e-6)

            if np.any(mean_deviation > 3.0):  # 3-sigma rule
                faults.append({
                    'type': 'bias_drift',
                    'severity': float(np.mean(mean_deviation)),
                    'description': f'Significant bias drift detected: {np.mean(mean_deviation):.3f}'
                })

            if np.any(std_deviation > 2.0):  # Large change in noise
                faults.append({
                    'type': 'noise_change',
                    'severity': float(np.mean(std_deviation)),
                    'description': f'Significant noise change detected: {np.mean(std_deviation):.3f}'
                })

        return faults
```

## Hands-on Exercise: Implementing a Multi-Sensor Fusion System

In this exercise, you'll implement a complete multi-sensor fusion system that combines data from multiple sensors to estimate robot state.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of linear algebra and probability

### Exercise Objectives
1. Implement a Kalman filter for state estimation
2. Create a simulation environment with multiple sensors
3. Integrate sensor data using fusion techniques
4. Handle sensor failures and adapt the system
5. Evaluate fusion performance

### Exercise Steps

1. Implement the core fusion algorithm
2. Create sensor simulation
3. Add fault detection capabilities
4. Test with various scenarios
5. Evaluate and tune the system

### Expected Outcome
You should have a working multi-sensor fusion system that can estimate robot state using multiple sensors, handle sensor failures gracefully, and maintain accurate estimates despite noise and uncertainties.

### Sample Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class MultiSensorFusionExercise:
    """Hands-on exercise for multi-sensor fusion implementation"""

    def __init__(self):
        # Robot state [x, y, theta, vx, vy, omega]
        self.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Fusion filter
        self.fusion_filter = ExtendedKalmanFilter(state_dim=6, measurement_dim=4)

        # Simulated sensors
        self.sensors = {
            'imu': {
                'enabled': True,
                'noise_std': 0.01,
                'bias': np.array([0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001]),
                'data': np.zeros(6)
            },
            'camera': {
                'enabled': True,
                'noise_std': 0.02,
                'bias': np.array([0.0, 0.0]),
                'data': np.zeros(2)
            },
            'encoders': {
                'enabled': True,
                'noise_std': 0.005,
                'bias': np.array([0.0, 0.0]),
                'data': np.zeros(2)
            }
        }

        # Simulation parameters
        self.dt = 0.01  # 100 Hz
        self.simulation_time = 10.0  # 10 seconds
        self.steps = int(self.simulation_time / self.dt)

        # Results tracking
        self.state_history = []
        self.estimate_history = []
        self.measurement_history = []
        self.time_history = []

        # Fault injection (for testing robustness)
        self.injected_faults = []

    def simulate_robot_motion(self, control_input, dt):
        """Simulate robot motion based on control input"""
        x, y, theta, vx, vy, omega = self.true_state

        # Simple differential drive model
        v_linear = control_input[0]  # Linear velocity
        v_angular = control_input[1]  # Angular velocity

        # Update state with dynamics
        new_theta = theta + omega * dt
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = v_linear * np.cos(new_theta)  # Convert to world frame
        new_vy = v_linear * np.sin(new_theta)
        new_omega = v_angular

        self.true_state = np.array([new_x, new_y, new_theta, new_vx, new_vy, new_omega])

        return self.true_state.copy()

    def simulate_sensors(self, true_state):
        """Simulate sensor measurements"""
        measurements = {}

        # IMU: measures orientation and angular velocity
        if self.sensors['imu']['enabled']:
            imu_measurement = np.array([
                true_state[2],  # theta (orientation)
                true_state[5],  # omega (angular velocity)
                true_state[3],  # vx (linear velocity x)
                true_state[4],  # vy (linear velocity y)
                0,  # ax (would be from IMU)
                0   # ay (would be from IMU)
            ])

            # Add noise and bias
            noise = np.random.normal(0, self.sensors['imu']['noise_std'], 6)
            bias = self.sensors['imu']['bias']
            measurements['imu'] = imu_measurement + noise + bias

        # Camera: measures position
        if self.sensors['camera']['enabled']:
            camera_measurement = np.array([true_state[0], true_state[1]])  # x, y position

            # Add noise and bias
            noise = np.random.normal(0, self.sensors['camera']['noise_std'], 2)
            bias = self.sensors['camera']['bias']
            measurements['camera'] = camera_measurement + noise + bias

        # Encoders: measure wheel velocities (simplified as position change)
        if self.sensors['encoders']['enabled']:
            # In a real system, encoders measure wheel rotations
            # Here we'll simulate position change
            encoder_measurement = np.array([true_state[0], true_state[1]])  # x, y position

            # Add noise and bias
            noise = np.random.normal(0, self.sensors['encoders']['noise_std'], 2)
            bias = self.sensors['encoders']['bias']
            measurements['encoders'] = encoder_measurement + noise + bias

        return measurements

    def inject_sensor_fault(self, sensor_id, fault_type='bias_drift', severity=0.1):
        """Inject a fault into a sensor for testing"""
        if sensor_id in self.sensors:
            if fault_type == 'bias_drift':
                # Add bias to the sensor
                if sensor_id == 'imu':
                    fault_bias = np.random.normal(0, severity, 6)
                    self.sensors[sensor_id]['bias'] += fault_bias
                elif sensor_id == 'camera':
                    fault_bias = np.random.normal(0, severity, 2)
                    self.sensors[sensor_id]['bias'] += fault_bias
                elif sensor_id == 'encoders':
                    fault_bias = np.random.normal(0, severity, 2)
                    self.sensors[sensor_id]['bias'] += fault_bias

            elif fault_type == 'noise_increase':
                # Increase sensor noise
                self.sensors[sensor_id]['noise_std'] *= (1 + severity)

            elif fault_type == 'complete_failure':
                # Disable the sensor
                self.sensors[sensor_id]['enabled'] = False

            # Log the fault
            self.injected_faults.append({
                'sensor_id': sensor_id,
                'fault_type': fault_type,
                'severity': severity,
                'time': len(self.state_history) * self.dt
            })

    def run_simulation(self, inject_faults=True):
        """Run the complete simulation"""
        print("Starting Multi-Sensor Fusion Simulation...")

        # Inject faults at specific times for testing
        if inject_faults:
            # Inject fault after 3 seconds
            fault_step = int(3.0 / self.dt)
            self.inject_sensor_fault('camera', 'bias_drift', 0.05)

            # Inject another fault after 6 seconds
            fault_step2 = int(6.0 / self.dt)
            self.inject_sensor_fault('imu', 'noise_increase', 0.5)

        for step in range(self.steps):
            # Simulate time
            current_time = step * self.dt

            # Generate control input (simple trajectory)
            control_input = self.generate_control_input(current_time)

            # Simulate robot motion
            true_state = self.simulate_robot_motion(control_input, self.dt)

            # Simulate sensors
            measurements = self.simulate_sensors(true_state)

            # Perform sensor fusion
            estimated_state = self.perform_fusion(measurements, self.dt)

            # Store results
            self.state_history.append(true_state.copy())
            self.estimate_history.append(estimated_state.copy())
            self.measurement_history.append(measurements.copy())
            self.time_history.append(current_time)

            # Print progress
            if step % 1000 == 0:
                error = np.linalg.norm(true_state[:2] - estimated_state[:2])
                print(f"Time: {current_time:.2f}s, Position Error: {error:.3f}m")

        print("Simulation completed!")

        return {
            'true_states': np.array(self.state_history),
            'estimates': np.array(self.estimate_history),
            'measurements': self.measurement_history,
            'times': np.array(self.time_history)
        }

    def generate_control_input(self, time):
        """Generate control input for robot motion"""
        # Create a circular trajectory
        radius = 1.0
        angular_freq = 0.5  # rad/s

        v_linear = radius * angular_freq  # Constant linear velocity for circular motion
        v_angular = angular_freq          # Constant angular velocity

        # Add some variation
        v_linear += 0.1 * np.sin(2 * np.pi * time * 0.3)  # Small speed variation
        v_angular += 0.05 * np.cos(2 * np.pi * time * 0.2)  # Small angular variation

        return np.array([v_linear, v_angular])

    def perform_fusion(self, measurements, dt):
        """Perform sensor fusion using EKF"""
        # Prediction step
        self.fusion_filter.predict(dt)

        # Update with each available measurement
        for sensor_id, measurement in measurements.items():
            if self.sensors[sensor_id]['enabled']:
                # Convert measurement to appropriate format for fusion
                if sensor_id == 'imu':
                    # Use orientation and velocities
                    fused_measurement = np.array([
                        measurement[0],  # theta
                        measurement[1],  # omega
                        measurement[2],  # vx
                        measurement[3]   # vy
                    ])
                elif sensor_id in ['camera', 'encoders']:
                    # Use position measurements
                    fused_measurement = np.array([
                        measurement[0],  # x
                        measurement[1]   # y
                    ])

                # Update filter
                self.fusion_filter.update(fused_measurement)

        # Return current state estimate
        return self.fusion_filter.get_state()

    def evaluate_performance(self, results):
        """Evaluate fusion performance"""
        true_states = results['true_states']
        estimates = results['estimates']

        # Calculate errors
        position_errors = []
        orientation_errors = []

        for true, est in zip(true_states, estimates):
            pos_error = np.linalg.norm(true[:2] - est[:2])  # x, y position error
            orientation_error = abs(true[2] - est[2])       # theta orientation error

            position_errors.append(pos_error)
            orientation_errors.append(orientation_error)

        # Calculate statistics
        avg_pos_error = np.mean(position_errors)
        max_pos_error = np.max(position_errors)
        std_pos_error = np.std(position_errors)

        avg_orient_error = np.mean(orientation_errors)
        max_orient_error = np.max(orientation_errors)
        std_orient_error = np.std(orientation_errors)

        performance_metrics = {
            'position': {
                'avg_error': avg_pos_error,
                'max_error': max_pos_error,
                'std_error': std_pos_error,
                'rmse': np.sqrt(np.mean(np.array(position_errors)**2))
            },
            'orientation': {
                'avg_error': avg_orient_error,
                'max_error': max_orient_error,
                'std_error': std_orient_error,
                'rmse': np.sqrt(np.mean(np.array(orientation_errors)**2))
            }
        }

        print("\n=== Performance Evaluation ===")
        print(f"Position RMSE: {performance_metrics['position']['rmse']:.4f}m")
        print(f"Position Max Error: {performance_metrics['position']['max_error']:.4f}m")
        print(f"Orientation RMSE: {performance_metrics['orientation']['rmse']:.4f}rad")
        print(f"Orientation Max Error: {performance_metrics['orientation']['max_error']:.4f}rad")

        # Check for fault handling effectiveness
        if self.injected_faults:
            print("\nFault Handling:")
            for fault in self.injected_faults:
                print(f"  - {fault['fault_type']} on {fault['sensor_id']} at {fault['time']:.1f}s")

        return performance_metrics

    def visualize_results(self, results):
        """Visualize the simulation results"""
        true_states = results['true_states']
        estimates = results['estimates']
        times = results['times']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Trajectory
        ax1 = axes[0, 0]
        ax1.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True Trajectory', linewidth=2)
        ax1.plot(estimates[:, 0], estimates[:, 1], 'r--', label='Estimated Trajectory', linewidth=2)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Robot Trajectory')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Position Error Over Time
        ax2 = axes[0, 1]
        position_errors = [np.linalg.norm(true[:2] - est[:2]) for true, est in zip(true_states, estimates)]
        ax2.plot(times, position_errors, 'r-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error Over Time')
        ax2.grid(True)

        # Plot 3: Orientation Comparison
        ax3 = axes[1, 0]
        ax3.plot(times, true_states[:, 2], 'b-', label='True Orientation', linewidth=2)
        ax3.plot(times, estimates[:, 2], 'r--', label='Estimated Orientation', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Orientation (rad)')
        ax3.set_title('Orientation Comparison')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Velocity Comparison
        ax4 = axes[1, 1]
        ax4.plot(times, true_states[:, 3], 'b-', label='True Vx', linewidth=2)
        ax4.plot(times, estimates[:, 3], 'r--', label='Estimated Vx', linewidth=2)
        ax4.plot(times, true_states[:, 4], 'g-', label='True Vy', linewidth=2)
        ax4.plot(times, estimates[:, 4], 'm--', label='Estimated Vy', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocity Comparison')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

        # Also create a 3D plot of the trajectory
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection='3d')

        # Use orientation as a third dimension (color-coded)
        scatter = ax.scatter(
            true_states[:, 0], true_states[:, 1], times,
            c=true_states[:, 2], cmap='viridis', s=1, alpha=0.7
        )

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Time (s)')
        ax.set_title('3D Trajectory with Orientation')

        plt.colorbar(scatter)
        plt.show()

    def run_complete_exercise(self):
        """Run the complete multi-sensor fusion exercise"""
        print("Multi-Sensor Fusion Exercise")
        print("=" * 50)

        # Run simulation
        results = self.run_simulation()

        # Evaluate performance
        metrics = self.evaluate_performance(results)

        # Visualize results
        self.visualize_results(results)

        # Test fault handling
        print("\nTesting Fault Handling Capability...")
        self.reset_system()

        # Run with a sensor failure
        self.sensors['camera']['enabled'] = False
        fault_results = self.run_simulation(inject_faults=False)

        print("Simulation completed with camera failure.")
        print("Notice how the system adapts to the missing sensor data.")

        return results, metrics

    def reset_system(self):
        """Reset the system to initial state"""
        self.true_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.fusion_filter = ExtendedKalmanFilter(state_dim=6, measurement_dim=4)

        # Reset sensor states
        for sensor_id in self.sensors:
            self.sensors[sensor_id]['enabled'] = True
            self.sensors[sensor_id]['bias'] = np.zeros_like(self.sensors[sensor_id]['bias'])

        # Clear histories
        self.state_history = []
        self.estimate_history = []
        self.measurement_history = []
        self.time_history = []
        self.injected_faults = []

# Example usage
def run_multi_sensor_fusion_exercise():
    """Run the complete multi-sensor fusion exercise"""
    exercise = MultiSensorFusionExercise()
    results, metrics = exercise.run_complete_exercise()

    print("\nExercise completed successfully!")
    print(f"Final position error: {metrics['position']['rmse']:.4f}m")
    print(f"Final orientation error: {metrics['orientation']['rmse']:.4f}rad")

    return exercise, results, metrics

# Uncomment to run the exercise
# exercise, results, metrics = run_multi_sensor_fusion_exercise()
```

## Advanced Fusion Techniques

### Information-Theoretic Fusion

Information-theoretic approaches use concepts like entropy and mutual information to optimally combine sensor data.

```python
class InformationTheoreticFusion:
    """Information-theoretic approach to sensor fusion"""

    def __init__(self):
        self.sensor_entropies = {}
        self.mutual_information_matrix = None
        self.fusion_weights = {}

    def calculate_entropy(self, sensor_data, bins=50):
        """Calculate entropy of sensor data distribution"""
        # Flatten data for histogram calculation
        flat_data = np.array(sensor_data).flatten()

        # Create histogram
        hist, bin_edges = np.histogram(flat_data, bins=bins)

        # Normalize to get probabilities
        prob = hist / np.sum(hist)

        # Calculate entropy (excluding zero probabilities)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))

        return entropy

    def calculate_mutual_information(self, sensor1_data, sensor2_data, bins=20):
        """Calculate mutual information between two sensors"""
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(
            np.array(sensor1_data).flatten(),
            np.array(sensor2_data).flatten(),
            bins=bins
        )

        # Normalize to get joint probability
        pxy = hist_2d / np.sum(hist_2d)

        # Marginal probabilities
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Calculate mutual information
        # MI = sum(sum(p(x,y) * log(p(x,y)/(p(x)*p(y))))
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

        return mi

    def information_based_weighting(self, sensor_data_list):
        """Calculate fusion weights based on information content"""
        n_sensors = len(sensor_data_list)

        # Calculate entropy for each sensor
        entropies = []
        for data in sensor_data_list:
            entropy = self.calculate_entropy(data)
            entropies.append(entropy)

        # Calculate mutual information between sensors
        mutual_info_matrix = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    mi = self.calculate_mutual_information(sensor_data_list[i], sensor_data_list[j])
                    mutual_info_matrix[i, j] = mi

        # Calculate weights based on information content
        # Lower entropy = more information = higher weight
        # Lower mutual information with other sensors = more unique information = higher weight
        weights = []
        for i in range(n_sensors):
            # Information content based on entropy
            info_content = 1.0 / (entropies[i] + 1e-6)  # Add small value to avoid division by zero

            # Uniqueness based on mutual information with other sensors
            avg_mi = np.mean(mutual_info_matrix[i]) if n_sensors > 1 else 0
            uniqueness = 1.0 / (avg_mi + 1e-6)

            # Combined weight
            weight = info_content * uniqueness
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return weights

    def fuse_with_information_weights(self, sensor_measurements):
        """Fuse measurements using information-theoretic weights"""
        # Calculate weights
        weights = self.information_based_weighting(sensor_measurements)

        # Weighted average fusion
        fused_measurement = np.zeros_like(sensor_measurements[0])
        for i, measurement in enumerate(sensor_measurements):
            fused_measurement += weights[i] * np.array(measurement)

        return fused_measurement, weights
```

### Deep Learning-Based Fusion

Modern approaches use neural networks to learn optimal fusion strategies from data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepFusionNetwork(nn.Module):
    """Deep neural network for sensor fusion"""

    def __init__(self, input_dims, output_dim, fusion_type='dense'):
        super(DeepFusionNetwork, self).__init__()

        self.input_dims = input_dims  # List of dimensions for each sensor
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        if fusion_type == 'dense':
            # Dense fusion network
            total_input_dim = sum(input_dims)
            self.network = nn.Sequential(
                nn.Linear(total_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.sensor_embeddings = nn.ModuleList([
                nn.Linear(dim, 64) for dim in input_dims
            ])
            self.attention_layer = nn.MultiheadAttention(embed_dim=64, num_heads=4)
            self.output_layer = nn.Linear(64, output_dim)

        elif fusion_type == 'graph':
            # Graph neural network fusion (simplified)
            self.sensor_processors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                ) for dim in input_dims
            ])
            self.graph_conv = nn.Linear(64 * len(input_dims), 128)
            self.output_layer = nn.Linear(128, output_dim)

    def forward(self, sensor_inputs):
        """Forward pass through the fusion network"""
        if self.fusion_type == 'dense':
            # Concatenate all sensor inputs
            concatenated = torch.cat(sensor_inputs, dim=-1)
            return self.network(concatenated)

        elif self.fusion_type == 'attention':
            # Process each sensor through embedding
            embedded_sensors = []
            for i, (sensor_input, processor) in enumerate(zip(sensor_inputs, self.sensor_embeddings)):
                embedded = processor(sensor_input)
                embedded_sensors.append(embedded.unsqueeze(0))  # Add sequence dimension

            # Stack and apply attention
            stacked = torch.cat(embedded_sensors, dim=0)
            attended, _ = self.attention_layer(stacked, stacked, stacked)

            # Average across sensors and apply output layer
            fused = torch.mean(attended, dim=0)
            return self.output_layer(fused)

        elif self.fusion_type == 'graph':
            # Process each sensor independently
            processed_sensors = []
            for sensor_input, processor in zip(sensor_inputs, self.sensor_processors):
                processed = processor(sensor_input)
                processed_sensors.append(processed)

            # Concatenate and apply graph convolution
            concatenated = torch.cat(processed_sensors, dim=-1)
            graph_output = torch.relu(self.graph_conv(concatenated))
            return self.output_layer(graph_output)

class DeepSensorFusion:
    """Deep learning-based sensor fusion system"""

    def __init__(self, sensor_configs, output_dim):
        self.sensor_configs = sensor_configs  # List of (sensor_type, input_dim)
        self.output_dim = output_dim

        # Extract input dimensions
        input_dims = [config[1] for config in sensor_configs]

        # Create fusion network
        self.fusion_network = DeepFusionNetwork(input_dims, output_dim, fusion_type='attention')
        self.optimizer = optim.Adam(self.fusion_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Training data
        self.training_data = []
        self.validation_data = []

    def prepare_training_data(self, sensor_data_list, ground_truth):
        """Prepare training data from sensor measurements and ground truth"""
        # sensor_data_list: list of [batch_size, sensor_dim] tensors
        # ground_truth: [batch_size, output_dim] tensor

        return list(zip(sensor_data_list, ground_truth))

    def train_fusion_network(self, training_data, epochs=100):
        """Train the fusion network"""
        self.fusion_network.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (sensor_inputs, target) in enumerate(training_data):
                # Forward pass
                output = self.fusion_network(sensor_inputs)

                # Calculate loss
                loss = self.criterion(output, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(training_data)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

    def fuse_sensors(self, sensor_inputs):
        """Fuse sensor inputs using the trained network"""
        self.fusion_network.eval()

        with torch.no_grad():
            # Convert to tensors if needed
            tensor_inputs = []
            for inp in sensor_inputs:
                if not isinstance(inp, torch.Tensor):
                    inp = torch.FloatTensor(inp).unsqueeze(0)  # Add batch dimension
                tensor_inputs.append(inp)

            # Fuse using network
            fused_output = self.fusion_network(tensor_inputs)

            return fused_output.squeeze(0).numpy()  # Remove batch dimension

    def evaluate_fusion(self, test_data):
        """Evaluate fusion performance"""
        self.fusion_network.eval()

        total_error = 0.0
        n_samples = 0

        with torch.no_grad():
            for sensor_inputs, target in test_data:
                output = self.fusion_network(sensor_inputs)
                error = torch.mean((output - target)**2)
                total_error += error.item()
                n_samples += 1

        mse = total_error / n_samples
        rmse = np.sqrt(mse)

        return {'mse': mse, 'rmse': rmse}
```

## Sensor Management and Resource Allocation

Efficiently managing computational resources for sensor processing is crucial for real-time applications.

```python
class SensorResourceManager:
    """Manage computational resources for sensor processing"""

    def __init__(self):
        self.sensors = {}
        self.resource_limits = {
            'cpu_percent': 80.0,  # Maximum CPU usage percent
            'memory_mb': 1024,    # Maximum memory usage in MB
            'bandwidth_kbps': 10000  # Maximum network bandwidth in kbps
        }
        self.priority_weights = {}  # Priority weights for each sensor
        self.current_resource_usage = {
            'cpu': 0.0,
            'memory': 0.0,
            'bandwidth': 0.0
        }

    def register_sensor(self, sensor_id, processing_requirements, priority=1.0):
        """Register a sensor with its processing requirements"""
        self.sensors[sensor_id] = {
            'requirements': processing_requirements,  # {'cpu': 5.0, 'memory': 10.0, 'bandwidth': 100.0}
            'priority': priority,
            'enabled': True,
            'last_update_time': 0,
            'processing_interval': 0.01  # Default 100 Hz
        }
        self.priority_weights[sensor_id] = priority

    def allocate_resources(self, available_resources):
        """Allocate resources to sensors based on priority and requirements"""
        # Sort sensors by priority (highest first)
        sorted_sensors = sorted(
            self.sensors.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )

        allocated_resources = {}
        remaining_resources = available_resources.copy()

        for sensor_id, sensor_info in sorted_sensors:
            if not sensor_info['enabled']:
                continue

            # Check if we can allocate resources to this sensor
            can_allocate = True
            for resource, required in sensor_info['requirements'].items():
                if remaining_resources.get(resource, 0) < required:
                    can_allocate = False
                    break

            if can_allocate:
                # Allocate resources
                allocated_resources[sensor_id] = sensor_info['requirements'].copy()
                for resource, required in sensor_info['requirements'].items():
                    remaining_resources[resource] -= required
            else:
                # Disable sensor if we can't allocate resources
                self.sensors[sensor_id]['enabled'] = False
                print(f"Warning: Disabling sensor {sensor_id} due to resource constraints")

        return allocated_resources

    def dynamic_resource_allocation(self, current_usage, performance_metrics):
        """Dynamically adjust resource allocation based on performance"""
        # Adjust sensor priorities based on performance
        for sensor_id, metrics in performance_metrics.items():
            if sensor_id in self.sensors:
                # If performance is poor, increase priority
                if metrics.get('accuracy', 1.0) < 0.7:
                    self.sensors[sensor_id]['priority'] *= 1.1
                # If performance is good and resource usage is high, decrease priority
                elif (metrics.get('accuracy', 1.0) > 0.9 and
                      current_usage.get('cpu', 0) > 70.0):
                    self.sensors[sensor_id]['priority'] *= 0.95

                # Constrain priority to reasonable range
                self.sensors[sensor_id]['priority'] = np.clip(
                    self.sensors[sensor_id]['priority'], 0.1, 10.0
                )

    def get_optimal_sampling_rates(self, computational_budget):
        """Calculate optimal sampling rates for sensors given computational budget"""
        # Use utility-based allocation
        # Each sensor has a utility function based on its importance and diminishing returns
        sensor_utilities = {}

        for sensor_id, sensor_info in self.sensors.items():
            if not sensor_info['enabled']:
                continue

            # Utility decreases with higher sampling rates (diminishing returns)
            # and increases with priority
            base_utility = sensor_info['priority']
            max_rate = 1.0 / sensor_info['requirements'].get('cpu', 0.01)  # Inverse of CPU requirement
            current_rate = 1.0 / sensor_info['processing_interval']

            # Diminishing returns: utility increases but at decreasing rate
            utility = base_utility * (1 - np.exp(-current_rate / max_rate))
            sensor_utilities[sensor_id] = utility

        # Normalize utilities
        total_utility = sum(sensor_utilities.values()) if sensor_utilities else 1.0
        normalized_utilities = {k: v/total_utility for k, v in sensor_utilities.items()}

        # Allocate budget based on normalized utilities
        optimal_rates = {}
        for sensor_id, norm_utility in normalized_utilities.items():
            optimal_rates[sensor_id] = computational_budget * norm_utility

        return optimal_rates
```

## Evaluation Metrics for Fusion Systems

Evaluating the performance of multi-sensor fusion systems is critical for ensuring reliable operation in humanoid robots. Here are key metrics and methodologies:

```python
class FusionSystemEvaluator:
    """Comprehensive evaluation framework for multi-sensor fusion systems"""

    def __init__(self):
        self.metrics = {}
        self.baseline_comparisons = {}
        self.robustness_tests = []

    def calculate_basic_metrics(self, true_states, estimated_states):
        """Calculate basic performance metrics"""
        # Position RMSE
        position_errors = [np.linalg.norm(true[:2] - est[:2])
                          for true, est in zip(true_states, estimated_states)]
        pos_rmse = np.sqrt(np.mean(np.array(position_errors)**2))

        # Orientation RMSE
        orientation_errors = [abs(true[2] - est[2])
                             for true, est in zip(true_states, estimated_states)]
        orient_rmse = np.sqrt(np.mean(np.array(orientation_errors)**2))

        # Velocity RMSE
        velocity_errors = [np.linalg.norm(true[3:] - est[3:])
                          for true, est in zip(true_states, estimated_states)]
        vel_rmse = np.sqrt(np.mean(np.array(velocity_errors)**2))

        # Calculate ATE (Absolute Trajectory Error)
        ate = np.mean(position_errors)

        # Calculate RTE (Relative Trajectory Error)
        rte = self.calculate_relative_errors(true_states, estimated_states)

        return {
            'position_rmse': pos_rmse,
            'orientation_rmse': orient_rmse,
            'velocity_rmse': vel_rmse,
            'ate': ate,
            'rte': rte,
            'max_position_error': max(position_errors),
            'std_position_error': np.std(position_errors)
        }

    def calculate_relative_errors(self, true_states, estimated_states):
        """Calculate relative trajectory errors"""
        rel_errors = []

        for i in range(1, len(true_states)):
            # True relative transformation
            true_rel_pos = true_states[i][:2] - true_states[i-1][:2]
            est_rel_pos = estimated_states[i][:2] - estimated_states[i-1][:2]

            # Calculate relative error
            rel_error = np.linalg.norm(true_rel_pos - est_rel_pos)
            rel_errors.append(rel_error)

        return np.mean(rel_errors)

    def calculate_consistency_metrics(self, estimates, covariances):
        """Calculate consistency metrics (NEES, NIS)"""
        # Normalized Estimation Error Squared (NEES)
        nees_values = []
        for i, (est, cov, true) in enumerate(zip(estimates, covariances, self.ground_truth)):
            error = est - true
            try:
                inv_cov = np.linalg.inv(cov)
                nees = error.T @ inv_cov @ error
                nees_values.append(nees)
            except np.linalg.LinAlgError:
                # If covariance is singular, skip this calculation
                continue

        # Calculate NEES average and consistency
        if nees_values:
            avg_nees = np.mean(nees_values)
            nees_consistency = {
                'avg_nees': avg_nees,
                'consistent': avg_nees < len(estimates[0]) if estimates else 0  # Degrees of freedom
            }
        else:
            nees_consistency = {'avg_nees': float('inf'), 'consistent': False}

        return nees_consistency

    def calculate_efficiency_metrics(self, processing_times, update_rates):
        """Calculate computational efficiency metrics"""
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        std_processing_time = np.std(processing_times)

        # Real-time factor (should be < 1.0 for real-time operation)
        avg_update_interval = 1.0 / np.mean(update_rates) if update_rates else 1.0
        real_time_factor = avg_processing_time / avg_update_interval

        return {
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max_processing_time * 1000,
            'std_processing_time_ms': std_processing_time * 1000,
            'real_time_factor': real_time_factor,
            'utilization_percent': min(100.0, real_time_factor * 100)
        }

    def calculate_robustness_metrics(self, sensor_health_data, degradation_data):
        """Calculate robustness metrics under various conditions"""
        # Sensor outage resilience
        outage_resilience = self.calculate_outage_resilience(sensor_health_data)

        # Noise tolerance
        noise_tolerance = self.calculate_noise_tolerance(degradation_data)

        # Failure detection capability
        failure_detection = self.calculate_failure_detection_capability(sensor_health_data)

        return {
            'outage_resilience': outage_resilience,
            'noise_tolerance': noise_tolerance,
            'failure_detection_rate': failure_detection
        }

    def calculate_outage_resilience(self, sensor_health_data):
        """Calculate how well the system handles sensor outages"""
        # This would analyze performance degradation when sensors fail
        resilience_score = 0.8  # Placeholder - would be calculated from actual data
        return resilience_score

    def calculate_noise_tolerance(self, degradation_data):
        """Calculate system's tolerance to increased sensor noise"""
        # This would analyze how performance degrades with increasing noise
        tolerance_score = 0.7  # Placeholder - would be calculated from actual data
        return tolerance_score

    def calculate_failure_detection_capability(self, sensor_health_data):
        """Calculate how effectively the system detects sensor failures"""
        # This would analyze fault detection rates
        detection_rate = 0.95  # Placeholder - would be calculated from actual data
        return detection_rate

    def run_comprehensive_evaluation(self, fusion_system, test_scenarios):
        """Run comprehensive evaluation across multiple scenarios"""
        evaluation_results = {}

        for scenario_name, scenario_data in test_scenarios.items():
            # Run scenario
            results = self.run_single_scenario(fusion_system, scenario_data)

            # Calculate metrics
            metrics = self.calculate_scenario_metrics(results)
            evaluation_results[scenario_name] = metrics

        # Aggregate results
        overall_metrics = self.aggregate_results(evaluation_results)

        return {
            'scenario_results': evaluation_results,
            'overall_metrics': overall_metrics,
            'recommendations': self.generate_recommendations(overall_metrics)
        }

    def run_single_scenario(self, fusion_system, scenario_data):
        """Run fusion system on a single test scenario"""
        # This would execute the fusion system with the given scenario
        # and collect performance data
        return {
            'true_states': scenario_data.get('ground_truth', []),
            'estimated_states': [],
            'processing_times': [],
            'covariances': []
        }

    def calculate_scenario_metrics(self, results):
        """Calculate metrics for a single scenario"""
        # Placeholder implementation
        return {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.88
        }

    def aggregate_results(self, evaluation_results):
        """Aggregate results across all scenarios"""
        # Calculate overall performance across scenarios
        accuracy_values = [result['accuracy'] for result in evaluation_results.values()]
        precision_values = [result['precision'] for result in evaluation_results.values()]
        recall_values = [result['recall'] for result in evaluation_results.values()]

        return {
            'avg_accuracy': np.mean(accuracy_values),
            'avg_precision': np.mean(precision_values),
            'avg_recall': np.mean(recall_values),
            'std_accuracy': np.std(accuracy_values)
        }

    def generate_recommendations(self, metrics):
        """Generate recommendations based on evaluation results"""
        recommendations = []

        if metrics['avg_accuracy'] < 0.8:
            recommendations.append("Consider improving sensor calibration procedures")

        if metrics['avg_precision'] < 0.8:
            recommendations.append("Investigate noise reduction techniques")

        if metrics['std_accuracy'] > 0.1:
            recommendations.append("System performance varies significantly; investigate consistency issues")

        return recommendations

class FusionBenchmarkSuite:
    """Standardized benchmark suite for multi-sensor fusion"""

    def __init__(self):
        self.test_scenarios = self.define_standard_scenarios()
        self.reference_implementations = {}
        self.performance_baselines = {}

    def define_standard_scenarios(self):
        """Define standard test scenarios for fusion evaluation"""
        return {
            'static_calibration': {
                'description': 'Static calibration and initial alignment',
                'duration': 10.0,
                'motion_profile': 'static',
                'sensor_config': 'all_sensors_active'
            },
            'dynamic_tracking': {
                'description': 'Dynamic target tracking with known trajectory',
                'duration': 30.0,
                'motion_profile': 'circular_trajectory',
                'sensor_config': 'all_sensors_active'
            },
            'sensor_outage': {
                'description': 'Performance under sensor outage conditions',
                'duration': 30.0,
                'motion_profile': 'linear_trajectory',
                'sensor_config': 'camera_outage'
            },
            'high_noise': {
                'description': 'Performance under high noise conditions',
                'duration': 20.0,
                'motion_profile': 'stationary',
                'sensor_config': 'high_noise_profile'
            },
            'fast_motion': {
                'description': 'Performance under fast motion conditions',
                'duration': 25.0,
                'motion_profile': 'aggressive_maneuvers',
                'sensor_config': 'all_sensors_active'
            }
        }

    def run_benchmark(self, fusion_system):
        """Run the complete benchmark suite"""
        results = {}

        for scenario_name, scenario_config in self.test_scenarios.items():
            print(f"Running benchmark: {scenario_name}")

            # Set up scenario
            self.setup_scenario(scenario_config)

            # Run fusion system
            scenario_results = fusion_system.run_with_config(scenario_config)

            # Evaluate results
            evaluator = FusionSystemEvaluator()
            metrics = evaluator.calculate_basic_metrics(
                scenario_results['ground_truth'],
                scenario_results['estimates']
            )

            results[scenario_name] = {
                'metrics': metrics,
                'passed': self.check_pass_criteria(metrics, scenario_name)
            }

        # Generate final report
        final_report = self.generate_benchmark_report(results)

        return {
            'scenario_results': results,
            'final_report': final_report,
            'overall_score': self.calculate_overall_score(results)
        }

    def setup_scenario(self, config):
        """Set up a specific test scenario"""
        # This would configure the simulation environment for the scenario
        pass

    def check_pass_criteria(self, metrics, scenario_name):
        """Check if the system passes the criteria for a scenario"""
        criteria = {
            'static_calibration': {'position_rmse': 0.01},  # 1cm accuracy
            'dynamic_tracking': {'position_rmse': 0.05},    # 5cm accuracy
            'sensor_outage': {'position_rmse': 0.10},       # 10cm accuracy
            'high_noise': {'position_rmse': 0.03},          # 3cm accuracy
            'fast_motion': {'position_rmse': 0.08}          # 8cm accuracy
        }

        required_rmse = criteria.get(scenario_name, {}).get('position_rmse', 0.10)
        actual_rmse = metrics.get('position_rmse', float('inf'))

        return actual_rmse <= required_rmse

    def generate_benchmark_report(self, results):
        """Generate a comprehensive benchmark report"""
        passed_scenarios = [name for name, result in results.items() if result['passed']]
        failed_scenarios = [name for name, result in results.items() if not result['passed']]

        return {
            'total_scenarios': len(results),
            'passed_scenarios': len(passed_scenarios),
            'failed_scenarios': len(failed_scenarios),
            'pass_rate': len(passed_scenarios) / len(results) if results else 0,
            'detailed_results': results
        }

    def calculate_overall_score(self, results):
        """Calculate an overall benchmark score"""
        if not results:
            return 0.0

        # Weight different scenarios based on importance
        scenario_weights = {
            'static_calibration': 0.15,
            'dynamic_tracking': 0.35,
            'sensor_outage': 0.25,
            'high_noise': 0.15,
            'fast_motion': 0.10
        }

        weighted_score = 0.0
        for scenario_name, result in results.items():
            weight = scenario_weights.get(scenario_name, 0.2)  # Default weight
            score = 1.0 if result['passed'] else 0.0
            weighted_score += weight * score

        return weighted_score
```

## Safety and Ethical Considerations

Multi-sensor fusion systems in humanoid robots raise important safety and ethical considerations that must be addressed to ensure responsible deployment in human environments.

### Safety Considerations

#### Sensor-Based Safety Systems
Humanoid robots operating in human environments require robust safety systems that can detect and respond to potential hazards in real-time.

```python
class SafetyRiskAssessment:
    """Comprehensive safety risk assessment for multi-sensor fusion systems"""

    def __init__(self):
        self.hazard_database = self.initialize_hazard_database()
        self.safety_protocols = self.initialize_safety_protocols()
        self.emergency_procedures = self.initialize_emergency_procedures()
        self.safety_metrics = {}
        self.monitoring_systems = {}

    def initialize_hazard_database(self):
        """Initialize database of potential hazards"""
        return {
            'collision_risk': {
                'description': 'Risk of collision with humans or objects',
                'detection_method': 'proximity_sensors, vision',
                'severity': 'high',
                'mitigation': 'collision_avoidance, emergency_stop'
            },
            'environmental_hazard': {
                'description': 'Environmental risks like slippery surfaces',
                'detection_method': 'force_torque, vision, tactile',
                'severity': 'medium',
                'mitigation': 'gait_adaptation, warning_system'
            },
            'sensor_failure': {
                'description': 'Failure of critical sensors',
                'detection_method': 'health_monitoring',
                'severity': 'high',
                'mitigation': 'redundancy, fallback_modes'
            },
            'human_behavior': {
                'description': 'Unpredictable human behavior',
                'detection_method': 'vision, audio, proximity',
                'severity': 'medium',
                'mitigation': 'behavior_prediction, safe_distance'
            }
        }

    def initialize_safety_protocols(self):
        """Initialize safety protocols for different scenarios"""
        return {
            'approach_protocol': {
                'enabled': True,
                'minimum_distance': 1.0,  # meters
                'approach_speed_limit': 0.5  # m/s
            },
            'collision_avoidance': {
                'enabled': True,
                'prediction_horizon': 2.0,  # seconds
                'safety_margin': 0.3  # meters
            },
            'emergency_stop': {
                'enabled': True,
                'trigger_conditions': ['immediate_collision', 'sensor_failure', 'human_fall_detected'],
                'response_time': 0.1  # seconds
            }
        }

    def initialize_emergency_procedures(self):
        """Initialize emergency procedures"""
        return {
            'sensor_failure': {
                'action': 'switch_to_backup_sensors',
                'timeout': 5.0,
                'fallback_behavior': 'return_to_safe_pose'
            },
            'human_fall_detected': {
                'action': 'stop_immediately_and_alert',
                'timeout': 0.5,
                'fallback_behavior': 'wait_for_assistance'
            },
            'unexpected_object': {
                'action': 'pause_and_assess',
                'timeout': 3.0,
                'fallback_behavior': 'navigate_around_or_request_help'
            }
        }

    def assess_environmental_safety(self, sensor_data):
        """Assess environmental safety based on sensor inputs"""
        safety_assessment = {
            'collision_risk': 0.0,
            'environmental_hazard': 0.0,
            'human_proximity': 0.0,
            'navigation_safety': 0.0
        }

        # Analyze proximity sensors for collision risk
        if 'proximity' in sensor_data:
            min_distance = min(sensor_data['proximity']) if sensor_data['proximity'] else float('inf')
            if min_distance < 0.5:  # High risk if closer than 0.5m
                safety_assessment['collision_risk'] = 1.0 - min(min_distance / 0.5, 1.0)

        # Analyze vision data for environmental hazards
        if 'vision' in sensor_data:
            vision_data = sensor_data['vision']
            # Check for slippery surfaces, obstacles, etc.
            if vision_data.get('surface_type') == 'slippery':
                safety_assessment['environmental_hazard'] = 0.8
            elif vision_data.get('obstacle_density', 0) > 0.5:
                safety_assessment['environmental_hazard'] = 0.6

        # Analyze human detection
        if 'human_detection' in sensor_data:
            humans = sensor_data['human_detection']
            if len(humans) > 0:
                closest_human = min(humans, key=lambda h: h.get('distance', float('inf')))
                distance = closest_human.get('distance', float('inf'))
                if distance < 2.0:  # Within 2m of human
                    safety_assessment['human_proximity'] = max(0.0, 1.0 - distance/2.0)

        # Overall navigation safety
        safety_assessment['navigation_safety'] = 1.0 - max(safety_assessment.values())

        return safety_assessment

    def trigger_safety_protocol(self, hazard_type, sensor_data):
        """Trigger appropriate safety protocol based on hazard"""
        if hazard_type in self.emergency_procedures:
            procedure = self.emergency_procedures[hazard_type]

            # Execute safety action
            if procedure['action'] == 'switch_to_backup_sensors':
                self.switch_to_backup_sensors()
            elif procedure['action'] == 'stop_immediately_and_alert':
                self.emergency_stop()
                self.send_alert('human_fall_detected')
            elif procedure['action'] == 'pause_and_assess':
                self.pause_operation()

            return True
        return False

    def switch_to_backup_sensors(self):
        """Switch to backup sensor configurations"""
        print("Switching to backup sensors due to primary sensor failure")
        # Implementation would switch fusion system to use backup sensors
        pass

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        print("Executing emergency stop")
        # Implementation would immediately stop all robot motion
        pass

    def pause_operation(self):
        """Pause robot operation for assessment"""
        print("Pausing operation for safety assessment")
        # Implementation would pause robot while maintaining safe pose
        pass

    def send_alert(self, alert_type):
        """Send safety alert"""
        print(f"Sending safety alert: {alert_type}")
        # Implementation would send alerts to operators or emergency services
        pass

    def calculate_safety_metrics(self, assessment_results):
        """Calculate safety metrics from assessment results"""
        return {
            'risk_score': np.mean([result.get('collision_risk', 0) for result in assessment_results]),
            'safety_compliance_rate': 0.95,  # Placeholder
            'emergency_response_time': 0.08,  # Placeholder
            'hazard_detection_rate': 0.92     # Placeholder
        }

class CollisionAvoidanceSystem:
    """Advanced collision avoidance using multi-sensor fusion"""

    def __init__(self):
        self.prediction_model = self.initialize_prediction_model()
        self.human_behavior_model = HumanBehaviorPredictor()
        self.safety_controller = SafetyController()
        self.threat_assessment = ThreatAssessment()

    def initialize_prediction_model(self):
        """Initialize collision prediction model"""
        return {
            'horizon': 3.0,  # seconds
            'update_rate': 10.0,  # Hz
            'confidence_threshold': 0.8
        }

    def assess_collision_risk(self, fused_state, environment_model):
        """Assess collision risk using fused sensor data"""
        # Predict future positions of robot and obstacles
        robot_trajectory = self.predict_robot_trajectory(fused_state)
        obstacle_trajectories = self.predict_obstacle_trajectories(environment_model)

        # Calculate collision probability
        collision_risk = self.calculate_collision_probability(
            robot_trajectory, obstacle_trajectories
        )

        return collision_risk

    def predict_robot_trajectory(self, current_state):
        """Predict robot's future trajectory"""
        # Simplified prediction based on current state
        # In practice, this would use more sophisticated motion models
        trajectory = []
        dt = 0.1  # time step
        steps = int(self.prediction_model['horizon'] / dt)

        state = current_state.copy()
        for i in range(steps):
            # Apply motion model
            new_state = self.apply_motion_model(state, dt)
            trajectory.append(new_state)
            state = new_state

        return trajectory

    def predict_obstacle_trajectories(self, environment_model):
        """Predict trajectories of dynamic obstacles"""
        trajectories = {}

        for obj_id, obj_data in environment_model.items():
            if obj_data.get('dynamic', False):
                # Predict based on observed motion
                trajectory = self.predict_dynamic_object_trajectory(obj_data)
                trajectories[obj_id] = trajectory

        return trajectories

    def predict_dynamic_object_trajectory(self, obj_data):
        """Predict trajectory for a dynamic object"""
        trajectory = []
        dt = 0.1
        steps = int(self.prediction_model['horizon'] / dt)

        position = np.array(obj_data['position'])
        velocity = np.array(obj_data['velocity'])

        for i in range(steps):
            new_position = position + velocity * dt
            trajectory.append(new_position)
            # Update for next step (could include acceleration)
            position = new_position

        return trajectory

    def calculate_collision_probability(self, robot_traj, obstacle_trajs):
        """Calculate probability of collision"""
        collision_prob = 0.0

        for robot_state in robot_traj:
            for obj_id, obj_traj in obstacle_trajs.items():
                for obj_state in obj_traj:
                    # Calculate distance between robot and obstacle
                    dist = np.linalg.norm(robot_state[:2] - obj_state[:2])

                    # If distance is less than safety margin, potential collision
                    safety_margin = 0.5  # meters
                    if dist < safety_margin:
                        # Calculate collision probability based on uncertainty
                        collision_prob = max(collision_prob, 1.0 - (dist / safety_margin))

        return min(collision_prob, 1.0)

    def apply_motion_model(self, state, dt):
        """Apply motion model to predict next state"""
        # Simplified constant velocity model
        new_state = state.copy()
        # Update position based on velocity
        new_state[0] += new_state[3] * dt  # x += vx * dt
        new_state[1] += new_state[4] * dt  # y += vy * dt
        # Orientation update
        new_state[2] += new_state[5] * dt  # theta += omega * dt
        return new_state

class HumanBehaviorPredictor:
    """Predict human behavior for safety-aware navigation"""

    def __init__(self):
        self.behavior_patterns = self.initialize_behavior_patterns()
        self.context_awareness = ContextAwarenessSystem()

    def initialize_behavior_patterns(self):
        """Initialize common human behavior patterns"""
        return {
            'walking_straight': {
                'probability': 0.6,
                'duration_range': (1.0, 5.0),
                'direction_variance': 0.1
            },
            'turning': {
                'probability': 0.2,
                'duration_range': (0.5, 2.0),
                'angular_velocity_range': (-1.0, 1.0)
            },
            'stopping': {
                'probability': 0.15,
                'duration_range': (0.5, 10.0),
                'trigger_conditions': ['object_interaction', 'conversation']
            },
            'avoidance': {
                'probability': 0.05,
                'trigger_conditions': ['robot_approach', 'sudden_movement']
            }
        }

    def predict_human_trajectory(self, human_state, context):
        """Predict human's future trajectory based on current state and context"""
        # Use behavior patterns and context to predict movement
        predicted_trajectory = []

        # Determine most likely behavior based on context
        likely_behavior = self.determine_likely_behavior(human_state, context)

        # Generate trajectory based on behavior
        dt = 0.1
        steps = 30  # Predict 3 seconds ahead

        current_pos = np.array(human_state['position'])
        current_vel = np.array(human_state['velocity'])

        for i in range(steps):
            # Apply behavior-specific motion model
            new_pos, new_vel = self.apply_behavior_model(
                current_pos, current_vel, likely_behavior, dt
            )
            predicted_trajectory.append({
                'position': new_pos,
                'velocity': new_vel,
                'timestamp': i * dt
            })
            current_pos, current_vel = new_pos, new_vel

        return predicted_trajectory

    def determine_likely_behavior(self, human_state, context):
        """Determine the most likely human behavior"""
        # Analyze current state and context to select behavior
        # This is a simplified version - in practice would use ML models
        if context.get('robot_distance', float('inf')) < 1.0:
            # If robot is close, human might change behavior
            return 'avoidance'
        elif abs(human_state.get('velocity_magnitude', 0)) < 0.1:
            # If human is stationary
            return 'stopping'
        else:
            # Default to walking straight
            return 'walking_straight'

    def apply_behavior_model(self, pos, vel, behavior, dt):
        """Apply behavior-specific motion model"""
        if behavior == 'walking_straight':
            # Continue in current direction with slight variation
            new_pos = pos + vel * dt
            new_vel = vel  # Maintain velocity
        elif behavior == 'turning':
            # Apply angular change
            angular_change = np.random.uniform(-0.5, 0.5) * dt
            # Rotate velocity vector
            cos_a, sin_a = np.cos(angular_change), np.sin(angular_change)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            new_vel = rot_matrix @ vel
            new_pos = pos + new_vel * dt
        elif behavior == 'stopping':
            # Decelerate to stop
            deceleration = 0.5
            speed = np.linalg.norm(vel)
            if speed > deceleration * dt:
                new_vel = vel - (vel / speed) * deceleration * dt
            else:
                new_vel = np.array([0.0, 0.0])
            new_pos = pos + new_vel * dt
        elif behavior == 'avoidance':
            # Move away from robot
            robot_pos = np.array([0, 0])  # Placeholder - actual robot position
            avoidance_dir = pos - robot_pos
            avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
            new_vel = avoidance_dir * min(np.linalg.norm(vel), 0.8)  # Max speed 0.8 m/s
            new_pos = pos + new_vel * dt
        else:
            # Default behavior
            new_pos = pos + vel * dt
            new_vel = vel

        return new_pos, new_vel

class SafetyController:
    """Safety controller that enforces safety constraints"""

    def __init__(self):
        self.safety_constraints = {
            'minimum_distance_to_human': 0.8,  # meters
            'maximum_approach_speed': 0.3,     # m/s
            'maximum_angular_velocity': 0.5,   # rad/s
            'maximum_acceleration': 0.5        # m/s^2
        }
        self.safety_buffer = 0.2  # Additional safety margin

    def enforce_safety_constraints(self, desired_motion, human_positions, obstacle_positions):
        """Enforce safety constraints on desired motion"""
        safe_motion = desired_motion.copy()

        # Check minimum distance to humans
        for human_pos in human_positions:
            dist_to_human = np.linalg.norm(safe_motion['position'] - human_pos)
            min_safe_dist = self.safety_constraints['minimum_distance_to_human']

            if dist_to_human < min_safe_dist:
                # Adjust motion to maintain safe distance
                direction_to_human = (safe_motion['position'] - human_pos) / dist_to_human
                safe_position = human_pos + direction_to_human * min_safe_dist
                safe_motion['position'] = safe_position

        # Limit approach speed
        approach_speed = np.linalg.norm(safe_motion.get('velocity', [0, 0]))
        max_speed = self.safety_constraints['maximum_approach_speed']
        if approach_speed > max_speed:
            if approach_speed > 0:
                safe_motion['velocity'] = (
                    np.array(safe_motion['velocity']) * max_speed / approach_speed
                )

        return safe_motion

class ThreatAssessment:
    """Assess potential threats in the environment"""

    def __init__(self):
        self.threat_levels = {
            'low': 0.0,
            'medium': 0.3,
            'high': 0.7,
            'critical': 0.9
        }

    def assess_threat_level(self, environment_data):
        """Assess overall threat level in the environment"""
        threat_score = 0.0

        # Assess based on various factors
        if 'humans' in environment_data:
            human_density = len(environment_data['humans']) / environment_data.get('area', 1.0)
            threat_score += min(human_density * 0.2, 0.3)  # Up to 0.3 from human density

        if 'obstacles' in environment_data:
            obstacle_density = len(environment_data['obstacles']) / environment_data.get('area', 1.0)
            threat_score += min(obstacle_density * 0.1, 0.2)  # Up to 0.2 from obstacles

        # Add other threat factors as needed
        # Child presence, moving objects, environmental hazards, etc.

        return self.categorize_threat_level(threat_score)

    def categorize_threat_level(self, score):
        """Categorize threat level based on score"""
        if score >= self.threat_levels['critical']:
            return 'critical'
        elif score >= self.threat_levels['high']:
            return 'high'
        elif score >= self.threat_levels['medium']:
            return 'medium'
        else:
            return 'low'

class ContextAwarenessSystem:
    """System that maintains awareness of context for safety decisions"""

    def __init__(self):
        self.context_map = {}
        self.situation_awareness = SituationAwareness()

    def update_context(self, sensor_data, timestamp):
        """Update context based on sensor data"""
        # Update various context elements
        self.context_map['timestamp'] = timestamp
        self.context_map['location'] = self.estimate_location(sensor_data)
        self.context_map['environment_type'] = self.classify_environment(sensor_data)
        self.context_map['activity_type'] = self.recognize_activity(sensor_data)
        self.context_map['social_context'] = self.understand_social_context(sensor_data)

    def estimate_location(self, sensor_data):
        """Estimate current location"""
        # Use SLAM or localization data
        return sensor_data.get('location', 'unknown')

    def classify_environment(self, sensor_data):
        """Classify environment type"""
        # Determine if indoor/outdoor, structured/unstructured, etc.
        return sensor_data.get('environment_type', 'indoor')

    def recognize_activity(self, sensor_data):
        """Recognize ongoing activities"""
        # Use vision and other sensors to recognize activities
        return sensor_data.get('activities', [])

    def understand_social_context(self, sensor_data):
        """Understand social context"""
        # Analyze human interactions and social norms
        return sensor_data.get('social_context', {})
```

### Ethical Considerations

#### Privacy Preservation
Multi-sensor fusion systems often process sensitive data, particularly from cameras and microphones. Privacy preservation is essential:

```python
class PrivacyPreservingFusion:
    """Fusion system with privacy preservation capabilities"""

    def __init__(self):
        self.privacy_filters = {
            'face_blurring': True,
            'voice_anonymization': True,
            'location_masking': True
        }
        self.data_retention_policy = {
            'temporary_data': 300,  # 5 minutes
            'processed_data': 86400,  # 24 hours
            'anonymized_data': 2592000  # 30 days
        }

    def apply_privacy_filters(self, sensor_data):
        """Apply privacy filters to sensor data"""
        filtered_data = sensor_data.copy()

        # Apply face blurring to camera data
        if 'camera' in filtered_data and self.privacy_filters['face_blurring']:
            filtered_data['camera'] = self.blur_faces(filtered_data['camera'])

        # Apply voice anonymization to audio data
        if 'audio' in filtered_data and self.privacy_filters['voice_anonymization']:
            filtered_data['audio'] = self.anonymize_voice(filtered_data['audio'])

        return filtered_data

    def blur_faces(self, image_data):
        """Apply face blurring to protect identity"""
        # This would use computer vision techniques to detect and blur faces
        # For simulation, we'll return the data unchanged
        return image_data

    def anonymize_voice(self, audio_data):
        """Anonymize voice data to protect identity"""
        # This would use audio processing to anonymize voices
        # For simulation, we'll return the data unchanged
        return audio_data

    def enforce_data_retention(self, data, data_type):
        """Enforce data retention policies"""
        import time
        current_time = time.time()

        # Determine retention period
        retention_period = self.data_retention_policy.get(data_type, 300)

        # Mark data for deletion after retention period
        data['retention_expiry'] = current_time + retention_period
        return data
```

#### Cultural and Social Adaptation
Humanoid robots must adapt to cultural and social norms:

```python
class SocialNormsManager:
    """Manage cultural and social adaptation for humanoid robots"""

    def __init__(self):
        self.cultural_databases = self.load_cultural_databases()
        self.social_behavior_rules = self.initialize_social_rules()
        self.ethical_guidelines = self.load_ethical_guidelines()

    def load_cultural_databases(self):
        """Load cultural databases for different regions"""
        return {
            'japan': {
                'personal_space': 1.2,  # meters
                'greeting_protocol': 'bow',
                'eye_contact_norms': 'respectful_avoidance',
                'interaction_style': 'formal_polite'
            },
            'usa': {
                'personal_space': 0.9,  # meters
                'greeting_protocol': 'handshake',
                'eye_contact_norms': 'maintain_respectfully',
                'interaction_style': 'friendly_direct'
            },
            'middle_east': {
                'personal_space': 1.0,  # meters
                'greeting_protocol': 'respectful_distance',
                'eye_contact_norms': 'moderate_avoidance',
                'interaction_style': 'respectful_courteous'
            }
        }

    def initialize_social_rules(self):
        """Initialize general social behavior rules"""
        return {
            'respect_personal_space': True,
            'adapt_to_local_norms': True,
            'respect_privacy': True,
            'provide_transparency': True,
            'avoid_discrimination': True
        }

    def adapt_to_cultural_context(self, location, cultural_group):
        """Adapt behavior to local cultural context"""
        culture_info = self.cultural_databases.get(cultural_group.lower())
        if culture_info:
            # Adjust personal space norms
            self.social_behavior_rules['personal_space'] = culture_info['personal_space']

            # Adjust interaction style
            self.social_behavior_rules['interaction_style'] = culture_info['interaction_style']

            # Adjust eye contact norms
            self.social_behavior_rules['eye_contact_norms'] = culture_info['eye_contact_norms']

    def evaluate_ethical_compliance(self, robot_action, context):
        """Evaluate if robot action complies with ethical guidelines"""
        compliance_score = 1.0  # Start with full compliance

        # Check respect for autonomy
        if not self.respects_autonomy(robot_action, context):
            compliance_score -= 0.3

        # Check non-maleficence (do no harm)
        if not self.avoids_harm(robot_action, context):
            compliance_score -= 0.4

        # Check beneficence (do good)
        if not self.provides_benefit(robot_action, context):
            compliance_score -= 0.2

        # Check justice (fair treatment)
        if not self.treats_fairly(robot_action, context):
            compliance_score -= 0.3

        return max(0.0, compliance_score)  # Ensure non-negative score

    def respects_autonomy(self, action, context):
        """Check if action respects human autonomy"""
        # Implementation would check if action respects human choices and decisions
        return True  # Placeholder

    def avoids_harm(self, action, context):
        """Check if action avoids harm to humans"""
        # Implementation would check for potential physical or psychological harm
        return True  # Placeholder

    def provides_benefit(self, action, context):
        """Check if action provides benefit"""
        # Implementation would check if action serves human welfare
        return True  # Placeholder

    def treats_fairly(self, action, context):
        """Check if action treats all humans fairly"""
        # Implementation would check for discrimination or bias
        return True  # Placeholder

    def load_ethical_guidelines(self):
        """Load ethical guidelines for robot behavior"""
        return {
            'asimov_laws_compliance': True,
            'ieee_ethical_standards': True,
            'local_regulations': True,
            'cultural_sensitivity': True
        }
```

## Summary

This chapter covered the essential concepts of multi-sensor fusion for humanoid robots, from fundamental techniques like Kalman filtering and particle filtering to advanced approaches using machine learning and information theory.

We explored different fusion architectures including centralized and distributed approaches, with practical implementations for each. The chapter provided detailed examples of integrating specific sensor types like vision-inertial systems and force-tactile fusion for manipulation tasks.

We also covered critical aspects of practical fusion systems including fault detection and isolation, which is essential for maintaining robust operation in real-world scenarios. The hands-on exercise provided practical experience implementing a complete fusion system with evaluation metrics.

## Practical Examples and Exercises

### Example 1: Indoor Navigation with Multiple Sensors
Let's walk through a practical example of fusing data from multiple sensors for indoor navigation:

```python
class IndoorNavigationFusion:
    """Practical example of multi-sensor fusion for indoor navigation"""

    def __init__(self):
        # Initialize multiple sensor types
        self.imu_filter = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)  # orientation only
        self.vision_tracker = self.initialize_vision_tracker()
        self.wheel_encoders = self.initialize_encoder_odometry()
        self.lidar_mapper = self.initialize_lidar_mapper()

        # Global fusion filter
        self.global_fusion = UnscentedKalmanFilter(state_dim=6, measurement_dim=2)  # position only

        # State variables
        self.position_estimate = np.zeros(2)
        self.orientation_estimate = 0.0
        self.velocity_estimate = np.zeros(2)

    def initialize_vision_tracker(self):
        """Initialize visual feature tracker"""
        return {
            'features': [],
            'feature_poses': [],
            'camera_matrix': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
            'tracked_features': 0
        }

    def initialize_encoder_odometry(self):
        """Initialize wheel encoder odometry"""
        return {
            'left_ticks': 0,
            'right_ticks': 0,
            'wheel_radius': 0.05,  # 5cm wheels
            'wheel_base': 0.3,     # 30cm wheel base
            'last_position': np.zeros(2),
            'last_orientation': 0.0
        }

    def initialize_lidar_mapper(self):
        """Initialize LiDAR-based mapping"""
        return {
            'map': np.zeros((100, 100)),  # 10x10m map at 10cm resolution
            'map_resolution': 0.1,       # 10cm per cell
            'obstacles_detected': 0
        }

    def process_sensor_data(self, imu_data, vision_data, encoder_data, lidar_data):
        """Process and fuse data from all sensors"""
        # Process IMU data for orientation
        imu_orientation = self.process_imu_data(imu_data)

        # Process vision data for position relative to landmarks
        vision_position = self.process_vision_data(vision_data)

        # Process encoder data for odometry
        encoder_position = self.process_encoder_data(encoder_data)

        # Process LiDAR data for obstacle detection and mapping
        self.process_lidar_data(lidar_data)

        # Fuse all estimates using UKF
        final_estimate = self.fuse_all_sensors(
            imu_orientation, vision_position, encoder_position
        )

        return final_estimate

    def process_imu_data(self, imu_data):
        """Process IMU data for orientation estimation"""
        # Extract accelerometer and gyroscope data
        accel = np.array(imu_data['accelerometer'])
        gyro = np.array(imu_data['gyroscope'])

        # Update IMU filter
        dt = 0.01  # 100Hz
        self.imu_filter.predict(dt)

        # Create measurement vector [roll, pitch, yaw]
        # Simplified: using accelerometer for tilt, gyroscope for rotation
        orientation_measurement = self.estimate_orientation_from_imu(accel, gyro, dt)

        self.imu_filter.update(orientation_measurement)

        # Extract orientation estimate
        state = self.imu_filter.get_state()
        return state[2]  # yaw angle

    def estimate_orientation_from_imu(self, accel, gyro, dt):
        """Estimate orientation from accelerometer and gyroscope"""
        # Integrate gyroscope for rotation
        delta_orientation = gyro[2] * dt  # Assuming z-axis is up

        # Use accelerometer to correct for drift
        # This is a simplified approach - in practice, more sophisticated fusion is needed
        pitch = np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        roll = np.arctan2(-accel[1], accel[2])

        return np.array([roll, pitch, delta_orientation])

    def process_vision_data(self, vision_data):
        """Process vision data for position estimation"""
        # Detect and track visual features
        features = vision_data.get('features', [])

        if len(features) >= 3:  # Need at least 3 features for position estimation
            # Perform visual odometry
            position_estimate = self.visual_odometry(features)
            return position_estimate
        else:
            # Not enough features, return last known position
            return self.position_estimate

    def visual_odometry(self, features):
        """Perform visual odometry from tracked features"""
        # This would implement feature tracking and position estimation
        # For simplicity, we'll return a placeholder
        return self.position_estimate + np.random.normal(0, 0.01, 2)  # Small random walk

    def process_encoder_data(self, encoder_data):
        """Process wheel encoder data for odometry"""
        left_ticks = encoder_data['left']
        right_ticks = encoder_data['right']

        # Calculate wheel displacements
        left_dist = (left_ticks - self.wheel_encoders['left_ticks']) * 2 * np.pi * \
                   self.wheel_encoders['wheel_radius'] / 1000  # Assuming 1000 ticks per revolution
        right_dist = (right_ticks - self.wheel_encoders['right_ticks']) * 2 * np.pi * \
                    self.wheel_encoders['wheel_radius'] / 1000

        # Update stored tick counts
        self.wheel_encoders['left_ticks'] = left_ticks
        self.wheel_encoders['right_ticks'] = right_ticks

        # Calculate robot displacement
        avg_dist = (left_dist + right_dist) / 2
        delta_theta = (right_dist - left_dist) / self.wheel_encoders['wheel_base']

        # Update orientation
        new_orientation = self.wheel_encoders['last_orientation'] + delta_theta

        # Calculate position change in robot frame
        local_dx = avg_dist * np.cos(delta_theta / 2)
        local_dy = avg_dist * np.sin(delta_theta / 2)

        # Transform to global frame and update position
        global_dx = local_dx * np.cos(self.wheel_encoders['last_orientation']) - \
                   local_dy * np.sin(self.wheel_encoders['last_orientation'])
        global_dy = local_dx * np.sin(self.wheel_encoders['last_orientation']) + \
                   local_dy * np.cos(self.wheel_encoders['last_orientation'])

        new_position = self.wheel_encoders['last_position'] + np.array([global_dx, global_dy])

        # Update stored values
        self.wheel_encoders['last_position'] = new_position
        self.wheel_encoders['last_orientation'] = new_orientation

        return new_position

    def process_lidar_data(self, lidar_data):
        """Process LiDAR data for mapping and obstacle detection"""
        ranges = lidar_data.get('ranges', [])
        angles = lidar_data.get('angles', [])

        # Convert to Cartesian coordinates
        points = []
        for r, theta in zip(ranges, angles):
            if r < 5.0:  # Only consider points within 5m
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append((x, y))

        # Update map with detected obstacles
        for x, y in points:
            # Convert to map coordinates
            map_x = int((x / self.lidar_mapper['map_resolution']) + 50)  # Center at (50,50)
            map_y = int((y / self.lidar_mapper['map_resolution']) + 50)

            if 0 <= map_x < 100 and 0 <= map_y < 100:
                self.lidar_mapper['map'][map_x, map_y] = 1  # Mark as occupied

        self.lidar_mapper['obstacles_detected'] = len(points)

    def fuse_all_sensors(self, imu_orientation, vision_position, encoder_position):
        """Fuse all sensor estimates using UKF"""
        # Create measurement vector from sensor estimates
        # For position: use vision and encoder data
        measurement = np.concatenate([vision_position, encoder_position]).reshape(-1)

        # Since we have two position estimates, we'll use a weighted combination
        # In practice, the UKF would handle the fusion more elegantly
        fused_position = (vision_position * 0.6 + encoder_position * 0.4)  # Weighted average

        # Update orientation from IMU
        self.orientation_estimate = imu_orientation

        # Update position estimate
        self.position_estimate = fused_position

        # For a complete implementation, we would update the UKF with the fused measurement
        # self.global_fusion.update(fused_position)

        return {
            'position': fused_position,
            'orientation': imu_orientation,
            'timestamp': time.time()
        }

# Example usage of the indoor navigation fusion system
def example_indoor_navigation():
    """Example of using the indoor navigation fusion system"""
    fusion_system = IndoorNavigationFusion()

    # Simulate sensor data over time
    for t in range(100):  # 100 time steps
        # Simulate sensor readings
        imu_data = {
            'accelerometer': np.random.normal(0, 0.01, 3),
            'gyroscope': np.random.normal(0, 0.001, 3)
        }

        vision_data = {
            'features': np.random.rand(5, 2) * 10  # 5 random features
        }

        encoder_data = {
            'left': 100 + t * 10 + np.random.normal(0, 1),
            'right': 100 + t * 10 + np.random.normal(0, 1)
        }

        lidar_data = {
            'ranges': np.random.uniform(0.5, 5.0, 360),  # 360 degree scan
            'angles': np.linspace(0, 2*np.pi, 360)
        }

        # Process and fuse sensor data
        estimate = fusion_system.process_sensor_data(imu_data, vision_data, encoder_data, lidar_data)

        print(f"Time {t*0.01:.2f}s - Position: ({estimate['position'][0]:.3f}, {estimate['position'][1]:.3f}), "
              f"Orientation: {estimate['orientation']:.3f} rad")
```

### Example 2: Manipulation with Force-Torque and Vision Fusion
Here's an example of fusing force-torque and vision data for precise manipulation:

```python
class ManipulationFusion:
    """Fusion system for precise manipulation using force-torque and vision"""

    def __init__(self):
        self.vision_tracker = self.initialize_vision_tracker()
        self.force_estimator = self.initialize_force_estimator()
        self.contact_detector = self.initialize_contact_detector()
        self.impedance_controller = self.initialize_impedance_controller()

    def initialize_vision_tracker(self):
        """Initialize vision-based object tracking"""
        return {
            'object_pose': np.zeros(6),  # [x, y, z, rx, ry, rz]
            'tracking_confidence': 0.0,
            'feature_points': [],
            'camera_calibration': np.eye(3)
        }

    def initialize_force_estimator(self):
        """Initialize force estimation from multiple sensors"""
        return {
            'force_filter': KalmanFilter(state_dim=6, measurement_dim=6),  # [fx, fy, fz, tx, ty, tz]
            'estimated_contact_force': np.zeros(3),
            'estimated_contact_torque': np.zeros(3)
        }

    def initialize_contact_detector(self):
        """Initialize contact detection system"""
        return {
            'contact_threshold': 2.0,  # Newtons
            'contact_history': [],
            'contact_state': 'no_contact'
        }

    def initialize_impedance_controller(self):
        """Initialize impedance control parameters"""
        return {
            'stiffness': np.diag([1000, 1000, 1000, 100, 100, 100]),  # High for position, lower for orientation
            'damping_ratio': 1.0,  # Critical damping
            'desired_force': np.zeros(6)
        }

    def process_manipulation_task(self, vision_data, force_torque_data, desired_trajectory):
        """Process manipulation task with fused sensory feedback"""
        # Process vision data for object pose estimation
        object_pose = self.process_vision_data(vision_data)

        # Process force-torque data for contact force estimation
        contact_force = self.process_force_torque_data(force_torque_data)

        # Detect contact state
        contact_state = self.detect_contact(contact_force)

        # Plan manipulation based on fused information
        control_command = self.plan_manipulation(
            object_pose, contact_force, contact_state, desired_trajectory
        )

        return control_command

    def process_vision_data(self, vision_data):
        """Process vision data for object pose estimation"""
        # Extract object pose from vision data
        # This would typically involve object detection and pose estimation
        if 'object_pose' in vision_data:
            # Update object pose estimate with vision data
            self.vision_tracker['object_pose'] = vision_data['object_pose']
            self.vision_tracker['tracking_confidence'] = vision_data.get('confidence', 0.8)
        else:
            # Use feature-based tracking
            features = vision_data.get('features', [])
            if len(features) >= 4:
                pose = self.estimate_pose_from_features(features)
                self.vision_tracker['object_pose'] = pose
                self.vision_tracker['tracking_confidence'] = 0.7

        return self.vision_tracker['object_pose']

    def estimate_pose_from_features(self, features):
        """Estimate object pose from visual features"""
        # Simplified pose estimation
        # In practice, this would use PnP algorithms or template matching
        if len(features) > 0:
            # Calculate centroid as position estimate
            centroid = np.mean(features, axis=0)
            # Add placeholder orientation
            pose = np.array([centroid[0], centroid[1], 0.1, 0, 0, 0])  # z=0.1m height
            return pose
        else:
            return self.vision_tracker['object_pose']  # Return last known pose

    def process_force_torque_data(self, force_torque_data):
        """Process force-torque sensor data"""
        # Extract force and torque measurements
        force = np.array(force_torque_data[:3])  # [fx, fy, fz]
        torque = np.array(force_torque_data[3:])  # [tx, ty, tz]

        # Apply filtering to reduce noise
        measurement = np.concatenate([force, torque])

        # Update force filter (predict and update cycle)
        dt = 0.001  # 1kHz
        self.force_estimator['force_filter'].predict(dt)
        self.force_estimator['force_filter'].update(measurement)

        # Extract filtered estimates
        state = self.force_estimator['force_filter'].get_state()
        estimated_force = state[:3]
        estimated_torque = state[3:6]

        # Store estimates
        self.force_estimator['estimated_contact_force'] = estimated_force
        self.force_estimator['estimated_contact_torque'] = estimated_torque

        return estimated_force

    def detect_contact(self, contact_force):
        """Detect contact based on force measurements"""
        force_magnitude = np.linalg.norm(contact_force)

        # Update contact history
        self.contact_detector['contact_history'].append(force_magnitude)
        if len(self.contact_detector['contact_history']) > 10:
            self.contact_detector['contact_history'].pop(0)

        # Detect contact if force exceeds threshold
        if force_magnitude > self.contact_detector['contact_threshold']:
            self.contact_detector['contact_state'] = 'in_contact'
        else:
            self.contact_detector['contact_state'] = 'no_contact'

        return self.contact_detector['contact_state']

    def plan_manipulation(self, object_pose, contact_force, contact_state, desired_trajectory):
        """Plan manipulation based on fused sensory information"""
        # Calculate desired end-effector pose based on object pose and task
        desired_ee_pose = self.calculate_desired_ee_pose(object_pose, desired_trajectory)

        # Calculate required forces based on contact state
        required_forces = self.calculate_required_forces(
            contact_force, contact_state, desired_trajectory
        )

        # Generate control command combining position and force control
        control_command = self.generate_control_command(
            desired_ee_pose, required_forces, contact_state
        )

        return control_command

    def calculate_desired_ee_pose(self, object_pose, desired_trajectory):
        """Calculate desired end-effector pose"""
        # This would implement the specific manipulation task
        # For example, grasping, pushing, lifting, etc.
        obj_pos = object_pose[:3]
        obj_rot = object_pose[3:]

        # Example: approach object from above
        approach_offset = np.array([0, 0, 0.05])  # 5cm above object
        desired_pos = obj_pos + approach_offset

        return np.concatenate([desired_pos, obj_rot])

    def calculate_required_forces(self, contact_force, contact_state, desired_trajectory):
        """Calculate required forces for manipulation"""
        # This would implement force control based on task
        if contact_state == 'in_contact':
            # Apply desired contact force
            desired_force = desired_trajectory.get('desired_force', np.zeros(3))
        else:
            # Zero force when not in contact
            desired_force = np.zeros(3)

        return desired_force

    def generate_control_command(self, desired_pose, desired_forces, contact_state):
        """Generate control command combining position and force control"""
        # Implement hybrid position/force control
        if contact_state == 'in_contact':
            # Use impedance control to achieve desired forces
            command = self.impedance_control(desired_pose, desired_forces)
        else:
            # Use position control to reach desired pose
            command = self.position_control(desired_pose)

        return command

    def impedance_control(self, desired_pose, desired_forces):
        """Implement impedance control"""
        # Calculate position and force errors
        current_pose = np.zeros(6)  # Placeholder - would come from robot state
        pos_error = desired_pose[:3] - current_pose[:3]
        force_error = desired_forces - self.force_estimator['estimated_contact_force']

        # Apply impedance control law
        stiffness = self.impedance_controller['stiffness'][:3, :3]
        damping = 2 * self.impedance_controller['damping_ratio'] * np.sqrt(stiffness)

        # Calculate control output
        position_term = stiffness @ pos_error
        force_term = force_error  # Desired force correction

        control_output = position_term + force_term

        return {
            'type': 'impedance',
            'command': control_output,
            'desired_pose': desired_pose,
            'desired_forces': desired_forces
        }

    def position_control(self, desired_pose):
        """Implement position control"""
        return {
            'type': 'position',
            'command': desired_pose,
            'desired_pose': desired_pose
        }

# Example usage of the manipulation fusion system
def example_manipulation_task():
    """Example of using the manipulation fusion system"""
    fusion_system = ManipulationFusion()

    # Define a simple manipulation task
    desired_trajectory = {
        'task': 'grasp_object',
        'object_pose': np.array([0.5, 0.2, 0.1, 0, 0, 0]),  # Object at (0.5, 0.2, 0.1)
        'approach_direction': np.array([0, 0, -1]),  # Approach from above
        'grasp_force': 5.0  # 5N grasp force
    }

    # Simulate sensor data over time
    for t in range(50):  # 50 time steps
        # Simulate vision data
        vision_data = {
            'object_pose': np.array([0.5 + t*0.001, 0.2 + t*0.0005, 0.1, 0, 0, 0]),
            'confidence': 0.9
        }

        # Simulate force-torque data
        force_torque_data = np.random.normal(0, 0.1, 6)  # Small noise initially
        if t > 20:  # After some time, contact occurs
            force_torque_data[:3] += np.array([0, 0, 3.0])  # Contact force in z direction

        # Process manipulation task
        control_command = fusion_system.process_manipulation_task(
            vision_data, force_torque_data, desired_trajectory
        )

        print(f"Time {t*0.01:.2f}s - Control type: {control_command['type']}, "
              f"Contact state: {fusion_system.contact_detector['contact_state']}")
```

### Exercise: Implement Your Own Fusion System

Now it's your turn to implement a fusion system. The following exercise will guide you through creating a custom fusion system for a specific application:

```python
class CustomFusionExercise:
    """Template for implementing your own fusion system"""

    def __init__(self, application_type):
        """
        Initialize fusion system for a specific application
        application_type: 'navigation', 'manipulation', 'human_interaction', etc.
        """
        self.application_type = application_type
        self.sensors = {}
        self.fusion_algorithm = None
        self.performance_metrics = {}

    def add_sensor(self, sensor_name, sensor_type, data_callback):
        """Add a sensor to the fusion system"""
        self.sensors[sensor_name] = {
            'type': sensor_type,
            'callback': data_callback,
            'data_buffer': [],
            'calibration_params': {},
            'health_status': 'nominal'
        }

    def design_fusion_algorithm(self):
        """Design your fusion algorithm based on application requirements"""
        # This is where you implement your fusion logic
        # Consider:
        # - What sensors do you have?
        # - What is the state you want to estimate?
        # - What are the noise characteristics?
        # - What are the computational constraints?
        pass

    def implement_fusion_step(self, sensor_measurements):
        """Implement a single fusion step"""
        # Your fusion algorithm implementation goes here
        # Should return fused state estimate and uncertainty
        pass

    def evaluate_performance(self, ground_truth_data):
        """Evaluate the performance of your fusion system"""
        # Calculate relevant metrics for your application
        # e.g., RMSE, success rate, computational efficiency
        pass

    def run_exercise(self):
        """Run the complete fusion exercise"""
        print(f"Starting fusion exercise for {self.application_type}")

        # Step 1: Define your application requirements
        requirements = self.define_requirements()

        # Step 2: Select appropriate sensors
        self.select_sensors(requirements)

        # Step 3: Design fusion algorithm
        self.design_fusion_algorithm()

        # Step 4: Implement and test
        self.implement_and_test()

        # Step 5: Evaluate performance
        self.evaluate_performance(None)  # Add ground truth data

        print("Fusion exercise completed!")

    def define_requirements(self):
        """Define requirements for your fusion system"""
        # Example requirements - customize for your application
        requirements = {
            'accuracy': 'high',      # Required accuracy level
            'update_rate': 100,      # Required update rate (Hz)
            'robustness': 'high',    # Required robustness to sensor failures
            'computational_budget': 'medium',  # Available computational resources
            'safety_requirements': 'critical'  # Safety criticality
        }
        return requirements

    def select_sensors(self, requirements):
        """Select appropriate sensors based on requirements"""
        # Based on requirements, select sensors
        if self.application_type == 'navigation':
            self.add_sensor('imu', 'inertial', lambda: np.random.normal(0, 0.01, 6))
            self.add_sensor('camera', 'vision', lambda: np.random.rand(640, 480))
            self.add_sensor('lidar', 'range', lambda: np.random.uniform(0.1, 10, 360))
        elif self.application_type == 'manipulation':
            self.add_sensor('force_torque', 'force', lambda: np.random.normal(0, 0.1, 6))
            self.add_sensor('joint_encoders', 'position', lambda: np.random.uniform(-np.pi, np.pi, 7))
            self.add_sensor('tactile', 'touch', lambda: np.random.rand(8, 4))

    def implement_and_test(self):
        """Implement and test the fusion system"""
        # Implement your fusion algorithm
        # Test with simulated or real data
        # Iterate and improve
        pass

# Example of using the exercise template
def run_custom_fusion_exercise():
    """Run a custom fusion exercise"""
    # Example: Navigation fusion system
    nav_fusion = CustomFusionExercise('navigation')
    nav_fusion.run_exercise()

    # Example: Manipulation fusion system
    manip_fusion = CustomFusionExercise('manipulation')
    manip_fusion.run_exercise()

# Uncomment to run the exercises
# run_custom_fusion_exercise()
```

## Summary

This chapter covered the essential concepts of multi-sensor fusion for humanoid robots, from fundamental techniques like Kalman filtering and particle filtering to advanced approaches using machine learning and information theory.

We explored different fusion architectures including centralized and distributed approaches, with practical implementations for each. The chapter provided detailed examples of integrating specific sensor types like vision-inertial systems and force-tactile fusion for manipulation tasks.

We also covered critical aspects of practical fusion systems including fault detection and isolation, which is essential for maintaining robust operation in real-world scenarios. The hands-on exercise provided practical experience implementing a complete fusion system with evaluation metrics.

Advanced topics included information-theoretic fusion approaches that optimize based on information content, deep learning methods for learning fusion strategies from data, and resource management techniques for efficient real-time operation.

The chapter emphasized that successful multi-sensor fusion requires careful consideration of sensor characteristics, environmental conditions, computational constraints, and the specific requirements of the humanoid robot application.

## Key Takeaways

- Kalman filtering provides optimal state estimation for linear Gaussian systems
- Particle filtering handles non-linear, non-Gaussian systems effectively
- Sensor fusion improves accuracy, reliability, and robustness compared to single sensors
- Fault detection and isolation are crucial for safety-critical humanoid applications
- Information-theoretic approaches can optimize fusion based on information content
- Deep learning enables data-driven fusion strategy learning
- Resource management ensures efficient real-time operation
- Different fusion architectures suit different application requirements
- Safety and ethical considerations are paramount in humanoid robot deployment
- Evaluation metrics are essential for validating fusion system performance
- Practical implementation requires careful integration of multiple sensor types
- Real-world deployment needs robust handling of environmental variations

## Next Steps

In the next chapter, we'll explore system integration and deployment considerations, including real-time performance optimization, hardware integration, and field deployment strategies for humanoid robots. We'll also examine how the multi-sensor fusion systems developed in this chapter integrate with higher-level control and decision-making systems to enable truly autonomous humanoid robot operation in complex environments.

## References and Further Reading

1. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. John Wiley & Sons.
2. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
3. Reif, K., Günther, S., Yaz, E., & Unbehauen, R. (1999). Stochastic stability of the discrete-time extended Kalman filter. IEEE Transactions on Automatic Control.
4. Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of Nonlinear Filtering.
5. Chen, L. (2003). Bayesian filtering: From Kalman filters to particle filters, and beyond. Statistics.
6. Raiko, T., Valpola, H., & Ypma, A. (2005). Fast and simple gradient-based adaptive learning rate method. Neural Networks.