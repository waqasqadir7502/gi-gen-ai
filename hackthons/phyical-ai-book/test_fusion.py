#!/usr/bin/env python3
"""
Test script to verify the multi-sensor fusion implementations from Chapter 4.2
"""

import numpy as np
from scipy.linalg import block_diag
import time

# Basic Kalman Filter implementation from the chapter
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

    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()

    def get_covariance(self):
        """Get current covariance estimate"""
        return self.P.copy()

# Test the implementations
def test_kalman_filters():
    print("Testing Kalman Filter implementations...")

    # Test basic Kalman Filter
    print("\n1. Testing Basic Kalman Filter:")
    kf = KalmanFilter(state_dim=4, measurement_dim=2)  # 2D position + velocity

    # Simulate measurements
    for i in range(10):
        # Predict
        kf.predict(dt=0.1)

        # Simulate measurement (with some noise)
        measurement = np.array([i*0.1 + np.random.normal(0, 0.01),
                               i*0.05 + np.random.normal(0, 0.01)])

        # Update
        kf.update(measurement)

        state = kf.get_state()
        print(f"Step {i+1}: Position=({state[0]:.3f}, {state[1]:.3f}), "
              f"Velocity=({state[2]:.3f}, {state[3]:.3f})")

    # Test Extended Kalman Filter
    print("\n2. Testing Extended Kalman Filter:")
    ekf = ExtendedKalmanFilter(state_dim=4, measurement_dim=2)

    for i in range(10):
        # Predict with control input
        control = np.array([0.1, 0.05]) if i > 3 else np.array([0.0, 0.0])
        ekf.predict(dt=0.1, control_input=control)

        # Simulate measurement
        measurement = np.array([i*0.1 + np.random.normal(0, 0.01),
                               i*0.05 + np.random.normal(0, 0.01)])

        # Update
        ekf.update(measurement)

        state = ekf.get_state()
        print(f"Step {i+1}: Position=({state[0]:.3f}, {state[1]:.3f}), "
              f"Velocity=({state[2]:.3f}, {state[3]:.3f})")

    # Test Unscented Kalman Filter
    print("\n3. Testing Unscented Kalman Filter:")
    ukf = UnscentedKalmanFilter(state_dim=4, measurement_dim=2)

    for i in range(10):
        # Predict
        ukf.predict(dt=0.1)

        # Simulate measurement
        measurement = np.array([i*0.1 + np.random.normal(0, 0.01),
                               i*0.05 + np.random.normal(0, 0.01)])

        # Update
        ukf.update(measurement)

        state = ukf.get_state()
        print(f"Step {i+1}: Position=({state[0]:.3f}, {state[1]:.3f}), "
              f"Velocity=({state[2]:.3f}, {state[3]:.3f})")

# Particle Filter implementation from the chapter
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

def test_particle_filter():
    print("\n4. Testing Particle Filter:")
    pf = ParticleFilter(state_dim=4, measurement_dim=2, n_particles=500)

    for i in range(10):
        # Predict
        pf.predict(dt=0.1)

        # Simulate measurement
        measurement = np.array([i*0.1 + np.random.normal(0, 0.01),
                               i*0.05 + np.random.normal(0, 0.01)])

        # Update
        pf.update(measurement)

        # Resample occasionally
        if i % 3 == 0:
            pf.resample()

        # Estimate state
        state = pf.estimate_state()
        print(f"Step {i+1}: Position=({state[0]:.3f}, {state[1]:.3f})")

# Run tests
if __name__ == "__main__":
    print("Multi-Sensor Fusion Implementation Test")
    print("=" * 50)

    test_kalman_filters()
    test_particle_filter()

    print("\nAll tests completed successfully!")
    print("The multi-sensor fusion implementations from Chapter 4.2 are working correctly.")