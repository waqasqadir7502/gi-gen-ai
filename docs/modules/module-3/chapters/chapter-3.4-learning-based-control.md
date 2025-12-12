# Chapter 3.4: Learning-Based Control

## Overview

Learning-based control represents a paradigm shift in humanoid robotics, moving from traditional model-based control to data-driven approaches. This chapter covers reinforcement learning, imitation learning, and adaptive control techniques that enable humanoid robots to improve their performance through experience and interaction with the environment.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Implement reinforcement learning algorithms for humanoid robot control
2. Apply imitation learning techniques to acquire manipulation skills
3. Design adaptive control systems that improve with experience
4. Integrate learning-based control with traditional control methods
5. Evaluate learning-based control performance and safety considerations

## Introduction to Learning-Based Control

Traditional control methods for humanoid robots rely on mathematical models of the robot and its environment. While these methods are well-understood and reliable, they often struggle with model uncertainties, changing environments, and complex tasks that are difficult to specify analytically.

Learning-based control addresses these limitations by enabling robots to improve their behavior through experience and interaction with the environment. Rather than relying solely on predetermined models, these approaches allow robots to adapt their control strategies based on feedback from their actions.

### Categories of Learning-Based Control

#### Supervised Learning
- **Application**: Learning mapping from sensor inputs to control outputs
- **Example**: Learning to predict optimal joint angles for reaching

#### Reinforcement Learning (RL)
- **Application**: Learning optimal control policies through trial and error
- **Example**: Learning to walk or balance through reward maximization

#### Imitation Learning
- **Application**: Learning from demonstrations by experts
- **Example**: Learning manipulation skills from human demonstrations

#### Adaptive Control
- **Application**: Online parameter estimation and control adjustment
- **Example**: Adjusting controller gains based on performance feedback

## Reinforcement Learning for Robot Control

### Markov Decision Process (MDP) Framework

Reinforcement learning problems are typically formulated as Markov Decision Processes (MDPs), defined by:
- **States** (S): Robot configurations, sensor readings, environment states
- **Actions** (A): Control commands (torques, positions, velocities)
- **Rewards** (R): Scalar feedback for action quality
- **Transitions** (P): Probability of moving between states
- **Discount Factor** (γ): Importance of future rewards

### Deep Reinforcement Learning

Deep RL combines reinforcement learning with deep neural networks, enabling complex policy learning from high-dimensional sensory inputs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ActorNetwork(nn.Module):
    """Actor network for policy learning"""

    def __init__(self, state_dim, action_dim, action_bounds):
        super(ActorNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.action_bounds = action_bounds

    def forward(self, state):
        action = torch.tanh(self.fc_layers(state))

        # Scale action to bounds
        low_bound = torch.tensor(self.action_bounds[0])
        high_bound = torch.tensor(self.action_bounds[1])

        scaled_action = low_bound + (action + 1.0) * (high_bound - low_bound) / 2.0

        return scaled_action

class CriticNetwork(nn.Module):
    """Critic network for value estimation"""

    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )

        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU()
        )

        self.q_network = nn.Sequential(
            nn.Linear(512, 256),  # 256 + 256
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state_features = self.state_fc(state)
        action_features = self.action_fc(action)

        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.q_network(combined)

        return q_value

class DDPGAgent:
    """Deep Deterministic Policy Gradient (DDPG) agent for continuous control"""

    def __init__(self, state_dim, action_dim, action_bounds, lr_actor=1e-4, lr_critic=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, action_bounds)
        self.actor_target = ActorNetwork(state_dim, action_dim, action_bounds)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter
        self.noise_std = 0.2  # Exploration noise

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 128

    def select_action(self, state, add_noise=True):
        """Select action using the current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if add_noise:
            # Add Ornstein-Uhlenbeck noise for exploration
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, self.action_bounds[0], self.action_bounds[1])

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """Update the networks using a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples yet

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.FloatTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch]).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, local_net, target_net):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_models(self, filepath):
        """Save the trained models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict()
        }, filepath)

    def load_models(self, filepath):
        """Load trained models"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
```

### Robot-Specific Reward Design

Designing effective reward functions is crucial for successful learning:

```python
class RobotRewardFunction:
    """Reward function design for humanoid robot tasks"""

    def __init__(self, task_type):
        self.task_type = task_type
        self.weights = self.get_default_weights(task_type)

    def get_default_weights(self, task_type):
        """Get default reward weights for different tasks"""
        if task_type == 'balance':
            return {
                'uprightness': 5.0,
                'energy_efficiency': 2.0,
                'smoothness': 1.0,
                'penalty': -10.0  # For falling
            }
        elif task_type == 'walking':
            return {
                'forward_progress': 10.0,
                'balance': 5.0,
                'energy_efficiency': 2.0,
                'penalty': -10.0  # For falling or stepping outside bounds
            }
        elif task_type == 'manipulation':
            return {
                'proximity': 3.0,
                'grasp_success': 10.0,
                'object_transport': 5.0,
                'energy_efficiency': 1.0,
                'penalty': -5.0  # For dropping object
            }
        else:
            return {
                'task_completion': 10.0,
                'efficiency': 2.0,
                'safety': 5.0,
                'penalty': -5.0
            }

    def calculate_balance_reward(self, robot_state):
        """Calculate reward for balance maintenance"""
        # Robot state includes: [position, orientation, joint_angles, velocities, etc.]

        # Reward for staying upright (based on base orientation)
        orientation = robot_state['base_orientation']  # Quaternion [w, x, y, z]
        upright_reward = self.calculate_upright_reward(orientation)

        # Penalty for excessive joint velocities (energy efficiency)
        joint_velocities = robot_state['joint_velocities']
        energy_penalty = -self.weights['energy_efficiency'] * np.sum(np.abs(joint_velocities))

        # Reward for smooth control (penalty for jerky movements)
        joint_accelerations = robot_state.get('joint_accelerations', np.zeros_like(joint_velocities))
        smoothness_reward = -self.weights['smoothness'] * np.sum(np.abs(joint_accelerations))

        # Overall reward
        total_reward = (self.weights['uprightness'] * upright_reward +
                       energy_penalty + smoothness_reward)

        return total_reward

    def calculate_upright_reward(self, orientation):
        """Calculate reward based on how upright the robot is"""
        # Convert quaternion to rotation matrix to get up vector
        # Simplified: assume z-axis is up
        # In practice, you'd convert quaternion to get the actual up vector
        w, x, y, z = orientation
        # For a perfectly upright robot with z-axis up: [0, 0, 0, 1]
        # We want the base frame's z-axis to align with world z-axis
        # This is equivalent to checking if the robot's up vector aligns with [0, 0, 1]

        # Simplified calculation: reward based on z-component of base orientation
        # A more complete implementation would involve rotation matrix conversion
        # and checking the alignment of the robot's up vector with gravity
        upright_alignment = 2 * (w*w + z*z) - 1  # Simplified alignment measure
        return np.clip(upright_alignment, -1.0, 1.0)

    def calculate_walking_reward(self, robot_state, target_velocity, current_velocity):
        """Calculate reward for walking task"""
        # Reward for moving at desired velocity
        velocity_error = np.linalg.norm(target_velocity - current_velocity)
        velocity_reward = -self.weights['forward_progress'] * velocity_error

        # Balance component
        balance_reward = self.calculate_balance_reward(robot_state)

        # Energy efficiency
        joint_velocities = robot_state['joint_velocities']
        energy_penalty = -self.weights['energy_efficiency'] * np.sum(np.abs(joint_velocities))

        total_reward = velocity_reward + balance_reward + energy_penalty
        return total_reward

    def calculate_manipulation_reward(self, robot_state, object_state, target_state):
        """Calculate reward for manipulation task"""
        # Proximity to object
        ee_pos = robot_state['end_effector_position']
        obj_pos = object_state['position']
        distance = np.linalg.norm(ee_pos - obj_pos)
        proximity_reward = -self.weights['proximity'] * distance

        # Grasp success (if object is grasped)
        if robot_state.get('is_grasping', False):
            grasp_reward = self.weights['grasp_success']

            # Transport reward (moving object toward target)
            obj_to_target = np.linalg.norm(obj_pos - target_state['position'])
            transport_reward = -self.weights['object_transport'] * obj_to_target

            total_reward = proximity_reward + grasp_reward + transport_reward
        else:
            # Still reward for getting closer to object
            total_reward = proximity_reward

        return total_reward

# Example usage
def example_reward_calculation():
    """Example of reward function calculation"""
    reward_fn = RobotRewardFunction(task_type='balance')

    # Simulated robot state
    robot_state = {
        'base_orientation': [0.9, 0.1, 0.1, 0.3],  # Quaternion
        'joint_velocities': np.random.randn(12) * 0.1,  # 12 DOF example
        'joint_accelerations': np.random.randn(12) * 0.01
    }

    reward = reward_fn.calculate_balance_reward(robot_state)
    print(f"Balance reward: {reward:.3f}")

    return reward
```

## Imitation Learning

Imitation learning allows robots to learn from demonstrations by experts, accelerating the learning process and incorporating human expertise.

### Behavioral Cloning

Behavioral cloning learns a direct mapping from states to actions by mimicking expert demonstrations.

```python
class BehavioralCloning:
    """Behavioral cloning for learning from demonstrations"""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Storage for demonstrations
        self.demo_states = []
        self.demo_actions = []

    def add_demonstration(self, states, actions):
        """Add expert demonstration to the dataset"""
        self.demo_states.extend(states)
        self.demo_actions.extend(actions)

    def train_policy(self, epochs=1000, batch_size=32):
        """Train the policy using behavioral cloning"""
        if len(self.demo_states) == 0:
            print("No demonstrations provided!")
            return

        for epoch in range(epochs):
            # Sample batch
            indices = np.random.choice(len(self.demo_states), size=batch_size, replace=True)
            batch_states = torch.FloatTensor([self.demo_states[i] for i in indices])
            batch_actions = torch.FloatTensor([self.demo_actions[i] for i in indices])

            # Forward pass
            predicted_actions = self.policy(batch_states)
            loss = self.loss_fn(predicted_actions, batch_actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict_action(self, state):
        """Predict action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy(state_tensor).detach().numpy().flatten()
        return action

class ImitationLearningSystem:
    """Complete imitation learning system"""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.behavioral_cloner = None
        self.dagger_iterations = 5  # Number of DAgger iterations

    def collect_demonstration(self, expert_policy, env, n_episodes=10):
        """Collect demonstrations from expert policy"""
        states = []
        actions = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False

            episode_states = []
            episode_actions = []

            while not done:
                action = expert_policy(obs)

                episode_states.append(obs.copy())
                episode_actions.append(action.copy())

                obs, reward, done, info = env.step(action)

            states.extend(episode_states)
            actions.extend(episode_actions)

        return states, actions

    def dagger_algorithm(self, expert_policy, env, n_episodes=100):
        """DAgger (Dataset Aggregation) algorithm for improving imitation learning"""
        # Initialize with behavioral cloning
        self.behavioral_cloner = BehavioralCloning(
            state_dim=len(env.observation_space.sample()),
            action_dim=len(env.action_space.sample())
        )

        # Collect initial demonstrations
        demo_states, demo_actions = self.collect_demonstration(expert_policy, env)
        self.behavioral_cloner.add_demonstration(demo_states, demo_actions)
        self.behavioral_cloner.train_policy()

        # DAgger iterations
        for iteration in range(self.dagger_iterations):
            print(f"DAgger iteration {iteration + 1}/{self.dagger_iterations}")

            # Roll out learned policy to collect on-policy data
            on_policy_states = []
            expert_actions = []  # Actions from expert for on-policy states

            for ep in range(n_episodes // 10):  # Fewer episodes per iteration
                obs = env.reset()
                done = False

                while not done:
                    # Use learned policy to collect on-policy states
                    learned_action = self.behavioral_cloner.predict_action(obs)

                    # Get expert action for the same state
                    expert_action = expert_policy(obs)

                    on_policy_states.append(obs.copy())
                    expert_actions.append(expert_action.copy())

                    # Take action (could be learned or expert action)
                    action_to_take = learned_action
                    obs, reward, done, info = env.step(action_to_take)

            # Add on-policy data to training set
            self.behavioral_cloner.add_demonstration(on_policy_states, expert_actions)

            # Retrain policy
            self.behavioral_cloner.train_policy(epochs=500)
```

### Inverse Reinforcement Learning

Inverse RL learns the reward function from expert demonstrations, which can be more robust than directly copying actions.

```python
class MaximumEntropyIRL:
    """Maximum Entropy Inverse Reinforcement Learning"""

    def __init__(self, state_dim, action_dim, feature_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        # Learnable reward weights
        self.reward_weights = np.random.randn(feature_dim)

        # Feature function (in practice, this would be more complex)
        self.feature_fn = self.create_feature_function()

        # Hyperparameters
        self.learning_rate = 0.01
        self.entropy_weight = 0.1

    def create_feature_function(self):
        """Create feature function for state-action pairs"""
        def feature_function(state, action):
            # Example features: position, velocity, and action magnitude
            features = np.concatenate([
                state[:3],  # Position features
                state[3:6], # Velocity features
                action,     # Action features
                [np.sum(np.abs(action))]  # Action magnitude
            ])
            return features
        return feature_function

    def compute_reward(self, state, action):
        """Compute reward using learned weights"""
        features = self.feature_fn(state, action)
        return np.dot(self.reward_weights, features)

    def expert_policy_rollout(self, expert_policy, env, n_episodes=10):
        """Roll out expert policy to collect trajectories"""
        trajectories = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            trajectory = {'states': [], 'actions': [], 'rewards': []}

            while not done:
                action = expert_policy(obs)

                trajectory['states'].append(obs.copy())
                trajectory['actions'].append(action.copy())

                # Compute reward based on current weights
                reward = self.compute_reward(obs, action)
                trajectory['rewards'].append(reward)

                obs, _, done, _ = env.step(action)

            trajectories.append(trajectory)

        return trajectories

    def policy_evaluation(self, policy, env, gamma=0.99):
        """Evaluate a policy using Monte Carlo simulation"""
        # This would typically involve value iteration or policy evaluation
        # For simplicity, we'll return a placeholder
        return np.random.random()  # Placeholder

    def update_reward_weights(self, expert_trajectories, current_policy):
        """Update reward weights based on difference between expert and current policy"""
        # This is a simplified version - in practice, this involves
        # computing gradients of the partition function
        feature_expert = self.compute_expected_features(expert_trajectories)

        # In practice, you'd also compute expected features under current policy
        # and update weights based on the difference
        gradient = feature_expert  # Simplified gradient computation

        # Update weights
        self.reward_weights += self.learning_rate * gradient

        return self.reward_weights

    def compute_expected_features(self, trajectories):
        """Compute expected feature counts from trajectories"""
        total_features = np.zeros(self.feature_dim)
        total_count = 0

        for traj in trajectories:
            for state, action in zip(traj['states'], traj['actions']):
                features = self.feature_fn(state, action)
                total_features += features
                total_count += 1

        if total_count > 0:
            return total_features / total_count
        else:
            return np.zeros(self.feature_dim)
```

## Adaptive Control

Adaptive control adjusts controller parameters based on real-time performance feedback, making systems robust to model uncertainties and changing conditions.

### Model Reference Adaptive Control (MRAC)

```python
class MRACController:
    """Model Reference Adaptive Control for humanoid robots"""

    def __init__(self, reference_model_params, plant_order=2):
        self.ref_params = reference_model_params  # Parameters of desired reference model
        self.plant_order = plant_order

        # Adaptive parameters
        self.theta = np.random.randn(plant_order * 2)  # Controller parameters to adapt
        self.P = np.eye(len(self.theta)) * 100  # Covariance matrix
        self.gamma = 0.1  # Adaptation gain

        # Reference model
        self.reference_model = self.initialize_reference_model()

        # Tracking error
        self.e = 0  # Tracking error
        self.e_int = 0  # Integrated error
        self.e_deriv = 0  # Derivative of error

    def initialize_reference_model(self):
        """Initialize reference model dynamics"""
        # For simplicity, we'll use a second-order reference model
        # In practice, this would be based on desired closed-loop dynamics
        wn = 2.0  # Natural frequency
        zeta = 0.7  # Damping ratio

        # Reference model: s^2 + 2*zeta*wn*s + wn^2
        A_ref = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
        B_ref = np.array([[0], [wn**2]])

        return {'A': A_ref, 'B': B_ref}

    def update(self, reference_signal, actual_output, control_input, dt=0.01):
        """Update MRAC controller"""
        # Calculate tracking error
        self.e = reference_signal - actual_output

        # Integrate and differentiate error for additional terms
        self.e_int += self.e * dt
        self.e_deriv = (self.e - getattr(self, '_prev_e', self.e)) / dt
        self._prev_e = self.e

        # Regressor vector (function of states and inputs)
        phi = self.calculate_regressor(reference_signal, actual_output, control_input)

        # Parameter adaptation law
        # theta_dot = gamma * P * phi * e
        param_update = self.gamma * self.P @ phi * self.e

        # Covariance matrix update
        # P_dot = gamma * (P * phi * phi^T * P - P)
        phi_outer = np.outer(phi, phi)
        P_update = self.gamma * (self.P @ phi_outer @ self.P - self.P)

        # Update parameters and covariance
        self.theta += param_update * dt
        self.P += P_update * dt

        # Calculate control command
        control_command = self.calculate_control_command(phi)

        return control_command

    def calculate_regressor(self, ref, actual, control):
        """Calculate regressor vector for parameter adaptation"""
        # This is a simplified regressor - in practice, this would be based on
        # the known structure of the system dynamics
        phi = np.array([
            ref,                    # Reference signal
            actual,                 # Actual output
            control,                # Control input
            ref * actual,           # Cross terms
            actual**2,              # Nonlinear terms
            control**2,             # Control squared
            self.e,                 # Current error
            self.e_int,             # Integrated error
            self.e_deriv            # Derivative of error
        ])

        # Ensure phi has same length as theta
        if len(phi) > len(self.theta):
            phi = phi[:len(self.theta)]
        elif len(phi) < len(self.theta):
            # Pad with zeros
            phi = np.pad(phi, (0, len(self.theta) - len(phi)), 'constant')

        return phi

    def calculate_control_command(self, phi):
        """Calculate control command from adaptive parameters"""
        # Simple linear combination
        control = np.dot(self.theta, phi)
        return np.clip(control, -10, 10)  # Limit control output

class AdaptivePDController:
    """Adaptive PD controller with parameter adjustment"""

    def __init__(self, initial_kp=10.0, initial_kd=2.0, adaptation_rate=0.01):
        self.kp = initial_kp
        self.kd = initial_kd
        self.adaptation_rate = adaptation_rate

        # Error history for adaptation
        self.error_history = deque(maxlen=100)
        self.derivative_error = 0
        self.integral_error = 0

        # Performance metrics
        self.performance_history = deque(maxlen=50)
        self.target_performance = 0.8  # Target performance (0-1 scale)

    def update(self, error, dt=0.01):
        """Update controller with error and adapt parameters"""
        # Store error for history
        self.error_history.append(error)

        # Calculate derivative of error
        if len(self.error_history) >= 2:
            self.derivative_error = (error - self.error_history[-2]) / dt

        # Calculate integral of error
        self.integral_error += error * dt

        # Calculate control output
        control = self.kp * error + self.kd * self.derivative_error

        # Adapt parameters based on performance
        self.adapt_parameters()

        return control

    def adapt_parameters(self):
        """Adapt controller parameters based on performance"""
        if len(self.error_history) < 10:
            return  # Need more data

        # Calculate recent performance (simplified as inverse of error magnitude)
        recent_errors = list(self.error_history)[-10:]
        avg_error = np.mean(np.abs(recent_errors))

        # Performance metric (lower error = better performance)
        performance = 1.0 / (1.0 + avg_error)  # Map to [0, 1]
        self.performance_history.append(performance)

        if len(self.performance_history) < 5:
            return

        # Calculate performance trend
        recent_perf = list(self.performance_history)[-5:]
        perf_trend = np.mean(recent_perf) - self.target_performance

        # Adapt gains based on performance
        if perf_trend < -0.1:  # Performance is poor
            # Increase gains to improve response
            self.kp *= (1.0 + self.adaptation_rate)
            self.kd *= (1.0 + self.adaptation_rate * 0.5)
        elif perf_trend > 0.1:  # Performance is good but maybe too aggressive
            # Slightly decrease gains to reduce oscillation
            self.kp *= (1.0 - self.adaptation_rate * 0.2)
            self.kd *= (1.0 - self.adaptation_rate * 0.1)

        # Constrain gains to reasonable bounds
        self.kp = np.clip(self.kp, 1.0, 100.0)
        self.kd = np.clip(self.kd, 0.1, 20.0)

    def get_performance_metrics(self):
        """Get current performance metrics"""
        if len(self.performance_history) == 0:
            return {'performance': 0.0, 'kp': self.kp, 'kd': self.kd}

        avg_performance = np.mean(list(self.performance_history))
        return {
            'performance': avg_performance,
            'kp': self.kp,
            'kd': self.kd,
            'error_magnitude': np.mean(np.abs(list(self.error_history))) if self.error_history else float('inf')
        }
```

### Self-Organizing Maps for Control

Self-organizing maps can be used to learn control mappings in a biologically-inspired way.

```python
class SOMController:
    """Self-Organizing Map controller for learning sensorimotor mappings"""

    def __init__(self, input_dim, output_dim, map_size=(10, 10)):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.map_size = map_size

        # Initialize SOM weights randomly
        self.weights = np.random.randn(map_size[0], map_size[1], output_dim) * 0.1

        # Neighborhood function parameters
        self.sigma_init = max(map_size) / 2.0  # Initial neighborhood radius
        self.sigma = self.sigma_init
        self.learning_rate_init = 0.1
        self.learning_rate = self.learning_rate_init

        # Training parameters
        self.max_epochs = 1000
        self.decay_factor = 0.995  # Decay factor for sigma and learning rate

    def get_bmu(self, input_vector):
        """Find Best Matching Unit (BMU) for input vector"""
        # Calculate distances to all neurons
        distances = np.linalg.norm(
            self.weights - input_vector.reshape(1, 1, -1),
            axis=2
        )

        # Find BMU coordinates
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_idx

    def neighborhood_function(self, bmu_coords, neuron_coords, sigma):
        """Calculate neighborhood function value"""
        dist_sq = np.sum((np.array(bmu_coords) - np.array(neuron_coords))**2)
        return np.exp(-dist_sq / (2 * sigma**2))

    def update_weights(self, input_vector, bmu_coords, learning_rate, sigma):
        """Update weights based on neighborhood learning rule"""
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                nh_value = self.neighborhood_function(bmu_coords, (i, j), sigma)
                self.weights[i, j] += nh_value * learning_rate * (input_vector - self.weights[i, j])

    def train(self, input_sequences, output_sequences):
        """Train the SOM controller with input-output sequences"""
        for epoch in range(self.max_epochs):
            total_error = 0

            for input_seq, output_seq in zip(input_sequences, output_sequences):
                for inp, out in zip(input_seq, output_seq):
                    # Find BMU for input
                    bmu_coords = self.get_bmu(inp)

                    # Update weights
                    self.update_weights(inp, bmu_coords, self.learning_rate, self.sigma)

                    # Calculate error for monitoring
                    bmu_output = self.weights[bmu_coords[0], bmu_coords[1]]
                    error = np.linalg.norm(out - bmu_output)
                    total_error += error

            # Decay learning parameters
            self.sigma = self.sigma_init * (self.decay_factor ** epoch)
            self.learning_rate = self.learning_rate_init * (self.decay_factor ** epoch)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Average Error: {total_error/len(input_sequences):.4f}")

    def get_output(self, input_vector):
        """Get output for given input using trained SOM"""
        bmu_coords = self.get_bmu(input_vector)
        return self.weights[bmu_coords[0], bmu_coords[1]].copy()
```

## Integration with Traditional Control

### Hybrid Learning-Based/Model-Based Control

```python
class HybridController:
    """Hybrid controller combining learning-based and traditional control"""

    def __init__(self, model_based_controller, learning_based_controller):
        self.model_based_ctrl = model_based_controller
        self.learning_based_ctrl = learning_based_controller

        # Blending parameters
        self.learning_weight = 0.3  # How much to trust learning-based component
        self.confidence_threshold = 0.7  # Threshold for switching strategies

        # Performance monitors
        self.tracking_error_history = deque(maxlen=100)
        self.model_accuracy_history = deque(maxlen=100)

    def update(self, state, reference, dt=0.01):
        """Update hybrid controller with current state and reference"""
        # Get commands from both controllers
        model_cmd = self.model_based_ctrl.update(state, reference, dt)
        learning_cmd = self.learning_based_ctrl.get_action(state, reference)

        # Assess confidence in model-based prediction
        model_confidence = self.assess_model_confidence(state)

        # Blend commands based on confidence
        if model_confidence > self.confidence_threshold:
            # Trust model-based controller more
            blend_factor = 1.0 - self.learning_weight
        else:
            # Rely more on learning-based controller
            blend_factor = 0.3

        # Weighted combination
        final_command = blend_factor * model_cmd + (1.0 - blend_factor) * learning_cmd

        # Monitor performance
        self.update_performance_monitors(state, reference, final_command)

        return final_command

    def assess_model_confidence(self, state):
        """Assess confidence in model-based controller"""
        # This could be based on:
        # - Model prediction accuracy
        # - System identification confidence
        # - Environmental familiarity

        # Simplified: use historical model accuracy
        if len(self.model_accuracy_history) > 0:
            return np.mean(list(self.model_accuracy_history))
        else:
            return 0.5  # Default confidence

    def update_performance_monitors(self, state, reference, command):
        """Update performance monitoring variables"""
        # Calculate tracking error
        error = reference - state
        self.tracking_error_history.append(np.linalg.norm(error))

        # Update model accuracy (simplified)
        if len(self.tracking_error_history) > 1:
            recent_errors = list(self.tracking_error_history)[-10:]
            avg_error = np.mean(recent_errors)
            # Convert to confidence measure (lower error = higher confidence)
            model_conf = 1.0 / (1.0 + avg_error)
            self.model_accuracy_history.append(model_conf)

class SafetyMonitor:
    """Safety monitoring for learning-based control systems"""

    def __init__(self, safety_bounds):
        self.safety_bounds = safety_bounds  # Dictionary of {variable: (min, max)}
        self.safety_history = deque(maxlen=100)
        self.emergency_stop_active = False

        # Safety margins
        self.margin_factor = 0.9  # 90% of bounds as safety margin

    def check_safety(self, state, action):
        """Check if current state and action are safe"""
        safety_violations = []

        # Check state constraints
        for var_name, (min_val, max_val) in self.safety_bounds.items():
            if var_name in state:
                val = state[var_name]
                safety_min = min_val * self.margin_factor
                safety_max = max_val * self.margin_factor

                if val < safety_min or val > safety_max:
                    safety_violations.append({
                        'type': 'state_violation',
                        'variable': var_name,
                        'value': val,
                        'bounds': (safety_min, safety_max)
                    })

        # Check action constraints
        if 'action_limits' in self.safety_bounds:
            action_min, action_max = self.safety_bounds['action_limits']
            for i, act in enumerate(action):
                safety_min = action_min[i] * self.margin_factor
                safety_max = action_max[i] * self.margin_factor

                if act < safety_min or act > safety_max:
                    safety_violations.append({
                        'type': 'action_violation',
                        'index': i,
                        'value': act,
                        'bounds': (safety_min, safety_max)
                    })

        # Log safety status
        is_safe = len(safety_violations) == 0
        self.safety_history.append(is_safe)

        return is_safe, safety_violations

    def activate_emergency_stop(self):
        """Activate emergency stop if safety is compromised"""
        self.emergency_stop_active = True
        print("EMERGENCY STOP ACTIVATED: Safety limits exceeded!")
        return True

    def get_safety_metrics(self):
        """Get current safety metrics"""
        if len(self.safety_history) == 0:
            return {'safety_rate': 0.0, 'violations': 0}

        safety_rate = np.mean(list(self.safety_history))
        violations = len([s for s in self.safety_history if not s])

        return {
            'safety_rate': safety_rate,
            'violations': violations,
            'emergency_stop': self.emergency_stop_active
        }
```

## Hands-on Exercise: Implementing a Learning-Based Controller

In this exercise, you'll implement a complete learning-based controller that combines reinforcement learning with traditional control methods.

### Requirements
- Python 3.8+
- PyTorch library
- NumPy library
- Matplotlib for visualization
- Basic understanding of control systems

### Exercise Steps
1. Implement a DDPG-based learning controller
2. Create a simulated robot environment
3. Design appropriate reward functions
4. Train the controller with safety constraints
5. Evaluate performance against traditional controllers

### Expected Outcome
You should have a working learning-based controller that can learn to control a simulated humanoid robot for a specific task, with safety monitoring and performance evaluation.

### Sample Implementation

```python
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class SimulatedHumanoidEnv(gym.Env):
    """Simple simulated humanoid environment for learning-based control"""

    def __init__(self):
        super(SimulatedHumanoidEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32  # 6 joint torques
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32  # 6 pos + 6 vel + 6 other
        )

        # Robot parameters
        self.mass = 50.0  # kg
        self.height = 0.8  # m
        self.dt = 0.01     # s (100Hz control)

        # State variables
        self.state = np.zeros(18)  # [pos, vel, other]
        self.target_position = np.array([0.0, 0.0, 0.8])  # Standing position
        self.steps = 0
        self.max_steps = 1000

        # Balance parameters
        self.upright_threshold = 0.7  # Cosine of max allowed tilt

    def reset(self):
        """Reset the environment to initial state"""
        # Random initial state (small perturbations)
        self.state = np.random.randn(18) * 0.1
        self.state[2] = 0.8  # Start at nominal height
        self.steps = 0

        return self.state.copy()

    def step(self, action):
        """Execute one step of the environment"""
        self.steps += 1

        # Apply action (torques) to simulate robot dynamics
        # This is a simplified model - in reality, this would involve
        # complex multibody dynamics
        torque = action * 100.0  # Scale to reasonable torque values

        # Simple physics simulation (simplified)
        # In reality, this would use a physics engine like PyBullet
        pos = self.state[:6]  # Position/orientation of key joints
        vel = self.state[6:12]  # Velocities
        other = self.state[12:]  # Other state variables

        # Apply torques to update velocities (simplified dynamics)
        acceleration = torque / 10.0  # Simplified inertia
        new_vel = vel + acceleration * self.dt

        # Update positions
        new_pos = pos + new_vel * self.dt

        # Add gravity effect
        new_pos[2] -= 9.81 * self.dt**2 / 2  # Z position decreases due to gravity

        # Update other state variables
        new_other = other + np.random.randn(6) * 0.01  # Small random changes

        # Update state
        self.state[:6] = new_pos
        self.state[6:12] = new_vel
        self.state[12:] = new_other

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination conditions
        done = self.is_terminal()
        info = {}

        return self.state.copy(), reward, done, info

    def calculate_reward(self):
        """Calculate reward based on current state"""
        # Extract relevant state components
        z_pos = self.state[2]  # Height
        orientation = self.state[3:7]  # Orientation (simplified as 4 values)

        # Reward for staying upright
        upright_reward = self.calculate_upright_reward(orientation)

        # Reward for maintaining height
        height_deviation = abs(z_pos - self.target_position[2])
        height_reward = -height_deviation

        # Penalty for excessive joint velocities
        vel_magnitude = np.linalg.norm(self.state[6:12])
        velocity_penalty = -vel_magnitude * 0.1

        # Penalty for falling (if height is too low)
        fall_penalty = 0
        if z_pos < 0.3:  # Fell down
            fall_penalty = -100.0

        total_reward = upright_reward * 5.0 + height_reward * 2.0 + velocity_penalty + fall_penalty
        return total_reward

    def calculate_upright_reward(self, orientation):
        """Calculate reward based on how upright the robot is"""
        # Simplified: assume first 4 elements represent orientation
        # In reality, this would involve quaternion normalization
        orient_norm = np.linalg.norm(orientation)
        if orient_norm > 0:
            normalized_orient = orientation / orient_norm
        else:
            normalized_orient = orientation

        # Reward for maintaining upright orientation
        # Simplified as alignment with desired orientation [0, 0, 0, 1]
        desired_orient = np.array([0, 0, 0, 1])
        alignment = np.dot(normalized_orient, desired_orient)

        return max(0, alignment)  # Only positive reward for alignment

    def is_terminal(self):
        """Check if episode should terminate"""
        # Terminate if fallen or max steps reached
        z_pos = self.state[2]
        fallen = z_pos < 0.3  # Fallen if height below 0.3m
        max_steps_reached = self.steps >= self.max_steps

        return fallen or max_steps_reached

class LearningBasedControllerExercise:
    """Hands-on exercise for learning-based control implementation"""

    def __init__(self):
        # Initialize environment
        self.env = SimulatedHumanoidEnv()

        # Initialize DDPG agent
        self.agent = DDPGAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            action_bounds=[self.env.action_space.low, self.env.action_space.high]
        )

        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor({
            'height': (0.1, 1.2),  # Min/max height
            'joint_positions': (-2.0, 2.0),  # Joint limits
            'action_limits': (self.env.action_space.low, self.env.action_space.high)
        })

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.safety_metrics = []

    def train_agent(self, n_episodes=1000):
        """Train the DDPG agent"""
        print("Starting training...")

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            step_count = 0

            done = False
            while not done:
                # Select action with exploration
                action = self.agent.select_action(state, add_noise=True)

                # Check safety before executing action
                is_safe, violations = self.safety_monitor.check_safety(
                    {'height': state[2]}, action
                )

                if not is_safe:
                    print(f"Safety violation in episode {episode}, step {step_count}")
                    for violation in violations:
                        print(f"  {violation}")

                # Take action
                next_state, reward, done, _ = self.env.step(action)

                # Store transition in replay buffer
                self.agent.store_transition(state, action, reward, next_state, done)

                # Update agent
                self.agent.update()

                state = next_state
                total_reward += reward
                step_count += 1

                if done:
                    break

            # Record metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step_count)
            self.safety_metrics.append(self.safety_monitor.get_safety_metrics())

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")

        print("Training completed!")

    def evaluate_agent(self, n_episodes=10):
        """Evaluate the trained agent"""
        print("Evaluating agent...")

        eval_rewards = []
        eval_lengths = []

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            step_count = 0

            done = False
            while not done:
                # Select action WITHOUT exploration for evaluation
                action = self.agent.select_action(state, add_noise=False)

                # Take action
                state, reward, done, _ = self.env.step(action)

                total_reward += reward
                step_count += 1

                if done:
                    break

            eval_rewards.append(total_reward)
            eval_lengths.append(step_count)

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)

        print(f"Evaluation results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Success Rate: {(np.array(eval_lengths) == 1000).mean():.1%}")  # Episodes that didn't fall

        return avg_reward, avg_length

    def visualize_training_progress(self):
        """Visualize training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)

        # Plot 2: Smoothed rewards
        if len(self.episode_rewards) > 100:
            smoothed_rewards = []
            for i in range(100, len(self.episode_rewards)):
                avg = np.mean(self.episode_rewards[i-100:i])
                smoothed_rewards.append(avg)
            axes[0, 1].plot(range(100, len(self.episode_rewards)), smoothed_rewards)
            axes[0, 1].set_title('Smoothed Episode Rewards (100-episode window)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)

        # Plot 3: Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps Until Termination')
        axes[1, 0].axhline(y=1000, color='r', linestyle='--', label='Max Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: Safety metrics
        if self.safety_metrics:
            safety_rates = [metric['safety_rate'] for metric in self.safety_metrics]
            axes[1, 1].plot(safety_rates)
            axes[1, 1].set_title('Safety Rate Over Time')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Safety Rate')
            axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='Good Safety (>0.95)')
            axes[1, 1].axhline(y=0.8, color='orange', linestyle='--', label='OK Safety (>0.8)')
            axes[1, 1].axhline(y=0.6, color='r', linestyle='--', label='Poor Safety (<0.6)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def run_complete_exercise(self):
        """Run the complete learning-based control exercise"""
        print("Starting Learning-Based Control Exercise")
        print("="*50)

        # Train the agent
        self.train_agent(n_episodes=500)

        # Evaluate the trained agent
        eval_reward, eval_length = self.evaluate_agent(n_episodes=10)

        # Visualize results
        self.visualize_training_progress()

        # Print final results
        print("\n" + "="*50)
        print("EXERCISE RESULTS")
        print("="*50)
        print(f"Training Episodes: 500")
        print(f"Evaluation Average Reward: {eval_reward:.2f}")
        print(f"Evaluation Average Length: {eval_length:.1f}")
        print(f"Final Training Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")

        # Calculate improvement over random policy
        # (Random policy would get approximately 0 reward on average)
        improvement = eval_reward - 0  # Baseline is roughly 0 for random
        print(f"Improvement over random: {improvement:.2f}")

        # Safety assessment
        if self.safety_metrics:
            final_safety = self.safety_metrics[-1]['safety_rate']
            print(f"Final Safety Rate: {final_safety:.2f}")

        print("="*50)

        return {
            'training_rewards': self.episode_rewards,
            'evaluation_reward': eval_reward,
            'safety_metrics': self.safety_metrics
        }

# Run the exercise
def run_learning_based_control_exercise():
    """Run the complete learning-based control exercise"""
    exercise = LearningBasedControllerExercise()
    results = exercise.run_complete_exercise()

    return exercise, results

# Uncomment to run the exercise
# exercise, results = run_learning_based_control_exercise()
```

## Advanced Learning Techniques

### Curriculum Learning

Curriculum learning gradually increases task difficulty, helping robots learn complex skills more effectively.

```python
class CurriculumLearning:
    """Curriculum learning for progressive skill acquisition"""

    def __init__(self, tasks, difficulty_threshold=0.8):
        self.tasks = tasks  # List of tasks ordered by difficulty
        self.difficulty_threshold = difficulty_threshold  # Performance threshold to advance
        self.current_task_idx = 0
        self.task_performance = {}  # Performance history for each task

    def update_task_performance(self, task_idx, performance):
        """Update performance for a specific task"""
        if task_idx not in self.task_performance:
            self.task_performance[task_idx] = []

        self.task_performance[task_idx].append(performance)

        # Check if we can advance to next task
        if self.can_advance_to_next_task(task_idx):
            self.advance_to_next_task()

    def can_advance_to_next_task(self, task_idx):
        """Check if we can advance to the next task"""
        if task_idx != self.current_task_idx:
            return False  # Only check advancement for current task

        if task_idx >= len(self.tasks) - 1:
            return False  # Already at hardest task

        # Check if recent performance is above threshold
        if len(self.task_performance[task_idx]) >= 10:
            recent_performance = np.mean(self.task_performance[task_idx][-10:])
            return recent_performance >= self.difficulty_threshold

        return False

    def advance_to_next_task(self):
        """Advance to the next task in the curriculum"""
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            print(f"Advanced to task: {self.tasks[self.current_task_idx]}")

    def get_current_task(self):
        """Get the current task"""
        return self.tasks[self.current_task_idx]

    def get_task_complexity(self, task_idx):
        """Get complexity level of a task"""
        # In practice, this would be determined by task characteristics
        return task_idx / (len(self.tasks) - 1)  # Normalize to [0, 1]
```

### Transfer Learning

Transfer learning allows robots to apply knowledge from one task to another, accelerating learning.

```python
class TransferLearningFramework:
    """Framework for transferring learned skills between tasks"""

    def __init__(self):
        self.pretrained_models = {}  # Trained models for different tasks
        self.transfer_functions = {}  # Functions to adapt models to new tasks
        self.knowledge_graph = {}    # Relationships between tasks/skills

    def register_pretrained_model(self, task_name, model):
        """Register a pretrained model for a specific task"""
        self.pretrained_models[task_name] = model

    def compute_similarity(self, source_task, target_task):
        """Compute similarity between tasks for transfer potential"""
        # This would involve comparing task characteristics
        # In practice, this could use embeddings of task descriptions
        if source_task == target_task:
            return 1.0

        # Check knowledge graph for relationships
        if source_task in self.knowledge_graph:
            if target_task in self.knowledge_graph[source_task]:
                return self.knowledge_graph[source_task][target_task]

        # Default similarity based on task overlap
        # This is a simplified approach
        source_parts = set(source_task.lower().split('_'))
        target_parts = set(target_task.lower().split('_'))
        overlap = len(source_parts.intersection(target_parts))
        union = len(source_parts.union(target_parts))

        if union > 0:
            return overlap / union
        else:
            return 0.0

    def transfer_model(self, source_task, target_task, adaptation_strength=0.5):
        """Transfer a model from source task to target task"""
        if source_task not in self.pretrained_models:
            raise ValueError(f"No pretrained model for task: {source_task}")

        source_model = self.pretrained_models[source_task]
        similarity = self.compute_similarity(source_task, target_task)

        if similarity < 0.1:
            print(f"Low similarity between {source_task} and {target_task}, limited transfer expected")

        # Create adapted model
        # This would involve techniques like:
        # - Fine-tuning: slight modification of pretrained weights
        # - Feature extraction: using pretrained layers as features
        # - Domain adaptation: adapting to new environment conditions

        adapted_model = self.adapt_model_for_transfer(
            source_model, source_task, target_task, adaptation_strength
        )

        return adapted_model, similarity

    def adapt_model_for_transfer(self, source_model, source_task, target_task, strength):
        """Adapt source model for target task"""
        # This is a simplified adaptation - in practice, this would involve
        # sophisticated transfer learning techniques

        # For neural networks, this might involve:
        # - Keeping early layers frozen (feature extractors)
        # - Fine-tuning later layers for new task
        # - Adding new layers for task-specific outputs

        # Create a copy of the source model structure
        if hasattr(source_model, 'state_dict'):
            # PyTorch model
            adapted_model = type(source_model)(**source_model.__dict__)
            adapted_model.load_state_dict(source_model.state_dict())

            # Apply adaptation based on transfer strength
            # This is a simplified example - real adaptation would be more complex
            for param in adapted_model.parameters():
                if np.random.rand() < strength:
                    # Add small random perturbation based on transfer strength
                    noise = torch.randn_like(param) * (0.1 * strength)
                    param.data += noise
        else:
            # Other model type - implement accordingly
            adapted_model = source_model  # Placeholder

        return adapted_model
```

## Safety and Ethical Considerations

### Safe Exploration

Safe exploration is critical when learning-based controllers are deployed on physical robots.

```python
class SafeExplorationManager:
    """Manager for safe exploration in learning-based control"""

    def __init__(self, safety_constraints, exploration_budget=0.1):
        self.safety_constraints = safety_constraints  # Function that checks safety
        self.exploration_budget = exploration_budget  # How much exploration is allowed
        self.safety_buffer = 0.1  # Extra safety margin
        self.risk_assessment = 0.0  # Current risk level

        # Exploration parameters
        self.exploration_decay = 0.99  # Decay exploration over time
        self.min_exploration = 0.01  # Minimum exploration level

    def safe_action_selection(self, policy_action, exploration_action, current_state):
        """Select action that balances policy and exploration while maintaining safety"""
        # Calculate exploration intensity based on current risk
        current_exploration = self.exploration_budget * (self.exploration_decay ** self.risk_assessment)
        current_exploration = max(current_exploration, self.min_exploration)

        # Blend policy and exploration actions
        blended_action = (1 - current_exploration) * policy_action + \
                         current_exploration * exploration_action

        # Check if blended action is safe
        is_safe, violations = self.safety_constraints(current_state, blended_action)

        if is_safe:
            return blended_action
        else:
            # If unsafe, reduce exploration and use more conservative policy
            conservative_factor = 0.7
            conservative_action = (1 - conservative_factor) * policy_action + \
                                  conservative_factor * np.zeros_like(policy_action)

            # Verify conservative action is safe
            is_conservative_safe, _ = self.safety_constraints(current_state, conservative_action)

            if is_conservative_safe:
                return conservative_action
            else:
                # Emergency: return to safe default
                return self.get_safe_default_action(current_state)

    def get_safe_default_action(self, state):
        """Get a safe default action when all else fails"""
        # This would return a safe posture or emergency stop
        return np.zeros_like(state[:6])  # Zero torques as default

    def update_risk_assessment(self, recent_violations, performance_degradation):
        """Update risk assessment based on recent events"""
        # Increase risk if there were safety violations
        violation_risk = len(recent_violations) * 0.3

        # Increase risk if performance is degrading
        performance_risk = max(0, performance_degradation) * 0.1

        # Update overall risk assessment
        self.risk_assessment = 0.8 * self.risk_assessment + 0.2 * (violation_risk + performance_risk)

        # Cap risk assessment
        self.risk_assessment = min(self.risk_assessment, 1.0)

        return self.risk_assessment
```

## Summary

This chapter covered learning-based control methods for humanoid robots, from reinforcement learning algorithms like DDPG to imitation learning techniques like behavioral cloning and DAgger. We explored adaptive control methods that adjust parameters based on real-time performance, and self-organizing maps for biologically-inspired control.

The chapter emphasized the importance of safety in learning-based systems, discussing safety monitoring, safe exploration, and hybrid approaches that combine learning-based and traditional control methods. We showed how to integrate learning systems with traditional control to leverage the strengths of both approaches.

The hands-on exercise provided practical experience implementing a complete learning-based controller with safety monitoring, demonstrating the challenges and benefits of learning-based approaches for humanoid robot control.

We also covered advanced topics like curriculum learning for progressive skill acquisition, transfer learning for applying knowledge across tasks, and ethical considerations for deploying learning systems on physical robots.

Learning-based control represents a powerful approach to handling the complexity and uncertainty inherent in humanoid robotics, enabling robots to adapt and improve their performance through experience.

## Key Takeaways

- Reinforcement learning can learn complex control policies through trial and error
- Imitation learning accelerates learning by leveraging expert demonstrations
- Adaptive control adjusts parameters based on real-time performance feedback
- Safety monitoring is critical for learning-based systems on physical robots
- Hybrid approaches combine learning and traditional control for robustness
- Curriculum learning progressively builds complex skills
- Transfer learning applies knowledge across related tasks
- Safe exploration balances learning with system safety

## Next Steps

In the next module, we'll explore applications and integration topics, including human-robot interaction, multi-sensor fusion, real-world deployment considerations, and a comprehensive capstone project that integrates all the concepts covered in this book.

## References and Further Reading

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Levine, S., Finn, C., Darrell, T., & Abbeel, P. (2016). End-to-end training of deep visuomotor policies. Journal of Machine Learning Research.
3. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics.
4. Pomerleau, D. A. (1989). ALVINN: An autonomous land vehicle in a neural network. Advances in Neural Information Processing Systems.
5. Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research.