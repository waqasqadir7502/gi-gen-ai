# Chapter 2.3: Actuators and Control Systems

## Overview

Actuators are the muscles of humanoid robots, converting control signals into physical motion. Control systems orchestrate these actuators to achieve desired movements while maintaining stability and safety. This chapter covers different types of actuators used in humanoid robotics, control architectures, and implementation of stable control systems.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Identify and compare different types of actuators used in humanoid robots
2. Understand control system architectures and feedback mechanisms
3. Implement PID controllers for actuator control
4. Design stable control systems for humanoid robots
5. Apply control techniques for safe human-robot interaction

## Introduction to Robot Actuators

Actuators are devices that convert energy (typically electrical) into mechanical motion. In humanoid robots, actuators must provide precise control, sufficient power, and safe operation around humans.

### Requirements for Humanoid Robot Actuators

- **Backdrivability**: Ability to be moved by external forces (important for safety)
- **High Torque-to-Weight Ratio**: Sufficient power while keeping robot lightweight
- **Precision**: Accurate positioning and force control
- **Safety**: Safe operation in human environments
- **Energy Efficiency**: Long operation times between charges
- **Durability**: Withstanding repeated loading cycles

### Actuator Performance Metrics

- **Torque**: Rotational force output (Nm)
- **Speed**: Maximum angular velocity (rad/s)
- **Power**: Rate of work (W)
- **Efficiency**: Power output vs. power input
- **Resolution**: Smallest controllable movement
- **Bandwidth**: Frequency response of the actuator

## Types of Actuators

### Electric Motors

#### DC Motors
- **Construction**: Permanent magnets and wound armature
- **Control**: Voltage controls speed, current controls torque
- **Advantages**: Simple control, high efficiency
- **Disadvantages**: Requires gearing for high torque, brushes wear out
- **Applications**: Low-cost joints, simple applications

#### Brushless DC (BLDC) Motors
- **Construction**: Electronic commutation instead of brushes
- **Control**: Requires 3-phase drive electronics
- **Advantages**: Higher efficiency, longer life, better control
- **Disadvantages**: More complex drive electronics
- **Applications**: High-performance joints in modern robots

#### Servo Motors
- **Construction**: Motor + gearhead + position sensor + control electronics
- **Control**: Position-based control with feedback
- **Advantages**: Precise position control, integrated solution
- **Disadvantages**: Limited bandwidth, fixed control parameters
- **Applications**: Hobby robotics, simple applications

### Series Elastic Actuators (SEA)

#### Design Principle
Series Elastic Actuators place a spring in series between the motor and the output, allowing for:
- **Force control**: Spring deflection indicates output force
- **Compliance**: Inherent safety and shock absorption
- **Backdrivability**: Easy to move when powered down

#### Advantages
- Safe human interaction
- Accurate force control
- Shock tolerance
- Energy storage and return

#### Disadvantages
- Reduced position accuracy
- Added complexity
- Potential for oscillation

#### Applications
- Humanoid robots (e.g., Boston Dynamics robots)
- Rehabilitation robots
- Physical therapy devices

### Pneumatic and Hydraulic Actuators

#### Pneumatic Actuators
- **Medium**: Compressed air
- **Advantages**: Light weight, high power-to-weight ratio, inherent compliance
- **Disadvantages**: Compressibility effects, requires air supply
- **Applications**: Lightweight robots, applications requiring compliance

#### Hydraulic Actuators
- **Medium**: Pressurized fluid
- **Advantages**: Very high power, precise control, good for heavy loads
- **Disadvantages**: Complex plumbing, potential for leaks, heavy
- **Applications**: Heavy-duty humanoid robots, industrial applications

## Control System Architectures

### Open-Loop vs. Closed-Loop Control

#### Open-Loop Control
- **Principle**: Control input determined without feedback
- **Advantages**: Simple, no sensor requirements
- **Disadvantages**: No error correction, sensitive to disturbances
- **Applications**: Simple, predictable tasks

#### Closed-Loop Control
- **Principle**: Control input adjusted based on feedback
- **Advantages**: Error correction, disturbance rejection
- **Disadvantages**: Requires sensors, potential for instability
- **Applications**: Precise control tasks

### Hierarchical Control Architecture

Humanoid robots typically use multiple control layers:

#### High-Level Planner
- **Function**: Generate overall motion plans
- **Input**: Task goals, environment information
- **Output**: Desired trajectories

#### Mid-Level Controller
- **Function**: Track planned trajectories
- **Input**: Desired trajectories
- **Output**: Joint-level commands

#### Low-Level Actuator Control
- **Function**: Control individual actuators
- **Input**: Joint commands
- **Output**: Motor commands

## PID Control

Proportional-Integral-Derivative (PID) control is fundamental to robot control systems.

### PID Controller Equation

```
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
```

Where:
- u(t): Control output
- e(t): Error (desired - actual)
- Kp: Proportional gain
- Ki: Integral gain
- Kd: Derivative gain

### PID Components

#### Proportional (P) Control
- **Effect**: Response proportional to current error
- **Kp too high**: Oscillations, instability
- **Kp too low**: Slow response, steady-state error

#### Integral (I) Control
- **Effect**: Eliminates steady-state error by integrating past errors
- **Ki too high**: Oscillations, overshoot
- **Ki too low**: Slow elimination of steady-state error

#### Derivative (D) Control
- **Effect**: Dampens response by considering rate of error change
- **Kd too high**: Noise amplification, sluggish response
- **Kd too low**: Oscillations, poor damping

### PID Tuning Methods

#### Ziegler-Nichols Method
1. Set Ki = 0, Kd = 0
2. Increase Kp until system oscillates
3. Record critical gain (Kc) and oscillation period (Pc)
4. Use formulas:
   - Kp = 0.6 * Kc
   - Ki = 2 * Kp / Pc
   - Kd = Kp * Pc / 8

#### Trial and Error
1. Start with P control only
2. Increase Kp until response is fast but not oscillating
3. Add small Ki to eliminate steady-state error
4. Add Kd to dampen oscillations

### Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    """Simple PID controller implementation"""

    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.error_prev = 0
        self.integral = 0

    def update(self, error):
        """Update PID controller with new error"""
        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.error_prev) / self.dt
        d_term = self.kd * derivative

        # Store current error for next iteration
        self.error_prev = error

        # Calculate output
        output = p_term + i_term + d_term

        return output

def simulate_motor_system():
    """Simulate a motor control system with PID"""
    # System parameters
    inertia = 0.1  # kg*m^2
    damping = 0.05  # N*m*s/rad
    dt = 0.01  # 100 Hz control loop

    # PID gains (tuned for this system)
    pid = PIDController(kp=10.0, ki=1.0, kd=0.5, dt=dt)

    # Simulation parameters
    time = np.arange(0, 5, dt)  # 5 seconds
    desired_position = np.ones_like(time) * np.pi/2  # 90 degrees
    actual_position = np.zeros_like(time)
    actual_velocity = np.zeros_like(time)
    control_effort = np.zeros_like(time)

    # Initial conditions
    actual_position[0] = 0.0
    actual_velocity[0] = 0.0

    # Simulation loop
    for i in range(1, len(time)):
        # Calculate error
        error = desired_position[i-1] - actual_position[i-1]

        # PID control
        torque = pid.update(error)
        control_effort[i] = torque

        # Apply torque to system (simple physics model)
        # tau = I*alpha + b*omega
        # alpha = (tau - b*omega) / I
        angular_acceleration = (torque - damping * actual_velocity[i-1]) / inertia

        # Update state
        actual_velocity[i] = actual_velocity[i-1] + angular_acceleration * dt
        actual_position[i] = actual_position[i-1] + actual_velocity[i] * dt

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time, desired_position, 'r--', label='Desired')
    plt.plot(time, actual_position, 'b-', label='Actual')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('Position Response')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, actual_velocity)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.title('Velocity Response')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, control_effort)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Effort (Nm)')
    plt.title('Control Effort')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    error = desired_position - actual_position
    plt.plot(time, error)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Position Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print performance metrics
    final_error = abs(error[-1])
    rise_time_idx = np.where(abs(actual_position - desired_position[0]) >= 0.1 * abs(desired_position[0] - desired_position[0]))[0]
    rise_time = time[rise_time_idx[0]] if len(rise_time_idx) > 0 else float('inf')

    print(f"Final position error: {final_error:.4f} rad")
    print(f"Rise time (to 90%): {rise_time:.3f} s")
    print(f"Steady-state error: {abs(np.mean(error[-100:])):.4f} rad")

# Run simulation
simulate_motor_system()
```

## Advanced Control Techniques

### Impedance Control

Impedance control regulates the dynamic relationship between position and force, making the robot behave like a mechanical system with specified mass, damping, and stiffness.

#### Impedance Controller Equation

```
M_d * (d²x_e/dt²) + B_d * (dx_e/dt) + K_d * x_e = F_cmd
```

Where:
- x_e: Position error (x_d - x_actual)
- M_d, B_d, K_d: Desired mass, damping, stiffness
- F_cmd: Commanded force

### Admittance Control

Admittance control is the dual of impedance control, where force input produces position output.

```
dx_d/dt = α * (F_ext - F_d)
```

Where:
- α: Admittance (inverse of impedance)
- F_ext: External force
- F_d: Desired force

### Model-Based Control

#### Computed Torque Control
Uses robot dynamics model to linearize system:

```
τ = M(q) * (q_dd_des + Kd*q_d_err + Kp*q_err) + C(q,q_d)*q_d + g(q)
```

Where:
- M(q): Inertia matrix
- C(q,q_d): Coriolis matrix
- g(q): Gravity vector

#### Operational Space Control
Controls task space variables directly:

```
τ = J^T * (λ * (ẍ_d + Kd*ẋ_err + Kp*x_err)) + h
```

Where:
- J: Jacobian matrix
- λ: Task space inertia
- h: Coriolis and gravity terms

## Safety and Compliance

### Force Limiting
- **Current limiting**: Limit motor current to limit torque
- **Torque limiting**: Directly limit commanded torque
- **Impact detection**: Detect external impacts and react

### Compliance Control
- **Variable stiffness**: Adjust virtual spring constants
- **Active compliance**: Use feedback to adjust impedance
- **Passive compliance**: Mechanical compliance in transmission

### Collision Detection and Response
- **Current monitoring**: Sudden current increases indicate collisions
- **Model-based detection**: Compare commanded vs. actual motion
- **Response strategies**: Stop, back away, or compliant behavior

## Control System Implementation

### Real-Time Considerations
- **Control frequency**: Higher frequencies for better performance
- **Latency**: Minimize delay between sensing and actuation
- **Jitter**: Consistent timing for stable control

### Hardware-in-the-Loop (HIL) Testing
- **Simulation**: Test controllers in simulation first
- **Gradual deployment**: Move from simulation to hardware incrementally
- **Safety limits**: Always have hardware safety limits

### Control System Architecture Example

```python
import time
import threading
import numpy as np

class JointController:
    """Example joint controller with safety features"""

    def __init__(self, joint_id, kp=10.0, ki=1.0, kd=0.1):
        self.joint_id = joint_id
        self.pid = PIDController(kp, ki, kd, dt=0.001)

        # Safety limits
        self.position_limits = (-np.pi, np.pi)
        self.velocity_limit = 5.0  # rad/s
        self.torque_limit = 50.0   # Nm

        # State
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_torque = 0.0
        self.desired_position = 0.0

        # Threading
        self.running = False
        self.control_thread = None

    def set_desired_position(self, position):
        """Set desired position with limits"""
        self.desired_position = np.clip(position,
                                       self.position_limits[0],
                                       self.position_limits[1])

    def update_sensors(self, position, velocity):
        """Update sensor readings"""
        self.current_position = position
        self.current_velocity = velocity

    def compute_control(self):
        """Compute control output"""
        # Calculate error
        position_error = self.desired_position - self.current_position

        # Apply PID control
        torque_cmd = self.pid.update(position_error)

        # Apply safety limits
        torque_cmd = np.clip(torque_cmd, -self.torque_limit, self.torque_limit)

        # Check velocity limit
        if abs(self.current_velocity) > self.velocity_limit:
            # Add damping to reduce velocity
            torque_cmd -= np.sign(self.current_velocity) * 5.0

        self.current_torque = torque_cmd
        return torque_cmd

    def start_control_loop(self, frequency=1000):  # 1kHz
        """Start the control loop in a separate thread"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop,
                                              args=(1.0/frequency,))
        self.control_thread.start()

    def _control_loop(self, dt):
        """Internal control loop"""
        while self.running:
            start_time = time.time()

            # Compute control
            torque_cmd = self.compute_control()

            # Here you would send the torque command to the actual motor
            # motor.set_torque(self.joint_id, torque_cmd)

            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stop the control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

# Example usage
def example_joint_control():
    """Example of using the joint controller"""
    # Create controller
    controller = JointController(joint_id=0, kp=15.0, ki=2.0, kd=0.2)

    # Set desired position
    controller.set_desired_position(np.pi/4)  # 45 degrees

    # Simulate sensor updates and control
    for t in np.arange(0, 2, 0.01):  # 2 seconds of simulation
        # Simulate sensor readings (in real robot, these would come from encoders)
        # For simulation, we'll just use the desired position with some delay
        sim_position = controller.current_position + 0.1 * (controller.desired_position - controller.current_position)
        sim_velocity = (sim_position - controller.current_position) / 0.01

        # Update sensors
        controller.update_sensors(sim_position, sim_velocity)

        # Compute control
        torque = controller.compute_control()

        print(f"Time: {t:.2f}s, Position: {sim_position:.3f}, Torque: {torque:.3f}")

        time.sleep(0.01)  # Simulate 100Hz control loop

    controller.stop()

# Run example
example_joint_control()
```

## Hands-on Exercise: Implementing a Joint Controller

In this exercise, you'll implement a complete joint controller with PID control, safety features, and simulation.

### Requirements
- Python 3.8+
- NumPy library
- Matplotlib for visualization
- Basic understanding of control systems

### Exercise Steps
1. Implement a PID controller class
2. Create a simulated motor system
3. Add safety features to the controller
4. Test the controller with different trajectories
5. Analyze performance and stability

### Expected Outcome
You should have a working joint controller that can track position commands while maintaining safety limits and stability.

### Sample Implementation
```python
import numpy as np
import matplotlib.pyplot as plt

class SafePIDController:
    """PID controller with safety features"""

    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.error_prev = 0
        self.integral = 0
        self.output_prev = 0

        # Safety limits
        self.position_limits = (-2*np.pi, 2*np.pi)
        self.velocity_limit = 10.0
        self.output_limit = 100.0  # Torque or voltage limit

    def update(self, desired, actual):
        """Update controller with desired and actual values"""
        # Calculate error
        error = desired - actual

        # Apply anti-windup to integral term
        if abs(self.output_prev) < self.output_limit * 0.9:  # Only integrate if not saturated
            self.integral += error * self.dt

        # Calculate derivative with noise filtering
        derivative_filtered = (error - self.error_prev) / self.dt
        # Apply simple low-pass filter to derivative
        derivative = 0.2 * derivative_filtered + 0.8 * (self.error_prev - self.error_prev_prev) / self.dt if hasattr(self, 'error_prev_prev') else derivative_filtered

        # Calculate PID output
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        # Apply output limits
        output = np.clip(output, -self.output_limit, self.output_limit)

        # Store values for next iteration
        self.error_prev_prev = self.error_prev
        self.error_prev = error
        self.output_prev = output

        return output

def simulate_joint_control():
    """Simulate a joint control system"""
    # System parameters
    inertia = 0.05  # kg*m^2
    damping = 0.1   # N*m*s/rad
    dt = 0.005      # 200 Hz control loop

    # Create controller
    controller = SafePIDController(kp=20.0, ki=5.0, kd=1.0, dt=dt)

    # Simulation parameters
    duration = 4.0  # seconds
    time_steps = int(duration / dt)
    time_vec = np.linspace(0, duration, time_steps)

    # Desired trajectory (smooth step with minimum jerk)
    desired_pos = np.zeros(time_steps)
    for i in range(time_steps):
        if time_vec[i] < 1.0:
            # Smooth start (5th order polynomial)
            t_norm = time_vec[i] / 1.0
            desired_pos[i] = 0.5 * (1 - np.cos(np.pi * t_norm))  # Smooth start
        elif time_vec[i] < 3.0:
            desired_pos[i] = 1.0  # Hold position
        else:
            # Smooth stop
            t_norm = (time_vec[i] - 3.0) / 1.0
            desired_pos[i] = 1.0 - 0.5 * (1 - np.cos(np.pi * t_norm))  # Smooth stop

    # Scale to 90 degrees
    desired_pos = desired_pos * np.pi/2

    # Initialize state arrays
    actual_pos = np.zeros(time_steps)
    actual_vel = np.zeros(time_steps)
    actual_acc = np.zeros(time_steps)
    control_effort = np.zeros(time_steps)
    errors = np.zeros(time_steps)

    # Initial conditions
    actual_pos[0] = 0.0
    actual_vel[0] = 0.0

    # Simulation loop
    for i in range(1, time_steps):
        # Calculate error
        error = desired_pos[i-1] - actual_pos[i-1]
        errors[i] = error

        # PID control
        control_output = controller.update(desired_pos[i-1], actual_pos[i-1])
        control_effort[i] = control_output

        # Apply control to system (forward dynamics)
        # tau = I*alpha + b*omega
        # alpha = (tau - b*omega) / I
        acceleration = (control_output - damping * actual_vel[i-1]) / inertia
        actual_acc[i] = acceleration

        # Update velocity and position (numerical integration)
        actual_vel[i] = actual_vel[i-1] + acceleration * dt
        actual_pos[i] = actual_pos[i-1] + actual_vel[i] * dt

    # Calculate performance metrics
    final_error = abs(errors[-1])
    rise_time_idx = np.where(abs(actual_pos - 0.1) >= 0.9 * abs(1.0 - 0.1))[0]
    rise_time = time_vec[rise_time_idx[0]] if len(rise_time_idx) > 0 else float('inf')
    overshoot = (np.max(actual_pos) - 1.0) / 1.0 * 100 if 1.0 != 0 else 0

    print(f"Performance Metrics:")
    print(f"  Final error: {final_error:.4f} rad")
    print(f"  Rise time: {rise_time:.3f} s")
    print(f"  Overshoot: {overshoot:.2f}%")
    print(f"  Max control effort: {np.max(np.abs(control_effort)):.2f}")

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(time_vec, desired_pos, 'r--', label='Desired', linewidth=2)
    plt.plot(time_vec, actual_pos, 'b-', label='Actual', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('Position Response')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(time_vec, actual_vel)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.title('Velocity Response')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(time_vec, actual_acc)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (rad/s²)')
    plt.title('Acceleration Response')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(time_vec, control_effort)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Effort')
    plt.title('Control Effort')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(time_vec, errors)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Position Error')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(actual_pos, actual_vel)
    plt.xlabel('Position (rad)')
    plt.ylabel('Velocity (rad/s)')
    plt.title('Phase Portrait')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_joint_control()
```

## Control Challenges in Humanoid Robotics

### Underactuation
Humanoid robots are often underactuated (fewer actuators than degrees of freedom), requiring:
- Advanced control techniques like hybrid zero dynamics
- Careful gait design to maintain stability
- Passive dynamics utilization

### Contact Transitions
Switching between different contact states (e.g., foot contact during walking):
- Require careful control design
- May cause discontinuities in dynamics
- Need smooth transition strategies

### Disturbance Rejection
External disturbances from environment or human interaction:
- Require robust control design
- May need adaptive control strategies
- Balance performance vs. robustness

## Summary

This chapter covered actuators and control systems, which are essential for humanoid robot movement and interaction. We explored different types of actuators including electric motors, series elastic actuators, and fluid-based actuators, each with their own advantages and applications.

We examined control system architectures from high-level planning to low-level actuator control, with PID control as a fundamental technique. We implemented practical examples of PID controllers and discussed advanced techniques like impedance control and model-based control.

The chapter also covered safety considerations in control systems, including force limiting, compliance control, and collision detection. We provided hands-on exercises to implement and test joint controllers with safety features.

Understanding actuators and control systems is crucial for developing humanoid robots that can move precisely, safely interact with humans and environments, and maintain stability during complex tasks.

## Key Takeaways

- Different actuators (electric, hydraulic, pneumatic) have trade-offs in power, precision, and safety
- PID control is fundamental but must be carefully tuned for each application
- Safety features are essential for human-robot interaction
- Control system architecture typically involves multiple hierarchical levels
- Real-time performance is critical for stable control
- Advanced techniques like impedance control provide better human interaction
- Underactuation and contact transitions present unique control challenges

## Next Steps

In the next chapter, we'll explore basic locomotion patterns in humanoid robots, building on the kinematic, perception, and control foundations we've established. We'll cover walking patterns, balance control, and how to implement stable locomotion in simulation.

## References and Further Reading

1. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). Robot Modeling and Control. John Wiley & Sons.
2. Craig, J. J. (2005). Introduction to Robotics: Mechanics and Control (3rd ed.). Pearson Prentice Hall.
3. Pratt, J., & Krupp, B. (2008). Series elastic actuators for high fidelity force control. Industrial Robot: An International Journal.
4. Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K., & Hirukawa, H. (2003). Biped walking pattern generation by using preview control of zero-moment point. IEEE International Conference on Robotics and Automation.
5. Hofmann, A., Iida, F., & Pfeifer, R. (2008). Designing a force-controllable ankle for a biped robot: Physical and numerical experiments. IEEE International Conference on Robotics and Automation.