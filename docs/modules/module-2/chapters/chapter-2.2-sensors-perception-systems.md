# Chapter 2.2: Sensors and Perception Systems

## Overview

Sensors and perception systems are the eyes, ears, and sensory organs of humanoid robots. They enable robots to understand their environment, interact with objects, maintain balance, and navigate safely. This chapter covers the various types of sensors used in humanoid robotics, how they work, and how to process their data for effective perception.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Identify and classify different types of sensors used in humanoid robots
2. Understand how vision, touch, and proprioceptive sensors work
3. Process sensor data for environmental perception
4. Implement basic sensor fusion techniques
5. Apply perception systems to robot control and navigation

## Introduction to Robot Perception

Perception in humanoid robots involves acquiring, processing, and interpreting sensory information from the environment. This information is crucial for:

- **Navigation**: Understanding spatial relationships and avoiding obstacles
- **Manipulation**: Locating and grasping objects
- **Balance**: Maintaining stability and posture
- **Interaction**: Understanding human behavior and environmental changes

### The Perception-Action Cycle

Robot perception follows a continuous cycle:
1. **Sensing**: Acquiring data from various sensors
2. **Processing**: Filtering, calibrating, and interpreting sensor data
3. **Fusion**: Combining data from multiple sensors
4. **Understanding**: Creating meaningful representations of the environment
5. **Action**: Using perception data to guide robot behavior
6. **Feedback**: Using action outcomes to refine perception

## Types of Sensors in Humanoid Robots

### Proprioceptive Sensors

Proprioceptive sensors provide information about the robot's own state:

#### Joint Encoders
- **Purpose**: Measure joint angles and positions
- **Types**: Absolute encoders (provide exact position) and incremental encoders (measure changes from a reference)
- **Applications**: Joint position control, kinematic calculations, motion planning

#### Inertial Measurement Units (IMUs)
- **Components**: Accelerometers, gyroscopes, and sometimes magnetometers
- **Purpose**: Measure orientation, angular velocity, and linear acceleration
- **Applications**: Balance control, posture estimation, motion tracking

#### Force/Torque Sensors
- **Purpose**: Measure forces and torques at joints or end-effectors
- **Applications**: Grasping control, contact detection, compliance control

#### Tactile Sensors
- **Purpose**: Detect contact, pressure, and texture
- **Applications**: Grasping, manipulation, object recognition

### Exteroceptive Sensors

Exteroceptive sensors provide information about the external environment:

#### Vision Systems
- **Cameras**: RGB, stereo, depth, and thermal cameras
- **Purpose**: Object recognition, scene understanding, navigation
- **Applications**: Face recognition, object detection, SLAM (Simultaneous Localization and Mapping)

#### Range Sensors
- **LIDAR**: Light Detection and Ranging
- **Ultrasonic**: Sound-based distance measurement
- **Purpose**: Distance measurement, obstacle detection, mapping
- **Applications**: Navigation, mapping, collision avoidance

#### Audio Sensors
- **Microphones**: Sound detection and processing
- **Purpose**: Voice recognition, sound localization
- **Applications**: Human-robot interaction, environmental monitoring

## Vision Systems in Humanoid Robots

Vision is one of the most important sensory modalities for humanoid robots, providing rich information about the environment.

### Camera Types and Configurations

#### Monocular Cameras
- **Advantages**: Simple, lightweight, low computational requirements
- **Limitations**: No depth information from a single image
- **Applications**: Object recognition, face detection, color-based segmentation

#### Stereo Cameras
- **Principle**: Two cameras separated by a known distance, similar to human eyes
- **Advantages**: Depth information from stereo disparity
- **Applications**: 3D reconstruction, obstacle detection, grasping

#### RGB-D Cameras
- **Components**: RGB camera + depth sensor (e.g., Microsoft Kinect, Intel RealSense)
- **Advantages**: Color and depth information in a single device
- **Applications**: Scene understanding, object recognition with spatial context

### Computer Vision for Robotics

#### Feature Detection
Key algorithms for identifying important points in images:
- **SIFT (Scale-Invariant Feature Transform)**: Robust to scale and rotation changes
- **SURF (Speeded Up Robust Features)**: Faster alternative to SIFT
- **ORB (Oriented FAST and Rotated BRIEF)**: Fast and efficient for real-time applications

#### Object Detection and Recognition
- **Template Matching**: Finding predefined objects in images
- **Machine Learning Approaches**: Using trained models (e.g., CNNs) for object detection
- **Deep Learning**: Modern approaches like YOLO, R-CNN for real-time object detection

#### Visual SLAM
Simultaneous Localization and Mapping using visual information:
- **Feature-based SLAM**: Tracking visual features across frames
- **Direct SLAM**: Using pixel intensities directly
- **Applications**: Navigation, mapping, augmented reality

### Vision Processing Pipeline

```
Raw Image → Preprocessing → Feature Extraction → Object Detection → Scene Understanding
```

1. **Preprocessing**: Noise reduction, color correction, distortion correction
2. **Feature Extraction**: Detecting corners, edges, textures
3. **Object Detection**: Identifying and localizing objects
4. **Scene Understanding**: Interpreting the scene context

## Tactile and Force Sensing

Tactile and force sensing are crucial for manipulation and physical interaction.

### Force/Torque Sensors

#### 6-Axis Force/Torque Sensors
- **Measurement**: 3 forces (X, Y, Z) and 3 torques (roll, pitch, yaw)
- **Applications**: Grasping, assembly, haptic feedback
- **Implementation**: Strain gauges arranged in a specific configuration

#### Applications in Manipulation
- **Grasping**: Detecting contact and adjusting grip force
- **Assembly**: Applying precise forces for delicate operations
- **Human Safety**: Limiting forces during human-robot interaction

### Tactile Sensors

#### Types of Tactile Sensors
- **Pressure Sensors**: Detect contact and pressure distribution
- **Temperature Sensors**: Detect thermal properties of objects
- **Slip Sensors**: Detect when objects are slipping from grasp
- **Texture Sensors**: Detect surface properties

#### Tactile Sensor Arrays
- **GelSight**: High-resolution tactile sensing using optical imaging of a deformable gel
- **BioTac**: Biomimetic tactile sensors with fluid-filled skin
- **Barrel-Snap**: Tactile sensors with microstructures for enhanced sensitivity

## Range Sensing and Environment Mapping

### LIDAR Systems

#### How LIDAR Works
- **Principle**: Time-of-flight measurement of laser pulses
- **Output**: 2D or 3D point cloud representing distances to objects
- **Advantages**: High accuracy, works in various lighting conditions
- **Limitations**: Expensive, can be affected by reflective surfaces

#### Applications
- **Mapping**: Creating 2D/3D maps of environments
- **Navigation**: Path planning and obstacle avoidance
- **Localization**: Determining robot position in known maps

### Ultrasonic Sensors

#### Working Principle
- **Method**: Measure time-of-flight of ultrasonic pulses
- **Range**: Typically 3-4 meters with moderate accuracy
- **Applications**: Simple obstacle detection, rough distance estimation

### Depth Cameras

#### Stereo Vision
- **Principle**: Triangulation from disparity between two cameras
- **Accuracy**: Depends on baseline distance and image resolution
- **Limitations**: Performance degrades with textureless surfaces

#### Structured Light
- **Method**: Project known light patterns and measure distortions
- **Examples**: Microsoft Kinect (first generation)
- **Advantages**: Good accuracy at close range
- **Limitations**: Sensitive to ambient light

## Sensor Fusion

Sensor fusion combines data from multiple sensors to create a more accurate and robust perception system than any single sensor could provide.

### Why Sensor Fusion?

- **Redundancy**: Multiple sensors can verify each other
- **Complementarity**: Different sensors provide different types of information
- **Robustness**: If one sensor fails, others can still function
- **Accuracy**: Combined data is often more accurate than individual sensors

### Fusion Techniques

#### Kalman Filtering
- **Purpose**: Estimate state from noisy sensor measurements
- **Assumption**: Linear system with Gaussian noise
- **Applications**: IMU fusion, object tracking, state estimation

#### Extended Kalman Filter (EKF)
- **Purpose**: Handle non-linear systems
- **Applications**: SLAM, robot localization

#### Particle Filtering
- **Purpose**: Handle non-linear, non-Gaussian systems
- **Method**: Represent probability distribution with particles
- **Applications**: Robot localization in complex environments

#### Sensor Fusion Example: Robot Localization

```python
import numpy as np
from scipy.stats import multivariate_normal

class SimpleSensorFusion:
    """Simple example of fusing IMU and camera data for position estimation"""

    def __init__(self, initial_state, process_noise, measurement_noise):
        # State: [x, y, vx, vy] (position and velocity)
        self.state = initial_state
        self.covariance = np.eye(4)  # Initial uncertainty

        # Noise parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, dt):
        """Predict state forward in time using IMU data"""
        # Simple motion model: x_new = x_old + v*dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurement):
        """Update state with camera measurement (position only)"""
        # Measurement matrix (camera only measures position)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Innovation
        y = measurement - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.measurement_noise

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

# Example usage
def example_sensor_fusion():
    """Demonstrate simple sensor fusion"""
    # Initial state: [x, y, vx, vy] = [0, 0, 0.1, 0.05]
    initial_state = np.array([0.0, 0.0, 0.1, 0.05])

    # Process and measurement noise
    process_noise = 0.01 * np.eye(4)
    measurement_noise = 0.1 * np.eye(2)

    fusion = SimpleSensorFusion(initial_state, process_noise, measurement_noise)

    # Simulate measurements over time
    dt = 0.1  # 100ms time steps
    true_positions = []
    estimated_positions = []

    for t in range(100):
        # Predict step using IMU (simulated)
        fusion.predict(dt)

        # Simulate camera measurement (noisy)
        true_x = 0.1 * t + 0.01 * t**2  # True position with acceleration
        true_y = 0.05 * t
        measurement = np.array([true_x, true_y]) + np.random.normal(0, 0.2, 2)

        # Update step with camera measurement
        fusion.update(measurement)

        true_positions.append([true_x, true_y])
        estimated_positions.append([fusion.state[0], fusion.state[1]])

    print("Sensor fusion example completed")
    print(f"Final true position: ({true_x:.2f}, {true_y:.2f})")
    print(f"Final estimated position: ({fusion.state[0]:.2f}, {fusion.state[1]:.2f})")
    print(f"Estimation error: {np.linalg.norm([true_x - fusion.state[0], true_y - fusion.state[1]]):.3f}")

# Run example
example_sensor_fusion()
```

## Hands-on Exercise: Processing Camera Data for Object Detection

In this exercise, you'll implement a simple object detection system using camera data and apply basic sensor fusion with IMU data.

### Requirements
- Python 3.8+
- OpenCV library
- NumPy
- PyBullet for simulation (optional)
- Camera access (or sample images)

### Exercise Steps
1. Capture or load an image from a camera
2. Implement basic color-based object detection
3. Apply sensor fusion with simulated IMU data
4. Visualize the results
5. Evaluate the performance

### Expected Outcome
You should have a working system that can detect objects in camera images and combine this information with simulated IMU data to improve position estimation.

### Sample Implementation
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_red_objects(image):
    """Detect red objects in an image using color filtering"""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color
    # Note: red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour (assuming it's our target object)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w//2
            center_y = y + h//2
            return (center_x, center_y), (x, y, w, h)

    return None, None

def simulate_imu_data(prev_position, dt=0.01):
    """Simulate IMU-based position estimation with noise"""
    # Simulate simple motion with some acceleration
    acceleration = np.array([0.1, 0.05])  # m/s^2
    velocity = np.array([0.5, 0.3])       # m/s (would be integrated from acceleration)

    # Add some noise to simulate IMU inaccuracy
    noise = np.random.normal(0, 0.01, 2)  # 1cm standard deviation
    displacement = velocity * dt + 0.5 * acceleration * dt**2
    new_position = prev_position + displacement + noise

    return new_position

def sensor_fusion(camera_pos, imu_pos, camera_uncertainty=50, imu_uncertainty=20):
    """Simple weighted fusion of camera and IMU position estimates"""
    # Convert uncertainties to weights (inverse of variance)
    camera_weight = 1.0 / (camera_uncertainty ** 2)
    imu_weight = 1.0 / (imu_uncertainty ** 2)

    # Weighted average
    fused_x = (camera_weight * camera_pos[0] + imu_weight * imu_pos[0]) / (camera_weight + imu_weight)
    fused_y = (camera_weight * camera_pos[1] + imu_weight * imu_pos[1]) / (camera_weight + imu_weight)

    return np.array([fused_x, fused_y])

def run_perception_exercise():
    """Run the complete perception exercise"""
    # Create a sample image with a red object
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red rectangle (simulating a red object)
    cv2.rectangle(img, (250, 200), (350, 300), (0, 0, 255), -1)

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Detect red object
    object_center, bbox = detect_red_objects(img)

    if object_center:
        print(f"Detected red object at: {object_center}")
        print(f"Bounding box: {bbox}")

        # Simulate IMU data (assuming previous position was known)
        prev_position = np.array([300, 250])  # Previous estimate in pixels
        imu_position = simulate_imu_data(prev_position)

        print(f"IMU-based position estimate: {imu_position}")

        # Perform sensor fusion
        fused_position = sensor_fusion(np.array(object_center), imu_position)
        print(f"Fused position estimate: {fused_position}")

        # Visualize results
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.circle(img, object_center, 5, (0, 255, 0), -1)
            cv2.circle(img, (int(fused_position[0]), int(fused_position[1])), 5, (255, 0, 255), -1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Object Detection Results')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(prev_position[0], prev_position[1], 'bo', label='Previous Position', markersize=10)
        plt.plot(object_center[0], object_center[1], 'ro', label='Camera Detection', markersize=8)
        plt.plot(imu_position[0], imu_position[1], 'go', label='IMU Estimate', markersize=8)
        plt.plot(fused_position[0], fused_position[1], 'mo', label='Fused Estimate', markersize=8)
        plt.legend()
        plt.title('Sensor Fusion Results')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("No red objects detected")

# Run the exercise
run_perception_exercise()
```

## Perception for Locomotion

Sensory perception is crucial for humanoid robot locomotion, enabling safe and stable movement.

### Balance and Posture Control
- **Center of Mass (CoM)**: Calculated using IMU and joint encoder data
- **Zero Moment Point (ZMP)**: Critical for dynamic balance
- **Ankle Strategies**: Using foot sensors for balance recovery

### Terrain Perception
- **Step Detection**: Identifying stairs, curbs, and obstacles
- **Surface Classification**: Distinguishing between different ground types
- **Slip Detection**: Using tactile and force sensors

### Visual Navigation
- **Gait Adaptation**: Adjusting walking pattern based on visual input
- **Footstep Planning**: Selecting safe footholds
- **Obstacle Avoidance**: Detecting and avoiding obstacles in the path

## Challenges in Robot Perception

### Sensor Limitations
- **Noise**: All sensors have inherent noise that must be filtered
- **Latency**: Processing time can cause delays in perception
- **Range Limitations**: Sensors have limited effective ranges
- **Environmental Sensitivity**: Performance varies with lighting, weather, etc.

### Computational Complexity
- **Real-time Processing**: Perception must keep up with robot motion
- **Resource Constraints**: Limited computational resources on robots
- **Power Consumption**: Sensors and processing consume power

### Data Association
- **Feature Matching**: Correctly matching features across frames
- **Object Tracking**: Maintaining identity of objects over time
- **Loop Closure**: Recognizing previously visited locations

## Summary

This chapter covered the essential sensors and perception systems used in humanoid robots. We explored proprioceptive sensors (encoders, IMUs, force sensors) that provide information about the robot's own state, and exteroceptive sensors (cameras, LIDAR, ultrasonic) that provide information about the environment.

We examined vision systems in detail, including different camera types, computer vision techniques, and visual SLAM. We also covered tactile and force sensing for manipulation, range sensing for mapping and navigation, and sensor fusion techniques that combine data from multiple sensors for more robust perception.

The hands-on exercise provided practical experience with object detection and sensor fusion, demonstrating how different sensory modalities can be combined to improve robot perception capabilities.

Understanding perception systems is crucial for developing humanoid robots that can safely and effectively interact with their environment, whether for navigation, manipulation, or human interaction.

## Key Takeaways

- Proprioceptive sensors provide information about the robot's own state (position, orientation, forces)
- Exteroceptive sensors provide information about the environment (vision, range, audio)
- Vision systems are crucial for object recognition, navigation, and scene understanding
- Tactile and force sensing enable safe and effective manipulation
- Sensor fusion combines multiple sensors for more robust perception
- Perception systems must operate in real-time with computational constraints
- Locomotion requires specialized perception for balance and navigation

## Next Steps

In the next chapter, we'll explore actuators and control systems that work with perception systems to enable humanoid robots to move and interact with their environment. We'll cover different types of actuators, control architectures, and how to implement stable control systems.

## References and Further Reading

1. Siciliano, B., & Khatib, O. (Eds.). (2016). Springer Handbook of Robotics. Springer.
2. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
3. Sibley, G., Mei, C., Baldwin, G., & Mahon, I. (2010). On the motion problem in visual SLAM. IEEE International Conference on Robotics and Automation.
4. Okada, K., & Inaba, M. (2014). Humanoid robot platform HRP-4 - Physical capability and system structure. IEEE-RAS International Conference on Humanoid Robots.
5. Pratt, J., & Krupp, B. (2008). Series elastic actuators for high fidelity force control. Industrial Robot: An International Journal.