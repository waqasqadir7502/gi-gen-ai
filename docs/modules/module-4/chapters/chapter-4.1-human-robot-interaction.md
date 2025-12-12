# Chapter 4.1: Human-Robot Interaction

## Overview

Human-Robot Interaction (HRI) is a critical component of humanoid robotics, focusing on the design, development, and evaluation of robots that can effectively communicate and collaborate with humans. This chapter covers the principles of HRI, social robotics, communication modalities, and the design of robots that can safely and naturally interact with humans in various environments.

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the principles of effective human-robot interaction
2. Design robots with appropriate social behaviors and communication modalities
3. Implement perception systems for detecting and interpreting human intentions
4. Create safe interaction protocols that respect human comfort zones
5. Evaluate human-robot interaction quality and effectiveness

## Introduction to Human-Robot Interaction

Human-Robot Interaction (HRI) is an interdisciplinary field that combines robotics, psychology, cognitive science, human-computer interaction, and design to create robots that can interact naturally and effectively with humans. Unlike traditional industrial robots that operate in isolation, humanoid robots designed for HRI must navigate complex social norms, communication patterns, and safety considerations.

### Key Challenges in HRI

#### Anthropomorphic Expectations
Humans naturally attribute human-like qualities to robots, especially humanoid robots. This can lead to unrealistic expectations about the robot's capabilities and understanding, creating potential frustration when the robot fails to meet these expectations.

#### Social Norms and Etiquette
Human social interactions follow unwritten rules about personal space, turn-taking, eye contact, and appropriate behavior in different contexts. Robots must be programmed to respect these norms to avoid appearing awkward or threatening.

#### Safety and Trust
Trust is fundamental to effective human-robot interaction. Humans need to feel safe around robots, especially in close-proximity interactions. This requires not only physical safety mechanisms but also predictable and understandable behavior patterns.

#### Communication Modalities
Humans communicate through multiple channels: speech, gestures, facial expressions, posture, and proxemics (use of space). Effective HRI systems must be able to interpret and utilize these various communication modalities.

### The HRI Design Process

#### User-Centered Design
HRI design must begin with understanding the needs, capabilities, and preferences of the intended users. This includes considering factors such as age, cultural background, technical literacy, and specific use cases.

#### Iterative Prototyping
Due to the complexity of social interaction, HRI systems benefit greatly from iterative prototyping and user testing. Early prototypes may be low-fidelity (e.g., Wizard of Oz studies) and gradually increase in fidelity as the interaction design becomes clearer.

#### Evaluation and Validation
HRI systems must be evaluated not only for technical performance but also for social acceptability, usability, and effectiveness in achieving their intended goals.

## Social Robotics Fundamentals

### Social Presence

Social presence refers to the extent to which a robot is perceived as a social entity rather than merely a machine. Key factors that contribute to social presence include:

#### Anthropomorphism
The attribution of human characteristics to non-human entities. Appropriate anthropomorphism can make robots more relatable and easier to interact with, but excessive anthropomorphism can lead to the "uncanny valley" effect, where robots become unsettling.

#### Agency
The perception that the robot acts with intention and autonomy rather than merely responding to pre-programmed stimuli. This includes the appearance of decision-making and goal-directed behavior.

#### Expressiveness
The ability to convey emotions, intentions, and mental states through facial expressions, gestures, tone of voice, and other modalities.

### Proxemics in HRI

Proxemics, the study of human use of space and the effects of population density on behavior, is crucial for robot design. Edward T. Hall identified four distance zones:

#### Intimate Distance (0-45 cm)
Reserved for close relationships and intimate activities. Robots should rarely enter this space without explicit permission.

#### Personal Distance (45-120 cm)
For interactions among friends and family. This is often appropriate for collaborative tasks.

#### Social Distance (120-360 cm)
For formal interactions and group conversations. This is typically the most comfortable distance for casual human-robot interactions.

#### Public Distance (360+ cm)
For public speaking and formal presentations. Robots may operate at this distance for information dissemination or monitoring tasks.

### Social Signal Processing

Social signals include facial expressions, gestures, posture, gaze, and vocal patterns that convey social information. Effective HRI systems must be able to recognize and respond to these signals.

```python
import numpy as np
import cv2
from enum import Enum

class SocialSignal(Enum):
    GREETING = 1
    ATTENTION = 2
    APPROACH = 3
    AVOIDANCE = 4
    CONFUSION = 5
    POSITIVE_FEEDBACK = 6
    NEGATIVE_FEEDBACK = 7

class SocialSignalDetector:
    """Detect social signals from human behavior"""

    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Emotion recognition model (simplified)
        self.emotion_model = self.load_emotion_model()

        # Gesture recognition
        self.gesture_recognizer = self.initialize_gesture_recognizer()

        # State tracking
        self.previous_signals = []
        self.confidence_threshold = 0.7

    def load_emotion_model(self):
        """Load emotion recognition model (placeholder)"""
        # In a real implementation, this would load a trained model
        # such as a CNN for emotion classification
        return lambda face_image: self.simple_emotion_classification(face_image)

    def initialize_gesture_recognizer(self):
        """Initialize gesture recognition system"""
        return {
            'wave_threshold': 50,  # Movement threshold for wave detection
            'hand_raised_threshold': 0.8,  # Confidence for raised hand detection
            'pointing_angle_threshold': 30  # Degrees for pointing detection
        }

    def detect_social_signals(self, rgb_image, depth_map=None):
        """Detect social signals from image and depth data"""
        signals = []

        # Detect faces
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Detect emotions
            emotion = self.emotion_model(face_roi)

            # Detect gaze direction (simplified)
            gaze_direction = self.estimate_gaze_direction(face_roi, (x, y, w, h))

            # Determine if attention is directed toward robot
            if self.is_attention_directed_toward_robot(gaze_direction):
                signals.append(SocialSignal.ATTENTION)

            # Detect approach/avoidance based on distance change
            distance_to_robot = self.estimate_distance_to_robot(depth_map, (x, y, w, h))
            if distance_to_robot < 1.5:  # Within 1.5m
                signals.append(SocialSignal.APPROACH)

            # Interpret emotions
            if emotion == 'happy' or emotion == 'surprised':
                signals.append(SocialSignal.POSITIVE_FEEDBACK)
            elif emotion == 'sad' or emotion == 'angry':
                signals.append(SocialSignal.NEGATIVE_FEEDBACK)
            elif emotion == 'confused':
                signals.append(SocialSignal.CONFUSION)

        # Detect gestures
        gestures = self.detect_gestures(rgb_image)
        if SocialSignal.GREETING in gestures:
            signals.extend(gestures)

        return signals

    def simple_emotion_classification(self, face_image):
        """Simple emotion classification (placeholder)"""
        # In reality, this would use a trained model
        # For now, return a simple classification based on face features
        # This is just a placeholder implementation

        # Calculate simple features
        mean_intensity = np.mean(face_image)
        intensity_variance = np.var(face_image)

        # Simple heuristic classification
        if mean_intensity > 150 and intensity_variance > 1000:
            return 'happy'
        elif mean_intensity < 100 and intensity_variance < 500:
            return 'sad'
        elif intensity_variance > 2000:
            return 'surprised'
        else:
            return 'neutral'

    def estimate_gaze_direction(self, face_image, face_bbox):
        """Estimate gaze direction (simplified)"""
        # This would normally use eye tracking and head pose estimation
        # For simplification, we'll just return a default direction
        return np.array([0, 0, 1])  # Looking forward

    def is_attention_directed_toward_robot(self, gaze_direction):
        """Check if gaze is directed toward robot"""
        # Simplified: assume robot is at origin and facing forward
        robot_forward = np.array([0, 0, 1])
        angle_threshold = np.deg2rad(30)  # 30 degree threshold

        cos_angle = np.dot(gaze_direction, robot_forward) / (
            np.linalg.norm(gaze_direction) * np.linalg.norm(robot_forward)
        )

        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) < angle_threshold

    def estimate_distance_to_robot(self, depth_map, bbox):
        """Estimate distance to robot from depth data"""
        if depth_map is None:
            return float('inf')  # Unknown distance

        x, y, w, h = bbox
        # Get average depth in face region
        face_depth_region = depth_map[y:y+h, x:x+w]

        if face_depth_region.size > 0:
            return np.mean(face_depth_region[face_depth_region > 0])  # Ignore invalid depth values
        else:
            return float('inf')

    def detect_gestures(self, image):
        """Detect gestures in the image"""
        gestures = []

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple wave detection (simplified)
        # In reality, this would use more sophisticated gesture recognition
        # such as pose estimation and temporal analysis

        # For this example, we'll simulate gesture detection
        # by looking for hand-like regions above a certain height in the image

        # This is a very simplified approach - real implementation would use
        # pose estimation (MediaPipe, OpenPose, etc.)

        # Look for bright regions that might be hands
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Large enough to be a hand
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Check if in upper portion of image (likely to be hand)
                if y < image.shape[0] * 0.5:
                    # This could be a waving hand or raised hand
                    gestures.append(SocialSignal.GREETING)

        return gestures

# Example usage
def example_social_signal_detection():
    """Example of social signal detection"""
    detector = SocialSignalDetector()

    # In a real implementation, we would have actual image data
    # For this example, we'll create a synthetic image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.uniform(1.0, 3.0, (480, 640))

    signals = detector.detect_social_signals(dummy_image, dummy_depth)

    print("Detected social signals:")
    for signal in signals:
        print(f"  - {signal.name}")

    return signals
```

## Communication Modalities

### Speech and Natural Language Processing

Voice interaction is one of the most natural forms of human communication and should be a primary modality for HRI systems.

```python
import speech_recognition as sr
import pyttsx3
import nltk
from transformers import pipeline

class NaturalLanguageInterface:
    """Natural language processing for HRI"""

    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.setup_tts_settings()

        # Initialize intent classification
        self.intent_classifier = self.initialize_intent_classifier()

        # Initialize dialogue manager
        self.dialogue_manager = DialogueManager()

        # Common responses
        self.responses = {
            'greeting': ["Hello!", "Hi there!", "Nice to meet you!"],
            'goodbye': ["Goodbye!", "See you later!", "Take care!"],
            'confused': ["I'm sorry, I didn't understand. Could you please repeat that?",
                         "Could you rephrase that for me?"]
        }

    def setup_tts_settings(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)  # Use first available voice
        self.tts_engine.setProperty('rate', 150)  # Words per minute
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

    def initialize_intent_classifier(self):
        """Initialize intent classification model"""
        # In a real implementation, this would load a trained NLP model
        # For this example, we'll use a simple keyword-based classifier
        return KeywordIntentClassifier()

    def listen_and_understand(self):
        """Listen to user and understand their intent"""
        try:
            with self.microphone as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5.0)

            # Convert speech to text
            text = self.recognizer.recognize_google(audio)
            print(f"Heard: {text}")

            # Classify intent
            intent = self.intent_classifier.classify_intent(text)

            return text, intent

        except sr.WaitTimeoutError:
            print("No speech detected")
            return "", "no_input"
        except sr.UnknownValueError:
            print("Could not understand audio")
            return "", "unknown"
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return "", "error"

    def speak(self, text):
        """Speak text using TTS engine"""
        print(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def process_user_input(self, user_text, intent):
        """Process user input and generate appropriate response"""
        response = self.dialogue_manager.generate_response(user_text, intent)

        # Speak the response
        self.speak(response)

        return response

class KeywordIntentClassifier:
    """Simple keyword-based intent classifier"""

    def __init__(self):
        self.intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'goodbye': ['goodbye', 'bye', 'see you', 'farewell', 'catch you later'],
            'help': ['help', 'assist', 'support', 'what can you do', 'how do i'],
            'command': ['move', 'go', 'walk', 'dance', 'turn', 'jump', 'sit', 'stand'],
            'question': ['what', 'who', 'when', 'where', 'why', 'how', 'can you', 'do you'],
            'affirmation': ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay'],
            'negation': ['no', 'nope', 'nah', 'negative', 'not really']
        }

    def classify_intent(self, text):
        """Classify intent based on keywords"""
        text_lower = text.lower()

        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent

        return 'unknown'

class DialogueManager:
    """Manage conversation flow and context"""

    def __init__(self):
        self.context = {}
        self.conversation_history = []
        self.response_templates = {
            'greeting': ["Hello! How can I assist you today?",
                        "Hi there! What can I help you with?"],
            'goodbye': ["It was nice talking with you!",
                       "Goodbye! Feel free to come back anytime."],
            'help': ["I can help with various tasks. You can ask me to move, dance, or answer questions.",
                    "I'm here to assist. Try asking me to do something or asking a question."],
            'question': ["That's an interesting question. Let me think about that...",
                        "I'll do my best to answer your question."],
            'command': ["I'll try to do that for you.",
                       "Okay, I'll attempt that task."]
        }

    def generate_response(self, user_text, intent):
        """Generate appropriate response based on user input and intent"""
        import random

        # Update context and history
        self.conversation_history.append({'user': user_text, 'intent': intent})

        # Generate response based on intent
        if intent in self.response_templates:
            response = random.choice(self.response_templates[intent])
        else:
            response = "I understand you said: " + user_text

        # Add contextual responses
        if len(self.conversation_history) > 1:
            # If this is a follow-up, acknowledge the context
            response = "Regarding your previous comment, " + response.lower()

        return response
```

### Non-Verbal Communication

Non-verbal communication is crucial for natural human-robot interaction, including gestures, facial expressions, and body language.

```python
import time
import numpy as np

class NonVerbalCommunication:
    """Handle non-verbal communication for the robot"""

    def __init__(self, robot_interface):
        self.robot_interface = robot_interface  # Interface to control robot actuators
        self.expressions = self.define_expressions()
        self.gestures = self.define_gestures()

    def define_expressions(self):
        """Define facial expressions the robot can display"""
        return {
            'happy': {
                'eye_lid': 0.8,
                'eyebrow': 0.3,
                'mouth': 0.7
            },
            'sad': {
                'eye_lid': 0.3,
                'eyebrow': -0.5,
                'mouth': -0.7
            },
            'surprised': {
                'eye_lid': 1.0,
                'eyebrow': 0.8,
                'mouth': 0.9
            },
            'confused': {
                'eye_lid': 0.6,
                'eyebrow': 0.4,
                'mouth': 0.0
            },
            'neutral': {
                'eye_lid': 0.7,
                'eyebrow': 0.0,
                'mouth': 0.0
            }
        }

    def define_gestures(self):
        """Define body gestures the robot can perform"""
        return {
            'wave': self.wave_gesture,
            'nod': self.nod_gesture,
            'shake_head': self.shake_head_gesture,
            'point': self.point_gesture,
            'beckon': self.beckon_gesture
        }

    def display_expression(self, expression_name):
        """Display a facial expression"""
        if expression_name in self.expressions:
            expression = self.expressions[expression_name]
            # In a real implementation, this would control facial actuators
            print(f"Displaying expression: {expression_name}")
            print(f"Eye lid: {expression['eye_lid']}, Eyebrow: {expression['eyebrow']}, Mouth: {expression['mouth']}")

            # Simulate the expression change
            self.robot_interface.set_facial_expression(expression)

    def perform_gesture(self, gesture_name, duration=1.0):
        """Perform a body gesture"""
        if gesture_name in self.gestures:
            print(f"Performing gesture: {gesture_name}")
            self.gestures[gesture_name](duration)
        else:
            print(f"Unknown gesture: {gesture_name}")

    def wave_gesture(self, duration):
        """Perform a waving gesture"""
        # This would control the robot's arm to wave
        print("Waving gesture initiated")

        # Simulate wave motion
        for i in range(int(duration * 10)):  # 10 Hz simulation
            angle = np.sin(i * 0.5) * 0.5  # Oscillating motion
            # In real implementation: self.robot_interface.move_arm(angle)
            time.sleep(0.1)

    def nod_gesture(self, duration):
        """Perform a nodding gesture (yes)"""
        print("Nodding gesture initiated")

        for i in range(int(duration * 5)):  # 5 Hz nodding
            angle = np.sin(i * 0.8) * 0.3
            # In real implementation: self.robot_interface.move_head(angle, axis='pitch')
            time.sleep(0.2)

    def shake_head_gesture(self, duration):
        """Perform a head shaking gesture (no)"""
        print("Shaking head gesture initiated")

        for i in range(int(duration * 5)):  # 5 Hz shaking
            angle = np.sin(i * 0.8) * 0.4
            # In real implementation: self.robot_interface.move_head(angle, axis='yaw')
            time.sleep(0.2)

    def point_gesture(self, direction='front', duration=2.0):
        """Point in a specific direction"""
        print(f"Pointing gesture to the {direction}")

        # In real implementation: move arm to point
        # self.robot_interface.point_to_direction(direction)
        time.sleep(duration)

    def beckon_gesture(self, duration=2.0):
        """Gesture to beckon someone closer"""
        print("Beckoning gesture initiated")

        # Simulate beckoning motion
        for i in range(int(duration * 8)):  # 8 Hz beckoning
            if i % 2 == 0:
                # Move arm toward robot
                print("Arm moving toward robot")
            else:
                # Move arm away
                print("Arm moving away")
            time.sleep(0.125)

    def react_to_social_signals(self, signals):
        """React to detected social signals with appropriate expressions/gestures"""
        for signal in signals:
            if signal == SocialSignal.GREETING:
                self.display_expression('happy')
                self.perform_gesture('wave', 1.5)
            elif signal == SocialSignal.ATTENTION:
                self.display_expression('neutral')
                self.perform_gesture('nod', 0.5)
            elif signal == SocialSignal.NEGATIVE_FEEDBACK:
                self.display_expression('confused')
            elif signal == SocialSignal.POSITIVE_FEEDBACK:
                self.display_expression('happy')
            elif signal == SocialSignal.CONFUSION:
                self.display_expression('confused')
                self.perform_gesture('shrug', 1.0)  # Assuming we have a shrug gesture

class RobotInterface:
    """Interface to control the physical robot"""

    def __init__(self):
        self.facial_expression = {}
        self.head_position = np.array([0, 0, 0])
        self.arm_position = np.array([0, 0, 0])

    def set_facial_expression(self, expression):
        """Set the robot's facial expression"""
        self.facial_expression = expression
        print(f"Facial expression set to: {expression}")

    def move_head(self, angle, axis='pitch'):
        """Move the robot's head"""
        if axis == 'pitch':
            self.head_position[1] = angle
        elif axis == 'yaw':
            self.head_position[2] = angle
        print(f"Head moved to {axis} angle: {angle}")

    def move_arm(self, angle):
        """Move the robot's arm"""
        self.arm_position[0] = angle
        print(f"Arm moved to angle: {angle}")

    def point_to_direction(self, direction):
        """Point to a specific direction"""
        print(f"Pointing to {direction}")
```

## Interaction Design Principles

### Creating Natural Interactions

Natural interactions feel intuitive to users and follow human social conventions. Here are key principles for creating natural HRI:

#### Predictability
Users should be able to anticipate the robot's behavior based on its current state and context. This means consistent responses to similar situations and clear feedback when actions are being processed.

#### Legibility
The robot's intentions should be clearly communicated through its behavior, expressions, and actions. Users should understand why the robot is doing something.

#### Forgiveness
The system should be tolerant of user errors and provide helpful recovery options rather than failing completely.

#### Appropriate Personality
The robot should exhibit personality traits appropriate to its role and context, neither too human-like nor too mechanical.

### Social Navigation

When robots move around humans, they must follow social conventions for navigation and spatial interaction.

```python
class SocialNavigation:
    """Handle social navigation for the robot"""

    def __init__(self, robot_radius=0.3):
        self.robot_radius = robot_radius
        self.human_proxemics = {
            'intimate': 0.45,
            'personal': 1.2,
            'social': 3.6,
            'public': 7.6
        }
        self.current_human_positions = []
        self.social_routes = []

    def update_human_positions(self, human_positions):
        """Update positions of detected humans"""
        self.current_human_positions = human_positions

    def plan_social_route(self, start, goal):
        """Plan a route that respects human social spaces"""
        # This would typically use a modified path planning algorithm
        # that treats human proxemic zones as costly areas

        # For this example, we'll implement a simple approach
        # that adds cost to areas near humans

        # Create a cost map based on human proximity
        cost_map = self.create_social_cost_map(start, goal)

        # Use A* or other path planning algorithm with the cost map
        route = self.plan_path_with_costs(start, goal, cost_map)

        return route

    def create_social_cost_map(self, start, goal):
        """Create a cost map that penalizes proximity to humans"""
        # This is a simplified representation
        # In reality, this would be a grid or graph with cost values
        cost_map = {}

        # For each human, create a cost field around them
        for human_pos in self.current_human_positions:
            # Add cost to areas around the human based on proxemic zones
            for x in range(int(min(start[0], goal[0], human_pos[0]) - 5),
                          int(max(start[0], goal[0], human_pos[0]) + 5)):
                for y in range(int(min(start[1], goal[1], human_pos[1]) - 5),
                              int(max(start[1], goal[1], human_pos[1]) + 5)):

                    dist_to_human = np.sqrt((x - human_pos[0])**2 + (y - human_pos[1])**2)

                    # Higher cost for being in personal space
                    if dist_to_human < self.human_proxemics['personal']:
                        cost = 100.0
                    elif dist_to_human < self.human_proxemics['social']:
                        cost = 10.0
                    elif dist_to_human < self.human_proxemics['public']:
                        cost = 2.0
                    else:
                        cost = 1.0  # Base cost

                    cost_map[(x, y)] = cost

        return cost_map

    def plan_path_with_costs(self, start, goal, cost_map):
        """Plan path using cost map (simplified A* implementation)"""
        # This is a simplified version - a full implementation would use
        # proper A* with the cost map
        path = []

        # For this example, we'll create a simple path that goes around humans
        current = np.array(start)
        target = np.array(goal)

        # Simple approach: move to avoid humans if they're in the direct path
        for i in range(20):  # Max 20 steps
            direction = target - current
            distance = np.linalg.norm(direction)

            if distance < 0.5:  # Close enough to goal
                path.append(tuple(target))
                break

            # Normalize direction and move
            direction = direction / distance
            next_pos = current + direction * 0.5  # Move 0.5m at a time

            # Check if this position is too close to humans
            too_close = False
            for human_pos in self.current_human_positions:
                if np.linalg.norm(next_pos - human_pos) < self.human_proxemics['personal']:
                    too_close = True
                    # Try to go around the human
                    # Move perpendicular to the direction to human
                    perp_direction = np.array([-direction[1], direction[0]])  # Perpendicular
                    next_pos = current + perp_direction * 0.5
                    break

            if not too_close:
                current = next_pos
            else:
                # If still too close, move in the perpendicular direction
                current = current + np.array([-direction[1], direction[0]]) * 0.3

            path.append(tuple(current))

        return path

    def adjust_behavior_for_human_density(self, human_density):
        """Adjust navigation behavior based on human density"""
        if human_density > 0.1:  # High density area
            # Move more cautiously, slower, with wider berth
            self.speed_multiplier = 0.7
            self.proxemics_multiplier = 1.2
        elif human_density < 0.01:  # Low density
            # Can move faster with normal proxemics
            self.speed_multiplier = 1.0
            self.proxemics_multiplier = 1.0
        else:  # Medium density
            self.speed_multiplier = 0.9
            self.proxemics_multiplier = 1.1

    def request_passage(self, human_position):
        """Request passage politely when path is blocked by humans"""
        print(f"Requesting passage near human at {human_position}")

        # Turn toward the human to get attention
        # In real implementation: self.robot_interface.turn_toward(human_position)

        # Display polite expression
        # In real implementation: self.non_verbal.display_expression('neutral_polite')

        # Speak request
        # In real implementation: self.natural_language.speak("Excuse me, may I pass?")

        # Wait for acknowledgment or path clearing
        # In real implementation: self.wait_for_acknowledgment()

        print("Requested passage, waiting for response")
```

## Safety and Comfort in HRI

### Physical Safety

Physical safety is paramount in HRI. This includes not just preventing harm, but also making humans feel physically safe around the robot.

```python
class SafetySystem:
    """Safety system for human-robot interaction"""

    def __init__(self):
        self.safety_zones = {
            'danger': 0.3,      # Immediate danger zone
            'caution': 0.8,     # Zone requiring caution
            'safe': 1.5        # Safe interaction distance
        }

        self.emergency_stop_active = False
        self.speed_limits = {
            'slow': 0.1,    # m/s for close proximity
            'normal': 0.5,  # m/s for normal operation
            'fast': 1.0     # m/s for open areas
        }

        self.force_limits = {
            'light_touch': 5.0,   # N for intentional light contact
            'accidental': 20.0,   # N for accidental contact
            'maximum': 50.0       # N maximum allowable force
        }

    def check_proximity_safety(self, human_positions, robot_position):
        """Check if robot is in safe proximity to humans"""
        safety_status = {
            'closest_human_dist': float('inf'),
            'zone': 'safe',
            'needs_attention': False
        }

        for human_pos in human_positions:
            dist = np.linalg.norm(np.array(human_pos[:2]) - np.array(robot_position[:2]))
            safety_status['closest_human_dist'] = min(safety_status['closest_human_dist'], dist)

            if dist < self.safety_zones['danger']:
                safety_status['zone'] = 'danger'
                safety_status['needs_attention'] = True
            elif dist < self.safety_zones['caution']:
                safety_status['zone'] = 'caution'
                safety_status['needs_attention'] = True
            elif dist < self.safety_zones['safe']:
                safety_status['zone'] = 'safe_but_nearby'
                safety_status['needs_attention'] = True

        return safety_status

    def adjust_speed_for_safety(self, safety_zone):
        """Adjust robot speed based on safety zone"""
        if safety_zone == 'danger':
            return self.speed_limits['slow'] * 0.5  # Very slow
        elif safety_zone == 'caution':
            return self.speed_limits['slow']
        elif safety_zone == 'safe_but_nearby':
            return self.speed_limits['normal'] * 0.8
        else:  # 'safe'
            return self.speed_limits['normal']

    def emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        print("EMERGENCY STOP ACTIVATED")
        # In real implementation: self.robot_interface.emergency_stop()
        return True

    def check_force_limits(self, applied_forces):
        """Check if applied forces are within safe limits"""
        violations = []

        for i, force in enumerate(applied_forces):
            if force > self.force_limits['maximum']:
                violations.append({
                    'joint': i,
                    'force': force,
                    'limit': self.force_limits['maximum'],
                    'severity': 'critical'
                })
            elif force > self.force_limits['accidental']:
                violations.append({
                    'joint': i,
                    'force': force,
                    'limit': self.force_limits['accidental'],
                    'severity': 'warning'
                })

        return violations

    def comfort_assessment(self, interaction_data):
        """Assess human comfort during interaction"""
        # Analyze various comfort indicators:
        # - Proximity (already handled by proximity check)
        # - Interaction duration
        # - Human behavior (receding, tense posture, etc.)
        # - Physiological indicators (if available)

        comfort_score = 1.0  # 1.0 = very comfortable, 0.0 = very uncomfortable

        # Decrease comfort if humans are too close for too long
        if interaction_data.get('duration', 0) > 30:  # 30 seconds
            comfort_score *= 0.8

        # Increase comfort if interaction is positive
        if interaction_data.get('positive_feedback', 0) > interaction_data.get('negative_feedback', 0):
            comfort_score = min(1.0, comfort_score * 1.2)

        # Check for signs of discomfort
        if interaction_data.get('avoidance_behaviors', 0) > 0:
            comfort_score *= 0.7

        return max(0.0, min(1.0, comfort_score))
```

### Psychological Safety

Psychological safety involves making humans feel comfortable and not threatened by the robot's presence or behavior.

```python
class PsychologicalSafety:
    """Psychological safety aspects of HRI"""

    def __init__(self):
        self.appropriateness_rules = {
            'personal_space_violations': 0,
            'inappropriate_gestures': 0,
            'intimidating_behavior': 0
        }

        self.transparency_level = 0.7  # How transparent the robot's intentions are (0-1)
        self.predictability_score = 0.8  # How predictable the robot's behavior is (0-1)

    def assess_psychological_safety(self, human_reactions, robot_behavior):
        """Assess psychological safety based on human reactions"""
        safety_score = 1.0  # Start optimistic

        # Penalize for negative reactions
        if 'startled' in human_reactions:
            safety_score -= 0.3
        if 'uncomfortable' in human_reactions:
            safety_score -= 0.2
        if 'avoiding' in human_reactions:
            safety_score -= 0.25
        if 'frightened' in human_reactions:
            safety_score -= 0.4

        # Adjust for transparency and predictability
        safety_score *= (0.5 + 0.5 * self.transparency_level)
        safety_score *= (0.5 + 0.5 * self.predictability_score)

        return max(0.0, min(1.0, safety_score))

    def adjust_behavior_for_comfort(self, safety_score, current_behavior):
        """Adjust robot behavior based on psychological safety"""
        adjusted_behavior = current_behavior.copy()

        if safety_score < 0.3:  # Very uncomfortable
            # Significantly reduce expressiveness and movement
            adjusted_behavior['movement_amplitude'] *= 0.3
            adjusted_behavior['expression_intensity'] *= 0.2
            adjusted_behavior['interaction_frequency'] *= 0.1
        elif safety_score < 0.6:  # Somewhat uncomfortable
            # Reduce expressiveness moderately
            adjusted_behavior['movement_amplitude'] *= 0.6
            adjusted_behavior['expression_intensity'] *= 0.5
        elif safety_score > 0.8:  # Very comfortable
            # Can be more expressive and interactive
            adjusted_behavior['movement_amplitude'] *= 1.2
            adjusted_behavior['expression_intensity'] *= 1.1

        # Ensure adjustments stay within safe bounds
        adjusted_behavior['movement_amplitude'] = np.clip(
            adjusted_behavior['movement_amplitude'], 0.1, 1.0
        )
        adjusted_behavior['expression_intensity'] = np.clip(
            adjusted_behavior['expression_intensity'], 0.1, 1.0
        )

        return adjusted_behavior
```

## Cultural and Ethical Considerations

### Cultural Sensitivity

Different cultures have varying norms for personal space, eye contact, gestures, and communication styles. HRI systems should be adaptable to different cultural contexts.

```python
class CulturalAdaptation:
    """Adapt HRI behavior to different cultural contexts"""

    def __init__(self):
        self.cultural_profiles = {
            'default': {
                'personal_space': 1.2,
                'eye_contact_duration': 3.0,
                'gestures': ['wave', 'nod'],
                'formality_level': 0.5,
                'touch_acceptance': 0.3
            },
            'japanese': {
                'personal_space': 1.5,  # Larger personal space in Japan
                'eye_contact_duration': 1.0,  # Less direct eye contact
                'gestures': ['bow', 'nod'],  # Bowing is common
                'formality_level': 0.8,  # More formal interactions
                'touch_acceptance': 0.1   # Very low touch acceptance
            },
            'middle_eastern': {
                'personal_space': 1.0,
                'eye_contact_duration': 2.0,
                'gestures': ['hand_on_heart', 'nod'],
                'formality_level': 0.7,
                'touch_acceptance': 0.2,  # Gender-dependent
                'gender_interaction_rules': True
            },
            'mediterranean': {
                'personal_space': 0.8,  # Smaller personal space
                'eye_contact_duration': 4.0,
                'gestures': ['hand_gestures', 'embrace'],  # More expressive
                'formality_level': 0.4,  # Less formal
                'touch_acceptance': 0.6   # Higher touch acceptance
            }
        }

        self.current_culture = 'default'
        self.user_preferences = {}

    def set_cultural_context(self, culture):
        """Set the cultural context for interaction"""
        if culture in self.cultural_profiles:
            self.current_culture = culture
            print(f"Set cultural context to: {culture}")
        else:
            print(f"Unknown culture: {culture}, using default")
            self.current_culture = 'default'

    def get_cultural_norms(self):
        """Get current cultural norms"""
        return self.cultural_profiles[self.current_culture]

    def adapt_behavior_to_culture(self, base_behavior):
        """Adapt base behavior to current cultural context"""
        cultural_norms = self.cultural_profiles[self.current_culture]
        adapted_behavior = base_behavior.copy()

        # Adjust personal space
        adapted_behavior['personal_space_multiplier'] = cultural_norms['personal_space'] / 1.2

        # Adjust eye contact
        adapted_behavior['eye_contact_duration'] = cultural_norms['eye_contact_duration']

        # Adjust gestures
        adapted_behavior['allowed_gestures'] = cultural_norms['gestures']

        # Adjust formality
        adapted_behavior['formality_level'] = cultural_norms['formality_level']

        # Adjust touch sensitivity
        adapted_behavior['touch_threshold'] = cultural_norms['touch_acceptance']

        # Apply user preferences if available
        if self.user_preferences:
            adapted_behavior.update(self.user_preferences)

        return adapted_behavior

    def learn_user_preferences(self, user_id, interaction_feedback):
        """Learn individual user preferences based on feedback"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}

        # Update preferences based on positive/negative feedback
        for aspect, feedback in interaction_feedback.items():
            if feedback > 0:  # Positive feedback
                # Increase comfort with this behavior
                pass  # Implementation would adjust preferences
            elif feedback < 0:  # Negative feedback
                # Decrease likelihood of this behavior
                pass  # Implementation would adjust preferences
```

## Hands-on Exercise: Implementing an HRI System

In this exercise, you'll implement a complete HRI system that integrates perception, communication, and safety considerations.

### Requirements
- Python 3.8+
- OpenCV for computer vision
- Speech recognition libraries
- NumPy for numerical computations
- Matplotlib for visualization

### Exercise Steps
1. Implement a social signal detection system
2. Create a natural language interface
3. Integrate non-verbal communication
4. Implement safety monitoring
5. Test the system with simulated interactions

### Expected Outcome
You should have a working HRI system that can detect social signals, engage in simple conversations, respond with appropriate expressions and gestures, and maintain safety protocols during interactions.

### Sample Implementation

```python
import threading
import time
import random

class HRIExerciseSystem:
    """Complete HRI system for the hands-on exercise"""

    def __init__(self):
        # Initialize components
        self.social_detector = SocialSignalDetector()
        self.nlp_interface = NaturalLanguageInterface()
        self.robot_interface = RobotInterface()
        self.nonverbal = NonVerbalCommunication(self.robot_interface)
        self.safety_system = SafetySystem()
        self.psychological_safety = PsychologicalSafety()
        self.cultural_adapter = CulturalAdaptation()

        # System state
        self.running = False
        self.interaction_active = False
        self.human_positions = []

        # Interaction metrics
        self.interaction_log = []
        self.comfort_scores = []
        self.safety_incidents = []

    def simulate_human_presence(self):
        """Simulate human positions for testing"""
        # In a real system, this would come from perception systems
        # For simulation, we'll generate random positions
        humans = []
        for i in range(random.randint(1, 3)):  # 1-3 humans
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            humans.append((x, y, 0))  # x, y, z position
        return humans

    def perceive_environment(self):
        """Perceive the environment (simulated)"""
        # Simulate perception of humans
        self.human_positions = self.simulate_human_presence()

        # Simulate social signal detection
        # In a real system, this would process actual sensor data
        detected_signals = []
        if self.human_positions:
            # Randomly generate some signals for simulation
            if random.random() > 0.7:
                detected_signals.append(SocialSignal.GREETING)
            if random.random() > 0.8:
                detected_signals.append(SocialSignal.ATTENTION)
            if random.random() > 0.9:
                detected_signals.append(SocialSignal.POSITIVE_FEEDBACK)

        return self.human_positions, detected_signals

    def manage_interaction(self):
        """Main interaction management loop"""
        print("Starting HRI System...")

        while self.running:
            # Perceive environment
            human_positions, social_signals = self.perceive_environment()

            # Check safety
            safety_status = self.safety_system.check_proximity_safety(
                human_positions, (0, 0, 0)  # Robot position
            )

            if safety_status['zone'] == 'danger':
                print("DANGER: Too close to human, initiating safety protocol")
                self.safety_system.emergency_stop()
                continue

            # React to social signals
            if social_signals:
                print(f"Detected social signals: {[s.name for s in social_signals]}")
                self.nonverbal.react_to_social_signals(social_signals)

            # Listen for speech if humans are paying attention
            if SocialSignal.ATTENTION in social_signals:
                print("Human attention detected, listening for speech...")
                user_text, intent = self.nlp_interface.listen_and_understand()

                if user_text:
                    print(f"Processing: '{user_text}' (intent: {intent})")
                    response = self.nlp_interface.process_user_input(user_text, intent)

                    # Log the interaction
                    interaction_record = {
                        'timestamp': time.time(),
                        'user_input': user_text,
                        'intent': intent,
                        'response': response,
                        'signals': [s.name for s in social_signals]
                    }
                    self.interaction_log.append(interaction_record)

            # Adjust behavior based on safety and comfort
            safety_score = 1.0 if safety_status['zone'] != 'danger' else 0.0
            comfort_score = self.psychological_safety.assess_psychological_safety({
                'avoidance_behaviors': 0,
                'positive_feedback': len([s for s in social_signals if s in [SocialSignal.POSITIVE_FEEDBACK, SocialSignal.GREETING]]),
                'negative_feedback': len([s for s in social_signals if s in [SocialSignal.NEGATIVE_FEEDBACK, SocialSignal.AVOIDANCE]])
            })

            self.comfort_scores.append(comfort_score)

            # Adjust robot behavior based on comfort and safety
            if comfort_score < 0.5:
                print("Comfort level low, adjusting behavior...")
                # Implement behavior adjustment
                self.nonverbal.display_expression('neutral')

            # Sleep briefly to simulate real-time processing
            time.sleep(0.1)

    def start_interaction_system(self):
        """Start the HRI system"""
        self.running = True

        # Start the main interaction loop in a separate thread
        interaction_thread = threading.Thread(target=self.manage_interaction)
        interaction_thread.daemon = True
        interaction_thread.start()

        print("HRI System started. Press Ctrl+C to stop.")

        try:
            while self.running:
                # Main thread can handle other tasks
                time.sleep(1.0)

                # Print periodic status
                if len(self.interaction_log) % 10 == 0 and self.interaction_log:
                    avg_comfort = np.mean(self.comfort_scores) if self.comfort_scores else 0
                    print(f"Status: {len(self.interaction_log)} interactions, "
                          f"avg comfort: {avg_comfort:.2f}")

        except KeyboardInterrupt:
            print("\nStopping HRI system...")
            self.running = False
            interaction_thread.join(timeout=2.0)
            print("HRI system stopped.")

    def run_evaluation(self):
        """Run evaluation of the HRI system"""
        print("\n" + "="*50)
        print("HRI SYSTEM EVALUATION")
        print("="*50)

        # Interaction metrics
        total_interactions = len(self.interaction_log)
        avg_comfort = np.mean(self.comfort_scores) if self.comfort_scores else 0
        safety_incidents_count = len(self.safety_incidents)

        print(f"Total interactions: {total_interactions}")
        print(f"Average comfort score: {avg_comfort:.2f} (0-1 scale)")
        print(f"Safety incidents: {safety_incidents_count}")

        # Intent distribution
        if self.interaction_log:
            intents = [record['intent'] for record in self.interaction_log]
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            print(f"Intent distribution: {intent_counts}")

        # Comfort trend analysis
        if len(self.comfort_scores) > 10:
            early_comfort = np.mean(self.comfort_scores[:10])
            late_comfort = np.mean(self.comfort_scores[-10:])
            print(f"Early comfort: {early_comfort:.2f}, Late comfort: {late_comfort:.2f}")
            trend = "Improving" if late_comfort > early_comfort else "Declining"
            print(f"Comfort trend: {trend}")

        print("="*50)

        return {
            'total_interactions': total_interactions,
            'average_comfort': avg_comfort,
            'safety_incidents': safety_incidents_count,
            'intent_distribution': intent_counts if self.interaction_log else {},
            'comfort_trend': 'improving' if late_comfort > early_comfort else 'declining' if self.comfort_scores and len(self.comfort_scores) > 10 else 'stable'
        }

# Example usage and testing
def run_hri_exercise():
    """Run the complete HRI exercise"""
    print("Starting Human-Robot Interaction Exercise")
    print("This will simulate an HRI system with perception, communication, and safety features.")

    # Create the HRI system
    hri_system = HRIExerciseSystem()

    try:
        # Start the system
        hri_system.start_interaction_system()
    except KeyboardInterrupt:
        print("\nSystem stopped by user.")

    # Run evaluation
    evaluation_results = hri_system.run_evaluation()

    print(f"\nHRI Exercise completed!")
    print(f"Results: {evaluation_results}")

    return hri_system, evaluation_results

# Uncomment to run the exercise
# hri_system, results = run_hri_exercise()
```

## Advanced HRI Concepts

### Context-Aware Interaction

Context-aware systems adapt their behavior based on the environment, situation, and user state.

```python
class ContextAwareInteraction:
    """Context-aware interaction system"""

    def __init__(self):
        self.context_model = {
            'location': 'unknown',
            'time_of_day': 'unknown',
            'social_context': 'unknown',  # alone, group, formal, informal
            'user_activity': 'unknown',
            'environment_noise': 'low',
            'lighting_conditions': 'normal'
        }

        self.context_sensitive_behaviors = {
            'quiet_library': {
                'speech_volume': 0.3,
                'movement_speed': 0.5,
                'expressiveness': 0.4
            },
            'busy_restaurant': {
                'speech_volume': 0.8,
                'movement_speed': 0.7,
                'expressiveness': 0.8
            },
            'formal_meeting': {
                'speech_volume': 0.6,
                'movement_speed': 0.4,
                'expressiveness': 0.5,
                'formality': 0.9
            }
        }

    def update_context(self, new_context):
        """Update the system's understanding of context"""
        self.context_model.update(new_context)

        # Determine context profile
        location = self.context_model['location']
        social_context = self.context_model['social_context']
        context_profile = f"{location}_{social_context}".replace(" ", "_").lower()

        if context_profile in self.context_sensitive_behaviors:
            self.active_context_profile = context_profile
        else:
            self.active_context_profile = 'unknown'

    def get_context_appropriate_behavior(self, base_behavior):
        """Get behavior adapted to current context"""
        if self.active_context_profile == 'unknown':
            return base_behavior

        context_modifiers = self.context_sensitive_behaviors[self.active_context_profile]
        adapted_behavior = base_behavior.copy()

        # Apply context-specific modifications
        for aspect, modifier in context_modifiers.items():
            if aspect in adapted_behavior:
                adapted_behavior[aspect] = adapted_behavior[aspect] * modifier
            else:
                adapted_behavior[aspect] = modifier

        return adapted_behavior
```

### Multi-Party Interaction

Handling interactions with multiple people simultaneously requires sophisticated attention management and turn-taking protocols.

```python
class MultiPartyInteractionManager:
    """Manage interactions with multiple people"""

    def __init__(self):
        self.people_tracking = {}  # Track multiple people
        self.attention_model = {
            'speaker': None,
            'focus_person': None,
            'attention_history': []
        }
        self.turn_taking_rules = {
            'wait_time': 1.0,  # Seconds to wait before interrupting
            'gaze_cue_timeout': 3.0,  # Seconds to maintain gaze
            'acknowledgment_delay': 0.5  # Delay before acknowledging
        }

    def track_people(self, detected_people):
        """Track multiple people in the interaction space"""
        current_ids = set()

        for person_data in detected_people:
            person_id = person_data.get('id', self.assign_new_id())
            current_ids.add(person_id)

            if person_id not in self.people_tracking:
                self.people_tracking[person_id] = {
                    'position': person_data['position'],
                    'engagement_level': 0.0,
                    'last_interaction': time.time(),
                    'speaking': False,
                    'attention_received': 0
                }
            else:
                self.people_tracking[person_id].update({
                    'position': person_data['position'],
                    'speaking': person_data.get('speaking', False)
                })

        # Remove people who are no longer detected
        for person_id in list(self.people_tracking.keys()):
            if person_id not in current_ids:
                del self.people_tracking[person_id]

    def assign_new_id(self):
        """Assign a new unique ID to a person"""
        return len(self.people_tracking)

    def determine_attention_focus(self):
        """Determine who to focus attention on"""
        if not self.people_tracking:
            return None

        # Simple strategy: focus on the most engaged person
        # In reality, this would be more sophisticated
        most_engaged = None
        highest_engagement = -1

        for person_id, data in self.people_tracking.items():
            engagement = data['engagement_level']
            if data['speaking']:  # Speaking people get priority
                engagement += 0.5
            if data['attention_received'] < 5:  # People with less attention get priority
                engagement += 0.2

            if engagement > highest_engagement:
                highest_engagement = engagement
                most_engaged = person_id

        # Update attention history
        self.attention_model['attention_history'].append(most_engaged)
        if len(self.attention_model['attention_history']) > 10:
            self.attention_model['attention_history'] = self.attention_model['attention_history'][-10:]

        self.attention_model['focus_person'] = most_engaged
        self.attention_model['speaker'] = self.get_current_speaker()

        return most_engaged

    def get_current_speaker(self):
        """Get the person who is currently speaking"""
        for person_id, data in self.people_tracking.items():
            if data['speaking']:
                return person_id
        return None

    def manage_turn_taking(self, speech_detected_from):
        """Manage turn-taking in multi-party conversations"""
        current_speaker = self.get_current_speaker()

        if current_speaker != speech_detected_from:
            # New speaker detected, manage transition
            if current_speaker is not None:
                # Acknowledge the transition
                print(f"Transitioning from {current_speaker} to {speech_detected_from}")

            # Update attention focus
            self.attention_model['focus_person'] = speech_detected_from
            self.attention_model['speaker'] = speech_detected_from

    def interact_with_multiple_people(self, detected_people, speech_data=None):
        """Handle interaction with multiple people"""
        # Track people
        self.track_people(detected_people)

        # Determine attention focus
        focus_person = self.determine_attention_focus()

        # Manage turn-taking if speech is detected
        if speech_data and speech_data.get('speaker_id'):
            self.manage_turn_taking(speech_data['speaker_id'])

        # Update engagement levels
        for person_id in self.people_tracking:
            if person_id == focus_person:
                self.people_tracking[person_id]['engagement_level'] = min(
                    1.0, self.people_tracking[person_id]['engagement_level'] + 0.1
                )
                self.people_tracking[person_id]['attention_received'] += 1
            else:
                self.people_tracking[person_id]['engagement_level'] = max(
                    0.0, self.people_tracking[person_id]['engagement_level'] - 0.05
                )

        return {
            'focus_person': focus_person,
            'people_count': len(self.people_tracking),
            'attention_distribution': {
                pid: data['engagement_level'] for pid, data in self.people_tracking.items()
            }
        }
```

## Evaluation of HRI Systems

### Usability and Acceptance Metrics

Evaluating HRI systems requires both technical metrics and human-centered measures.

```python
class HRIEvaluationFramework:
    """Framework for evaluating HRI systems"""

    def __init__(self):
        self.metrics = {
            'usability': [],
            'acceptance': [],
            'safety': [],
            'effectiveness': [],
            'satisfaction': []
        }

        self.longitudinal_tracking = True
        self.comparative_analysis = True

    def evaluate_usability(self, user_performance_data):
        """Evaluate system usability"""
        # Calculate usability metrics
        task_completion_rate = user_performance_data.get('completed_tasks', 0) / max(
            user_performance_data.get('total_tasks', 1), 1
        )

        task_time_ratio = user_performance_data.get('actual_time', 1) / max(
            user_performance_data.get('expected_time', 1), 1
        )

        error_rate = user_performance_data.get('errors', 0) / max(
            user_performance_data.get('interactions', 1), 1
        )

        # Calculate composite usability score (0-1 scale)
        # Lower is better for time_ratio and error_rate
        usability_score = (task_completion_rate * 0.5 +
                          max(0, 1 - task_time_ratio) * 0.3 +
                          max(0, 1 - error_rate) * 0.2)

        self.metrics['usability'].append(usability_score)
        return usability_score

    def evaluate_acceptance(self, user_survey_responses):
        """Evaluate user acceptance"""
        # Process survey responses (typically Likert scale 1-5)
        if not user_survey_responses:
            return 0.0

        # Example survey dimensions
        hri_dimensions = {
            'ease_of_use': 'ease_rating',
            'trust': 'trust_rating',
            'comfort': 'comfort_rating',
            'naturalness': 'naturalness_rating',
            'helpfulness': 'helpfulness_rating'
        }

        total_score = 0
        valid_responses = 0

        for response in user_survey_responses:
            for dimension, key in hri_dimensions.items():
                if key in response:
                    # Normalize to 0-1 scale (assuming 1-5 scale)
                    rating = response[key]
                    normalized = (rating - 1) / 4.0  # Convert 1-5 to 0-1
                    total_score += normalized
                    valid_responses += 1

        if valid_responses > 0:
            acceptance_score = total_score / valid_responses
        else:
            acceptance_score = 0.0

        self.metrics['acceptance'].append(acceptance_score)
        return acceptance_score

    def evaluate_safety(self, safety_incident_data):
        """Evaluate safety performance"""
        # Calculate safety metrics
        incident_rate = len(safety_incident_data) / max(
            safety_incident_data.get('total_interaction_time', 1), 1
        )

        severe_incidents = sum(1 for incident in safety_incident_data if incident.get('severity') == 'high')
        severe_rate = severe_incidents / max(len(safety_incident_data), 1)

        # Safety score (inverted: fewer incidents = higher score)
        safety_score = max(0, 1 - incident_rate) * 0.7 + max(0, 1 - severe_rate) * 0.3

        self.metrics['safety'].append(safety_score)
        return safety_score

    def evaluate_effectiveness(self, task_outcome_data):
        """Evaluate interaction effectiveness"""
        # Calculate effectiveness based on goal achievement
        goals_achieved = task_outcome_data.get('successful_interactions', 0)
        total_goals = task_outcome_data.get('total_interactions', 1)

        # Consider quality of achievement, not just completion
        quality_weight = task_outcome_data.get('average_quality', 0.5)

        effectiveness_score = (goals_achieved / total_goals) * quality_weight

        self.metrics['effectiveness'].append(effectiveness_score)
        return effectiveness_score

    def evaluate_satisfaction(self, user_feedback_data):
        """Evaluate user satisfaction"""
        if not user_feedback_data:
            return 0.0

        # Process qualitative and quantitative feedback
        positive_sentiment = 0
        negative_sentiment = 0
        neutral_sentiment = 0
        numerical_ratings = []

        for feedback in user_feedback_data:
            if 'sentiment' in feedback:
                sentiment = feedback['sentiment']
                if sentiment > 0.1:
                    positive_sentiment += 1
                elif sentiment < -0.1:
                    negative_sentiment += 1
                else:
                    neutral_sentiment += 1
            if 'rating' in feedback:
                numerical_ratings.append(feedback['rating'])

        # Calculate sentiment-based satisfaction
        total_sentiments = positive_sentiment + negative_sentiment + neutral_sentiment
        if total_sentiments > 0:
            sentiment_satisfaction = (positive_sentiment - negative_sentiment) / total_sentiments
        else:
            sentiment_satisfaction = 0.0

        # Calculate rating-based satisfaction
        if numerical_ratings:
            rating_satisfaction = sum(numerical_ratings) / len(numerical_ratings)
            # Normalize from whatever scale to 0-1 (assuming 1-5 scale)
            rating_satisfaction = (rating_satisfaction - 1) / 4.0
        else:
            rating_satisfaction = 0.0

        # Weighted combination
        satisfaction_score = sentiment_satisfaction * 0.4 + rating_satisfaction * 0.6

        self.metrics['satisfaction'].append(satisfaction_score)
        return max(0.0, min(1.0, satisfaction_score))

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            'overall_scores': {},
            'trends': {},
            'recommendations': []
        }

        for metric_name, values in self.metrics.items():
            if values:
                report['overall_scores'][metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

                # Calculate trend
                if len(values) >= 5:
                    recent_avg = sum(values[-5:]) / 5
                    earlier_avg = sum(values[:5]) / min(5, len(values))
                    trend = 'improving' if recent_avg > earlier_avg else 'declining' if recent_avg < earlier_avg else 'stable'
                    report['trends'][metric_name] = trend

        # Generate recommendations based on performance
        for metric_name, scores in report['overall_scores'].items():
            if scores['average'] < 0.6:
                report['recommendations'].append(
                    f"{metric_name} scoring low (avg: {scores['average']:.2f}). "
                    f"Consider redesigning the {metric_name} aspects of the interaction."
                )

        return report

# Example usage
def example_evaluation():
    """Example of HRI evaluation"""
    evaluator = HRIEvaluationFramework()

    # Simulate some evaluation data
    user_performance = {
        'completed_tasks': 8,
        'total_tasks': 10,
        'actual_time': 120,
        'expected_time': 100,
        'errors': 3,
        'interactions': 15
    }

    usability = evaluator.evaluate_usability(user_performance)
    print(f"Usability score: {usability:.3f}")

    survey_responses = [
        {'ease_rating': 4, 'trust_rating': 3, 'comfort_rating': 4, 'naturalness_rating': 3, 'helpfulness_rating': 4},
        {'ease_rating': 5, 'trust_rating': 4, 'comfort_rating': 5, 'naturalness_rating': 4, 'helpfulness_rating': 5}
    ]

    acceptance = evaluator.evaluate_acceptance(survey_responses)
    print(f"Acceptance score: {acceptance:.3f}")

    # Generate report
    report = evaluator.generate_evaluation_report()
    print(f"Evaluation report: {report}")

    return evaluator
```

## Summary

This chapter covered the fundamental concepts of Human-Robot Interaction (HRI), which is crucial for humanoid robots that will operate in human environments. We explored the key challenges in HRI, including anthropomorphic expectations, social norms, safety, and communication modalities.

We implemented several core HRI components including social signal detection, natural language processing, non-verbal communication, and safety systems. The hands-on exercise provided practical experience in integrating these components into a complete HRI system.

We also covered advanced topics like social navigation, which allows robots to move naturally around humans while respecting social conventions, and cultural adaptation, which allows robots to modify their behavior for different cultural contexts.

The chapter emphasized the importance of safety in HRI, covering both physical safety measures and psychological safety considerations. We discussed how to evaluate HRI systems using various metrics including usability, acceptance, safety, effectiveness, and satisfaction.

Finally, we explored advanced concepts like context-aware interaction and multi-party interaction management, which are essential for robots operating in complex social environments.

Effective HRI is critical for the success of humanoid robots in human environments, as it determines whether humans will accept, trust, and effectively collaborate with robotic systems.

## Key Takeaways

- Social presence and anthropomorphism affect human expectations and interactions
- Proxemics (personal space) is crucial for comfortable interactions
- Multiple communication modalities (speech, gestures, expressions) enhance interaction
- Safety systems must address both physical and psychological safety
- Cultural adaptation improves interaction effectiveness across different populations
- Context-aware systems provide more natural interactions
- Multi-party interaction requires sophisticated attention management
- HRI evaluation requires human-centered metrics beyond technical performance
- Transparency and predictability build trust in HRI systems

## Next Steps

In the next chapter, we'll explore multi-sensor fusion techniques that enable humanoid robots to integrate information from various sensors to better understand their environment and interact more effectively with humans and objects.

## References and Further Reading

1. Goodrich, M. A., & Schultz, A. C. (2007). Human-robot interaction: a survey. Foundations and Trends in Human-Computer Interaction.
2. Breazeal, C. (2003). Toward sociable robots. Robotics and Autonomous Systems.
3. Mataric, M. J., & Scassellati, B. (2007). Socially assistive robotics. In Medical Robotics (pp. 645-666).
4. Tapus, A., Mataric, M. J., & Scassellati, B. (2007). The grand challenges in socially assistive robotics.
5. Kidd, C. D., & Breazeal, C. (2008). Robots at home: Understanding long-term human-robot interaction. IEEE International Conference on Robotics and Automation.