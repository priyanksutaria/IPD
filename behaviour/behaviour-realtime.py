import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import cv2
import mediapipe as mp
import json

class StudentBehaviorAnalysis:
    def __init__(self, num_classes=6, sequence_length=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the RNN model"""
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', 
                 input_shape=(self.sequence_length, 33*3)),  # 33 landmarks, each with x,y,z
            Dropout(0.2),
            LSTM(32, return_sequences=False, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        return model
    
    def extract_keypoints(self, frame):
        """Extract pose keypoints from a single frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks on the frame
            for landmark in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            pose = np.array([[landmark.x, landmark.y, landmark.z] 
                           for landmark in results.pose_landmarks.landmark]).flatten()
            return pose
        return np.zeros(33*3)  # Return zeros if no pose detected
    
    def preprocess_video(self, video_path):
        """Process video and extract pose sequences"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            frames.append(keypoints)
            
        cap.release()
        
        # Pad sequence if needed
        while len(frames) < self.sequence_length:
            frames.append(np.zeros(33*3))
            
        return np.array(frames)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        self.model.fit(
            X_train, 
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        return loss, accuracy
    
    def predict(self, video_path):
        """Predict behavior from a video"""
        sequence = self.preprocess_video(video_path)
        sequence = np.expand_dims(sequence, axis=0)
        
        prediction = self.model.predict(sequence)[0]
        return prediction
    
    def real_time_analysis(self, behaviors, camera_index=0):
        """Perform real-time behavior analysis using webcam"""
        cap = cv2.VideoCapture(camera_index)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract keypoints and overlay on the frame
            keypoints = self.extract_keypoints(frame)
            frames.append(keypoints)
            
            # Keep only the last sequence_length frames
            if len(frames) > self.sequence_length:
                frames = frames[-self.sequence_length:]
            
            # Make prediction when we have enough frames
            if len(frames) == self.sequence_length:
                sequence = np.array(frames)
                sequence = np.expand_dims(sequence, axis=0)
                prediction = self.model.predict(sequence)[0]
                
                # Display prediction on frame
                behavior = np.argmax(prediction)
                confidence = prediction[behavior]
                cv2.putText(frame, f"Behavior: {behaviors[behavior]}, Conf: {confidence:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame with keypoints
            cv2.imshow('Behavior Analysis with Keypoints', frame)
            
            # Quit real-time analysis with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Load behavior mapping
    with open('behavior_mapping.json', 'r') as f:
        behaviors = json.load(f)
        behaviors = {int(k): v for k, v in behaviors.items()}
    
    # Load the preprocessed dataset
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Initialize the analyzer
    analyzer = StudentBehaviorAnalysis(num_classes=len(behaviors))
    
    # Train the model
    print("Training the model...")
    analyzer.train(X_train, y_train, epochs=20, batch_size=32)
    
    # Save the trained model
    analyzer.model.save('student_behavior_model.h5')
    print("Model saved successfully.")
    
    # Evaluate the model
    print("\nEvaluating the model...")
    analyzer.evaluate(X_test, y_test)
    
    # Start real-time analysis
    print("\nStarting real-time analysis...")
    analyzer.real_time_analysis(behaviors)
