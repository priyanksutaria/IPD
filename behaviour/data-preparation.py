import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

class DataPreparationPipeline:
    def __init__(self, sequence_length=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_length = sequence_length
        
    def extract_keypoints(self, frame):
        """Extract pose keypoints from a single frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y, landmark.z] 
                           for landmark in results.pose_landmarks.landmark]).flatten()
            return pose
        return np.zeros(33*3)

    def process_video(self, video_path, label):
        """Process a single video and extract sequences"""
        sequences = []
        labels = []
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            keypoints = self.extract_keypoints(frame)
            frames.append(keypoints)
            
            # Create sequences of fixed length with overlap
            if len(frames) >= self.sequence_length:
                sequences.append(frames[-self.sequence_length:])
                labels.append(label)
                # Use 50% overlap for next sequence
                frames = frames[self.sequence_length//2:]
        
        cap.release()
        return sequences, labels

    def prepare_dataset_from_videos(self, data_dir):
        """
        Process videos from a directory structure:
        data_dir/
            behavior_1/
                video1.mp4
                video2.mp4
            behavior_2/
                video1.mp4
                video2.mp4
        """
        all_sequences = []
        all_labels = []
        behavior_mapping = {}
        
        # Process each behavior directory
        for behavior_idx, behavior in enumerate(os.listdir(data_dir)):
            behavior_path = os.path.join(data_dir, behavior)
            if not os.path.isdir(behavior_path):
                continue
                
            behavior_mapping[behavior_idx] = behavior
            print(f"Processing {behavior} videos...")
            
            # Process each video in the behavior directory
            for video_file in tqdm(os.listdir(behavior_path)):
                if not video_file.endswith(('.mp4', '.avi', '.mov')):
                    continue
                    
                video_path = os.path.join(behavior_path, video_file)
                sequences, labels = self.process_video(video_path, behavior_idx)
                
                all_sequences.extend(sequences)
                all_labels.extend(labels)
        
        # Save behavior mapping
        with open('behavior_mapping.json', 'w') as f:
            json.dump(behavior_mapping, f)
            
        return np.array(all_sequences), np.array(all_labels), behavior_mapping

    def record_training_data(self, save_dir, behavior, duration=5, camera_index=0):
        """Record video for training data"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        behavior_dir = os.path.join(save_dir, behavior)
        if not os.path.exists(behavior_dir):
            os.makedirs(behavior_dir)
            
        cap = cv2.VideoCapture(camera_index)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(behavior_dir, f"{behavior}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        start_time = datetime.now()
        frames_captured = 0
        
        print(f"Recording {behavior} for {duration} seconds...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add recording indicator
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Recording: {behavior}", (50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            cv2.imshow('Recording', frame)
            frames_captured += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Check if duration has elapsed
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Saved video to: {video_path}")
        return video_path

    def prepare_training_data(self, data_dir, test_size=0.2):
        """Prepare training and testing datasets"""
        # Process all videos and get sequences
        X, y, behavior_mapping = self.prepare_dataset_from_videos(data_dir)
        
        # Convert labels to one-hot encoding
        num_classes = len(behavior_mapping)
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_one_hot, test_size=test_size, random_state=42
        )
        
        # Save the preprocessed data
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        
        return X_train, X_test, y_train, y_test, behavior_mapping

# Example usage
def create_training_dataset():
    behaviors = [
        'SITTING',
        'STANDING',
        'WALKING',
        'RAISING_HAND',
        'WRITING',
        'USING_PHONE'
    ]
    
    # Initialize the pipeline
    pipeline = DataPreparationPipeline(sequence_length=30)
    
    # Create data directory
    data_dir = 'behavior_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Record training data for each behavior
    for behavior in behaviors:
        print(f"\nRecording {behavior}")
        input(f"Press Enter when ready to record {behavior}...")
        pipeline.record_training_data(data_dir, behavior, duration=5)
    
    # Prepare the final dataset
    X_train, X_test, y_train, y_test, behavior_mapping = pipeline.prepare_training_data(data_dir)
    
    print("\nDataset preparation completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("\nBehavior mapping:")
    for idx, behavior in behavior_mapping.items():
        print(f"{idx}: {behavior}")
    
    return X_train, X_test, y_train, y_test, behavior_mapping

# Run the data preparation
if __name__ == "__main__":
    create_training_dataset()
