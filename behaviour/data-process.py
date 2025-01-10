import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf

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
            
            if len(frames) >= self.sequence_length:
                sequences.append(frames[-self.sequence_length:])
                labels.append(label)
                frames = frames[self.sequence_length//2:]
        
        cap.release()
        return sequences, labels

    def prepare_dataset_from_videos(self, data_dir):
        """Process videos from a directory structure"""
        all_sequences = []
        all_labels = []
        behavior_mapping = {}
        
        # Process each behavior directory
        for behavior_idx, behavior in enumerate(sorted(os.listdir(data_dir))):
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
                
                if sequences:
                    all_sequences.extend(sequences)
                    all_labels.extend(labels)
        
        # Save behavior mapping
        with open('behavior_mapping.json', 'w') as f:
            json.dump(behavior_mapping, f)
            
        return np.array(all_sequences), np.array(all_labels), behavior_mapping

    def process_existing_data(self, data_dir):
        """Process existing videos and prepare the dataset"""
        print("Processing existing videos...")
        X, y, behavior_mapping = self.prepare_dataset_from_videos(data_dir)
        
        if len(X) == 0:
            raise ValueError("No sequences were extracted from the videos. Please check the video recordings.")
            
        # Ensure X is in the correct shape (samples, sequence_length, features)
        if len(X.shape) == 2:
            X = X.reshape(-1, self.sequence_length, 99)  # 99 = 33 landmarks * 3 coordinates
        
        # Convert labels to one-hot encoding
        num_classes = len(behavior_mapping)
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_one_hot, test_size=0.2, random_state=42
        )
        
        # Save the preprocessed data
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        
        print("\nDataset preparation completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print("\nBehavior mapping:")
        for idx, behavior in behavior_mapping.items():
            print(f"{idx}: {behavior}")
        
        return X_train, X_test, y_train, y_test, behavior_mapping

# Run just the processing part
if __name__ == "__main__":
    pipeline = DataPreparationPipeline(sequence_length=30)
    data_dir = 'behavior_data'  # Directory where your videos are saved
    try:
        X_train, X_test, y_train, y_test, behavior_mapping = pipeline.process_existing_data(data_dir)
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")