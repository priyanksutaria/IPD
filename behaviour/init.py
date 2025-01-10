import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_dataset(num_samples=100, frame_count=30, frame_size=(64, 64), num_classes=3):
    """
    Generate a synthetic dataset with random image sequences and labels.
    
    Parameters:
        num_samples (int): Number of video samples.
        frame_count (int): Number of frames per video.
        frame_size (tuple): Dimension of each frame (height, width).
        num_classes (int): Number of behavior classes.
    
    Returns:
        X (ndarray): Video data of shape (num_samples, frame_count, height, width, channels).
        y (ndarray): Labels corresponding to the samples.
    """
    # Generate random image sequences
    X = np.random.rand(num_samples, frame_count, frame_size[0], frame_size[1], 3)
    # Generate random labels
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y

# Parameters
num_samples = 1000
frame_count = 30
frame_size = (64, 64)
num_classes = 3

# Generate synthetic dataset
X, y = generate_synthetic_dataset(num_samples, frame_count, frame_size, num_classes)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
