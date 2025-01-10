import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "face_recognition_model.h5"  # Replace with the path to your model
model = load_model(model_path)

# Labels mapping (you should have this from your training)
labels_dict = {0: 'Priyank', 1: 'Sujal'}

# Directory containing the images to classify
image_folder = "Images for visualization"  # Replace with the path to your folder containing images

# Function to preprocess image to match model input
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (50, 50))  # Resize to 50x50
    img = img / 255.0  # Normalize
    img = img.reshape(1, 50, 50, 1)  # Reshape to match the input shape for the model
    return img

# Loop through the images in the folder and classify each
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Get prediction
    prediction = model.predict(processed_img)[0]
    label_idx = np.argmax(prediction)
    confidence = prediction[label_idx]
    
    # Get the label name and confidence score
    label = labels_dict[label_idx]
    print(f"Image: {img_name} | Predicted: {label} | Confidence: {confidence * 100:.2f}%")
