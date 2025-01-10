import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# Load the pre-trained model
model_path = "face_recognition_model.h5"
model = load_model(model_path)

# Labels for each class
labels_dict = {0: 'Priyank', 1: 'Sujal', 2: 'Sneh'}

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Confidence thresholds
high_confidence_threshold = 0.999  # For saving recognized faces
low_confidence_threshold = 0.999  # Below this, label as "Unknown"

# Directory to save recognized faces if it doesn't exist
save_dir = "recognized_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dictionary to track saved faces to prevent multiple saves
saved_faces = {}

# Preprocess function without normalization to match training
def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (50, 50))
    face_img = face_img.reshape(1, 50, 50, 1)
    return face_img

# Function to check if the face is likely half-visible or sideways
def is_half_visible_or_sideways(x, y, w, h, frame_width, frame_height):
    if x < 20 or y < 20 or (x + w) > (frame_width - 20) or (y + h) > (frame_height - 20):
        return True
    aspect_ratio = w / h
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        if is_half_visible_or_sideways(x, y, w, h, frame_width, frame_height):
            cv2.putText(frame, "Half-visible/Sideways", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            continue

        face_img = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_img)
        prediction = model.predict(processed_face)[0]

        label_idx = np.argmax(prediction)
        confidence = prediction[label_idx]

        if confidence >= low_confidence_threshold:
            label = labels_dict[label_idx]
            text = f"{label}: {confidence * 100:.2f}%"
            # Save face only if confidence exceeds high confidence threshold and hasn't been saved already
            if confidence >= high_confidence_threshold and label not in saved_faces:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"{label}_{timestamp}.jpg")
                cv2.imwrite(filename, face_img)
                saved_faces[label] = True  # Mark this label as saved
                print(f"Saved recognized face as {filename}")
        else:
            label = "Unknown"
            text = f"{label}: {confidence * 100:.2f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
