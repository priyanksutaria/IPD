import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = "face_recognition_model.h5"
model = load_model(model_path)

# Labels for each class
labels_dict = {0: 'Priyank', 1: 'Sujal', 2: 'Sneh'}

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up video capture (0 for the default camera, or replace with video file path)
cap = cv2.VideoCapture(0)

# Confidence threshold to classify as known vs unknown
confidence_threshold = 0.7  # Adjust this value as needed

# Updated preprocess function without normalization to match training
def preprocess_face(face_img):
    # Convert to grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Resize to 50x50 as expected by the model
    face_img = cv2.resize(face_img, (50, 50))
    # Reshape to match model input shape
    face_img = face_img.reshape(1, 50, 50, 1)
    return face_img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract each detected face
        face_img = frame[y:y+h, x:x+w]

        # Preprocess face (convert to grayscale, resize and reshape)
        processed_face = preprocess_face(face_img)

        # Get model prediction
        prediction = model.predict(processed_face)[0]
        
        # Debugging: Print raw model output
        print(f"Raw model prediction: {prediction}")

        # Get the highest probability label
        label_idx = np.argmax(prediction)
        confidence = prediction[label_idx]

        # Only label if confidence is above the threshold
        if confidence > confidence_threshold:
            label = labels_dict[label_idx]
            text = f"{label}: {confidence * 100:.2f}%"
        else:
            text = "Unknown"  # Mark as unknown if below confidence threshold

        # Draw rectangle around face and display label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
