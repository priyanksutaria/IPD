import cv2
import numpy as np

def detect_faces():
    # Load both frontal and profile face cascades
    face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance the image
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with adjusted parameters for frontal faces
        faces_frontal = face_cascade_frontal.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Reduced from 1.3 for better detection
            minNeighbors=3,       # Reduced from 5 for more detection
            minSize=(20, 20),     # Reduced minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect profile faces
        faces_profile = face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Also detect profile faces in flipped image for opposite side
        flipped = cv2.flip(gray, 1)
        faces_profile_flipped = face_cascade_profile.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles for frontal faces
        for (x, y, w, h) in faces_frontal:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw rectangles for profile faces
        for (x, y, w, h) in faces_profile:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw rectangles for profile faces from flipped image
        frame_width = frame.shape[1]
        for (x, y, w, h) in faces_profile_flipped:
            # Need to flip the coordinates back
            cv2.rectangle(frame, 
                        (frame_width - x - w, y), 
                        (frame_width - x, y+h), 
                        (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the detection
detect_faces()