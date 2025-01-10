import cv2
from ultralytics import YOLO

# Load the YOLOv8 model pre-trained on COCO dataset
model = YOLO('yolov8s.pt')  # You can use 'yolov8n.pt' (nano) for faster inference on low-resource devices

# Define the class ID for mobile phones in the COCO dataset (ID 67 for "cell phone")
MOBILE_CLASS_ID = 67

CONFIDENCE_THRESHOLD = 0.5  # Increase this value to make the model more strict

def detect_mobile_in_frame(model, frame):
    # Perform inference using YOLOv8
    results = model(frame)[0]
    
    # Loop through detected objects
    for result in results.boxes:
        cls = int(result.cls)  # Get the class ID
        if cls == MOBILE_CLASS_ID and result.conf >= CONFIDENCE_THRESHOLD:
            # Extract bounding box and confidence score
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            confidence = result.conf.item()  # Convert the tensor to a scalar value
            
            # Draw a rectangle around the detected mobile phone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label and confidence score to the bounding box
            label = f"Mobile: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

def real_time_mobile_detection():
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera")
        return
    
    # Get the frame width and height to set up the VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the output video file and codec
    output_filename = 'mobile_detection_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change this codec if needed
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera")
            break

        # Detect mobiles in the current frame
        processed_frame = detect_mobile_in_frame(model, frame)

        # Display the processed frame with detections
        cv2.imshow('Mobile Detection - YOLOv8', processed_frame)

        # Save the processed frame to the output video
        out.write(processed_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and VideoWriter, and close OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run the real-time mobile detection and save the results
real_time_mobile_detection()
