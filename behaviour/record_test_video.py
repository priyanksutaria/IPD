import cv2
import os
from datetime import datetime

def record_test_video(behavior, duration=5, save_dir='test_videos', camera_index=0):
    """
    Record a test video for a specific behavior.
    
    Args:
    - behavior (str): The behavior label for the video.
    - duration (int): Duration of the video in seconds.
    - save_dir (str): Directory to save the video.
    - camera_index (int): Index of the camera to use (default: 0).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the video file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(save_dir, f"{behavior}_{timestamp}.mp4")
    
    # Capture video from the webcam
    cap = cv2.VideoCapture(camera_index)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # You can adjust this based on your requirements
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Recording video for behavior: {behavior}")
    print(f"Press 'q' to stop early.")
    
    # Record for the specified duration
    start_time = datetime.now()
    while (datetime.now() - start_time).total_seconds() < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video.")
            break
        
        # Add a recording indicator and behavior label
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"Recording: {behavior}", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the video feed
        cv2.imshow('Recording Test Video', frame)
        
        # Write the frame to the video file
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Test video saved at: {video_path}")

# Example usage
if __name__ == "__main__":
    behavior = input("Enter the behavior (e.g., sitting, standing): ").strip()
    duration = int(input("Enter the duration of the video in seconds: ").strip())
    record_test_video(behavior, duration=duration)
