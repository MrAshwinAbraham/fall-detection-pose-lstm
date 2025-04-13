import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import collections
import warnings
warnings.filterwarnings('ignore')

#print("TensorFlow version:", tf.__version__)

# Load the trained LSTM model
model_path = 'model/fall_detection_model.h5'  
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit()

model = load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy')  # Silence metrics warning
print("Model loaded successfully!")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Parameters
sequence_length = 60
smoothing_window = 5  # Moving average filter window size
keypoint_buffer = collections.deque(maxlen=sequence_length)
prediction_buffer = collections.deque(maxlen=smoothing_window)

# Track last status to detect changes
last_status = None

def moving_average(data, window_size=10):
    """Apply moving average filter to smooth predictions."""
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(list(data)[-window_size:])

def process_video(source, is_webcam=True):
    """Process video from webcam or file for fall detection."""
    global last_status

    if is_webcam:
        cap = cv2.VideoCapture(0)
        source_name = "Webcam"
    else:
        if not os.path.exists(source):
            print(f"Error: Video file {source} not found.")
            return
        cap = cv2.VideoCapture(source)  # Video file
        source_name = os.path.basename(source)

    if not cap.isOpened():
        print(f"Error: Couldn’t open {source_name}.")
        return

    print(f"Processing {source_name}... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                print("Error: Couldn’t read frame from webcam.")
            else:
                print(f"End of video: {source_name}")
            break

        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Extract keypoints if pose detected
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoint_buffer.append(keypoints)

            # Draw keypoints on frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Predict when buffer is full
            if len(keypoint_buffer) == sequence_length:
                sequence = np.array(keypoint_buffer).reshape(1, sequence_length, -1)  # [1, 60, 132]
                prediction = model.predict(sequence, verbose=0)[0][0]  # Raw probability

                # Apply moving average smoothing
                prediction_buffer.append(prediction)
                smoothed_prediction = moving_average(prediction_buffer, smoothing_window)

                # Determine status
                label = "Fall" if smoothed_prediction > 0.6 else "Non-Fall"
                confidence = smoothed_prediction if smoothed_prediction > 0.5 else 1 - smoothed_prediction

                # Print alert message only when status changes
                if last_status != label and label=="Fall":
                    print(f"⚠️ ALERT: {label} Detected!")

                last_status = label

                # Display prediction on frame
                text = f"{label} ({confidence:.2f})"
                color = (0, 0, 255) if label == "Fall" else (0, 255, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show frame
        cv2.imshow(f'Fall Detection - {source_name}', frame)

        # Stop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    choice = input("Enter 'w' for webcam or 'v' for video file: ").lower()
    
    if choice == 'w':
        print("Starting webcam detection...")
        process_video(None, is_webcam=True)
    elif choice == 'v':
        video_path = input("Enter video file path (e.g., 'videos/fall_sample.mp4'): ")
        print(f"Starting video detection for {video_path}...")
        process_video(video_path, is_webcam=False)
    else:
        print("Invalid choice. Use 'w' for webcam or 'v' for video.")

# Cleanup
pose.close()
print("Program ended.")
