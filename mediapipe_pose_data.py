import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# GStreamer pipeline for CSI camera
gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Starting pose detection with data extraction...")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    h, w, c = frame.shape
    
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image
    results = pose.process(image)
    
    # Convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # ============ EXTRACT LANDMARK DATA ============
        landmarks = results.pose_landmarks.landmark
        
        # Example 1: Get specific landmark coordinates (normalized 0-1)
        # Landmark indices: https://google.github.io/mediapipe/solutions/pose.html
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Convert normalized coordinates to pixel coordinates
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        left_wrist_x, left_wrist_y = int(left_wrist.x * w), int(left_wrist.y * h)
        right_wrist_x, right_wrist_y = int(right_wrist.x * w), int(right_wrist.y * h)
        
        # Example 2: Calculate distances between landmarks
        def calculate_distance(landmark1, landmark2):
            return np.sqrt((landmark1.x - landmark2.x)**2 + 
                          (landmark1.y - landmark2.y)**2 + 
                          (landmark1.z - landmark2.z)**2)
        
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        
        # Example 3: Calculate angles
        def calculate_angle(point1, point2, point3):
            """Calculate angle at point2"""
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        
        # Calculate left elbow angle
        left_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        )
        
        # Example 4: Visibility and presence checks
        # visibility: likelihood landmark is visible (not occluded)
        # presence: likelihood landmark is in frame
        nose_visibility = nose.visibility
        nose_presence = nose.presence
        
        # Example 5: Display extracted data on frame
        cv2.putText(image, f"Nose: ({nose_x}, {nose_y})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"L Elbow Angle: {left_elbow_angle:.1f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Shoulder Width: {shoulder_width:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Nose Visibility: {nose_visibility:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Example 6: Get all 33 landmarks as a list
        all_landmarks = []
        for idx, landmark in enumerate(landmarks):
            all_landmarks.append({
                'id': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility,
                'presence': landmark.presence
            })
        
        # Print first landmark as example (optional - will spam console)
        # print(f"Landmark 0 (Nose): {all_landmarks[0]}")
        
        # Example 7: Check if person is raising hands
        left_hand_raised = left_wrist.y < nose.y
        right_hand_raised = right_wrist.y < nose.y
        
        if left_hand_raised and right_hand_raised:
            cv2.putText(image, "HANDS RAISED!", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    else:
        cv2.putText(image, "No person detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display
    cv2.imshow('Pose Data Extraction - Press Q to quit', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
