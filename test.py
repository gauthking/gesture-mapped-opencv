import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe hands instance
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Define the mapping of hand gestures to labels
gesture_labels = {
    'thumbs_up': 'Thumbs Up',
    'peace': 'Peace',
    'fist': 'Fist',
    'open_hand': 'Open Hand',
    'three_fingers': 'Three Fingers',
    'four_fingers': 'Four Fingers',
    'ok': 'OK',
    'rock': 'Rock',
    'palm_face': 'Palm to Face',
    'crossed_fingers': 'Crossed Fingers',
    # Add more gestures and labels as needed
}

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(frame_rgb)

    # Draw hand landmarks on the frame if detection is successful
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect specific hand gestures
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < index_tip.y:
                cv2.putText(
                    frame, gesture_labels['thumbs_up'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Add more gesture detection logic for other gestures
            elif thumb_tip.y > index_tip.y and thumb_tip.x > index_tip.x:
                cv2.putText(
                    frame, gesture_labels['peace'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif thumb_tip.y > index_tip.y and thumb_tip.x < index_tip.x:
                cv2.putText(
                    frame, gesture_labels['fist'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif thumb_tip.y > index_tip.y:
                cv2.putText(
                    frame, gesture_labels['open_hand'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Add more gesture detection logic for other gestures

    # Display output
    cv2.imshow('MediaPipe Hand Detection',
               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
