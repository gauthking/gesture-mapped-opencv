import cv2
import json
import mediapipe as mp
import paho.mqtt.publish as publish

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe hands instance
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

options = {
    1: {"name": "Light Intensity", "min_value": 0, "max_value": 100},
    2: {"name": "TV Volume", "min_value": 0, "max_value": 100},
    3: {"name": "AC Temperature", "min_value": 16, "max_value": 30}
}

show_options = True

feedback_message = ""
last_intensity = 0

selected_option = None

mqtt_broker = "localhost"
mqtt_port = 1883
mqtt_topic = "gesture-control"


def send_data_to_mqtt(option, intensity):
    try:
        data = {
            "intensity": intensity,
            "selected_option": option
        }
        payload = json.dumps(data)
        publish.single(mqtt_topic, payload,
                       hostname=mqtt_broker, port=mqtt_port)
        print("Data sent successfully via MQTT")
    except Exception as e:
        print(f"An error occurred while sending data via MQTT: {e}")


def process_frame(frame):
    global last_intensity, selected_option, show_options, feedback_message
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process frame with MediaPipe
    results = hands.process(frame_rgb)

    # draw hand landmarks on the frame if detection is successful using mediapipe hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if show_options:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # check if index finger tip is within the bounding box of any option
                for key, option in options.items():
                    if (option['bbox'][0] < index_finger_tip.x * frame.shape[1] < option['bbox'][2] and
                            option['bbox'][1] < index_finger_tip.y * frame.shape[0] < option['bbox'][3]):
                        # If finger is pointed to an option, switch to a new screen
                        show_options = False
                        feedback_message = f"Selected option: {options[key]['name']}"
                        selected_option = key
                        break
                else:
                    feedback_message = ""
                    show_options = True

            # detect palm and control intensity if options are not shown
            else:
                # check if both hands are detected
                if len(results.multi_hand_landmarks) == 2:
                    print("Both hands detected")
                    # if both hands are detected, send data via MQTT
                    if selected_option is not None:
                        send_data_to_mqtt(selected_option, last_intensity)
                    show_options = True

                else:
                    # calculate distance between thumb and index finger for pinch detection
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    distance_thumb_index = (
                        (thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

                    # map the distance to intensity range (0% to 100%)
                    last_intensity = max(
                        0, min(int((distance_thumb_index) * 100), 100))

                    # update the intensity of the selected option
                    if selected_option is not None:
                        options[selected_option]['current_value'] = last_intensity

                    # display intensity on the screen
                    if selected_option is not None:
                        label = "Intensity" if selected_option == 1 else "Volume" if selected_option == 2 else "Temperature"
                        cv2.putText(frame, f"{label}: {last_intensity}%", (
                            50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if show_options:
        # display options on the frame
        y = 150
        for key, option in options.items():
            cv2.putText(frame, f"{key}. {option['name']}",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            option['bbox'] = (
                50, y - 30, 50 + len(option['name']) * 20, y + 10)
            y += 50
    else:
        # display feedback message on the new screen
        cv2.putText(frame, feedback_message, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if selected_option is not None:
            label = "Intensity" if selected_option == 1 else "Volume" if selected_option == 2 else "Temperature"
            cv2.putText(frame, f"{label}: {last_intensity}%",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # display output
    cv2.namedWindow('Gesture-Controlled System', cv2.WINDOW_NORMAL)
    cv2.imshow('Gesture-Controlled System', frame)

    # update selected_option to None only if show_options is True
    if show_options:
        selected_option = None


def main():
    startup = False  # Set to True once thumbs-up is detected
    print("SHOW THUMBS UP GESTURE TO START THE APPLICATION..")
    while True:
        # read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # If startup is False, look for thumbs-up gesture to start the app
        if not startup:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y
                    if thumb_up:
                        print("Thumbs-up detected. Starting app...")
                        startup = True
                        process_frame(frame)

            # Display instructions until thumbs-up is detected
            cv2.putText(frame, "Show Thumbs-up Gesture to Start", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            process_frame(frame)

            # # check for thumbs-down gesture to kill the app
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = hands.process(frame_rgb)
            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         thumb_down = hand_landmarks.landmark[4].y > hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y
            #         if thumb_down:
            #             print("Thumbs-down detected. Exiting...")
            #             cap.release()
            #             cv2.destroyAllWindows()
            #             exit()

        # check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
