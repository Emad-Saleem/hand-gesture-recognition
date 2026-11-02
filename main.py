import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands and Drawing tools
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # Open the default laptop webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Check privacy settings or physical shutter.")
        return

    # Tips of fingers ID in MediaPipe: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    tip_ids = [4, 8, 12, 16, 20]

    print("Camera active. Press 'q' in the video window to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror-image effect and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        results = hands.process(rgb_frame)
        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton connections on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                # Thumb logic (checking X coordinate relative to joint 3)
                if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                    finger_count += 1

                # 4 Fingers logic (checking Y coordinate of tip vs lower joint)
                for idx in range(1, 5):
                    if landmarks[tip_ids[idx]].y < landmarks[tip_ids[idx] - 2].y:
                        finger_count += 1

            # Render the counted gesture on screen
            cv2.putText(frame, f'Count: {finger_count}', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show live feed
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()