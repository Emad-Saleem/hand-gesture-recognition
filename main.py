import math
import cv2
import mediapipe as mp


def calculate_distance(p1, p2):
  """Calculates Euclidean distance between two MediaPipe landmarks."""
  return math.hypot(p1.x - p2.x, p1.y - p2.y)


def main():
  # Initialize MediaPipe Hands and Drawing tools
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
      static_image_mode=False,
      max_num_hands=2,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7,
  )
  mp_draw = mp.solutions.drawing_utils

  # Open the default laptop webcam
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print(
        "Error: Could not open camera. Check privacy settings or physical"
        " shutter."
    )
    return

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

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand skeleton connections on the frame
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand_landmarks.landmark
        wrist = landmarks[0]
        finger_count = 0

        # --- 1. ROBUST THUMB LOGIC (Wrist-to-Tip vs Wrist-to-IP) ---
        # Thumb tip (4) vs Thumb IP joint (3) relative to wrist (0)
        dist_thumb_tip = calculate_distance(wrist, landmarks[4])
        dist_thumb_ip = calculate_distance(wrist, landmarks[3])
        if dist_thumb_tip > dist_thumb_ip * 1.1:
          finger_count += 1

        # --- 2. ROBUST 4-FINGER LOGIC (Wrist-to-Tip vs Wrist-to-PIP Ratios) ---
        # Pairs: (Tip Index, PIP Index)
        # Index: (8, 6), Middle: (12, 10), Ring: (16, 14), Pinky: (20, 18)
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]

        for tip_idx, pip_idx in finger_pairs:
          dist_tip = calculate_distance(wrist, landmarks[tip_idx])
          dist_pip = calculate_distance(wrist, landmarks[pip_idx])

          # If the tip is further from the wrist than the PIP joint (with a 1.15 buffer)
          if dist_tip > dist_pip * 1.15:
            finger_count += 1

        # Render the counted gesture for the current hand on screen
        # (Placing it near the wrist of each detected hand)
        text_x = int(wrist.x * frame.shape[1])
        text_y = int(wrist.y * frame.shape[0]) - 20
        cv2.putText(
            frame,
            f"Count: {finger_count}",
            (max(10, text_x - 50), max(30, text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    # Show live feed
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  # Clean up resources
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()