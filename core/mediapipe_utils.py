# core/mediapipe_utils.py

"""
This module will contain helper functions to:
- Initialize MediaPipe Hands.
- Capture frames from the webcam.
- Extract fingertip landmark coordinates for THUMB, INDEX, MIDDLE.
"""

def init_mediapipe_hands():
    """
    TODO (AI / TEAM):
    - Import mediapipe.
    - Create and return a Hands object configured for real-time video.
    """
    try:
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        return mp, hands
    except Exception:
        # Let the caller handle errors (e.g., mediapipe not installed)
        raise

def capture_frame_and_landmarks(hands_context):
    """
    TODO (AI / TEAM):
    - Capture a single frame from the default webcam (OpenCV).
    - Run MediaPipe hand detection on the frame.
    - If a hand is detected:
        * Extract fingertip coordinates for THUMB, INDEX, MIDDLE.
        * Return the frame (for display) and a dict like:
          {
              "THUMB": (x_thumb, y_thumb),
              "INDEX": (x_index, y_index),
              "MIDDLE": (x_middle, y_middle),
          }
      - If no hand is detected, return None or an empty dict.
    """
    # hands_context is expected to be the tuple returned by init_mediapipe_hands(): (mp, hands)
    import cv2

    mp, hands = hands_context

    # Open the default camera, capture one frame, then release.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None, {}

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None, {}

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = {}
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        # Landmark indices: thumb_tip=4, index_finger_tip=8, middle_finger_tip=12
        idx_map = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12}
        for name, idx in idx_map.items():
            lm = hand_lms.landmark[idx]
            # Return normalized coordinates (x, y) in [0, 1]
            landmarks[name] = (lm.x, lm.y)

    cap.release()
    return frame, landmarks
