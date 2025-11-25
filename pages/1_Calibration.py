import streamlit as st
from core import config
from core import mediapipe_utils
import time
import numpy as np
import cv2

st.set_page_config(page_title="Calibration", page_icon="🎯", layout="wide")

st.title("Step 1: Calibration")

st.markdown(
    """
    In this step, we capture a **baseline reference position** for your fingertips.

    **Instructions:**
    - Sit at a comfortable distance from your webcam.
    - Raise your **dominant hand** so it is fully visible.
    - Extend your **thumb, index, and middle fingers**.
    - Try to hold your hand as steady as possible inside the on-screen guide.
    """
)

st.warning("This tool is for educational purposes only and does **not** diagnose any condition.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Hand Preview (Browser)")
    st.markdown("Use the browser's webcam permission to see a live preview.")

    # Simple browser-based preview using Streamlit's camera_input
    preview = st.camera_input("Live preview (not used for calibration data)")
    if preview is not None:
        st.image(preview)

with col2:
    st.subheader("Calibration Control")

    st.markdown(
        f"""
        When you are ready and your hand is steady, click the button below.
        We will record **{config.CALIBRATION_DURATION_SECONDS} seconds** of baseline data.
        """
    )

    if st.button("▶ Run Calibration"):
        duration = config.CALIBRATION_DURATION_SECONDS

        st.info(
            "For calibration in Codespaces, we will capture a small set of "
            "browser webcam snapshots while you hold your hand steady."
        )

        samples = {f: [] for f in config.FINGERS_TO_TRACK}
        mp, hands = mediapipe_utils.init_mediapipe_hands()

        # Ask the user to take a few steady photos over the calibration window.
        num_shots = 3
        for i in range(num_shots):
            st.markdown(f"**Snapshot {i+1} of {num_shots}** – hold your hand steady and click 'Take Photo'.")
            img_file = st.camera_input(f"Calibration snapshot {i+1}")
            if img_file is None:
                st.warning("No image captured for this snapshot; skipping.")
                continue

            img_bytes = img_file.getvalue()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                st.warning("Could not decode image; skipping.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                for name, idx in {"THUMB": 4, "INDEX": 8, "MIDDLE": 12}.items():
                    lm = lms.landmark[idx]
                    samples[name].append((lm.x, lm.y))
            else:
                st.warning("No hand detected in this snapshot; try to keep your hand fully visible.")

        # compute mean baseline positions from all snapshots
        baseline = {}
        for f in config.FINGERS_TO_TRACK:
            arr = np.array(samples[f])
            if arr.size == 0:
                baseline[f] = None
            else:
                baseline[f] = tuple(arr.mean(axis=0).tolist())

        st.session_state["baseline_positions"] = baseline
        st.session_state["calibration_complete"] = True
        st.success("Calibration complete using browser-based snapshots!")

if st.session_state.get("calibration_complete"):
    st.success("Calibration complete! You can move on to the Live Test page.")
    st.caption("Use the sidebar to go to '2_Live_Test'.")
