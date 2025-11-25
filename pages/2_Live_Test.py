import streamlit as st
from core import config
from core import mediapipe_utils
import cv2
import time
import numpy as np
import tempfile
import os

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Live Test", page_icon="📊", layout="wide")

st.title("Step 2: Live Stability Test")

if "baseline_positions" not in st.session_state:
    st.error("Please complete Calibration first.")
    st.stop()

st.markdown(
    f"""
    Hold your hand in the **same position** as during calibration.

    We will record **{config.TEST_DURATION_SECONDS} seconds** of fingertip motion
    to estimate tremor, drift, and fatigue.
    """
)

st.divider()

if "raw_time_series" not in st.session_state:
    st.session_state["raw_time_series"] = {f: [] for f in config.FINGERS_TO_TRACK}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam & Landmark Tracking (Browser Stream)")

    st.markdown(
        "This uses your **browser webcam** via WebRTC. When you start the test, "
        "frames will stream from your browser to the server, where MediaPipe "
        "tracks your fingertips."
    )

    class HandTrackingTransformer(VideoTransformerBase):
        def __init__(self):
            self.mp, self.hands = mediapipe_utils.init_mediapipe_hands()
            self.start_time = None

        def recv(self, frame):
            import av

            img = frame.to_ndarray(format="bgr24")

            if st.session_state.get("webrtc_capturing"):
                if self.start_time is None:
                    self.start_time = time.time()
                t = time.time() - self.start_time

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    lms = results.multi_hand_landmarks[0]
                    for name, idx in {"THUMB": 4, "INDEX": 8, "MIDDLE": 12}.items():
                        lm = lms.landmark[idx]
                        st.session_state["raw_time_series"][name].append((t, lm.x, lm.y))

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="hand-tracking",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=HandTrackingTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Test Control & Status")

    if st.button("▶ Start 30s Test"):
        duration = config.TEST_DURATION_SECONDS

        if not webrtc_ctx or not webrtc_ctx.state.playing:
            st.error("WebRTC stream is not active. Make sure the webcam stream above is running.")
            st.stop()

        # reset previous data
        st.session_state["raw_time_series"] = {f: [] for f in config.FINGERS_TO_TRACK}
        st.session_state["webrtc_capturing"] = True

        with st.spinner(f"Recording via browser webcam for {duration} seconds..."):
            start = time.time()
            while time.time() - start < duration:
                time.sleep(0.1)

        st.session_state["webrtc_capturing"] = False

        st.session_state["test_complete"] = True
        st.success("Test complete using browser webcam stream.")

    if st.session_state.get("test_complete"):
        st.success("Test complete! Proceed to the Results page.")
        st.caption("Use the sidebar to go to '3_Results'.")
