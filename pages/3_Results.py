import streamlit as st
from core import config
from core import signal_processing, scoring, plotting_utils

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

st.title("Step 3: Results & Interpretation")

if not st.session_state.get("test_complete"):
    st.error("You need to complete a Live Test before viewing results.")
    st.stop()

raw_data = st.session_state.get("raw_time_series", {})
baseline = st.session_state.get("baseline_positions", {})

if not raw_data:
    st.error("No raw time series data found. Please rerun the Live Test.")
    st.stop()

displacement = signal_processing.compute_displacement_time_series(raw_data, baseline)
tremor = signal_processing.compute_tremor_metrics(displacement)
drift = signal_processing.compute_drift_metrics(displacement)
fatigue = signal_processing.compute_fatigue_metrics(displacement)

score_info = scoring.compute_stability_score(tremor, drift, fatigue)

st.subheader("Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Mean Tremor (all fingers)",
        f"{score_info['tremor_mean']:.4f}",
        "normalized units",
    )
with col2:
    st.metric(
        "Mean Drift",
        f"{score_info['drift_mean']:.4f}",
        "normalized units",
    )
with col3:
    st.metric("Mean Fatigue Index", f"{score_info['fatigue_mean']:.2f}")
with col4:
    st.metric("Stability Score", f"{score_info['score']:.1f}", "0–100")

st.divider()

st.subheader("Displacement Over Time")
st.markdown("Plots of fingertip displacement relative to baseline for each finger.")

fig_disp = plotting_utils.plot_displacement_time_series(displacement)
st.pyplot(fig_disp, use_container_width=True)

st.subheader("Fatigue & Coordination")

st.markdown("Fatigue index by finger (values > 1 suggest increasing tremor).")

fat_cols = st.columns(len(config.FINGERS_TO_TRACK))
for i, finger in enumerate(config.FINGERS_TO_TRACK):
    val = fatigue.get(finger, 1.0)
    with fat_cols[i]:
        st.metric(finger.title(), f"{val:.2f}")

st.divider()

st.subheader("Clinical-Style Interpretation")
st.markdown(
    """
    - **Tremor amplitude**: Higher values may correspond to less steady hands.
    - **Drift**: Large drift suggests difficulty maintaining a fixed posture.
    - **Fatigue index**: Values > 1 indicate increasing tremor over the test.
    - **Stability score**: Combines these into a single 0–100 index.

    ⚠️ **Important:** This is an educational simulation, not a diagnostic tool.
    """
)
