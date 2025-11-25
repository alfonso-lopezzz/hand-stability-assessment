# core/plotting_utils.py

"""
This module contains helper functions for creating plots of:
- Fingertip displacement over time
- Fatigue indices (bar chart)
- Optional correlation / similarity between fingers

All plotting should be:
- Simple
- Readable in a clinical / dashboard context
- Consistent with the color palette defined in core.config

We will generally return matplotlib Figure objects so Streamlit can display
them with st.pyplot(fig).
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from core import config


def plot_displacement_time_series(displacement_ts: Dict[str, List[Tuple[float, float]]]):
    """Create a line plot of displacement vs. time for each finger.

    Parameters
    ----------
    displacement_ts : dict
        Dictionary mapping finger name -> list of (t, displacement) samples.

        Example
        -------
        {
            "THUMB": [(t0, d0), (t1, d1), ...],
            "INDEX": [(t0, d0), (t1, d1), ...],
            "MIDDLE": [(t0, d0), (t1, d1), ...]
        }

    Returns
    -------
    matplotlib.figure.Figure
        A simple matplotlib Figure that can be shown with st.pyplot(fig).
    """

    fig, ax = plt.subplots(figsize=(6, 3))

    for finger, series in displacement_ts.items():
        if not series:
            continue
        ts = [t for t, _ in series]
        ds = [d for _, d in series]
        color = config.FINGER_COLORS.get(finger, "C0")
        ax.plot(ts, ds, label=finger.title(), color=color)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (normalized units)")
    ax.set_title("Fingertip Displacement vs. Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
