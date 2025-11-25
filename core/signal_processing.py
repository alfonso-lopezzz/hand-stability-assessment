# core/signal_processing.py

"""Simple signal processing utilities for the hand stability app.

All functions operate on plain Python containers (dicts and lists) so they
are easy to inspect and debug in Streamlit.
"""

from typing import Dict, List, Tuple
import math


def compute_displacement_time_series(
    raw_data: Dict[str, List[Tuple[float, float, float]]],
    baseline_positions: Dict[str, Tuple[float, float]]
) -> Dict[str, List[Tuple[float, float]]]:
    """Convert raw (t, x, y) into (t, displacement_from_baseline).

    - raw_data: finger -> list of (t, x, y) samples (normalized 0–1 coords)
    - baseline_positions: finger -> (x0, y0) baseline coordinates
    """

    displacement_ts: Dict[str, List[Tuple[float, float]]] = {}

    for finger, samples in raw_data.items():
        baseline = baseline_positions.get(finger)
        if not samples or baseline is None:
            displacement_ts[finger] = []
            continue

        x0, y0 = baseline
        series: List[Tuple[float, float]] = []
        for t, x, y in samples:
            dx = x - x0
            dy = y - y0
            d = math.sqrt(dx * dx + dy * dy)
            series.append((t, d))

        displacement_ts[finger] = series

    return displacement_ts


def _rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def compute_tremor_metrics(
    displacement_ts: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, float]:
    """Compute a simple tremor amplitude metric per finger.

    Here tremor is defined as RMS of displacement over the full test.
    """

    tremor: Dict[str, float] = {}
    for finger, series in displacement_ts.items():
        displacements = [d for _, d in series]
        tremor[finger] = _rms(displacements)
    return tremor


def compute_drift_metrics(
    displacement_ts: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, float]:
    """Compute slow drift away from baseline per finger.

    Defined here as: mean of the last 10% of samples minus mean of the first
    10% of samples. If there are too few samples we fall back to
    (last_value - first_value).
    """

    drift: Dict[str, float] = {}
    for finger, series in displacement_ts.items():
        n = len(series)
        if n < 2:
            drift[finger] = 0.0
            continue

        displacements = [d for _, d in series]
        k = max(1, n // 10)
        start_mean = sum(displacements[:k]) / k
        end_mean = sum(displacements[-k:]) / k
        drift[finger] = end_mean - start_mean

    return drift


def compute_fatigue_metrics(
    displacement_ts: Dict[str, List[Tuple[float, float]]]
) -> Dict[str, float]:
    """Compute a simple fatigue index per finger.

    - Split the time series into early and late halves by index.
    - Compute RMS in each half.
    - Fatigue index = late_rms / early_rms (1.0 = no change).
    """

    fatigue: Dict[str, float] = {}
    for finger, series in displacement_ts.items():
        n = len(series)
        if n < 4:
            fatigue[finger] = 1.0
            continue

        displacements = [d for _, d in series]
        mid = n // 2
        early = displacements[:mid]
        late = displacements[mid:]

        rms_early = _rms(early)
        rms_late = _rms(late)

        if rms_early <= 1e-9:
            fatigue[finger] = 1.0
        else:
            fatigue[finger] = rms_late / rms_early

    return fatigue
