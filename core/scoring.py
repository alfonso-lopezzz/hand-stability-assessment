# core/scoring.py

from typing import Dict

from core import config


def _mean_metric(values: Dict[str, float]) -> float:
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def compute_stability_score(
    tremor: Dict[str, float],
    drift: Dict[str, float],
    fatigue: Dict[str, float],
) -> Dict[str, float]:
    """Compute an overall 0–100 stability score and components.

    We use simple heuristic normalization:
    - Tremor: assume 0–0.05 is typical; larger values incur more penalty.
    - Drift: use absolute value, with 0–0.05 as typical.
    - Fatigue: ideal ~1.0; distance from 1.0 incurs penalty.

    Returns a dict with:
    - "score": overall stability (0–100, higher is better)
    - "tremor_mean", "drift_mean", "fatigue_mean": aggregated metrics
    """

    tremor_mean = _mean_metric(tremor)
    drift_mean = _mean_metric(drift)
    fatigue_mean = _mean_metric(fatigue)

    # Normalize each metric into [0, 1] penalty (0 = no penalty).
    # These thresholds are heuristic and can be tuned.
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    tremor_penalty = clamp01(tremor_mean / 0.05)
    drift_penalty = clamp01(abs(drift_mean) / 0.05)
    fatigue_penalty = clamp01(abs(fatigue_mean - 1.0) / 0.5)

    total_penalty = (
        config.WEIGHT_TREMOR * tremor_penalty
        + config.WEIGHT_DRIFT * drift_penalty
        + config.WEIGHT_FATIGUE * fatigue_penalty
    )

    # Map penalty in [0, 1] to score in [0, 100]
    stability_score = (1.0 - clamp01(total_penalty)) * 100.0

    return {
        "score": stability_score,
        "tremor_mean": tremor_mean,
        "drift_mean": drift_mean,
        "fatigue_mean": fatigue_mean,
    }
