from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
from scipy.stats import skew

from .constants import CHANNELS, GCS_SCORE


SUBPERIODS = (
    ("first100", 0.0, 1.0),
    ("first10", 0.0, 0.1),
    ("first25", 0.0, 0.25),
    ("first50", 0.0, 0.5),
    ("last10", 0.9, 1.0),
    ("last25", 0.75, 1.0),
    ("last50", 0.5, 1.0),
)
STAT_NAMES = ("min", "max", "mean", "std", "skew", "len")


def numeric_event_value(channel: str, value: object) -> float | None:
    if channel == "Capillary refill rate":
        return float(value)
    if channel == "Glascow coma scale total":
        return float(value)
    if channel.startswith("Glascow coma scale"):
        return GCS_SCORE.get((channel, str(value)))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _statistics(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full(6, np.nan, dtype=np.float32)
    # scipy intentionally returns NaN for constant/nearly constant windows. That
    # undefined skew is a structural missing feature handled by the train-only
    # imputer, so avoid emitting the same precision warning for every stay.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skew_value = skew(values)
    return np.asarray(
        [
            np.min(values),
            np.max(values),
            np.mean(values),
            np.std(values),
            skew_value,
            float(values.size),
        ],
        dtype=np.float32,
    )


def extract_lr_features(events: Iterable[tuple[float, str, object]]) -> np.ndarray:
    """Reproduce the benchmark's 17 x 7 x 6 sparse-event feature extractor."""
    by_channel: dict[str, list[tuple[float, float]]] = {channel: [] for channel in CHANNELS}
    for hour, channel, value in events:
        number = numeric_event_value(channel, value)
        if number is not None:
            by_channel[channel].append((float(hour), number))

    extracted: list[np.ndarray] = []
    for channel in CHANNELS:
        points = sorted(by_channel[channel])
        if not points:
            extracted.extend(np.full(6, np.nan, dtype=np.float32) for _ in SUBPERIODS)
            continue
        start, end = points[0][0], points[-1][0]
        span = end - start
        for _, left_fraction, right_fraction in SUBPERIODS:
            left = start + span * left_fraction
            right = start + span * right_fraction
            values = np.asarray(
                [value for hour, value in points if left - 1e-6 < hour < right + 1e-6],
                dtype=np.float64,
            )
            extracted.append(_statistics(values))
    result = np.concatenate(extracted).astype(np.float32)
    if result.shape != (714,):
        raise AssertionError(f"Expected 714 LR features, got {result.shape}")
    return result


def lr_feature_names() -> list[str]:
    return [
        f"{channel}|{period}|{stat}"
        for channel in CHANNELS
        for period, _, _ in SUBPERIODS
        for stat in STAT_NAMES
    ]
