"""
Phase 2: Build per-second survival/retention curves for selected videos.

For each video of duration D seconds:
    S(t) = fraction of viewers with watch_time >= t, for t in [1, D]

Usage:
    uv run python experiments/retention/02_retention_curves.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("cache/tsinghua")
SELECTED = CACHE / "selected"


def build_retention_curve(watch_times: np.ndarray, duration: float) -> dict:
    """Build a per-second survival curve from individual watch times.

    Args:
        watch_times: Array of watch_time values (seconds) for one video
        duration: Video duration in seconds

    Returns:
        dict with:
            - seconds: [1, 2, ..., D]
            - survival: S(t) at each second
            - scalar metrics (half_life, early_drop, completion_rate, etc.)
    """
    duration_int = int(np.ceil(duration))
    seconds = np.arange(1, duration_int + 1)
    n_viewers = len(watch_times)

    # S(t) = fraction of viewers still watching at time t
    survival = np.array([(watch_times >= t).sum() / n_viewers for t in seconds])

    # Derived scalar metrics
    half_life_idx = np.where(survival <= 0.5)[0]
    half_life = float(seconds[half_life_idx[0]]) if len(half_life_idx) > 0 else float(duration)

    early_drop = float(survival[min(2, len(survival) - 1)] / survival[0]) if survival[0] > 0 else 0.0
    completion_rate = float(survival[-1]) if len(survival) > 0 else 0.0
    rewatch_rate = float((watch_times > duration).sum() / n_viewers)
    retention_auc = float(np.trapezoid(survival, seconds) / duration) if duration > 0 else 0.0

    # Slope: first derivative of survival curve (negative = drop-off)
    slope = np.gradient(survival, seconds)

    # Cliff detection: where does the steepest drop happen?
    cliff_idx = int(np.argmin(slope))
    cliff_second = int(seconds[cliff_idx])
    cliff_magnitude = float(slope[cliff_idx])

    return {
        "seconds": seconds.tolist(),
        "survival": survival.tolist(),
        "slope": slope.tolist(),
        "n_viewers": n_viewers,
        "half_life": half_life,
        "early_drop_3s": early_drop,
        "completion_rate": completion_rate,
        "rewatch_rate": rewatch_rate,
        "retention_auc": retention_auc,
        "cliff_second": cliff_second,
        "cliff_magnitude": cliff_magnitude,
    }


def main():
    # Load selected videos
    selected = pd.read_csv(SELECTED / "selected_videos.csv")
    interactions = pd.read_csv(CACHE / "interaction_filtered.csv")

    print(f"Building retention curves for {len(selected)} videos...")

    # Filter interactions to selected videos only
    selected_ids = set(selected["video_id"].tolist())
    interactions = interactions[interactions["video_id"].isin(selected_ids)]
    print(f"Filtered interactions: {len(interactions):,}")

    # Build curves
    results = {}
    for _, row in selected.iterrows():
        vid = row["video_id"]
        duration = row["duration"]

        watch_times = interactions[interactions["video_id"] == vid]["watch_time"].values

        if len(watch_times) < 5:
            print(f"  Skipping {vid}: only {len(watch_times)} viewers")
            continue

        curve = build_retention_curve(watch_times, duration)
        curve["video_id"] = vid
        curve["duration"] = duration
        results[str(vid)] = curve

    print(f"\nBuilt {len(results)} retention curves")

    # Summary statistics
    metrics = pd.DataFrame([
        {
            "video_id": v["video_id"],
            "duration": v["duration"],
            "n_viewers": v["n_viewers"],
            "half_life": v["half_life"],
            "early_drop_3s": v["early_drop_3s"],
            "completion_rate": v["completion_rate"],
            "rewatch_rate": v["rewatch_rate"],
            "retention_auc": v["retention_auc"],
            "cliff_second": v["cliff_second"],
        }
        for v in results.values()
    ])

    print("\nRetention metrics summary:")
    print(metrics[["half_life", "completion_rate", "rewatch_rate", "retention_auc"]].describe())

    # Save
    output_path = SELECTED / "retention_curves.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics.to_csv(SELECTED / "retention_metrics.csv", index=False)

    print(f"\nSaved curves: {output_path}")
    print(f"Saved metrics: {SELECTED / 'retention_metrics.csv'}")
    print(f"Next: run 03_modal_tribe.py to get brain timeseries")


if __name__ == "__main__":
    main()
