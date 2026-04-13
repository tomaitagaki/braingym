"""
Phase 2: Build per-second retention/survival curves.

Usage:
    python experiments/retention/02_retention_curves.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("cache/tsinghua")
SELECTED = CACHE / "selected"


def build_retention_curve(watch_times: np.ndarray, duration: float) -> dict:
    duration_int = int(np.ceil(duration))
    seconds = np.arange(1, duration_int + 1)
    n_viewers = len(watch_times)

    survival = np.array([(watch_times >= t).sum() / n_viewers for t in seconds])

    half_life_idx = np.where(survival <= 0.5)[0]
    half_life = float(seconds[half_life_idx[0]]) if len(half_life_idx) > 0 else float(duration)

    early_drop = float(survival[min(2, len(survival) - 1)] / max(survival[0], 1e-10))
    completion_rate = float(survival[-1]) if len(survival) > 0 else 0.0
    rewatch_rate = float((watch_times > duration).sum() / n_viewers)
    retention_auc = float(np.trapezoid(survival, seconds) / duration) if duration > 0 else 0.0

    slope = np.gradient(survival, seconds)
    cliff_idx = int(np.argmin(slope))

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
        "cliff_second": int(seconds[cliff_idx]),
        "cliff_magnitude": float(slope[cliff_idx]),
    }


def main():
    selected = pd.read_csv(SELECTED / "selected_videos.csv")
    interactions = pd.read_csv(CACHE / "interaction_sampled.csv")

    selected_ids = set(selected["pid"].tolist())
    interactions = interactions[interactions["pid"].isin(selected_ids)]
    print(f"Building retention curves for {len(selected)} videos ({len(interactions):,} interactions)...")

    results = {}
    for _, row in selected.iterrows():
        vid = row["pid"]
        duration = row["duration"]
        watch_times = interactions[interactions["pid"] == vid]["watch_time"].values

        if len(watch_times) < 5:
            continue

        curve = build_retention_curve(watch_times, duration)
        curve["video_id"] = int(vid)
        curve["duration"] = float(duration)
        results[str(vid)] = curve

    print(f"Built {len(results)} curves")

    # Summary
    metrics = pd.DataFrame([
        {k: v for k, v in c.items() if k not in ("seconds", "survival", "slope")}
        for c in results.values()
    ])
    print(metrics[["half_life", "completion_rate", "retention_auc", "early_drop_3s"]].describe())

    # Save
    with open(SELECTED / "retention_curves.json", "w") as f:
        json.dump(results, f)
    metrics.to_csv(SELECTED / "retention_metrics.csv", index=False)
    print(f"Saved to {SELECTED}/")


if __name__ == "__main__":
    main()
