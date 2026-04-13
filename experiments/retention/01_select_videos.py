"""
Phase 1: Select ~100 videos with enough viewers for retention curves.

Usage:
    python experiments/retention/01_select_videos.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("cache/tsinghua")
OUTPUT = CACHE / "selected"


def main():
    print("Loading Tsinghua ShortVideo data...")
    df = pd.read_csv(CACHE / "interaction_sampled.csv")
    print(f"Rows: {len(df):,}, Videos: {df.pid.nunique():,}, Users: {df.user_id.nunique():,}")

    # Count viewers per video
    viewer_counts = df.groupby("pid")["user_id"].nunique().reset_index()
    viewer_counts.columns = ["pid", "viewer_count"]

    # Get video durations
    video_info = df.groupby("pid").agg(
        duration=("duration", "first"),
        mean_watch_time=("watch_time", "mean"),
        total_likes=("cvm_like", "sum"),
        total_comments=("comment", "sum"),
        total_forwards=("forward", "sum"),
    ).reset_index()

    merged = viewer_counts.merge(video_info, on="pid")
    merged["completion_rate"] = (merged["mean_watch_time"] / merged["duration"]).clip(0, 5)

    # Filter: 10-60s, >=15 viewers
    candidates = merged[
        (merged["viewer_count"] >= 15) &
        (merged["duration"] >= 10) &
        (merged["duration"] <= 60)
    ].copy()
    print(f"Candidates (>=15 viewers, 10-60s): {len(candidates)}")

    # Stratified sample by duration
    candidates["duration_bucket"] = pd.cut(
        candidates["duration"],
        bins=[10, 20, 30, 45, 60],
        labels=["10-20s", "20-30s", "30-45s", "45-60s"],
    )

    selected = []
    per_bucket = 25
    for bucket, group in candidates.groupby("duration_bucket", observed=True):
        group_sorted = group.sort_values("viewer_count", ascending=False)
        n = min(per_bucket, len(group_sorted))
        selected.append(group_sorted.head(n))
        print(f"  {bucket}: {n} / {len(group_sorted)}")

    result = pd.concat(selected, ignore_index=True).head(100)
    print(f"\nSelected {len(result)} videos")
    print(f"  Viewers: {result.viewer_count.min()}-{result.viewer_count.max()} (mean {result.viewer_count.mean():.0f})")
    print(f"  Duration: {result.duration.min():.0f}-{result.duration.max():.0f}s")

    # Save
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "selected_videos.csv", index=False)
    video_ids = result["pid"].tolist()
    with open(OUTPUT / "video_ids.json", "w") as f:
        json.dump(video_ids, f)

    print(f"Saved {len(video_ids)} video IDs to {OUTPUT}/video_ids.json")


if __name__ == "__main__":
    main()
