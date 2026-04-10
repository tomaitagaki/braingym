"""
Phase 1: Download interaction data and select ~100 videos with enough viewers
for meaningful per-second retention curves.

Dataset: Tsinghua ShortVideo (WWW 2025)
Download: http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset
Credentials: videodata / ShortVideo@10000

Usage:
    # First, manually download these files to cache/tsinghua/:
    #   - interaction_filtered.csv
    #   - video_info.csv
    # Then:
    uv run python experiments/retention/01_select_videos.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("cache/tsinghua")
OUTPUT = Path("cache/tsinghua/selected")


def load_interactions() -> pd.DataFrame:
    """Load the interaction CSV. Expected columns:
    user_id, video_id, watch_time, effective_view, cvm_like,
    comment, follow, collect, forward, hate, exposed_time, p_date, p_hour
    """
    path = CACHE / "interaction_filtered.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from:\n"
            "  http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset\n"
            "  User: videodata  Pass: ShortVideo@10000"
        )
    return pd.read_csv(path)


def load_video_info() -> pd.DataFrame:
    """Load video metadata. Expected columns include video_id, duration, etc."""
    path = CACHE / "video_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download alongside interactions.")
    return pd.read_csv(path)


def select_videos(
    interactions: pd.DataFrame,
    video_info: pd.DataFrame,
    min_viewers: int = 30,
    target_count: int = 100,
    duration_range: tuple[int, int] = (10, 60),
) -> pd.DataFrame:
    """Select videos with enough viewers, stratified by duration.

    Returns DataFrame with columns:
        video_id, duration, viewer_count, mean_watch_time, completion_rate
    """
    # Count unique viewers per video
    viewer_counts = (
        interactions.groupby("video_id")["user_id"]
        .nunique()
        .reset_index()
        .rename(columns={"user_id": "viewer_count"})
    )

    # Merge with video info for duration
    merged = viewer_counts.merge(video_info[["video_id", "duration"]], on="video_id", how="inner")

    # Filter
    mask = (
        (merged["viewer_count"] >= min_viewers)
        & (merged["duration"] >= duration_range[0])
        & (merged["duration"] <= duration_range[1])
    )
    candidates = merged[mask].copy()
    print(f"Candidates with ≥{min_viewers} viewers and {duration_range}s duration: {len(candidates)}")

    if len(candidates) == 0:
        # Relax constraints
        print("No candidates found. Relaxing min_viewers to 15...")
        mask = (
            (merged["viewer_count"] >= 15)
            & (merged["duration"] >= duration_range[0])
            & (merged["duration"] <= duration_range[1])
        )
        candidates = merged[mask].copy()
        print(f"Candidates with ≥15 viewers: {len(candidates)}")

    # Compute summary engagement stats
    engagement = (
        interactions.groupby("video_id")
        .agg(
            mean_watch_time=("watch_time", "mean"),
            median_watch_time=("watch_time", "median"),
            total_likes=("cvm_like", "sum"),
            total_comments=("comment", "sum"),
            total_shares=("forward", "sum"),
        )
        .reset_index()
    )
    candidates = candidates.merge(engagement, on="video_id", how="left")

    # Merge duration for completion rate calc
    candidates["completion_rate"] = (
        candidates["mean_watch_time"] / candidates["duration"]
    ).clip(0, 5)

    # Stratified sample by duration buckets
    candidates["duration_bucket"] = pd.cut(
        candidates["duration"],
        bins=[10, 20, 30, 45, 60],
        labels=["10-20s", "20-30s", "30-45s", "45-60s"],
    )

    selected = []
    per_bucket = target_count // candidates["duration_bucket"].nunique()

    for bucket, group in candidates.groupby("duration_bucket", observed=True):
        # Prioritize videos with more viewers (better retention curves)
        group_sorted = group.sort_values("viewer_count", ascending=False)
        n = min(per_bucket, len(group_sorted))
        selected.append(group_sorted.head(n))
        print(f"  {bucket}: selected {n} / {len(group_sorted)} available")

    result = pd.concat(selected, ignore_index=True)

    # If under target, fill from remaining candidates
    if len(result) < target_count:
        remaining = candidates[~candidates["video_id"].isin(result["video_id"])]
        remaining = remaining.sort_values("viewer_count", ascending=False)
        extra = remaining.head(target_count - len(result))
        result = pd.concat([result, extra], ignore_index=True)

    return result.head(target_count)


def main():
    print("Loading Tsinghua ShortVideo data...")
    interactions = load_interactions()
    video_info = load_video_info()

    print(f"Interactions: {len(interactions):,} rows")
    print(f"Videos: {len(video_info):,}")
    print(f"Users: {interactions['user_id'].nunique():,}")

    # Distribution of viewers per video
    vc = interactions.groupby("video_id")["user_id"].nunique()
    print(f"\nViewers per video: median={vc.median():.0f}, mean={vc.mean():.1f}, "
          f"max={vc.max()}, ≥30: {(vc >= 30).sum()}, ≥15: {(vc >= 15).sum()}")

    selected = select_videos(interactions, video_info)
    print(f"\nSelected {len(selected)} videos")
    print(selected[["video_id", "duration", "viewer_count", "completion_rate"]].describe())

    # Save selection
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT / "selected_videos.csv", index=False)

    # Save video IDs for MP4 download
    video_ids = selected["video_id"].tolist()
    with open(OUTPUT / "video_ids.json", "w") as f:
        json.dump(video_ids, f)

    print(f"\nSaved to {OUTPUT}/")
    print(f"Next: download MP4s for {len(video_ids)} videos, then run 02_retention_curves.py")


if __name__ == "__main__":
    main()
