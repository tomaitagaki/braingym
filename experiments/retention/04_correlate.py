"""
Phase 4: Correlate brain metric timeseries with retention curves.

Analyses:
  4a. Cross-video per-timepoint: brain(t) vs S(t) across videos
  4b. Within-video temporal: brain timeseries vs retention curve per video
  4c. Lagged correlation: does brain state predict future retention change?
  4d. Scalar summary: replicates pilot (DAN → share_rate)

Usage:
    uv run python experiments/retention/04_correlate.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
from dataclasses import dataclass

CACHE = Path("cache/tsinghua/selected")


@dataclass
class AlignedVideo:
    """A single video with brain timeseries and retention curve aligned to same time axis."""
    video_id: str
    duration: float
    seconds: np.ndarray           # [1, 2, ..., D]
    survival: np.ndarray          # S(t) at each second
    brain: dict[str, np.ndarray]  # network_name -> activation at each second
    attention_ts: np.ndarray      # DAN - DMN at each second
    cognitive_load_ts: np.ndarray # FPN + DAN - DMN at each second


def align_timeseries(
    retention: dict,
    brain: dict,
) -> AlignedVideo | None:
    """Align brain timeseries and retention curve to 1Hz (per-second).

    TRIBEv2 outputs at ~1 TR (~1.5s). Interpolate to match retention's 1Hz.
    """
    video_id = str(retention["video_id"])
    duration = retention["duration"]
    seconds = np.array(retention["seconds"])
    survival = np.array(retention["survival"])

    # Brain timeseries — need to interpolate to match retention seconds
    brain_ts = brain.get("timeseries", {})
    if not brain_ts:
        return None

    n_brain = len(next(iter(brain_ts.values())))
    if n_brain < 2:
        return None

    # TRIBEv2 timepoints → assume evenly spaced over video duration
    brain_times = np.linspace(0, duration, n_brain)

    # Interpolate each network to 1Hz
    aligned_brain = {}
    for net_name, values in brain_ts.items():
        values = np.array(values)
        aligned_brain[net_name] = np.interp(seconds, brain_times, values)

    # Derived timeseries
    attention_ts = aligned_brain["DAN"] - aligned_brain["DMN"]
    cognitive_load_ts = aligned_brain["FPN"] + aligned_brain["DAN"] - aligned_brain["DMN"]

    return AlignedVideo(
        video_id=video_id,
        duration=duration,
        seconds=seconds,
        survival=survival,
        brain=aligned_brain,
        attention_ts=attention_ts,
        cognitive_load_ts=cognitive_load_ts,
    )


def analysis_4a_cross_video(videos: list[AlignedVideo]) -> pd.DataFrame:
    """Cross-video, per-timepoint: at each second t, correlate brain(t) with S(t) across videos.

    Only includes videos long enough to have data at time t.
    """
    max_t = max(len(v.seconds) for v in videos)
    results = []

    for t in range(1, min(max_t + 1, 61)):  # cap at 60s
        # Gather brain metrics and survival at time t across videos
        brain_at_t = {net: [] for net in ["DAN", "DMN", "FPN", "VAN", "attention", "cognitive_load"]}
        survival_at_t = []

        for v in videos:
            if t > len(v.seconds):
                continue
            idx = t - 1
            survival_at_t.append(v.survival[idx])
            for net, vals in v.brain.items():
                if net in brain_at_t:
                    brain_at_t[net].append(vals[idx])
            brain_at_t["attention"].append(v.attention_ts[idx])
            brain_at_t["cognitive_load"].append(v.cognitive_load_ts[idx])

        if len(survival_at_t) < 10:
            continue

        survival_arr = np.array(survival_at_t)
        for metric_name, metric_vals in brain_at_t.items():
            if len(metric_vals) != len(survival_at_t):
                continue
            r, p = stats.pearsonr(metric_vals, survival_arr)
            results.append({
                "second": t,
                "metric": metric_name,
                "r": r,
                "p": p,
                "n_videos": len(survival_at_t),
            })

    return pd.DataFrame(results)


def analysis_4b_within_video(videos: list[AlignedVideo]) -> pd.DataFrame:
    """Within-video: correlate brain timeseries with retention curve for each video."""
    results = []

    for v in videos:
        if len(v.seconds) < 5:
            continue

        for metric_name, metric_ts in [
            ("DAN", v.brain.get("DAN")),
            ("DMN", v.brain.get("DMN")),
            ("attention", v.attention_ts),
            ("cognitive_load", v.cognitive_load_ts),
        ]:
            if metric_ts is None:
                continue
            r, p = stats.pearsonr(metric_ts, v.survival)
            results.append({
                "video_id": v.video_id,
                "duration": v.duration,
                "metric": metric_name,
                "r": r,
                "p": p,
                "n_timepoints": len(v.seconds),
            })

    return pd.DataFrame(results)


def analysis_4c_lagged(videos: list[AlignedVideo], max_lag: int = 5) -> pd.DataFrame:
    """Lagged cross-correlation: does brain state at t predict retention change at t+k?

    Uses the derivative of survival (drop-off rate) as the target.
    """
    results = []

    for v in videos:
        if len(v.seconds) < 10:
            continue

        # Retention slope (drop-off rate)
        retention_slope = np.gradient(v.survival)

        for metric_name, metric_ts in [
            ("DAN", v.brain.get("DAN")),
            ("attention", v.attention_ts),
        ]:
            if metric_ts is None:
                continue

            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    brain_seg = metric_ts[:len(metric_ts) - lag] if lag > 0 else metric_ts
                    ret_seg = retention_slope[lag:]
                else:
                    brain_seg = metric_ts[-lag:]
                    ret_seg = retention_slope[:len(retention_slope) + lag]

                n = min(len(brain_seg), len(ret_seg))
                if n < 5:
                    continue

                r, p = stats.pearsonr(brain_seg[:n], ret_seg[:n])
                results.append({
                    "video_id": v.video_id,
                    "metric": metric_name,
                    "lag": lag,
                    "r": r,
                    "p": p,
                })

    return pd.DataFrame(results)


def analysis_4d_scalar(videos: list[AlignedVideo], retention_metrics: pd.DataFrame) -> pd.DataFrame:
    """Scalar brain metrics vs scalar retention metrics (replicates pilot approach)."""
    rows = []
    for v in videos:
        rows.append({
            "video_id": v.video_id,
            "mean_DAN": float(v.brain["DAN"].mean()),
            "mean_DMN": float(v.brain["DMN"].mean()),
            "mean_attention": float(v.attention_ts.mean()),
            "mean_cognitive_load": float(v.cognitive_load_ts.mean()),
            "peak_DAN": float(v.brain["DAN"].max()),
            "dan_dmn_corr": float(np.corrcoef(v.brain["DAN"], v.brain["DMN"])[0, 1]),
        })
    brain_df = pd.DataFrame(rows)

    merged = brain_df.merge(
        retention_metrics[["video_id", "half_life", "completion_rate", "rewatch_rate", "retention_auc"]],
        on="video_id",
    )

    brain_cols = ["mean_DAN", "mean_DMN", "mean_attention", "mean_cognitive_load", "peak_DAN", "dan_dmn_corr"]
    ret_cols = ["half_life", "completion_rate", "rewatch_rate", "retention_auc"]

    results = []
    for bc in brain_cols:
        for rc in ret_cols:
            valid = merged[[bc, rc]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[bc], valid[rc])
            results.append({"brain_metric": bc, "retention_metric": rc, "r": r, "p": p, "n": len(valid)})

    return pd.DataFrame(results)


def main():
    # Load data
    with open(CACHE / "retention_curves.json") as f:
        retention_data = json.load(f)

    with open(CACHE / "tribe_results.json") as f:
        brain_data = {r["video_id"]: r for r in json.load(f)}

    retention_metrics = pd.read_csv(CACHE / "retention_metrics.csv")

    # Align timeseries
    videos = []
    for vid, ret in retention_data.items():
        brain = brain_data.get(str(ret["video_id"]))
        if brain is None or "error" in brain:
            continue
        aligned = align_timeseries(ret, brain)
        if aligned:
            videos.append(aligned)

    print(f"Aligned {len(videos)} videos with both brain and retention data\n")

    if len(videos) < 5:
        print("Too few videos to analyze. Check that both retention curves and brain results exist.")
        return

    # 4a: Cross-video per-timepoint
    print("=" * 60)
    print("4a: Cross-video per-timepoint correlation")
    print("=" * 60)
    df_4a = analysis_4a_cross_video(videos)
    print(df_4a[df_4a["p"] < 0.05].sort_values("second").to_string(index=False))
    df_4a.to_csv(CACHE / "correlation_4a_cross_video.csv", index=False)

    # 4b: Within-video temporal
    print(f"\n{'=' * 60}")
    print("4b: Within-video temporal correlation")
    print("=" * 60)
    df_4b = analysis_4b_within_video(videos)
    summary = df_4b.groupby("metric").agg(
        mean_r=("r", "mean"),
        median_r=("r", "median"),
        sig_count=("p", lambda x: (x < 0.05).sum()),
        n_videos=("r", "count"),
    )
    print(summary)
    df_4b.to_csv(CACHE / "correlation_4b_within_video.csv", index=False)

    # 4c: Lagged correlation
    print(f"\n{'=' * 60}")
    print("4c: Lagged cross-correlation (brain leads retention?)")
    print("=" * 60)
    df_4c = analysis_4c_lagged(videos)
    lag_summary = df_4c.groupby(["metric", "lag"]).agg(mean_r=("r", "mean")).reset_index()
    for metric in lag_summary["metric"].unique():
        subset = lag_summary[lag_summary["metric"] == metric].sort_values("lag")
        peak = subset.loc[subset["mean_r"].abs().idxmax()]
        print(f"  {metric}: peak |r| at lag={peak['lag']:.0f}s (r={peak['mean_r']:.3f})")
    df_4c.to_csv(CACHE / "correlation_4c_lagged.csv", index=False)

    # 4d: Scalar summary
    print(f"\n{'=' * 60}")
    print("4d: Scalar brain metrics vs retention metrics")
    print("=" * 60)
    df_4d = analysis_4d_scalar(videos, retention_metrics)
    sig = df_4d[df_4d["p"] < 0.05].sort_values("p")
    if len(sig) > 0:
        print(sig.to_string(index=False))
    else:
        print("No significant scalar correlations at p<0.05")
    df_4d.to_csv(CACHE / "correlation_4d_scalar.csv", index=False)

    print(f"\nAll results saved to {CACHE}/")


if __name__ == "__main__":
    main()
