import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    mo.md("### Brain Timeseries × Retention Curves")
    return Path, json, mo, np, pd


@app.cell
def _(Path, json, mo, np, pd):
    CACHE = Path("cache/tsinghua/selected")

    # Load retention curves
    with open(CACHE / "retention_curves.json") as f:
        retention_data = json.load(f)

    # Load brain results
    brain_path = CACHE / "tribe_results.json"
    brain_data = {}
    if brain_path.exists():
        with open(brain_path) as f:
            brain_data = {r["video_id"]: r for r in json.load(f)}

    # Load retention metrics
    metrics = pd.read_csv(CACHE / "retention_metrics.csv")

    mo.md(f"""
    **Loaded:** {len(retention_data)} retention curves, {len(brain_data)} brain results, {len(metrics)} video metrics
    """)
    return CACHE, brain_data, metrics, retention_data


@app.cell
def _(metrics, mo, np, pd):
    """Retention metrics overview."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(metrics["completion_rate"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Completion Rate")
    axes[0, 0].set_title("How many viewers finish?")

    axes[0, 1].hist(metrics["half_life"], bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Half Life (seconds)")
    axes[0, 1].set_title("When do 50% of viewers leave?")

    axes[1, 0].hist(metrics["rewatch_rate"], bins=30, edgecolor="black", alpha=0.7, color="green")
    axes[1, 0].set_xlabel("Rewatch Rate")
    axes[1, 0].set_title("How many loop the video?")

    axes[1, 1].hist(metrics["retention_auc"], bins=30, edgecolor="black", alpha=0.7, color="red")
    axes[1, 1].set_xlabel("Retention AUC (normalized)")
    axes[1, 1].set_title("Overall retention strength")

    fig.suptitle("Retention Metrics Distribution", fontsize=14)
    fig.tight_layout()
    mo.output.replace(fig)
    return


@app.cell
def _(mo, np, pd, retention_data):
    """Overlay retention curves, colored by completion rate."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    curves = sorted(retention_data.values(), key=lambda x: x["completion_rate"], reverse=True)
    cmap = plt.cm.RdYlGn

    for i, curve in enumerate(curves):
        color = cmap(curve["completion_rate"])
        ax.plot(curve["seconds"], curve["survival"], color=color, alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Second")
    ax.set_ylabel("Fraction still watching — S(t)")
    ax.set_title("Retention Curves (green=high completion, red=low)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% mark")
    ax.legend()
    fig.tight_layout()
    mo.output.replace(fig)
    return


@app.cell
def _(brain_data, mo, np, retention_data):
    """Dual-axis plot: brain timeseries + retention curve for a single video."""
    import matplotlib.pyplot as plt

    # Pick a video that has both brain and retention data
    matched = [
        vid for vid in retention_data
        if vid in brain_data and "error" not in brain_data[vid]
    ]

    if not matched:
        mo.md("**No videos with both brain and retention data yet.** Run phases 2 and 3 first.")
    else:
        vid = matched[0]
        ret = retention_data[vid]
        brain = brain_data[vid]

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Retention curve
        ax1.plot(ret["seconds"], ret["survival"], "k-", linewidth=2, label="Retention S(t)")
        ax1.fill_between(ret["seconds"], ret["survival"], alpha=0.1, color="black")
        ax1.set_xlabel("Second")
        ax1.set_ylabel("Fraction watching", color="black")
        ax1.set_ylim(0, 1.05)

        # Brain timeseries on secondary axis
        ax2 = ax1.twinx()
        ts = brain["timeseries"]
        n_tp = len(ts["DAN"])
        brain_x = np.linspace(1, ret["duration"], n_tp)

        ax2.plot(brain_x, ts["DAN"], "r-", alpha=0.8, label="DAN")
        ax2.plot(brain_x, ts["DMN"], "b-", alpha=0.8, label="DMN")
        ax2.set_ylabel("Brain network activation", color="gray")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.set_title(f"Video {vid} — Retention × Brain Networks")
        fig.tight_layout()
        mo.output.replace(fig)
    return


@app.cell
def _(CACHE, mo, pd):
    """Correlation results summary."""
    results_4d = CACHE / "correlation_4d_scalar.csv"
    if results_4d.exists():
        df = pd.read_csv(results_4d)
        sig = df[df["p"] < 0.05].sort_values("p")
        mo.md(f"""
        ### Scalar Correlations (p < 0.05)
        {mo.as_html(sig) if len(sig) > 0 else "No significant correlations yet."}

        ### All Correlations
        {mo.as_html(df.round(4))}
        """)
    else:
        mo.md("**Run 04_correlate.py first** to generate correlation results.")
    return


if __name__ == "__main__":
    app.run()
