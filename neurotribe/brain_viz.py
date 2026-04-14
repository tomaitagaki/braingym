"""
Brain surface visualization for NeuroTribe.

Renders predicted fMRI onto fsaverage5 surface using nilearn + matplotlib.
No PyVista dependency — pure matplotlib for Streamlit compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def load_fsaverage5():
    """Load fsaverage5 mesh coordinates and faces."""
    from nilearn.datasets import fetch_surf_fsaverage
    fsaverage = fetch_surf_fsaverage("fsaverage5")
    return fsaverage


def _load_mesh(fsaverage, hemi):
    """Load mesh coordinates and faces for a hemisphere."""
    from nilearn.surface import load_surf_mesh
    key = f"pial_{hemi}"
    coords, faces = load_surf_mesh(fsaverage[key])
    return coords, faces


def _load_sulc(fsaverage, hemi):
    """Load sulcal depth for background shading."""
    from nilearn.surface import load_surf_data
    key = f"sulc_{hemi}"
    return load_surf_data(fsaverage[key])


def render_brain_surface(vertex_data, view="lateral", figsize=(12, 5),
                         cmap="RdYlBu_r", vmin=None, vmax=None, title=None):
    """
    Render vertex_data (20484,) onto fsaverage5 surface.

    Uses nilearn's plot_surf_stat_map for proper 3D rendering.
    Returns a matplotlib figure.
    """
    from nilearn import plotting

    fsaverage = load_fsaverage5()
    n_hemi = len(vertex_data) // 2
    lh_data = vertex_data[:n_hemi]
    rh_data = vertex_data[n_hemi:]

    if vmax is None:
        vmax = np.percentile(np.abs(vertex_data), 99)
    threshold = max(0, np.percentile(np.abs(vertex_data), 20))

    fig = plt.figure(figsize=figsize)

    views_hemis = [
        ("left", "lateral"), ("left", "medial"),
        ("right", "lateral"), ("right", "medial"),
    ]

    for i, (hemi, view_name) in enumerate(views_hemis):
        data = lh_data if hemi == "left" else rh_data
        mesh = fsaverage[f"pial_{hemi}"]
        bg = fsaverage[f"sulc_{hemi}"]

        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        plotting.plot_surf_stat_map(
            mesh, data,
            hemi=hemi, view=view_name,
            bg_map=bg,
            cmap=cmap, vmax=vmax,
            threshold=threshold,
            axes=ax,
            colorbar=False,
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def render_brain_timeline(viz_preds, viz_indices, n_timepoints,
                          cmap="RdYlBu_r", max_cols=5):
    """
    Render multiple timepoints as a grid of brain surfaces.

    viz_preds: list of (20484,) arrays
    viz_indices: which timepoint each corresponds to
    """
    from nilearn import plotting

    fsaverage = load_fsaverage5()
    n_frames = len(viz_preds)
    n_cols = min(n_frames, max_cols)
    n_rows = (n_frames + n_cols - 1) // n_cols

    # Compute global normalization
    all_data = np.array(viz_preds)
    vmax = np.percentile(np.abs(all_data), 99)

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i, (pred, t_idx) in enumerate(zip(viz_preds, viz_indices)):
        pred = np.array(pred)
        n_hemi = len(pred) // 2
        lh_data = pred[:n_hemi]

        # Left hemisphere lateral view only (for compactness)
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["pial_left"], lh_data,
            hemi="left", view="lateral",
            bg_map=fsaverage["sulc_left"],
            cmap=cmap, vmax=vmax,
            axes=ax,
            colorbar=False,
        )
        ax.set_title(f"t={t_idx}s", fontsize=10)

    fig.suptitle(
        f"Brain response over time ({n_timepoints}s total, showing {n_frames} samples)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def render_network_timeseries(timeseries, attention_ts=None, figsize=(14, 8)):
    """
    Plot network-level timeseries with interpretation annotations.

    timeseries: dict of network_name -> list of values
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    t = np.arange(len(list(timeseries.values())[0]))

    # Panel 1: All networks
    ax = axes[0]
    colors = {
        "DAN": "#ef4444", "DMN": "#3b82f6", "FPN": "#f59e0b",
        "VAN": "#8b5cf6", "Vis": "#22c55e", "SomMot": "#ec4899", "Limbic": "#06b6d4",
    }
    for net_name, values in timeseries.items():
        ax.plot(t, values, label=net_name, color=colors.get(net_name, "#999"),
                linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Activation", fontsize=11)
    ax.set_title("All brain networks", fontsize=12, fontweight="bold")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    # Panel 2: Attention (DAN - DMN) — the key engagement signal
    ax = axes[1]
    if attention_ts:
        attn = np.array(attention_ts)
    else:
        attn = np.array(timeseries["DAN"]) - np.array(timeseries["DMN"])
    ax.fill_between(t, attn, where=attn > 0, color="#22c55e", alpha=0.3, label="Engaged")
    ax.fill_between(t, attn, where=attn < 0, color="#ef4444", alpha=0.3, label="Mind-wandering")
    ax.plot(t, attn, "k-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("DAN - DMN", fontsize=11)
    ax.set_title("Attention signal (positive = engaged, negative = mind-wandering)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 3: DMN (engagement proxy — lower = more engaged)
    ax = axes[2]
    dmn = np.array(timeseries["DMN"])
    ax.plot(t, -dmn, color="#3b82f6", linewidth=2)
    ax.fill_between(t, -dmn, alpha=0.15, color="#3b82f6")
    ax.set_ylabel("-DMN (engagement)", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("Engagement signal (higher = more engaged, DMN suppressed)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig
