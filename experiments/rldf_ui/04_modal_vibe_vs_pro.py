"""
Vibe-coded vs professional UI: maximum visual contrast, same content.

Same chatbot conversation, two extremes:
- Vibe coded: rainbow gradient, Comic Sans, neon colors, blinking elements, ad banners
- Professional: Apple HIG-inspired, SF Pro, clean bubbles, proper spacing

This is the strongest possible test: if TRIBE can't tell these apart,
it definitively cannot distinguish UI design quality.

Usage:
    modal run experiments/rldf_ui/04_modal_vibe_vs_pro.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-vibe-vs-pro")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)
CACHE_DIR = "/cache"

tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7", "torchvision>=0.20,<0.22", "transformers>=4.57,<5",
        "numpy==2.2.6", "einops", "pyyaml", "huggingface_hub", "langdetect",
        "spacy", "soundfile", "Levenshtein", "julius", "x_transformers==1.27.20",
        "nibabel", "scipy", "Pillow", "matplotlib",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)


@app.function(
    image=tribe_image, gpu="A10G", volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600, memory=32768,
)
def run_vibe_vs_pro():
    import os, numpy as np, torch, nibabel as nib
    from urllib.request import urlretrieve
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    results_dir = Path(CACHE_DIR) / "vibe_vs_pro_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    videos = {
        "vibe_coded": Path(CACHE_DIR) / "rldf_videos" / "vibe_coded.mp4",
        "professional": Path(CACHE_DIR) / "rldf_videos" / "professional.mp4",
    }

    print("=" * 70)
    print("VIBE-CODED vs PROFESSIONAL: Maximum contrast UI test")
    print("=" * 70)
    for name, path in videos.items():
        print(f"  {name}: {'OK' if path.exists() else 'MISSING'}")

    # Load TRIBE
    from tribev2.demo_utils import TribeModel
    tribe = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE_DIR, device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
        },
    )

    # Parcellation
    SCHAEFER_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
    ALIASES = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
    }
    parc_dir = Path(CACHE_DIR) / "schaefer"; parc_dir.mkdir(exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parc_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        labels, _, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = ALIASES.get(parts[2], parts[2])
                if net not in networks:
                    networks[net] = {"lh": [], "rh": []}
                networks[net][hemi].extend(np.where(labels == idx)[0].tolist())
    for net in networks:
        for hemi in ("lh", "rh"):
            networks[net][hemi] = np.array(networks[net][hemi])
    N = 10242

    # Run TRIBE
    all_results = {}
    all_timeseries = {}

    for name, path in videos.items():
        print(f"\n--- {name} ---")
        df = tribe.get_events_dataframe(video_path=str(path))
        preds, segments = tribe.predict(events=df)
        print(f"  Predictions: {preds.shape}")

        preds_lh = preds[:, :N]
        preds_rh = preds[:, N:2*N]
        net_signals = {}
        for net_name, masks in networks.items():
            lh_idx, rh_idx = masks["lh"], masks["rh"]
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = (sig_lh + sig_rh) / 2

        dan = net_signals["DAN"]; dmn = net_signals["DMN"]
        van = net_signals["VAN"]; fpn = net_signals["FPN"]
        limbic = net_signals["Limbic"]

        metrics = {
            "n_timepoints": int(preds.shape[0]),
            "engagement": float(-np.mean(dmn)),
            "attention": float(np.mean(dan) - np.mean(dmn)),
            "salience": float(np.mean(van)),
            "cognitive_load": float(np.mean(fpn) + np.mean(dan) - np.mean(dmn)),
            "emotional": float(np.mean(limbic)),
            "brain_reward": float(
                0.4 * (-np.mean(dmn)) + 0.3 * (np.mean(dan) - np.mean(dmn))
                + 0.2 * np.mean(van) + 0.1 * np.mean(limbic)
            ),
            "net_DAN": float(np.mean(dan)), "net_DMN": float(np.mean(dmn)),
            "net_VAN": float(np.mean(van)), "net_FPN": float(np.mean(fpn)),
            "net_Limbic": float(np.mean(limbic)),
            "net_Vis": float(np.mean(net_signals["Vis"])),
            "net_SomMot": float(np.mean(net_signals["SomMot"])),
            "dan_slope": float(np.polyfit(np.arange(len(dan)), dan, 1)[0]),
            "dmn_slope": float(np.polyfit(np.arange(len(dmn)), dmn, 1)[0]),
        }
        all_results[name] = metrics
        all_timeseries[name] = {net: sig.tolist() for net, sig in net_signals.items()}

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 90)
    print("RESULTS: Vibe-Coded vs Professional")
    print("=" * 90)

    vibe = all_results["vibe_coded"]
    pro = all_results["professional"]

    metrics_list = [
        ("engagement", "Engagement (-DMN)"),
        ("attention", "Attention (DAN-DMN)"),
        ("salience", "Salience (VAN)"),
        ("cognitive_load", "Cognitive Load"),
        ("brain_reward", "Brain Reward"),
        ("net_DAN", "DAN"),
        ("net_DMN", "DMN"),
        ("net_VAN", "VAN"),
        ("net_FPN", "FPN"),
        ("net_Limbic", "Limbic"),
        ("net_Vis", "Visual"),
        ("net_SomMot", "SomMot"),
    ]

    print(f"\n{'Metric':>25s} {'Vibe':>10s} {'Pro':>10s} {'Diff':>10s} {'Winner':>10s}")
    print("-" * 70)
    for key, label in metrics_list:
        v, p = vibe[key], pro[key]
        diff = p - v
        winner = "PRO" if abs(diff) > 0.001 and (
            (key in ["engagement", "attention", "brain_reward", "net_DAN", "salience"] and diff > 0)
            or (key in ["net_FPN", "cognitive_load"] and diff < 0)
        ) else ("VIBE" if abs(diff) > 0.001 else "~TIE")
        print(f"{label:>25s} {v:>+10.4f} {p:>+10.4f} {diff:>+10.4f} {winner:>10s}")

    # Spatial correlation between the two prediction patterns
    # (are they producing fundamentally different brain maps?)
    print(f"\n--- Spatial pattern analysis ---")
    vibe_ts = all_timeseries["vibe_coded"]
    pro_ts = all_timeseries["professional"]

    # Per-timepoint correlation of network profiles
    min_t = min(len(vibe_ts["DAN"]), len(pro_ts["DAN"]))
    net_names = ["DAN", "DMN", "VAN", "FPN", "Limbic", "Vis", "SomMot"]
    correlations = []
    for t in range(min_t):
        vibe_profile = [vibe_ts[n][t] for n in net_names]
        pro_profile = [pro_ts[n][t] for n in net_names]
        r = np.corrcoef(vibe_profile, pro_profile)[0, 1]
        correlations.append(r)

    print(f"  Network profile correlation (vibe vs pro):")
    print(f"    Mean: r={np.mean(correlations):.4f}")
    print(f"    Range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
    if np.mean(correlations) > 0.9:
        print(f"    -> Nearly identical brain patterns despite visual difference")
    elif np.mean(correlations) > 0.5:
        print(f"    -> Partially different brain patterns")
    else:
        print(f"    -> Substantially different brain patterns")

    # ============================================================
    # PLOT
    # ============================================================
    net_colors = {
        "DAN": "#e63946", "DMN": "#457b9d", "VAN": "#f4a261",
        "FPN": "#2a9d8f", "Limbic": "#9b59b6",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, (name, title, ls) in zip(axes, [
        ("vibe_coded", "Vibe-Coded (rainbow, Comic Sans, blinking)", "-"),
        ("professional", "Professional (Apple HIG, clean, minimal)", "-"),
    ]):
        ts = all_timeseries[name]
        t = np.arange(len(ts["DAN"]))
        for net in ["DAN", "DMN", "VAN", "FPN", "Limbic"]:
            ax.plot(t, ts[net], color=net_colors[net], linewidth=2,
                    alpha=0.8, label=net)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Activation")
        ax.set_xlabel("Time (seconds)")
        ax.legend(fontsize=8, ncol=5, loc="upper right")

    plt.tight_layout()
    plt.savefig(results_dir / "vibe_vs_pro_timeseries.png", dpi=150)

    # Overlay plot
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for net in ["DAN", "DMN", "FPN"]:
        vt = np.array(vibe_ts[net][:min_t])
        pt = np.array(pro_ts[net][:min_t])
        t = np.arange(min_t)
        ax2.plot(t, vt, color=net_colors[net], linestyle="--", alpha=0.5, linewidth=1.5,
                 label=f"{net} (vibe)")
        ax2.plot(t, pt, color=net_colors[net], linestyle="-", alpha=0.9, linewidth=2,
                 label=f"{net} (pro)")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_title("Overlay: solid=Professional, dashed=Vibe-Coded", fontweight="bold")
    ax2.set_ylabel("Activation")
    ax2.set_xlabel("Time (seconds)")
    ax2.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(results_dir / "vibe_vs_pro_overlay.png", dpi=150)

    # Bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bar_metrics = ["engagement", "attention", "salience", "cognitive_load", "brain_reward"]
    bar_labels = ["Engagement", "Attention", "Salience", "Cog Load", "Brain Reward"]
    x = np.arange(len(bar_metrics))
    width = 0.35
    v_vals = [vibe[m] for m in bar_metrics]
    p_vals = [pro[m] for m in bar_metrics]
    ax3.bar(x - width/2, v_vals, width, label="Vibe-Coded", color="#FF6B6B", alpha=0.8)
    ax3.bar(x + width/2, p_vals, width, label="Professional", color="#007AFF", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(bar_labels)
    ax3.legend()
    ax3.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax3.set_title("Vibe-Coded vs Professional: Cognitive Metrics")
    ax3.set_ylabel("Activation")
    plt.tight_layout()
    plt.savefig(results_dir / "vibe_vs_pro_bars.png", dpi=150)

    print(f"\n  Saved plots to {results_dir}")

    save_data = {
        "results": all_results,
        "timeseries": all_timeseries,
        "network_profile_correlation": {
            "mean": float(np.mean(correlations)),
            "per_timepoint": correlations,
        },
    }
    (results_dir / "vibe_vs_pro_results.json").write_text(json.dumps(save_data, indent=2))
    vol.commit()
    return save_data


@app.local_entrypoint()
def main():
    results = run_vibe_vs_pro.remote()
    Path("cache/rldf").mkdir(exist_ok=True)
    (Path("cache/rldf") / "vibe_vs_pro_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved local: cache/rldf/vibe_vs_pro_results.json")
