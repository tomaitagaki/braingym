"""
Step 3: TRIBE on designer-improved RLDF UIs (before vs after critique).

3 pairs where designers identified 5-7 specific issues each:
  - Dashboard (133): misaligned carets, empty sections, unclear icons
  - E-commerce search (128): misplaced search, missing reset, bad padding
  - Food delivery (138): poor hierarchy, misaligned text, broken layout

The "after" versions explicitly fix these issues based on designer feedback.
Much more dramatic visual differences than the ranking pairs.

Usage:
    modal run experiments/rldf_ui/03_modal_tribe_improved.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-rldf-improved")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "transformers>=4.57,<5",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "huggingface_hub",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "x_transformers==1.27.20",
        "nibabel",
        "scipy",
        "Pillow",
        "matplotlib",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def run_tribe_improved():
    import os
    import numpy as np
    import torch
    import nibabel as nib
    from urllib.request import urlretrieve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "rldf_improved_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path(CACHE_DIR) / "rldf_videos"
    pairs = {
        "dashboard_133": {
            "before": video_dir / "improved_133_before.mp4",
            "after": video_dir / "improved_133_after.mp4",
            "issues": "Misaligned carets, empty portfolio, unclear icons, bad table alignment (7 issues)",
        },
        "ecommerce_128": {
            "before": video_dir / "improved_128_before.mp4",
            "after": video_dir / "improved_128_after.mp4",
            "issues": "Misplaced search, bad padding, no reset button, poor visual hierarchy (6 issues)",
        },
        "food_delivery_138": {
            "before": video_dir / "improved_138_before.mp4",
            "after": video_dir / "improved_138_after.mp4",
            "issues": "Poor info hierarchy, misaligned text, broken layout, bad typography (6 issues)",
        },
    }

    print("=" * 70)
    print("RLDF Improved UIs: Before vs After Designer Critique")
    print("=" * 70)
    for cat, info in pairs.items():
        print(f"\n  {cat}:")
        print(f"    Issues: {info['issues']}")
        for v in ["before", "after"]:
            print(f"    {v}: {'OK' if info[v].exists() else 'MISSING'}")

    # Load TRIBE
    print("\nLoading TRIBE v2...")
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
    parc_dir = Path(CACHE_DIR) / "schaefer"
    parc_dir.mkdir(exist_ok=True)
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

    for cat, info in pairs.items():
        for variant in ["before", "after"]:
            label = f"{cat}_{variant}"
            path = info[variant]
            print(f"\n--- {label} ---")
            if not path.exists():
                print("  SKIPPED")
                continue

            try:
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

                dan = net_signals["DAN"]
                dmn = net_signals["DMN"]
                van = net_signals["VAN"]
                fpn = net_signals["FPN"]
                limbic = net_signals["Limbic"]

                metrics = {
                    "category": cat, "variant": variant,
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
                    "net_DAN": float(np.mean(dan)),
                    "net_DMN": float(np.mean(dmn)),
                    "net_VAN": float(np.mean(van)),
                    "net_FPN": float(np.mean(fpn)),
                    "net_Limbic": float(np.mean(limbic)),
                    "net_Vis": float(np.mean(net_signals["Vis"])),
                    "net_SomMot": float(np.mean(net_signals["SomMot"])),
                    # Trajectory features
                    "dan_slope": float(np.polyfit(np.arange(len(dan)), dan, 1)[0]),
                    "dmn_slope": float(np.polyfit(np.arange(len(dmn)), dmn, 1)[0]),
                    "fpn_slope": float(np.polyfit(np.arange(len(fpn)), fpn, 1)[0]),
                    "van_peak_time": int(np.argmax(van)),
                    "dan_sustained": float(-np.std(dan)),
                }

                all_results[label] = metrics
                all_timeseries[label] = {
                    net: sig.tolist() for net, sig in net_signals.items()
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 90)
    print("RESULTS: Before vs After Designer Critique")
    print("=" * 90)

    print(f"\n{'Label':>30s} {'Engage':>8s} {'Attn':>8s} {'Salien':>8s} "
          f"{'CogLoad':>8s} {'Reward':>8s} {'DAN':>8s} {'DMN':>8s} {'FPN':>8s} {'Vis':>8s}")
    print("-" * 110)
    for cat in pairs:
        for variant in ["before", "after"]:
            label = f"{cat}_{variant}"
            if label not in all_results:
                continue
            r = all_results[label]
            marker = "   " if variant == "before" else ">>>"
            print(f"{marker} {label:>27s} {r['engagement']:>+8.4f} "
                  f"{r['attention']:>+8.4f} {r['salience']:>+8.4f} "
                  f"{r['cognitive_load']:>+8.4f} {r['brain_reward']:>+8.4f} "
                  f"{r['net_DAN']:>+8.4f} {r['net_DMN']:>+8.4f} "
                  f"{r['net_FPN']:>+8.4f} {r['net_Vis']:>+8.4f}")
        print()

    # Pairwise: after should be "better" (designer-fixed)
    # Hypothesis: improved UI → lower FPN (less effort), higher engagement,
    # better visual hierarchy (higher Vis), more sustained attention
    print("=" * 90)
    print("PAIRWISE: Does designer-improved UI produce better brain metrics?")
    print("=" * 90)

    comparison_metrics = [
        ("engagement", "higher", "Less mind-wandering"),
        ("attention", "higher", "More focused attention"),
        ("brain_reward", "higher", "Higher composite reward"),
        ("net_FPN", "lower", "Less cognitive effort"),
        ("net_DAN", "higher", "Captures more attention"),
        ("salience", "higher", "Better visual landmarks"),
        ("net_Vis", "higher", "Stronger visual processing"),
    ]

    pair_results = {}
    for cat in pairs:
        before_label = f"{cat}_before"
        after_label = f"{cat}_after"
        if before_label not in all_results or after_label not in all_results:
            continue

        before = all_results[before_label]
        after = all_results[after_label]

        print(f"\n  {cat.upper()}:")
        print(f"  Issues fixed: {pairs[cat]['issues']}")
        wins = 0
        total = 0
        for metric, direction, rationale in comparison_metrics:
            b_val = before[metric]
            a_val = after[metric]
            diff = a_val - b_val

            if direction == "higher":
                after_wins = a_val > b_val
            else:
                after_wins = a_val < b_val

            wins += int(after_wins)
            total += 1
            status = " FIXED" if after_wins else "NO CHG"
            print(f"    [{status}] {metric:>15s}: before={b_val:+.4f} after={a_val:+.4f} "
                  f"diff={diff:+.4f}  ({rationale})")

        pair_results[cat] = {"wins": wins, "total": total}
        print(f"    Score: improved wins {wins}/{total}")

    total_wins = sum(p["wins"] for p in pair_results.values())
    total_total = sum(p["total"] for p in pair_results.values())
    print(f"\n  OVERALL: improved wins {total_wins}/{total_total} comparisons "
          f"({100*total_wins/max(total_total,1):.0f}%)")

    # Trajectory comparison
    print("\n" + "=" * 90)
    print("TRAJECTORY FEATURES")
    print("=" * 90)
    print(f"\n{'Label':>30s} {'DAN slope':>10s} {'DMN slope':>10s} {'FPN slope':>10s} "
          f"{'VAN peak @':>10s} {'DAN sust':>10s}")
    print("-" * 90)
    for cat in pairs:
        for variant in ["before", "after"]:
            label = f"{cat}_{variant}"
            if label not in all_results:
                continue
            r = all_results[label]
            marker = "   " if variant == "before" else ">>>"
            print(f"{marker} {label:>27s} {r['dan_slope']:>+10.5f} "
                  f"{r['dmn_slope']:>+10.5f} {r['fpn_slope']:>+10.5f} "
                  f"{r['van_peak_time']:>10d}s {r['dan_sustained']:>+10.5f}")
        print()

    # ============================================================
    # PLOTS
    # ============================================================
    net_colors = {
        "DAN": "#e63946", "DMN": "#457b9d", "VAN": "#f4a261",
        "FPN": "#2a9d8f", "Limbic": "#9b59b6",
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, cat in zip(axes, pairs.keys()):
        for variant, ls, alpha, lw in [("before", "--", 0.5, 1.5), ("after", "-", 0.9, 2)]:
            label = f"{cat}_{variant}"
            if label not in all_timeseries:
                continue
            ts = all_timeseries[label]
            t = np.arange(len(ts["DAN"]))
            for net in ["DAN", "DMN", "VAN", "FPN"]:
                show = (net in ["DAN", "FPN"])
                ax.plot(t, ts[net], color=net_colors[net], linestyle=ls,
                        linewidth=lw, alpha=alpha,
                        label=f"{net} ({variant})" if show else None)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        short_cat = cat.replace("_", " ").replace("133", "").replace("128", "").replace("138", "").strip().title()
        ax.set_title(f"{short_cat} — solid=AFTER (fixed), dashed=BEFORE (broken)",
                     fontweight="bold")
        ax.set_ylabel("Activation")
        ax.set_xlabel("Time (seconds)")
        ax.legend(fontsize=7, ncol=4, loc="upper right")

    plt.tight_layout()
    plt.savefig(results_dir / "improved_timeseries.png", dpi=150)
    print(f"\n  Saved: {results_dir / 'improved_timeseries.png'}")

    # Bar chart
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    bar_metrics = ["engagement", "attention", "salience", "cognitive_load", "brain_reward"]
    bar_labels = ["Engage", "Attn", "Salience", "CogLoad", "Reward"]

    for ax, cat in zip(axes2, pairs.keys()):
        before_label = f"{cat}_before"
        after_label = f"{cat}_after"
        if before_label not in all_results or after_label not in all_results:
            continue
        x = np.arange(len(bar_metrics))
        width = 0.35
        b_vals = [all_results[before_label][m] for m in bar_metrics]
        a_vals = [all_results[after_label][m] for m in bar_metrics]
        ax.bar(x - width/2, b_vals, width, label="Before", color="#FF5722", alpha=0.8)
        ax.bar(x + width/2, a_vals, width, label="After", color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right")
        short = cat.split("_")[0].title()
        ax.set_title(short)
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(results_dir / "improved_metrics_bars.png", dpi=150)
    print(f"  Saved: {results_dir / 'improved_metrics_bars.png'}")

    save_data = {
        "results": all_results,
        "timeseries": all_timeseries,
        "pair_comparisons": pair_results,
        "pair_descriptions": {cat: info["issues"] for cat, info in pairs.items()},
    }
    out_path = results_dir / "improved_tribe_results.json"
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"  Saved: {out_path}")

    vol.commit()
    return save_data


@app.local_entrypoint()
def main():
    results = run_tribe_improved.remote()
    local_dir = Path("cache/rldf")
    (local_dir / "improved_tribe_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved local: {local_dir / 'improved_tribe_results.json'}")
