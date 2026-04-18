"""
Layout-only experiment: same CSS/colors/fonts, different section order.

4 variants of one product page — only information hierarchy changes:
  - good_hierarchy: image → price → description → specs → CTA → reviews
  - reversed: reviews → specs → description → CTA → price → image
  - cta_first: image → CTA → price → reviews → description → specs
  - specs_first: image → specs → description → price → CTA → reviews

If TRIBE trajectories differ, it's responding to spatial/temporal content
organization, not pixel-level visual features.

Usage:
    modal run experiments/rldf_ui/08_modal_layout.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-layout")
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
def run_layout_experiment():
    import os, numpy as np, torch, nibabel as nib
    from urllib.request import urlretrieve
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    results_dir = Path(CACHE_DIR) / "layout_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path(CACHE_DIR) / "rldf_layout_videos"
    variants = {
        "good_hierarchy": "image → price → desc → specs → CTA → reviews",
        "specs_first": "image → specs → desc → price → CTA → reviews",
        "cta_first": "image → CTA → price → reviews → desc → specs",
        "reversed": "reviews → specs → desc → CTA → price → image",
    }

    print("=" * 70)
    print("LAYOUT EXPERIMENT: Same visual style, different section order")
    print("=" * 70)
    for name, desc in variants.items():
        path = video_dir / f"{name}.mp4"
        print(f"  {name:>20s}: {desc}")
        print(f"  {'':>20s}  {'OK' if path.exists() else 'MISSING'}")

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

    for name in variants:
        path = video_dir / f"{name}.mp4"
        print(f"\n--- {name} ---")
        if not path.exists():
            print("  SKIPPED"); continue

        try:
            df = tribe.get_events_dataframe(video_path=str(path))
            preds, segments = tribe.predict(events=df)
            print(f"  Predictions: {preds.shape}")

            preds_lh = preds[:, :N]; preds_rh = preds[:, N:2*N]
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
                "layout": variants[name],
                "n_timepoints": int(preds.shape[0]),
                "engagement": float(-np.mean(dmn)),
                "attention": float(np.mean(dan) - np.mean(dmn)),
                "salience": float(np.mean(van)),
                "cognitive_load": float(np.mean(fpn) + np.mean(dan) - np.mean(dmn)),
                "brain_reward": float(
                    0.4*(-np.mean(dmn)) + 0.3*(np.mean(dan)-np.mean(dmn))
                    + 0.2*np.mean(van) + 0.1*np.mean(limbic)
                ),
                "engage_plus_attn": float(-np.mean(dmn) + np.mean(dan) - np.mean(dmn)),
                "net_DAN": float(np.mean(dan)), "net_DMN": float(np.mean(dmn)),
                "net_VAN": float(np.mean(van)), "net_FPN": float(np.mean(fpn)),
                "net_Limbic": float(np.mean(limbic)),
                "net_Vis": float(np.mean(net_signals["Vis"])),
            }
            all_results[name] = metrics
            all_timeseries[name] = {net: sig.tolist() for net, sig in net_signals.items()}
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 100)
    print("RESULTS: Layout Variants — Same Visual Style, Different Section Order")
    print("=" * 100)

    ranked = sorted(all_results.keys(),
                    key=lambda k: all_results[k]["engage_plus_attn"], reverse=True)

    print(f"\n  Ranked by engage+attn (our best UI metric):\n")
    print(f"  {'Rank':>4s} {'Layout':>20s} {'E+A':>8s} {'Engage':>8s} {'Attn':>8s} "
          f"{'Reward':>8s} {'FPN':>8s} {'DAN':>8s} {'DMN':>8s} {'VAN':>8s}")
    print("  " + "-" * 95)
    for i, name in enumerate(ranked, 1):
        r = all_results[name]
        print(f"  {i:>4d} {name:>20s} {r['engage_plus_attn']:>+8.4f} "
              f"{r['engagement']:>+8.4f} {r['attention']:>+8.4f} "
              f"{r['brain_reward']:>+8.4f} {r['net_FPN']:>+8.4f} "
              f"{r['net_DAN']:>+8.4f} {r['net_DMN']:>+8.4f} {r['net_VAN']:>+8.4f}")

    print(f"\n  Section orders:")
    for name in ranked:
        print(f"    {name:>20s}: {variants[name]}")

    # Pairwise correlation of timeseries between layouts
    print(f"\n  Timeseries correlation (DAN) between layouts:")
    layout_names = list(all_timeseries.keys())
    min_len = min(len(all_timeseries[n]["DAN"]) for n in layout_names)
    print(f"  {'':>20s}", end="")
    for n in layout_names:
        print(f" {n[:12]:>12s}", end="")
    print()
    for ni in layout_names:
        print(f"  {ni:>20s}", end="")
        for nj in layout_names:
            dan_i = np.array(all_timeseries[ni]["DAN"][:min_len])
            dan_j = np.array(all_timeseries[nj]["DAN"][:min_len])
            r = np.corrcoef(dan_i, dan_j)[0, 1]
            print(f" {r:>12.4f}", end="")
        print()

    # ============================================================
    # PLOTS
    # ============================================================
    net_colors = {"DAN": "#e63946", "DMN": "#457b9d", "VAN": "#f4a261", "FPN": "#2a9d8f"}
    layout_styles = {
        "good_hierarchy": ("-", 2.5, 1.0),
        "specs_first": ("--", 1.5, 0.7),
        "cta_first": ("-.", 1.5, 0.7),
        "reversed": (":", 1.5, 0.7),
    }

    # All layouts overlaid
    fig, ax = plt.subplots(figsize=(14, 6))
    for layout_name in layout_names:
        ts = all_timeseries[layout_name]
        t = np.arange(len(ts["DAN"]))
        ls, lw, alpha = layout_styles.get(layout_name, ("-", 1.5, 0.7))
        for net in ["DAN", "DMN", "FPN"]:
            ax.plot(t, ts[net], color=net_colors[net], linestyle=ls,
                    linewidth=lw, alpha=alpha,
                    label=f"{net} ({layout_name})" if net == "DAN" else None)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title("Layout Experiment: DAN/DMN/FPN by Section Order\n"
                 "(solid=good, dashed=specs_first, dashdot=cta_first, dotted=reversed)",
                 fontweight="bold")
    ax.set_ylabel("Activation"); ax.set_xlabel("Time (seconds)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(results_dir / "layout_overlay.png", dpi=150)

    # Separate panels per network
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, net in zip(axes2, ["DAN", "DMN", "FPN"]):
        for layout_name in layout_names:
            ts = all_timeseries[layout_name]
            t = np.arange(len(ts[net]))
            ls, lw, alpha = layout_styles.get(layout_name, ("-", 1.5, 0.7))
            ax.plot(t, ts[net], linestyle=ls, linewidth=lw, alpha=alpha,
                    color=net_colors[net], label=layout_name)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{net} by Layout", fontweight="bold")
        ax.set_ylabel("Activation")
        ax.legend(fontsize=8, ncol=4)
    axes2[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(results_dir / "layout_per_network.png", dpi=150)

    print(f"\n  Saved: {results_dir / 'layout_overlay.png'}")
    print(f"  Saved: {results_dir / 'layout_per_network.png'}")

    save_data = {"results": all_results, "timeseries": all_timeseries, "variants": variants}
    (results_dir / "layout_results.json").write_text(json.dumps(save_data, indent=2))
    vol.commit()
    return save_data


@app.local_entrypoint()
def main():
    results = run_layout_experiment.remote()
    Path("cache/rldf/layout").mkdir(parents=True, exist_ok=True)
    (Path("cache/rldf/layout") / "layout_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved local: cache/rldf/layout/layout_results.json")
