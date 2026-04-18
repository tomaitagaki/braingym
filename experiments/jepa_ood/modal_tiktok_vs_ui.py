"""
TikTok vs UI: V-JEPA2 embeddings + TRIBE brain predictions.

Hypothesis (grounded in Yeo 7-network model):
  TikTok (designed for engagement):
    - High DAN (captures top-down attention)
    - Low DMN  (suppresses mind-wandering)
    - High VAN (novelty/surprise)
    - Low FPN  (effortless consumption)

  UI screen recordings (functional, mundane):
    - Lower DAN (less captivating)
    - Higher DMN (more mind-wandering)
    - Lower VAN (predictable, no surprise)
    - Higher FPN (effortful reading/navigation)

Phase 1: V-JEPA2 embedding comparison (are they in different regions?)
Phase 2: Full TRIBE brain predictions + network metrics + timeseries

Usage:
    modal run experiments/jepa_ood/modal_tiktok_vs_ui.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-tiktok-vs-ui")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

# Phase 1 image: just V-JEPA2
jepa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "transformers>=4.57,<5",
        "numpy==2.2.6",
        "huggingface_hub",
        "Pillow",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "einops",
        "av",
    )
)

# Phase 2 image: full TRIBE
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
        "scikit-learn",
        "matplotlib",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)


def extract_frames(video_path: str, fps: float = 2.0, max_frames: int = 60):
    """Extract frames from video at given fps using ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            f"{tmpdir}/frame_%04d.jpg",
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        frames = []
        for p in sorted(Path(tmpdir).glob("frame_*.jpg")):
            frames.append(Image.open(p).convert("RGB"))
    return frames


# ============================================================
# PHASE 1: V-JEPA2 Embedding Comparison
# ============================================================

@app.function(
    image=jepa_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    memory=32768,
)
def phase1_jepa_embeddings():
    """Compare V-JEPA2 embeddings: TikTok vs UI."""
    import os
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")

    results_dir = Path(CACHE_DIR) / "tiktok_vs_ui"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect videos
    videos = {}
    nat_dir = Path(CACHE_DIR) / "naturalistic_videos"
    if nat_dir.exists():
        for v in sorted(nat_dir.glob("tiktok_*.mp4")):
            videos[v.stem] = {"path": str(v), "category": "tiktok"}
        # Also grab a Tsinghua video as naturalistic baseline
        for v in sorted(nat_dir.glob("tsinghua_*.mp4"))[:2]:
            videos[v.stem] = {"path": str(v), "category": "naturalistic"}
        for v in sorted(nat_dir.glob("sintel_*.mp4"))[:1]:
            videos[v.stem] = {"path": str(v), "category": "naturalistic"}

    ui_dir = Path(CACHE_DIR) / "ui_videos"
    if ui_dir.exists():
        for v in sorted(ui_dir.glob("*.mov")) + sorted(ui_dir.glob("*.mp4")):
            videos[v.stem] = {"path": str(v), "category": "ui"}

    print("=" * 60)
    print("PHASE 1: V-JEPA2 Embeddings")
    print("=" * 60)
    for name, info in videos.items():
        print(f"  [{info['category']:>12s}] {name}")

    # Load V-JEPA2
    print("\nLoading V-JEPA2...")
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    model_name = "facebook/vjepa2-vitl-fpc64-256"
    processor = VJEPA2VideoProcessor.from_pretrained(model_name)
    model = VJEPA2Model.from_pretrained(model_name, dtype=torch.float16)
    model = model.to(device).eval()

    # Extract embeddings
    all_embeddings = {}
    all_categories = {}

    for name, info in videos.items():
        print(f"\n  Extracting: {name}")
        try:
            frames = extract_frames(info["path"], fps=2.0, max_frames=30)
            print(f"    {len(frames)} frames")
            if not frames:
                continue

            embeddings = []
            for frame in frames:
                inputs = processor(videos=[frame], return_tensors="pt")
                inputs = {k: v.to(device, dtype=torch.float16) if v.dtype.is_floating_point else v.to(device)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    emb = out.last_hidden_state[:, 0, :]
                embeddings.append(emb.float().cpu().numpy())

            all_embeddings[name] = np.concatenate(embeddings, axis=0)
            all_categories[name] = info["category"]
            print(f"    Shape: {all_embeddings[name].shape}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_embeddings:
        return {"error": "no embeddings"}

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    # Per-video norms
    print("\n--- L2 Norms ---")
    for name, emb in all_embeddings.items():
        norms = np.linalg.norm(emb, axis=1)
        print(f"  {name:>25s} ({all_categories[name]:>12s}): "
              f"mean={norms.mean():.2f}, std={norms.std():.2f}")

    # Pairwise cosine similarity between videos (mean frame embedding)
    print("\n--- Inter-Video Cosine Similarity (mean embeddings) ---")
    names_sorted = sorted(all_embeddings.keys())
    mean_embs = {n: all_embeddings[n].mean(axis=0, keepdims=True) for n in names_sorted}
    mean_matrix = np.concatenate([mean_embs[n] for n in names_sorted], axis=0)
    sim = cosine_similarity(mean_matrix)

    print(f"\n{'':>25s}", end="")
    for n in names_sorted:
        print(f" {n[:10]:>10s}", end="")
    print()
    for i, ni in enumerate(names_sorted):
        print(f"  {ni:>23s}", end="")
        for j in range(len(names_sorted)):
            print(f" {sim[i, j]:>10.4f}", end="")
        print(f"  ({all_categories[ni]})")

    # Within vs between category similarity
    print("\n--- Category-Level Similarity ---")
    cats = sorted(set(all_categories.values()))
    cat_frames = {}
    for name, emb in all_embeddings.items():
        cat = all_categories[name]
        cat_frames.setdefault(cat, []).append(emb)
    for cat in cat_frames:
        cat_frames[cat] = np.concatenate(cat_frames[cat], axis=0)

    cat_sim = {}
    for ci in cats:
        for cj in cats:
            s = cosine_similarity(cat_frames[ci], cat_frames[cj])
            if ci == cj:
                mask = ~np.eye(s.shape[0], dtype=bool)
                cat_sim[(ci, cj)] = float(s[mask].mean()) if mask.sum() > 0 else float(s.mean())
            else:
                cat_sim[(ci, cj)] = float(s.mean())
            print(f"  {ci:>12s} vs {cj:<12s}: {cat_sim[(ci, cj)]:.4f}")

    # PCA
    all_frames = np.concatenate(list(cat_frames.values()), axis=0)
    all_labels = []
    for cat in cats:
        all_labels.extend([cat] * len(cat_frames[cat]))

    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_frames)

    colors = {"tiktok": "#FF0050", "ui": "#4CAF50", "naturalistic": "#2196F3"}
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot by individual video
    offset = 0
    markers = {"tiktok": "o", "ui": "^", "naturalistic": "s"}
    for name in sorted(all_embeddings.keys()):
        cat = all_categories[name]
        n = len(all_embeddings[name])
        proj_vid = pca.transform(all_embeddings[name])
        ax.scatter(
            proj_vid[:, 0], proj_vid[:, 1],
            c=colors.get(cat, "gray"),
            marker=markers.get(cat, "o"),
            alpha=0.5, s=25,
            label=f"{name} ({cat})",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("V-JEPA2 Embeddings: TikTok vs UI vs Naturalistic")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "phase1_jepa_pca.png", dpi=150)
    print(f"\n  Saved: {results_dir / 'phase1_jepa_pca.png'}")

    vol.commit()

    return {
        "norms": {n: {"mean": float(np.linalg.norm(e, axis=1).mean()),
                       "std": float(np.linalg.norm(e, axis=1).std())}
                  for n, e in all_embeddings.items()},
        "inter_video_sim": {f"{names_sorted[i]}_vs_{names_sorted[j]}": float(sim[i, j])
                            for i in range(len(names_sorted))
                            for j in range(i + 1, len(names_sorted))},
        "category_sim": {f"{ci}_vs_{cj}": v for (ci, cj), v in cat_sim.items()},
    }


# ============================================================
# PHASE 2: Full TRIBE Brain Predictions
# ============================================================

@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def phase2_tribe_predictions():
    """Run full TRIBE on TikTok vs UI and compare brain metrics."""
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

    results_dir = Path(CACHE_DIR) / "tiktok_vs_ui"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect videos (TikTok + UI only for cleaner comparison)
    videos = {}
    nat_dir = Path(CACHE_DIR) / "naturalistic_videos"
    if nat_dir.exists():
        for v in sorted(nat_dir.glob("tiktok_*.mp4")):
            videos[v.stem] = {"path": str(v), "category": "tiktok"}

    ui_dir = Path(CACHE_DIR) / "ui_videos"
    if ui_dir.exists():
        for v in sorted(ui_dir.glob("*.mov")) + sorted(ui_dir.glob("*.mp4")):
            videos[v.stem] = {"path": str(v), "category": "ui"}

    print("=" * 60)
    print("PHASE 2: TRIBE Brain Predictions")
    print("=" * 60)
    for name, info in videos.items():
        print(f"  [{info['category']:>8s}] {name}")

    # Load TRIBE
    print("\nLoading TRIBE v2...")
    from tribev2.demo_utils import TribeModel
    tribe = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
        },
    )
    print(f"  Loaded on {device}")

    # Load Schaefer parcellation
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

    N = 10242  # fsaverage5 vertices per hemisphere

    # --- Run TRIBE on each video ---
    all_results = {}
    all_timeseries = {}

    for name, info in videos.items():
        print(f"\n--- Processing: {name} ({info['category']}) ---")
        try:
            df = tribe.get_events_dataframe(video_path=info["path"])
            preds, segments = tribe.predict(events=df)
            print(f"  Predictions: {preds.shape}")

            # Extract network signals
            preds_lh = preds[:, :N]
            preds_rh = preds[:, N:2*N]
            net_signals = {}
            for net_name, masks in networks.items():
                lh_idx, rh_idx = masks["lh"], masks["rh"]
                sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
                sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
                net_signals[net_name] = (sig_lh + sig_rh) / 2

            # Cognitive metrics
            dan = net_signals["DAN"]
            dmn = net_signals["DMN"]
            van = net_signals["VAN"]
            fpn = net_signals["FPN"]
            limbic = net_signals["Limbic"]

            n3 = min(3, len(dan))

            metrics = {
                "category": info["category"],
                "n_timepoints": int(preds.shape[0]),
                "duration_sec": int(preds.shape[0]),  # ~1 Hz output

                # Core metrics
                "engagement": float(-np.mean(dmn)),
                "attention": float(np.mean(dan) - np.mean(dmn)),
                "salience": float(np.mean(van)),
                "cognitive_load": float(np.mean(fpn) + np.mean(dan) - np.mean(dmn)),
                "emotional": float(np.mean(limbic)),

                # Temporal
                "hook_3s": float(np.mean(dan[:n3]) - np.mean(dmn[:n3])),
                "peak_salience": float(np.max(van)),
                "sustained_attention": float(-np.std(dmn)),

                # Per-network means
                "net_DAN": float(np.mean(dan)),
                "net_DMN": float(np.mean(dmn)),
                "net_VAN": float(np.mean(van)),
                "net_FPN": float(np.mean(fpn)),
                "net_Limbic": float(np.mean(limbic)),
                "net_Vis": float(np.mean(net_signals["Vis"])),
                "net_SomMot": float(np.mean(net_signals["SomMot"])),

                # Brain reward (from sandbox/core.py weights)
                "brain_reward": float(
                    0.4 * (-np.mean(dmn))
                    + 0.3 * (np.mean(dan) - np.mean(dmn))
                    + 0.2 * np.mean(van)
                    + 0.1 * np.mean(limbic)
                ),
            }

            all_results[name] = metrics
            all_timeseries[name] = {
                net: sig.tolist() for net, sig in net_signals.items()
            }
            all_timeseries[name]["_category"] = info["category"]

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        vol.commit()
        return {"error": "no predictions"}

    # ============================================================
    # PRINT RESULTS TABLE
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS: TikTok vs UI Brain Predictions")
    print("=" * 80)

    # Sort: tiktok first, then ui
    names_sorted = sorted(all_results.keys(),
                          key=lambda n: (0 if all_results[n]["category"] == "tiktok" else 1, n))

    print(f"\n{'Name':>25s} {'Cat':>8s} {'Engage':>8s} {'Attn':>8s} {'Salien':>8s} "
          f"{'CogLoad':>8s} {'Emot':>8s} {'Reward':>8s}")
    print("-" * 95)
    for name in names_sorted:
        r = all_results[name]
        print(f"{name:>25s} {r['category']:>8s} {r['engagement']:>+8.4f} "
              f"{r['attention']:>+8.4f} {r['salience']:>+8.4f} "
              f"{r['cognitive_load']:>+8.4f} {r['emotional']:>+8.4f} "
              f"{r['brain_reward']:>+8.4f}")

    print(f"\n{'Name':>25s} {'Cat':>8s} {'DAN':>8s} {'DMN':>8s} {'VAN':>8s} "
          f"{'FPN':>8s} {'Limbic':>8s} {'Vis':>8s} {'SomMot':>8s}")
    print("-" * 95)
    for name in names_sorted:
        r = all_results[name]
        print(f"{name:>25s} {r['category']:>8s} {r['net_DAN']:>+8.4f} "
              f"{r['net_DMN']:>+8.4f} {r['net_VAN']:>+8.4f} "
              f"{r['net_FPN']:>+8.4f} {r['net_Limbic']:>+8.4f} "
              f"{r['net_Vis']:>+8.4f} {r['net_SomMot']:>+8.4f}")

    # Category averages
    print("\n--- Category Averages ---")
    for cat in ["tiktok", "ui"]:
        cat_results = [r for r in all_results.values() if r["category"] == cat]
        if not cat_results:
            continue
        avg = {}
        for key in ["engagement", "attention", "salience", "cognitive_load",
                     "emotional", "brain_reward", "net_DAN", "net_DMN",
                     "net_VAN", "net_FPN", "net_Limbic", "net_Vis"]:
            avg[key] = np.mean([r[key] for r in cat_results])
        print(f"\n  {cat.upper():>8s}:")
        print(f"    Engagement={avg['engagement']:+.4f}  Attention={avg['attention']:+.4f}  "
              f"Salience={avg['salience']:+.4f}  CogLoad={avg['cognitive_load']:+.4f}")
        print(f"    DAN={avg['net_DAN']:+.4f}  DMN={avg['net_DMN']:+.4f}  "
              f"VAN={avg['net_VAN']:+.4f}  FPN={avg['net_FPN']:+.4f}  "
              f"Limbic={avg['net_Limbic']:+.4f}")

    # Hypothesis check
    print("\n" + "=" * 80)
    print("HYPOTHESIS CHECK")
    print("=" * 80)

    tiktok_results = [r for r in all_results.values() if r["category"] == "tiktok"]
    ui_results = [r for r in all_results.values() if r["category"] == "ui"]

    if tiktok_results and ui_results:
        checks = [
            ("DAN higher for TikTok",
             np.mean([r["net_DAN"] for r in tiktok_results]),
             np.mean([r["net_DAN"] for r in ui_results]),
             ">"),
            ("DMN lower for TikTok (more suppressed)",
             np.mean([r["net_DMN"] for r in tiktok_results]),
             np.mean([r["net_DMN"] for r in ui_results]),
             "<"),
            ("VAN higher for TikTok (more novelty)",
             np.mean([r["net_VAN"] for r in tiktok_results]),
             np.mean([r["net_VAN"] for r in ui_results]),
             ">"),
            ("FPN higher for UI (more effortful)",
             np.mean([r["net_FPN"] for r in tiktok_results]),
             np.mean([r["net_FPN"] for r in ui_results]),
             "<"),
            ("Engagement higher for TikTok",
             np.mean([r["engagement"] for r in tiktok_results]),
             np.mean([r["engagement"] for r in ui_results]),
             ">"),
            ("Brain reward higher for TikTok",
             np.mean([r["brain_reward"] for r in tiktok_results]),
             np.mean([r["brain_reward"] for r in ui_results]),
             ">"),
        ]

        n_pass = 0
        for desc, tiktok_val, ui_val, direction in checks:
            if direction == ">":
                passed = tiktok_val > ui_val
            else:
                passed = tiktok_val < ui_val
            status = "PASS" if passed else "FAIL"
            n_pass += int(passed)
            print(f"  [{status}] {desc}")
            print(f"         TikTok={tiktok_val:+.4f}  UI={ui_val:+.4f}  "
                  f"diff={tiktok_val - ui_val:+.4f}")

        print(f"\n  Result: {n_pass}/{len(checks)} hypotheses confirmed")
    else:
        print("  Cannot check — missing TikTok or UI results")

    # ============================================================
    # PLOTS
    # ============================================================

    # 1. Bar chart: cognitive metrics by video
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metric_groups = [
        ("Engagement & Attention", ["engagement", "attention", "salience"]),
        ("Cognitive Load & Control", ["cognitive_load", "net_FPN", "net_DAN"]),
        ("Network Activations", ["net_DAN", "net_DMN", "net_VAN", "net_FPN", "net_Limbic"]),
        ("Temporal Metrics", ["hook_3s", "peak_salience", "sustained_attention"]),
    ]

    cat_colors = {"tiktok": "#FF0050", "ui": "#4CAF50"}

    for ax, (title, metric_keys) in zip(axes.flat, metric_groups):
        x = np.arange(len(metric_keys))
        width = 0.35

        tiktok_vals = []
        ui_vals = []
        for key in metric_keys:
            tiktok_vals.append(np.mean([r[key] for r in all_results.values() if r["category"] == "tiktok"]))
            ui_vals.append(np.mean([r[key] for r in all_results.values() if r["category"] == "ui"]))

        bars1 = ax.bar(x - width/2, tiktok_vals, width, label="TikTok", color="#FF0050", alpha=0.8)
        bars2 = ax.bar(x + width/2, ui_vals, width, label="UI", color="#4CAF50", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("net_", "") for k in metric_keys], rotation=30, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(results_dir / "phase2_metrics_comparison.png", dpi=150)
    print(f"\n  Saved: {results_dir / 'phase2_metrics_comparison.png'}")

    # 2. Network timeseries for each video
    n_videos = len(all_timeseries)
    fig2, axes2 = plt.subplots(n_videos, 1, figsize=(14, 3.5 * n_videos), sharex=False)
    if n_videos == 1:
        axes2 = [axes2]

    net_colors = {
        "DAN": "#e63946", "DMN": "#457b9d", "VAN": "#f4a261",
        "FPN": "#2a9d8f", "Limbic": "#9b59b6", "Vis": "#95a5a6", "SomMot": "#d4a373",
    }

    for ax, name in zip(axes2, names_sorted):
        ts = all_timeseries[name]
        cat = ts["_category"]
        t = np.arange(len(ts["DAN"]))

        for net in ["DAN", "DMN", "VAN", "FPN", "Limbic"]:
            ax.plot(t, ts[net], label=net, color=net_colors[net], linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"{name} ({cat})", fontsize=11, fontweight="bold",
                     color=cat_colors.get(cat, "black"))
        ax.set_ylabel("Activation")
        ax.legend(fontsize=7, ncol=5, loc="upper right")
        ax.set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(results_dir / "phase2_timeseries.png", dpi=150)
    print(f"  Saved: {results_dir / 'phase2_timeseries.png'}")

    # Save results
    save_data = {
        "results": all_results,
        "timeseries": all_timeseries,
    }
    out_path = results_dir / "tiktok_vs_ui_results.json"
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"  Saved: {out_path}")

    vol.commit()
    return save_data


# ============================================================
# ENTRYPOINT: Run both phases
# ============================================================

@app.local_entrypoint()
def main():
    # Phase 1: JEPA embeddings
    print("=" * 60)
    print("LAUNCHING PHASE 1: V-JEPA2 Embeddings")
    print("=" * 60)
    jepa_results = phase1_jepa_embeddings.remote()

    # Phase 2: Full TRIBE
    print("\n" + "=" * 60)
    print("LAUNCHING PHASE 2: TRIBE Brain Predictions")
    print("=" * 60)
    tribe_results = phase2_tribe_predictions.remote()

    # Save locally
    local_dir = Path("cache/tiktok_vs_ui")
    local_dir.mkdir(parents=True, exist_ok=True)

    (local_dir / "phase1_jepa.json").write_text(json.dumps(jepa_results, indent=2))
    print(f"\nSaved: {local_dir / 'phase1_jepa.json'}")

    if "error" not in tribe_results:
        (local_dir / "phase2_tribe.json").write_text(json.dumps(tribe_results, indent=2))
        print(f"Saved: {local_dir / 'phase2_tribe.json'}")
