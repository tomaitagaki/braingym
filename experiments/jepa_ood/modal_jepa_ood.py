"""
V-JEPA2 OOD detection: are UI screen recordings in-distribution?

Approach:
  1. Extract V-JEPA2 frame embeddings from naturalistic videos (known in-distribution)
     and UI screen recordings (unknown)
  2. Compare embedding distributions:
     - L2 norms (OOD inputs often have unusual norms)
     - Pairwise cosine similarity within/between categories
     - PCA projection to visualize clustering
     - Mahalanobis distance from naturalistic centroid

If UI embeddings land close to naturalistic video in V-JEPA2 space,
TRIBE predictions for UI content may be meaningful.

Usage:
    modal run experiments/jepa_ood/modal_jepa_ood.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-jepa-ood")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

image = (
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
            img = Image.open(p).convert("RGB")
            frames.append(img)

    return frames


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def run_ood_analysis():
    """Extract V-JEPA2 embeddings and compare naturalistic vs UI videos."""
    import os
    import numpy as np
    import torch
    from pathlib import Path
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import mahalanobis
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "jepa_ood"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Collect videos by category ----
    print("=" * 60)
    print("STEP 1: Collecting videos")
    print("=" * 60)

    categories = {}

    # Naturalistic: uploaded to naturalistic_videos/ on volume
    nat_dir = Path(CACHE_DIR) / "naturalistic_videos"
    if nat_dir.exists():
        for v in sorted(nat_dir.glob("*.mp4")):
            categories.setdefault("naturalistic", []).append(
                {"path": str(v), "label": v.stem}
            )

    # Synthetic: Manim animations
    manim_dir = Path(CACHE_DIR) / "manim_experiment"
    if manim_dir.exists():
        for v in sorted(manim_dir.glob("*.mp4"))[:2]:
            categories.setdefault("synthetic", []).append(
                {"path": str(v), "label": f"manim_{v.stem}"}
            )

    # UI: screen recordings
    ui_dir = Path(CACHE_DIR) / "ui_videos"
    if ui_dir.exists():
        for v in sorted(ui_dir.glob("*.mp4")) + sorted(ui_dir.glob("*.mov")):
            categories.setdefault("ui", []).append(
                {"path": str(v), "label": f"ui_{v.stem}"}
            )

    for cat, vids in categories.items():
        print(f"  {cat}: {len(vids)} videos")
        for v in vids:
            print(f"    - {v['label']}")

    if not categories.get("naturalistic"):
        print("\n  WARNING: No naturalistic videos found!")
        print("  Expected in /cache/naturalistic_videos/")

    if not categories.get("ui"):
        print("\n  WARNING: No UI videos found in /cache/ui_videos/")

    # ---- Load V-JEPA2 ----
    print("\n" + "=" * 60)
    print("STEP 2: Loading V-JEPA2")
    print("=" * 60)

    # V-JEPA2 is a video ViT. For frame-level features we process
    # each frame as a single-frame video clip.
    # Using vitl (Large) for speed; vitg (Giant) is what TRIBE uses.
    model_name = "facebook/vjepa2-vitl-fpc64-256"
    print(f"  Loading {model_name}...")

    from transformers import VJEPA2Model, VJEPA2VideoProcessor

    processor = VJEPA2VideoProcessor.from_pretrained(model_name)
    model = VJEPA2Model.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"  Loaded on {device}")

    # ---- Extract embeddings ----
    print("\n" + "=" * 60)
    print("STEP 3: Extracting frame embeddings")
    print("=" * 60)

    all_embeddings = {}  # label -> (n_frames, dim) array
    all_categories = {}  # label -> category name

    FPS = 2.0       # 2 frames/sec (matches TRIBE config)
    MAX_FRAMES = 30  # Cap per video

    for cat, vids in categories.items():
        for vid_info in vids:
            label = vid_info["label"]
            path = vid_info["path"]
            print(f"\n  [{cat}] {label}")

            try:
                frames = extract_frames(path, fps=FPS, max_frames=MAX_FRAMES)
                print(f"    Extracted {len(frames)} frames")

                if len(frames) == 0:
                    print(f"    Skipping (no frames)")
                    continue

                # Process each frame as a 1-frame "video" through V-JEPA2
                # V-JEPA2 expects pixel_values of shape (batch, channels, n_frames, h, w)
                # For single frames, n_frames=1
                embeddings = []
                batch_size = 4

                for i in range(0, len(frames), batch_size):
                    batch_frames = frames[i:i + batch_size]

                    # Process each frame individually as a single-frame clip
                    batch_embeds = []
                    for frame in batch_frames:
                        # VJEPA2VideoProcessor expects a list of video frames
                        inputs = processor(
                            videos=[frame],  # single frame as 1-frame video
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(device, dtype=torch.float16) if v.dtype.is_floating_point else v.to(device)
                                  for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model(**inputs)

                        # Get CLS token or pooled output
                        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                            emb = outputs.pooler_output  # (1, dim)
                        elif hasattr(outputs, "last_hidden_state"):
                            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                        else:
                            emb = outputs[0][:, 0, :]

                        batch_embeds.append(emb.float().cpu().numpy())

                    embeddings.append(np.concatenate(batch_embeds, axis=0))

                embeddings = np.concatenate(embeddings, axis=0)
                print(f"    Embeddings: {embeddings.shape}")
                all_embeddings[label] = embeddings
                all_categories[label] = cat

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    if not all_embeddings:
        print("\nNo embeddings extracted. Exiting.")
        vol.commit()
        return {"error": "no embeddings"}

    # ---- Analysis ----
    print("\n" + "=" * 60)
    print("STEP 4: Distribution analysis")
    print("=" * 60)

    # 4a. Embedding norms per category
    print("\n--- L2 Norms ---")
    cat_norms = {}
    for label, emb in all_embeddings.items():
        cat = all_categories[label]
        norms = np.linalg.norm(emb, axis=1)
        cat_norms.setdefault(cat, []).extend(norms.tolist())
        print(f"  {label:>25s} ({cat:>12s}): "
              f"norm mean={norms.mean():.2f}, std={norms.std():.2f}, "
              f"range=[{norms.min():.2f}, {norms.max():.2f}]")

    print("\n  Per-category summary:")
    for cat, norms in cat_norms.items():
        norms = np.array(norms)
        print(f"  {cat:>12s}: mean={norms.mean():.2f}, std={norms.std():.2f}")

    # 4b. Cosine similarity within/between categories
    print("\n--- Cosine Similarity ---")
    from sklearn.metrics.pairwise import cosine_similarity

    cat_frames = {}
    for label, emb in all_embeddings.items():
        cat = all_categories[label]
        cat_frames.setdefault(cat, []).append(emb)

    for cat in cat_frames:
        cat_frames[cat] = np.concatenate(cat_frames[cat], axis=0)

    cat_names = sorted(cat_frames.keys())

    print(f"\n  {'':>15s}", end="")
    for c in cat_names:
        print(f"  {c:>12s}", end="")
    print()

    sim_matrix = {}
    for ci in cat_names:
        print(f"  {ci:>15s}", end="")
        for cj in cat_names:
            fi = cat_frames[ci]
            fj = cat_frames[cj]
            if len(fi) > 100:
                fi = fi[np.random.choice(len(fi), 100, replace=False)]
            if len(fj) > 100:
                fj = fj[np.random.choice(len(fj), 100, replace=False)]

            sims = cosine_similarity(fi, fj)
            if ci == cj:
                mask = ~np.eye(sims.shape[0], dtype=bool)
                if sims.shape[0] == sims.shape[1]:
                    mean_sim = sims[mask].mean() if mask.sum() > 0 else sims.mean()
                else:
                    mean_sim = sims.mean()
            else:
                mean_sim = sims.mean()
            sim_matrix[(ci, cj)] = float(mean_sim)
            print(f"  {mean_sim:>12.4f}", end="")
        print()

    # 4c. Mahalanobis distance from naturalistic distribution
    print("\n--- Mahalanobis Distance from Naturalistic Distribution ---")
    maha_results = {}
    if "naturalistic" in cat_frames:
        nat_frames = cat_frames["naturalistic"]

        n_components = min(50, nat_frames.shape[0] - 1, nat_frames.shape[1])
        pca_maha = PCA(n_components=n_components)
        nat_pca = pca_maha.fit_transform(nat_frames)
        nat_cov = np.cov(nat_pca.T) + np.eye(n_components) * 1e-4
        nat_cov_inv = np.linalg.inv(nat_cov)
        nat_mean_pca = nat_pca.mean(axis=0)

        for cat in cat_names:
            frames_pca = pca_maha.transform(cat_frames[cat])
            dists = [mahalanobis(f, nat_mean_pca, nat_cov_inv) for f in frames_pca]
            dists = np.array(dists)
            maha_results[cat] = {
                "mean": float(dists.mean()),
                "std": float(dists.std()),
                "median": float(np.median(dists)),
                "p95": float(np.percentile(dists, 95)),
            }
            print(f"  {cat:>12s}: mean={dists.mean():.2f}, "
                  f"std={dists.std():.2f}, "
                  f"median={np.median(dists):.2f}, "
                  f"p95={np.percentile(dists, 95):.2f}")

    # 4d. PCA visualization
    print("\n--- PCA Visualization ---")
    all_frames_arr = np.concatenate(list(cat_frames.values()), axis=0)
    all_labels_list = []
    for cat in cat_names:
        all_labels_list.extend([cat] * len(cat_frames[cat]))

    pca = PCA(n_components=3)
    projected = pca.fit_transform(all_frames_arr)
    print(f"  Explained variance: {pca.explained_variance_ratio_[:3].sum():.1%} "
          f"({pca.explained_variance_ratio_[0]:.1%}, "
          f"{pca.explained_variance_ratio_[1]:.1%}, "
          f"{pca.explained_variance_ratio_[2]:.1%})")

    colors = {"naturalistic": "#2196F3", "synthetic": "#FF9800", "ui": "#4CAF50"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cat in cat_names:
        mask = np.array(all_labels_list) == cat
        axes[0].scatter(
            projected[mask, 0], projected[mask, 1],
            c=colors.get(cat, "gray"), label=cat, alpha=0.5, s=15,
        )
        axes[1].scatter(
            projected[mask, 0], projected[mask, 2],
            c=colors.get(cat, "gray"), label=cat, alpha=0.5, s=15,
        )

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("V-JEPA2 Embedding Space (PC1 vs PC2)")
    axes[0].legend()
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    axes[1].set_title("V-JEPA2 Embedding Space (PC1 vs PC3)")
    axes[1].legend()

    plt.tight_layout()
    plot_path = results_dir / "jepa_ood_pca.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")

    # Per-video PCA plot
    markers = {"naturalistic": "o", "synthetic": "s", "ui": "^"}
    all_frames_by_label = []
    for label in sorted(all_embeddings.keys()):
        all_frames_by_label.append(all_embeddings[label])
    all_frames_by_label = np.concatenate(all_frames_by_label, axis=0)
    projected_by_label = pca.transform(all_frames_by_label)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    offset = 0
    for label in sorted(all_embeddings.keys()):
        cat = all_categories[label]
        n = len(all_embeddings[label])
        ax2.scatter(
            projected_by_label[offset:offset + n, 0],
            projected_by_label[offset:offset + n, 1],
            marker=markers.get(cat, "o"),
            alpha=0.4, s=20,
            label=f"{label} ({cat})",
        )
        offset += n

    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("V-JEPA2 Embeddings by Video")
    ax2.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plot_path2 = results_dir / "jepa_ood_per_video.png"
    plt.savefig(plot_path2, dpi=150)
    print(f"  Saved: {plot_path2}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = {
        "categories": {cat: int(len(frames)) for cat, frames in cat_frames.items()},
        "norms": {
            cat: {"mean": float(np.mean(norms)), "std": float(np.std(norms))}
            for cat, norms in cat_norms.items()
        },
        "cosine_similarity": {
            f"{ci}_vs_{cj}": sim_matrix[(ci, cj)]
            for ci in cat_names for cj in cat_names
        },
        "mahalanobis": maha_results,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "videos": {
            label: {
                "category": all_categories[label],
                "n_frames": int(len(emb)),
                "embedding_dim": int(emb.shape[1]),
                "norm_mean": float(np.linalg.norm(emb, axis=1).mean()),
                "norm_std": float(np.linalg.norm(emb, axis=1).std()),
            }
            for label, emb in all_embeddings.items()
        },
    }

    # Interpretation
    if "ui" in cat_frames and "naturalistic" in cat_frames:
        nat_ui_sim = sim_matrix.get(("naturalistic", "ui"), 0)
        nat_nat_sim = sim_matrix.get(("naturalistic", "naturalistic"), 0)
        ratio = nat_ui_sim / nat_nat_sim if nat_nat_sim > 0 else 0

        print(f"\n  Naturalistic within-sim: {nat_nat_sim:.4f}")
        print(f"  Naturalistic-UI cross-sim:  {nat_ui_sim:.4f}")
        print(f"  Ratio: {ratio:.2%}")

        if ratio > 0.85:
            verdict = "IN-DISTRIBUTION: UI embeddings overlap substantially with naturalistic video"
            results["verdict"] = "in_distribution"
        elif ratio > 0.60:
            verdict = "BORDERLINE: UI embeddings partially overlap — TRIBE may capture relative differences"
            results["verdict"] = "borderline"
        else:
            verdict = "OUT-OF-DISTRIBUTION: UI embeddings are far from naturalistic video — TRIBE predictions unreliable"
            results["verdict"] = "out_of_distribution"

        print(f"\n  VERDICT: {verdict}")
        results["interpretation"] = verdict
    else:
        print("\n  No UI videos — showing naturalistic vs synthetic comparison only")
        if "synthetic" in cat_frames and "naturalistic" in cat_frames:
            nat_syn_sim = sim_matrix.get(("naturalistic", "synthetic"), 0)
            nat_nat_sim = sim_matrix.get(("naturalistic", "naturalistic"), 0)
            print(f"  Naturalistic within-sim: {nat_nat_sim:.4f}")
            print(f"  Naturalistic-Synthetic cross-sim: {nat_syn_sim:.4f}")

    out_path = results_dir / "jepa_ood_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_path}")

    vol.commit()
    return results


@app.local_entrypoint()
def main():
    results = run_ood_analysis.remote()

    local_dir = Path("cache/jepa_ood")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "jepa_ood_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved local: {local_dir / 'jepa_ood_results.json'}")
