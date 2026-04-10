"""
Cross-modal spatial analysis: Dr. Seuss image vs Dr. Seuss poem (TTS).

Compare predicted brain activation patterns across visual and auditory modalities.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from moviepy import ImageClip

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================
# 1. STIMULI
# ============================================================

POEM = """
One fish two fish red fish blue fish.
Black fish blue fish old fish new fish.
This one has a little star. This one has a little car.
Say! What a lot of fish there are.
Yes. Some are red. And some are blue.
Some are old. And some are new.
Some are sad. And some are glad.
And some are very, very bad.
Why are they sad and glad and bad?
I do not know. Go ask your dad.
"""


def image_to_video(image_path, output_path, duration=5):
    """Convert a still image to a short video clip for TRIBEv2."""
    clip = ImageClip(str(image_path)).with_duration(duration).with_fps(24)
    clip.write_videofile(str(output_path), logger=None)
    clip.close()
    log(f"Created {duration}s video from image")


# ============================================================
# 2. RUN PREDICTIONS
# ============================================================

def predict_from_image(model, image_path):
    """Image → video → TRIBEv2 predicted brain response. Caches to disk."""
    cache_path = CACHE_FOLDER / "preds_image.npy"
    if cache_path.exists():
        log(f"Loading cached image predictions from {cache_path}")
        return np.load(cache_path)

    video_path = CACHE_FOLDER / "drseuss_image.mp4"
    image_to_video(image_path, video_path, duration=5)

    df = model.get_events_dataframe(video_path=str(video_path))
    preds, segments = model.predict(events=df)
    np.save(cache_path, preds)
    log(f"Saved image predictions to {cache_path}")
    return preds


def predict_from_text(model, text):
    """Text → TTS → TRIBEv2 predicted brain response. Caches to disk."""
    cache_path = CACHE_FOLDER / "preds_text.npy"
    if cache_path.exists():
        log(f"Loading cached text predictions from {cache_path}")
        return np.load(cache_path)

    text_path = CACHE_FOLDER / "drseuss_poem.txt"
    text_path.write_text(text.strip())

    df = model.get_events_dataframe(text_path=text_path)
    preds, segments = model.predict(events=df)
    np.save(cache_path, preds)
    log(f"Saved text predictions to {cache_path}")
    return preds


# ============================================================
# 3. SPATIAL ANALYSIS
# ============================================================

def spatial_analysis(preds_image, preds_text):
    """
    Compare spatial activation patterns between two modalities.

    Returns dict of analysis results.
    """
    from quantify import (
        extract_network_signals, compute_cognitive_metrics, N_VERTICES_HEMI
    )

    # --- Mean activation map per modality ---
    map_image = preds_image.mean(axis=0)  # (n_vertices,) average over time
    map_text = preds_text.mean(axis=0)

    # --- Vertex-wise spatial correlation ---
    spatial_corr = np.corrcoef(map_image, map_text)[0, 1]
    log(f"Spatial correlation (image vs text): r = {spatial_corr:.4f}")

    # --- Difference map: where do they diverge? ---
    # Normalize each to z-scores for fair comparison
    map_image_z = (map_image - map_image.mean()) / (map_image.std() + 1e-8)
    map_text_z = (map_text - map_text.mean()) / (map_text.std() + 1e-8)
    diff_map = map_image_z - map_text_z  # positive = more image, negative = more text

    # --- Network-level comparison ---
    signals_image = extract_network_signals(preds_image)
    signals_text = extract_network_signals(preds_text)
    metrics_image = compute_cognitive_metrics(signals_image)
    metrics_text = compute_cognitive_metrics(signals_text)

    # --- Find top vertices for each modality ---
    n_top = 500
    top_image_vertices = np.argsort(map_image_z)[-n_top:]  # most active for image
    top_text_vertices = np.argsort(map_text_z)[-n_top:]    # most active for text
    overlap = len(set(top_image_vertices) & set(top_text_vertices))
    log(f"Top-{n_top} vertex overlap: {overlap} ({overlap/n_top*100:.1f}%)")

    # --- Which hemisphere dominates for each? ---
    image_lh = map_image[:N_VERTICES_HEMI].mean()
    image_rh = map_image[N_VERTICES_HEMI:2*N_VERTICES_HEMI].mean()
    text_lh = map_text[:N_VERTICES_HEMI].mean()
    text_rh = map_text[N_VERTICES_HEMI:2*N_VERTICES_HEMI].mean()

    log(f"Image laterality: LH={image_lh:.4f}, RH={image_rh:.4f} "
        f"({'LH>RH' if image_lh > image_rh else 'RH>LH'})")
    log(f"Text  laterality: LH={text_lh:.4f}, RH={text_rh:.4f} "
        f"({'LH>RH' if text_lh > text_rh else 'RH>LH'})")

    return {
        "map_image": map_image,
        "map_text": map_text,
        "map_image_z": map_image_z,
        "map_text_z": map_text_z,
        "diff_map": diff_map,
        "spatial_corr": spatial_corr,
        "signals_image": signals_image,
        "signals_text": signals_text,
        "metrics_image": metrics_image,
        "metrics_text": metrics_text,
        "top_image_vertices": top_image_vertices,
        "top_text_vertices": top_text_vertices,
        "top_overlap": overlap,
    }


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_crossmodal(results, rows):
    """Plot the cross-modal spatial comparison with full metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cross-Modal Spatial Analysis: Dr. Seuss Image vs Poem",
                 fontsize=16, fontweight="bold")

    width = 0.35
    img_row = rows[0]  # "image"
    txt_row = rows[1]  # "text_audio"

    # (A) Network activation comparison
    ax = axes[0, 0]
    networks = ["Vis", "SomMot", "DAN", "VAN", "Limbic", "FPN", "DMN"]
    x = np.arange(len(networks))
    vals_img = [img_row[n] for n in networks]
    vals_txt = [txt_row[n] for n in networks]
    ax.bar(x - width/2, vals_img, width, label="Image", color="#8b5cf6")
    ax.bar(x + width/2, vals_txt, width, label="Text/Audio", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=30, fontsize=9)
    ax.set_title("Network Activation by Modality")
    ax.legend()

    # (B) Spatial region comparison (temporal vs visual etc.)
    ax = axes[0, 1]
    regions = ["visual_ctx", "temporal_ctx", "prefrontal", "parietal", "cingulate", "insular"]
    labels = ["Visual\nCortex", "Temporal\nCortex", "PFC", "Parietal", "Cingulate", "Insular"]
    x = np.arange(len(regions))
    vals_img = [img_row[r] for r in regions]
    vals_txt = [txt_row[r] for r in regions]
    ax.bar(x - width/2, vals_img, width, label="Image", color="#8b5cf6")
    ax.bar(x + width/2, vals_txt, width, label="Text/Audio", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Spatial Region Activation")
    ax.legend()

    # (C) Cognitive metrics comparison
    ax = axes[0, 2]
    metric_keys = ["cog_load", "attention", "engagement"]
    metric_labels = ["Cognitive\nLoad", "Attention", "Engagement"]
    x = np.arange(len(metric_keys))
    vals_img = [img_row[m] for m in metric_keys]
    vals_txt = [txt_row[m] for m in metric_keys]
    ax.bar(x - width/2, vals_img, width, label="Image", color="#8b5cf6")
    ax.bar(x + width/2, vals_txt, width, label="Text/Audio", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Cognitive Metrics by Modality")
    ax.legend()

    # (D) Spatial correlation scatter
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(results["map_image_z"]), 2000, replace=False)
    ax.scatter(results["map_image_z"][idx], results["map_text_z"][idx],
               alpha=0.3, s=3, c="#4a9eed")
    ax.plot([-3, 3], [-3, 3], "k--", alpha=0.3)
    ax.set_xlabel("Image activation (z)")
    ax.set_ylabel("Text activation (z)")
    ax.set_title(f"Vertex-wise Correlation (r={results['spatial_corr']:.3f})")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    # (E) Difference histogram
    ax = axes[1, 1]
    ax.hist(results["diff_map"], bins=80, color="#06b6d4", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Image - Text (z-scored)")
    ax.set_ylabel("Vertex count")
    ax.set_title("Spatial Difference Distribution")
    ax.annotate("Image\ndominant", xy=(0.85, 0.85), xycoords="axes fraction",
                ha="center", fontsize=10, color="#8b5cf6")
    ax.annotate("Text\ndominant", xy=(0.15, 0.85), xycoords="axes fraction",
                ha="center", fontsize=10, color="#f59e0b")

    # (F) Temporal vs Visual dominance
    ax = axes[1, 2]
    categories = ["Image", "Text/Audio"]
    temporal = [img_row["temporal_ctx"], txt_row["temporal_ctx"]]
    visual = [img_row["visual_ctx"], txt_row["visual_ctx"]]
    x = np.arange(2)
    ax.bar(x - width/2, visual, width, label="Visual Cortex", color="#8b5cf6", alpha=0.8)
    ax.bar(x + width/2, temporal, width, label="Temporal Cortex", color="#f59e0b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Temporal vs Visual Cortex Dominance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CACHE_FOLDER / "crossmodal_drseuss.png", dpi=150)
    plt.show()
    log(f"Saved to {CACHE_FOLDER / 'crossmodal_drseuss.png'}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    import torch
    from tribev2.demo_utils import TribeModel
    import sys

    # --- Get image path from arg or prompt ---
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        print("Usage: python crossmodal.py <path_to_drseuss_image.jpg>")
        print("  Provide any Dr. Seuss illustration (Cat in the Hat, etc.)")
        sys.exit(1)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"PyTorch {torch.__version__}, Device: {DEVICE}")

    log("Loading TRIBEv2...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_FOLDER,
        device=DEVICE,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )

    # --- Predict both modalities ---
    log("Predicting brain response to IMAGE...")
    preds_image = predict_from_image(model, image_path)
    log(f"  Image preds: {preds_image.shape}")

    log("Predicting brain response to TEXT/AUDIO...")
    preds_text = predict_from_text(model, POEM)
    log(f"  Text preds:  {preds_text.shape}")

    # --- Spatial analysis ---
    log("\n=== Spatial Analysis ===")
    results = spatial_analysis(preds_image, preds_text)

    # --- Full table: cognitive + spatial ---
    from quantify import compute_full_table, print_full_table
    rows = compute_full_table({"image": preds_image, "text_audio": preds_text})
    print_full_table(rows)

    # --- Plot ---
    plot_crossmodal(results, rows)


if __name__ == "__main__":
    main()
