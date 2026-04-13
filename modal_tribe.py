"""
Modal app: Run Tribe v2 on TikTok videos with GPU acceleration.

Usage:
    modal run modal_tribe.py              # process downloaded videos
    modal run modal_tribe.py --n-sample 50  # download & process N from dataset
"""

import json
import modal

app = modal.App("braingym-tribe")

# Container image with Tribe v2 + all dependencies
tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "moviepy>=2.2.1",
        "huggingface_hub",
        "gtts",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "transformers",
        "x_transformers==1.27.20",
        "nibabel",
        "scipy",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
    .pip_install("yt-dlp")
)

# Persistent volume for caching model weights + features
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"
VIDEO_DIR = "/cache/tiktok_videos"
RESULTS_DIR = "/cache/results"


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    timeout=3600,
    memory=32768,
)
def run_batch(videos: list[dict]) -> list[dict]:
    """Process all videos sequentially on a single GPU. Model loads once."""
    import os
    import subprocess
    import time
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load model once ---
    print("Loading Tribe v2...")
    t0 = time.time()
    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
        },
    )
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

    # --- Load parcellation once ---
    SCHAEFER_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
    NETWORK_ALIASES = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
    }

    def parse_parcellation(annot_path):
        labels, ctab, names = nib.freesurfer.read_annot(annot_path)
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        networks = {}
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = NETWORK_ALIASES.get(parts[2], parts[2])
                if net not in networks:
                    networks[net] = []
                networks[net].extend(np.where(labels == idx)[0].tolist())
        return {k: np.array(v) for k, v in networks.items()}

    parcellation_dir = Path(CACHE_DIR) / "schaefer"
    parcellation_dir.mkdir(parents=True, exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parcellation_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        networks[hemi] = parse_parcellation(local)

    N_VERTICES_HEMI = 10242

    # --- Process videos sequentially ---
    results = []
    for i, video in enumerate(videos):
        video_id = str(video["id"])
        video_url = video["url"]
        print(f"\n[{i+1}/{len(videos)}] Processing {video_id}...")

        # Check cache
        result_path = Path(RESULTS_DIR) / f"{video_id}.json"
        if result_path.exists():
            print(f"  Cached, skipping")
            results.append(json.loads(result_path.read_text()))
            continue

        # Download
        video_path = Path(VIDEO_DIR) / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  Downloading...")
            cmd = [
                "yt-dlp", "-o", str(video_path),
                "--format", "mp4", "--max-filesize", "50M",
                "--no-warnings", "--quiet",
                video_url,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0 or not video_path.exists():
                results.append({"video_id": video_id, "error": f"download failed: {proc.stderr[:200]}"})
                continue
            print(f"  Downloaded {video_path.stat().st_size / 1024 / 1024:.1f}MB")

        # Predict
        t0 = time.time()
        try:
            df = model.get_events_dataframe(video_path=str(video_path))
            preds, segments = model.predict(events=df)
        except Exception as e:
            results.append({"video_id": video_id, "error": str(e)[:500]})
            continue

        elapsed = time.time() - t0
        print(f"  Predicted {preds.shape} in {elapsed:.1f}s")

        # Extract metrics
        preds_lh = preds[:, :N_VERTICES_HEMI]
        preds_rh = preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]

        net_signals = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}
        result = {
            "video_id": video_id,
            "n_timepoints": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
            **{f"net_{k}": v for k, v in net_means.items()},
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            "attention": net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "timeseries": net_signals,
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)
        print(f"  Done: cog_load={result['cognitive_load']:.4f} attn={result['attention']:.4f}")

    return results


@app.local_entrypoint()
def main(n_sample: int = 0):
    """
    Run Tribe on TikTok videos.

    --n-sample N: download N new videos from dataset (stratified by engagement)
    0 (default): process videos already in cache/tiktok_valid.json
    """
    from pathlib import Path

    cache = Path("./cache")

    if n_sample > 0:
        # Download fresh sample from dataset
        print(f"Sampling {n_sample} videos from TikTok-10M...")
        from datasets import load_dataset

        ds = load_dataset("The-data-company/TikTok-10M", split="train", streaming=True)
        videos = []
        for i, row in enumerate(ds):
            if len(videos) >= n_sample:
                break
            if row.get("duration") and 10 <= row["duration"] <= 60 and row.get("url"):
                videos.append(row)

        with open(cache / "tiktok_batch.json", "w") as f:
            json.dump(videos, f, indent=2, default=str)
    else:
        # Use existing sample
        with open(cache / "tiktok_valid.json") as f:
            videos = json.load(f)

    print(f"Processing {len(videos)} videos...")
    results = run_batch.remote(videos)

    # Separate successes and errors
    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\nDone: {len(successes)} succeeded, {len(errors)} failed")
    for e in errors:
        print(f"  FAIL {e['video_id']}: {e['error'][:100]}")

    # Save results
    output_path = cache / "tribe_tiktok_results.json"
    with open(output_path, "w") as f:
        json.dump(successes, f, indent=2)
    print(f"Results saved to {output_path}")

    # Quick summary
    if successes:
        print(f"\n{'=' * 60}")
        print(f"{'Video':>20s} {'CogLoad':>8s} {'Attn':>8s} {'Engage':>8s} {'DMN':>8s}")
        print(f"{'=' * 60}")
        for r in successes:
            print(f"{r['video_id']:>20s} {r['cognitive_load']:8.4f} "
                  f"{r['attention']:8.4f} {r['engagement']:8.4f} {r['net_DMN']:8.4f}")
