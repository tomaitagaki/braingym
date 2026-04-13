"""
Modal app: Run Tribe v2 on 100 TikTok videos (stratified by engagement).
Excludes the 9 test-set videos from tiktok_valid.json.

Processes in batches of 10 on a single GPU to stay within timeout.
Results cached per-video on Modal volume — safe to rerun.

Usage:
    modal run modal_tribe_100.py
"""

import json
import modal

app = modal.App("braingym-tribe-100")

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
        "datasets",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
    .pip_install("yt-dlp")
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"
VIDEO_DIR = "/cache/tiktok_videos"
RESULTS_DIR = "/cache/results"


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def process_batch(videos: list[dict], batch_idx: int) -> list[dict]:
    """Process a batch of videos on a single GPU. Model loads once per batch."""
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

    # Persist HuggingFace cache on the volume so weights aren't re-downloaded
    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # --- Load model once ---
    print(f"[Batch {batch_idx}] Loading Tribe v2...")
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
    print(f"[Batch {batch_idx}] Model loaded in {time.time() - t0:.1f}s on {device}")

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

    # --- Process videos ---
    results = []
    for i, video in enumerate(videos):
        video_id = str(video["id"])
        video_url = video["url"]
        print(f"\n[Batch {batch_idx} | {i+1}/{len(videos)}] Processing {video_id}...")

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
                results.append({
                    "video_id": video_id,
                    "error": f"download failed: {proc.stderr[:200]}",
                })
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
def main():
    from pathlib import Path
    from datasets import load_dataset

    cache = Path("./cache")

    # --- Load test set IDs to exclude ---
    test_ids = set()
    test_path = cache / "tiktok_valid.json"
    if test_path.exists():
        with open(test_path) as f:
            test_ids = {str(v["id"]) for v in json.load(f)}
    print(f"Excluding {len(test_ids)} test-set videos")

    # --- Sample 100 videos stratified by engagement ---
    # Dataset is sorted by engagement (highest first)
    # Sample from different tiers to get spread
    train_path = cache / "tiktok_train_100.json"

    if train_path.exists():
        print(f"Loading existing sample from {train_path}")
        with open(train_path) as f:
            train_videos = json.load(f)
    else:
        print("Sampling 100 videos from TikTok-10M (stratified)...")
        ds = load_dataset("The-data-company/TikTok-10M", split="train", streaming=True)

        # Collect candidates from different positions in the dataset
        # Dataset is roughly sorted by play_count descending
        buckets = {
            "viral": [],      # first 200 rows (top viral)
            "high": [],       # rows 200-1000
            "mid": [],        # rows 1000-5000
            "low": [],        # rows 5000-20000
        }
        target_per_bucket = 25

        for i, row in enumerate(ds):
            if i >= 20000:
                break

            # Filter: 10-60s, has URL, not in test set
            if not (row.get("duration") and 10 <= row["duration"] <= 60 and row.get("url")):
                continue
            if str(row.get("id")) in test_ids:
                continue

            if i < 200 and len(buckets["viral"]) < target_per_bucket:
                buckets["viral"].append(row)
            elif 200 <= i < 1000 and len(buckets["high"]) < target_per_bucket:
                buckets["high"].append(row)
            elif 1000 <= i < 5000 and len(buckets["mid"]) < target_per_bucket:
                buckets["mid"].append(row)
            elif 5000 <= i < 20000 and len(buckets["low"]) < target_per_bucket:
                buckets["low"].append(row)

            # Early exit if all buckets full
            if all(len(b) >= target_per_bucket for b in buckets.values()):
                break

        train_videos = []
        for tier, vids in buckets.items():
            print(f"  {tier}: {len(vids)} videos")
            train_videos.extend(vids)

        # Save sample metadata
        with open(train_path, "w") as f:
            json.dump(train_videos, f, indent=2, default=str)
        print(f"Saved {len(train_videos)} videos to {train_path}")

    # --- Also save engagement metadata for later analysis ---
    meta_path = cache / "tiktok_train_100_meta.json"
    meta = [
        {
            "id": v["id"],
            "desc": str(v.get("desc", ""))[:100],
            "duration": v.get("duration"),
            "play_count": v.get("play_count"),
            "digg_count": v.get("digg_count"),
            "comment_count": v.get("comment_count"),
            "share_count": v.get("share_count"),
            "vq_score": v.get("vq_score"),
            "url": v.get("url"),
        }
        for v in train_videos
    ]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # --- Process in batches of 5 (each batch = ~30 min max) ---
    BATCH_SIZE = 5
    all_results = []

    for batch_idx in range(0, len(train_videos), BATCH_SIZE):
        batch = train_videos[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(train_videos) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} videos)")
        print(f"{'='*60}")

        results = process_batch.remote(batch, batch_num)
        all_results.extend(results)

        # Save intermediate results after each batch
        successes = [r for r in all_results if "error" not in r]
        errors = [r for r in all_results if "error" in r]
        output_path = cache / "tribe_tiktok_train_results.json"
        with open(output_path, "w") as f:
            json.dump(successes, f, indent=2)
        print(f"  Saved {len(successes)} results so far ({len(errors)} errors)")

    # --- Final summary ---
    successes = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    print(f"\n{'='*60}")
    print(f"DONE: {len(successes)} succeeded, {len(errors)} failed")
    print(f"{'='*60}")

    for e in errors:
        print(f"  FAIL {e['video_id']}: {e['error'][:80]}")

    output_path = cache / "tribe_tiktok_train_results.json"
    with open(output_path, "w") as f:
        json.dump(successes, f, indent=2)
    print(f"\nResults: {output_path}")
    print(f"Metadata: {meta_path}")
    print(f"Test set: {test_path} ({len(test_ids)} videos)")
