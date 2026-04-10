"""
Phase 3: Run TRIBEv2 on selected Tsinghua videos via Modal.

Adapted from modal_tribe_100.py — same GPU pipeline, different video source.
Downloads selected MP4s from Tsinghua, runs TRIBEv2 inference, extracts
7-network timeseries at ~1Hz for correlation with retention curves.

Usage:
    modal run experiments/retention/03_modal_tribe.py
"""

import json
import modal

app = modal.App("braingym-retention")

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
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"
VIDEO_DIR = "/cache/tsinghua_videos"
RESULTS_DIR = "/cache/tsinghua_results"


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def process_batch(video_paths: list[str], batch_idx: int) -> list[dict]:
    """Process a batch of videos. Model loads once per batch."""
    import os
    import time
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # Load model
    print(f"[Batch {batch_idx}] Loading TRIBEv2...")
    t0 = time.time()
    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
    )
    print(f"[Batch {batch_idx}] Model loaded in {time.time() - t0:.1f}s")

    # Load parcellation
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

    # Process videos
    results = []
    for video_path_str in video_paths:
        video_path = Path(video_path_str)
        video_id = video_path.stem

        result_path = Path(RESULTS_DIR) / f"{video_id}.json"
        if result_path.exists():
            print(f"  {video_id}: cached")
            results.append(json.loads(result_path.read_text()))
            continue

        if not video_path.exists():
            results.append({"video_id": video_id, "error": f"file not found: {video_path}"})
            continue

        print(f"  {video_id}: predicting...")
        t0 = time.time()
        try:
            df = model.get_events_dataframe(video_path=str(video_path))
            preds, segments = model.predict(events=df)
        except Exception as e:
            results.append({"video_id": video_id, "error": str(e)[:500]})
            continue

        elapsed = time.time() - t0
        print(f"  {video_id}: {preds.shape} in {elapsed:.1f}s")

        # Extract per-timepoint network activations
        preds_lh = preds[:, :N_VERTICES_HEMI]
        preds_rh = preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]

        net_signals = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        n_timepoints = preds.shape[0]
        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}

        result = {
            "video_id": video_id,
            "n_timepoints": n_timepoints,
            "n_vertices": int(preds.shape[1]),
            # Scalar summaries
            **{f"net_{k}": v for k, v in net_means.items()},
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            "attention": net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            # Full timeseries (the key output for retention correlation)
            "timeseries": net_signals,
            # Derived timeseries
            "attention_ts": [
                net_signals["DAN"][t] - net_signals["DMN"][t]
                for t in range(n_timepoints)
            ],
            "cognitive_load_ts": [
                net_signals["FPN"][t] + net_signals["DAN"][t] - net_signals["DMN"][t]
                for t in range(n_timepoints)
            ],
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)

    return results


@app.local_entrypoint()
def main():
    from pathlib import Path

    cache = Path("./cache/tsinghua/selected")

    # Load selected video IDs
    with open(cache / "video_ids.json") as f:
        video_ids = json.load(f)

    # Build paths to local MP4s (downloaded separately)
    video_dir = Path("./cache/tsinghua/videos")
    video_paths = []
    missing = []
    for vid in video_ids:
        mp4 = video_dir / f"{vid}.mp4"
        if mp4.exists():
            video_paths.append(str(mp4))
        else:
            missing.append(vid)

    if missing:
        print(f"WARNING: {len(missing)} videos not downloaded yet")
        print(f"Download MP4s to {video_dir}/ before running")
        if not video_paths:
            return

    print(f"Processing {len(video_paths)} videos...")

    # Upload videos to Modal volume first
    vol_local = modal.Volume.from_name("braingym-cache")
    for vp in video_paths:
        remote_path = f"tsinghua_videos/{Path(vp).name}"
        # Modal volume upload would go here
        # For now, videos need to be on the volume already

    # Process in batches of 5
    BATCH_SIZE = 5
    all_results = []

    for batch_idx in range(0, len(video_paths), BATCH_SIZE):
        batch = video_paths[batch_idx:batch_idx + BATCH_SIZE]
        # Remap to volume paths
        vol_paths = [f"{VIDEO_DIR}/{Path(p).name}" for p in batch]
        batch_num = batch_idx // BATCH_SIZE + 1
        total = (len(video_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total} ({len(batch)} videos)")

        results = process_batch.remote(vol_paths, batch_num)
        all_results.extend(results)

        # Save intermediate
        successes = [r for r in all_results if "error" not in r]
        errors = [r for r in all_results if "error" in r]
        with open(cache / "tribe_results.json", "w") as f:
            json.dump(successes, f, indent=2)
        print(f"  {len(successes)} succeeded, {len(errors)} failed")

    print(f"\nDone: {len([r for r in all_results if 'error' not in r])} videos processed")
