"""
Phase 3 (parallel): Run TRIBEv2 on Tsinghua videos via Modal.

Up to 10 GPUs in parallel. Each GPU loads the model once and processes a batch.

Usage:
    modal run experiments/retention/03_modal_tribe_parallel.py
"""

import json
import modal

app = modal.App("braingym-retention-parallel")

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
    max_containers=10,
)
def process_batch(video_names: list[str], batch_idx: int) -> list[dict]:
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
    print(f"[Batch {batch_idx}] Model loaded in {time.time() - t0:.1f}s on {device}")

    # Parcellation
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

    results = []
    for video_name in video_names:
        video_id = video_name.replace(".mp4", "")
        video_path = Path(VIDEO_DIR) / video_name

        result_path = Path(RESULTS_DIR) / f"{video_id}.json"
        if result_path.exists():
            print(f"  [{batch_idx}] {video_id}: cached")
            results.append(json.loads(result_path.read_text()))
            continue

        if not video_path.exists():
            results.append({"video_id": video_id, "error": f"not found: {video_path}"})
            continue

        print(f"  [{batch_idx}] {video_id}: predicting...")
        t0 = time.time()
        try:
            df = model.get_events_dataframe(video_path=str(video_path))
            preds, segments = model.predict(events=df)
        except Exception as e:
            results.append({"video_id": video_id, "error": str(e)[:500]})
            continue

        print(f"  [{batch_idx}] {video_id}: {preds.shape} in {time.time() - t0:.1f}s")

        preds_lh = preds[:, :N_VERTICES_HEMI]
        preds_rh = preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]

        net_signals = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        n_tp = preds.shape[0]
        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}

        result = {
            "video_id": video_id,
            "n_timepoints": n_tp,
            "n_vertices": int(preds.shape[1]),
            **{f"net_{k}": v for k, v in net_means.items()},
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            "attention": net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "timeseries": net_signals,
            "attention_ts": [net_signals["DAN"][t] - net_signals["DMN"][t] for t in range(n_tp)],
            "cognitive_load_ts": [net_signals["FPN"][t] + net_signals["DAN"][t] - net_signals["DMN"][t] for t in range(n_tp)],
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)

    return results


@app.local_entrypoint()
def main():
    from pathlib import Path

    cache = Path("./cache/tsinghua")
    video_dir = cache / "videos"
    selected_dir = cache / "selected"

    with open(selected_dir / "video_ids.json") as f:
        video_ids = json.load(f)

    # Check available videos
    available = [vid for vid in video_ids
                 if (video_dir / f"{vid}.mp4").exists() and (video_dir / f"{vid}.mp4").stat().st_size > 1000]
    print(f"Available: {len(available)} videos")

    if not available:
        print("No videos found.")
        return

    # Upload to volume
    print(f"Uploading {len(available)} videos...")
    _vol = modal.Volume.from_name("braingym-cache")
    with _vol.batch_upload(force=True) as batch:
        for _i, vid in enumerate(available):
            batch.put_file(str(video_dir / f"{vid}.mp4"), f"tsinghua_videos/{vid}.mp4")
            if (_i + 1) % 20 == 0:
                print(f"  {_i + 1}/{len(available)}")
    print("Upload complete")

    # Split into batches
    BATCH_SIZE = 5
    video_names = [f"{vid}.mp4" for vid in available]
    batches = [video_names[i:i + BATCH_SIZE] for i in range(0, len(video_names), BATCH_SIZE)]
    print(f"\nProcessing {len(video_names)} videos in {len(batches)} batches (up to 10 parallel GPUs)")

    # Fan out ALL batches in parallel
    all_results = []
    for result_batch in process_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(batches)],
        order_outputs=False,
    ):
        all_results.extend(result_batch)
        _successes = len([r for r in all_results if "error" not in r])
        _errors = len([r for r in all_results if "error" in r])
        print(f"  Progress: {_successes} succeeded, {_errors} errors")

    # Save
    successes = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    with open(selected_dir / "tribe_results.json", "w") as f:
        json.dump(successes, f, indent=2)

    print(f"\nDone: {len(successes)} succeeded, {len(errors)} failed")
    for e in errors:
        print(f"  FAIL {e['video_id']}: {e.get('error', '')[:80]}")
