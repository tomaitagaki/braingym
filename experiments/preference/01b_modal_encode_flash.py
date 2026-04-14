"""
Phase 1b: TRIBE encoding using the paper's flash protocol.

From TRIBEv2 paper section 5.9:
  "Images are presented for 1 second every 8 seconds"
  "We select the predicted response at t=5 after the image is shown"

This creates an 8-second video per image: 1s of image, 7s of gray screen.
Then takes the TRIBE prediction at t=5 (hemodynamic peak).

Usage:
    modal run experiments/preference/01b_modal_encode_flash.py
"""

import json
import modal

app = modal.App("braingym-preference-flash")

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
        "Pillow",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)
CACHE_DIR = "/cache"
IMAGE_DIR = "/cache/preference_images"
RESULTS_DIR = "/cache/preference_results_flash"


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def encode_flash_batch(image_names: list[str], batch_idx: int) -> list[dict]:
    """
    Encode images using the paper's flash protocol:
    1s image + 7s gray = 8s video. Take prediction at t=5.
    """
    import os
    import time
    import subprocess
    import tempfile
    import numpy as np
    import torch
    import nibabel as nib
    from pathlib import Path
    from urllib.request import urlretrieve

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # Load model
    print(f"[Flash Batch {batch_idx}] Loading TRIBE...")
    t0 = time.time()
    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
    )
    print(f"[Flash Batch {batch_idx}] Loaded in {time.time() - t0:.1f}s")

    # Load Schaefer parcellation
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

    parcellation_dir = Path(CACHE_DIR) / "schaefer"
    parcellation_dir.mkdir(parents=True, exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parcellation_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        labels, ctab, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        nets = {}
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = NETWORK_ALIASES.get(parts[2], parts[2])
                if net not in nets:
                    nets[net] = []
                nets[net].extend(np.where(labels == idx)[0].tolist())
        networks[hemi] = {k: np.array(v) for k, v in nets.items()}

    N_HEMI = 10242
    PEAK_T = 5  # Hemodynamic peak at t=5 per paper

    results = []
    for img_name in image_names:
        img_path = Path(IMAGE_DIR) / img_name
        result_path = Path(RESULTS_DIR) / f"tribe_flash_{img_name.rsplit('.', 1)[0]}.json"

        if result_path.exists():
            print(f"  [Flash {batch_idx}] {img_name}: cached")
            results.append(json.loads(result_path.read_text()))
            continue

        if not img_path.exists():
            results.append({"image": img_name, "error": f"not found: {img_path}"})
            continue

        print(f"  [Flash {batch_idx}] {img_name}: encoding (flash protocol)...")
        t0 = time.time()

        try:
            # Create 8-second video: 1s image + 7s gray
            # Step 1: Create a gray frame matching image dimensions
            from PIL import Image as PILImage
            img = PILImage.open(str(img_path))
            w, h = img.size
            # Make dimensions even for ffmpeg
            w = w - (w % 2)
            h = h - (h % 2)

            gray_path = tempfile.mktemp(suffix=".png")
            gray = PILImage.new("RGB", (w, h), (128, 128, 128))
            gray.save(gray_path)

            # Resize original to even dimensions
            img_resized_path = tempfile.mktemp(suffix=".jpg")
            img.resize((w, h)).save(img_resized_path)

            # Step 2: Create video with ffmpeg concat
            # 1-second image clip
            img_clip = tempfile.mktemp(suffix=".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1",
                "-i", img_resized_path,
                "-t", "1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", "25", "-an",
                img_clip,
            ], capture_output=True, check=True)

            # 7-second gray clip
            gray_clip = tempfile.mktemp(suffix=".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1",
                "-i", gray_path,
                "-t", "7",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", "25", "-an",
                gray_clip,
            ], capture_output=True, check=True)

            # Concatenate
            concat_list = tempfile.mktemp(suffix=".txt")
            with open(concat_list, "w") as f:
                f.write(f"file '{img_clip}'\n")
                f.write(f"file '{gray_clip}'\n")

            final_video = tempfile.mktemp(suffix=".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy", "-an",
                final_video,
            ], capture_output=True, check=True)

            # Run TRIBE
            df = model.get_events_dataframe(video_path=final_video)
            preds, segments = model.predict(events=df)

            # Cleanup temp files
            for f in [gray_path, img_resized_path, img_clip, gray_clip, concat_list, final_video]:
                try:
                    os.unlink(f)
                except OSError:
                    pass

        except Exception as e:
            results.append({"image": img_name, "error": str(e)[:500]})
            continue

        elapsed = time.time() - t0
        n_tp = preds.shape[0]
        print(f"  [Flash {batch_idx}] {img_name}: {preds.shape} in {elapsed:.1f}s")

        # Extract prediction at t=5 (hemodynamic peak) if available
        # Also save the full timeseries for analysis
        if n_tp > PEAK_T:
            peak_preds = preds[PEAK_T]  # (20484,) at t=5
        else:
            # If video too short for t=5, take the last available timepoint
            peak_preds = preds[-1]
            print(f"  [Flash {batch_idx}] WARNING: only {n_tp} timepoints, using t={n_tp-1} instead of t={PEAK_T}")

        # Also compute the average (for comparison with naive approach)
        avg_preds = preds.mean(axis=0)

        # Save full predictions + peak
        npy_path = Path(RESULTS_DIR) / f"tribe_flash_{img_name.rsplit('.', 1)[0]}_full.npy"
        np.save(str(npy_path), preds)

        npy_peak_path = Path(RESULTS_DIR) / f"tribe_flash_{img_name.rsplit('.', 1)[0]}_peak.npy"
        np.save(str(npy_peak_path), peak_preds)

        # Extract network means from the peak prediction
        preds_lh = peak_preds[:N_HEMI]
        preds_rh = peak_preds[N_HEMI:2 * N_HEMI]

        net_means_peak = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[lh_idx].mean() if len(lh_idx) > 0 else 0.0
            sig_rh = preds_rh[rh_idx].mean() if len(rh_idx) > 0 else 0.0
            net_means_peak[net_name] = float((sig_lh + sig_rh) / 2)

        # Also extract from average
        avg_lh = avg_preds[:N_HEMI]
        avg_rh = avg_preds[N_HEMI:2 * N_HEMI]

        net_means_avg = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = avg_lh[lh_idx].mean() if len(lh_idx) > 0 else 0.0
            sig_rh = avg_rh[rh_idx].mean() if len(rh_idx) > 0 else 0.0
            net_means_avg[net_name] = float((sig_lh + sig_rh) / 2)

        # Full timeseries per network (for temporal dynamics analysis)
        net_timeseries = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            ts = []
            for t in range(n_tp):
                sig_lh = preds[t, :N_HEMI][lh_idx].mean() if len(lh_idx) > 0 else 0.0
                sig_rh = preds[t, N_HEMI:2*N_HEMI][rh_idx].mean() if len(rh_idx) > 0 else 0.0
                ts.append(float((sig_lh + sig_rh) / 2))
            net_timeseries[net_name] = ts

        result = {
            "image": img_name,
            "model": "tribe_flash",
            "protocol": "1s_image_7s_gray",
            "n_timepoints": n_tp,
            "n_vertices": int(preds.shape[1]),
            "peak_timepoint": min(PEAK_T, n_tp - 1),
            "elapsed": round(elapsed, 1),
            "preds_full_path": str(npy_path),
            "preds_peak_path": str(npy_peak_path),
            # Peak (t=5) network means — the paper's recommended approach
            "network_means_peak": net_means_peak,
            "attention_peak": net_means_peak["DAN"] - net_means_peak["DMN"],
            "engagement_peak": -net_means_peak["DMN"],
            # Average network means — for comparison with naive 2s static approach
            "network_means_avg": net_means_avg,
            "attention_avg": net_means_avg["DAN"] - net_means_avg["DMN"],
            "engagement_avg": -net_means_avg["DMN"],
            # Full timeseries for temporal analysis
            "timeseries": net_timeseries,
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)

    return results


@app.local_entrypoint()
def main():
    from pathlib import Path

    cache = Path("cache/preference")
    pairs_path = cache / "selected_pairs.json"

    if not pairs_path.exists():
        print("Run 00_sample_pairs.py first!")
        return

    pairs = json.loads(pairs_path.read_text())
    print(f"Loaded {len(pairs)} pairs")

    all_images = set()
    for p in pairs:
        all_images.add(p["preferred_image"])
        all_images.add(p["nonpreferred_image"])
    all_images = sorted(all_images)
    print(f"Unique images: {len(all_images)}")

    # Images should already be on the volume from the naive run
    # But verify
    print("Checking images on volume...")

    BATCH_SIZE = 10  # Smaller batches — each video is 8s so encoding takes longer
    batches = [all_images[i:i + BATCH_SIZE] for i in range(0, len(all_images), BATCH_SIZE)]
    print(f"\n--- Flash protocol: {len(all_images)} images in {len(batches)} batches ---")

    results = []
    for result_batch in encode_flash_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(batches)],
        order_outputs=False,
    ):
        results.extend(result_batch)
        successes = len([r for r in results if "error" not in r])
        errors = len([r for r in results if "error" in r])
        print(f"  Progress: {successes} ok, {errors} errors")

    # Save results
    by_image = {r["image"]: r for r in results if "error" not in r}

    combined = []
    for pair in pairs:
        for side, is_pref in [("preferred_image", True), ("nonpreferred_image", False)]:
            img = pair[side]
            entry = {
                "pair_id": pair["pair_id"],
                "image": img,
                "is_preferred": is_pref,
                "prompt": pair["prompt"],
                "model": pair[f"model_{'preferred' if is_pref else 'nonpreferred'}"],
                "confidence": pair["confidence"],
            }
            if img in by_image:
                r = by_image[img]
                entry["tribe_flash_network_means_peak"] = r.get("network_means_peak", {})
                entry["tribe_flash_attention_peak"] = r.get("attention_peak")
                entry["tribe_flash_engagement_peak"] = r.get("engagement_peak")
                entry["tribe_flash_network_means_avg"] = r.get("network_means_avg", {})
                entry["tribe_flash_attention_avg"] = r.get("attention_avg")
                entry["tribe_flash_timeseries"] = r.get("timeseries", {})
            combined.append(entry)

    out_path = cache / "encoding_results_flash.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    ok = len([r for r in results if "error" not in r])
    errs = len([r for r in results if "error" in r])
    print(f"\n{'='*50}")
    print(f"Flash TRIBE: {ok}/{len(all_images)} encoded")
    print(f"Saved to {out_path}")

    for r in results:
        if "error" in r:
            print(f"  ERROR {r.get('image', '?')}: {r['error'][:80]}")
