"""
Compare BDDN (image→fMRI) vs Tribe (video→fMRI) on the same 5-second clip.

BDDN: extract frames → predict per-frame brain response → average per ROI
Tribe: full video pipeline → predict temporal brain response → average per ROI

Both should show similar activation patterns (visual cortex high, etc.)
but may differ in magnitude and temporal dynamics.
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path("bddn")))

CACHE = Path("./cache")
CACHE.mkdir(exist_ok=True)

SINTEL_VIDEO = CACHE / "sintel_trailer.mp4"
CLIP_START, CLIP_END = 20, 25  # 5 seconds of action scene


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================
# Extract frames from video
# ============================================================

def extract_frames(video_path, start, end, fps=2):
    """Extract frames at given fps from a video clip."""
    from moviepy import VideoFileClip

    clip = VideoFileClip(str(video_path)).subclipped(start, end)
    frames = []
    for t in np.arange(0, clip.duration, 1 / fps):
        frame = clip.get_frame(t)
        frames.append(Image.fromarray(frame))
    clip.close()
    return frames


# ============================================================
# BDDN predictions
# ============================================================

def load_bddn():
    from brainnet.config import get_cfg_defaults
    from brainnet.backbone import ModifiedCLIP
    from brainnet.plmodel import PLModel

    cfg = get_cfg_defaults()
    plmodel = PLModel(cfg, ModifiedCLIP(), skip_data=True)
    checkpoint = "https://raw.githubusercontent.com/huzeyann/BrainDecodesDeepNets/main/assets/weights/clip_factorTopy.pth"
    plmodel.model.load_state_dict(
        torch.hub.load_state_dict_from_url(checkpoint, progress=True)
    )
    return plmodel.eval(), cfg


def bddn_predict_frames(plmodel, cfg, frames):
    """Predict brain response for each frame."""
    transform = T.Compose([
        T.Resize(cfg.DATASET.RESOLUTION),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    preds = []
    for frame in frames:
        x = transform(frame.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            brain_value, _ = plmodel.forward(x)
        preds.append(brain_value.squeeze(0).numpy())
    return np.stack(preds)  # (n_frames, 37984)


def bddn_roi_means(preds):
    """Extract ROI means from BDDN predictions."""
    from brainnet.roi import roi_dict, nsdgeneral_indices

    nsd_set = set(nsdgeneral_indices.tolist())
    nsd_to_local = {v: i for i, v in enumerate(nsdgeneral_indices)}

    roi_means = {}
    for roi_name, fsavg_idx in roi_dict.items():
        local = [nsd_to_local[v] for v in fsavg_idx if v in nsd_set]
        if local:
            roi_means[roi_name] = float(preds[:, local].mean())
    return roi_means


# ============================================================
# Tribe predictions
# ============================================================

def load_tribe():
    from tribev2.demo_utils import TribeModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )


def tribe_predict_video(model, video_path, start, end):
    """Run Tribe on a short video clip."""
    from moviepy import VideoFileClip
    from tribev2.demo_utils import download_file

    clip_path = CACHE / "compare_clip.mp4"
    if not clip_path.exists():
        clip = VideoFileClip(str(video_path)).subclipped(start, end)
        clip.write_videofile(str(clip_path), logger=None)
        clip.close()

    df = model.get_events_dataframe(video_path=str(clip_path))
    preds, segments = model.predict(events=df)
    return preds  # (n_timesteps, 20484)


def tribe_roi_means(preds):
    """Extract network-level means from Tribe predictions."""
    from quantify import extract_network_signals
    signals = extract_network_signals(preds)
    return {k: float(np.mean(v)) for k, v in signals.items()}


# ============================================================
# Print comparison
# ============================================================

def print_comparison(bddn_rois, tribe_rois):
    scale = 30

    print(f"\n{'=' * 70}")
    print(f"  BDDN (image→fMRI) vs Tribe (video→fMRI)")
    print(f"  Clip: Sintel trailer {CLIP_START}-{CLIP_END}s")
    print(f"{'=' * 70}")

    print(f"\n  BDDN visual ROIs (per-frame, averaged):")
    print(f"  {'ROI':8s} {'Mean':>8s}  Distribution")
    print(f"  {'-' * 50}")
    for roi, val in sorted(bddn_rois.items(), key=lambda x: -x[1]):
        bar_len = int(abs(val) * scale)
        bar = "█" * bar_len if val >= 0 else "░" * bar_len
        print(f"  {roi:8s} {val:8.4f}  {'|':>5s}{bar}")

    print(f"\n  Tribe network ROIs (video+audio+text, temporal avg):")
    print(f"  {'Network':8s} {'Mean':>8s}  Distribution")
    print(f"  {'-' * 50}")
    for roi, val in sorted(tribe_rois.items(), key=lambda x: -x[1]):
        bar_len = int(abs(val) * scale)
        bar = "█" * bar_len if val >= 0 else "░" * bar_len
        print(f"  {roi:8s} {val:8.4f}  {'|':>5s}{bar}")

    # Key comparison: visual processing
    bddn_visual = np.mean([bddn_rois.get(r, 0) for r in ["V1", "V2", "V3", "V4"]])
    bddn_face = np.mean([bddn_rois.get(r, 0) for r in ["FFA", "OFA"]])
    bddn_scene = np.mean([bddn_rois.get(r, 0) for r in ["PPA", "OPA"]])
    bddn_body = np.mean([bddn_rois.get(r, 0) for r in ["EBA", "FBA"]])

    tribe_vis = tribe_rois.get("Vis", 0)
    tribe_dan = tribe_rois.get("DAN", 0)
    tribe_dmn = tribe_rois.get("DMN", 0)
    tribe_fpn = tribe_rois.get("FPN", 0)

    print(f"\n  {'=' * 50}")
    print(f"  Summary:")
    print(f"  {'─' * 50}")
    print(f"  BDDN early visual (V1-V4):  {bddn_visual:8.4f}")
    print(f"  BDDN face areas (FFA/OFA):  {bddn_face:8.4f}")
    print(f"  BDDN scene areas (PPA/OPA): {bddn_scene:8.4f}")
    print(f"  BDDN body areas (EBA/FBA):  {bddn_body:8.4f}")
    print(f"  {'─' * 50}")
    print(f"  Tribe visual network:       {tribe_vis:8.4f}")
    print(f"  Tribe attention (DAN):      {tribe_dan:8.4f}")
    print(f"  Tribe default mode (DMN):   {tribe_dmn:8.4f}")
    print(f"  Tribe control (FPN):        {tribe_fpn:8.4f}")


def main():
    # --- Download video if needed ---
    if not SINTEL_VIDEO.exists():
        log("Downloading Sintel trailer...")
        from tribev2.demo_utils import download_file
        download_file(
            "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
            SINTEL_VIDEO,
        )

    # --- Extract frames ---
    log(f"Extracting frames from {CLIP_START}-{CLIP_END}s at 2fps...")
    frames = extract_frames(SINTEL_VIDEO, CLIP_START, CLIP_END, fps=2)
    log(f"  {len(frames)} frames extracted")

    # Save frames for reference
    for i, f in enumerate(frames):
        f.save(CACHE / f"frame_{i:02d}.jpg")

    # --- BDDN ---
    log("Loading BDDN...")
    plmodel, cfg = load_bddn()
    log(f"BDDN loaded ({plmodel.n_vertices} vertices)")

    log("Running BDDN on frames...")
    t0 = time.time()
    bddn_preds = bddn_predict_frames(plmodel, cfg, frames)
    bddn_rois = bddn_roi_means(bddn_preds)
    log(f"  BDDN done: {bddn_preds.shape} in {time.time() - t0:.1f}s")

    # --- Tribe ---
    log("Loading Tribe v2...")
    tribe_model = load_tribe()
    log("Tribe loaded")

    log("Running Tribe on video clip...")
    t0 = time.time()
    tribe_preds = tribe_predict_video(tribe_model, SINTEL_VIDEO, CLIP_START, CLIP_END)
    tribe_rois = tribe_roi_means(tribe_preds)
    log(f"  Tribe done: {tribe_preds.shape} in {time.time() - t0:.1f}s")

    # --- Compare ---
    print_comparison(bddn_rois, tribe_rois)

    # Save raw predictions
    np.save(CACHE / "bddn_preds.npy", bddn_preds)
    np.save(CACHE / "tribe_preds.npy", tribe_preds)
    log("Saved raw predictions to cache/")
    log("Done!")


if __name__ == "__main__":
    main()
