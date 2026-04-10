"""Step 1: Run BDDN on frames from Sintel trailer. Use bddn venv."""

import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path("bddn")))

CACHE = Path("./cache")
CACHE.mkdir(exist_ok=True)
SINTEL_VIDEO = CACHE / "sintel_trailer.mp4"
CLIP_START, CLIP_END = 20, 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def extract_frames(video_path, start, end, fps=2):
    from moviepy import VideoFileClip
    clip = VideoFileClip(str(video_path)).subclipped(start, end)
    frames = []
    for t in np.arange(0, clip.duration, 1 / fps):
        frames.append(Image.fromarray(clip.get_frame(t)))
    clip.close()
    return frames


def main():
    # Download video if needed
    if not SINTEL_VIDEO.exists():
        import urllib.request
        log("Downloading Sintel trailer...")
        urllib.request.urlretrieve(
            "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
            SINTEL_VIDEO,
        )

    # Extract frames
    log(f"Extracting frames from {CLIP_START}-{CLIP_END}s at 2fps...")
    frames = extract_frames(SINTEL_VIDEO, CLIP_START, CLIP_END, fps=2)
    log(f"  {len(frames)} frames extracted")
    for i, f in enumerate(frames):
        f.save(CACHE / f"frame_{i:02d}.jpg")

    # Load BDDN
    log("Loading BDDN...")
    from brainnet.config import get_cfg_defaults
    from brainnet.backbone import ModifiedCLIP
    from brainnet.plmodel import PLModel

    cfg = get_cfg_defaults()
    plmodel = PLModel(cfg, ModifiedCLIP(), skip_data=True)
    checkpoint = "https://raw.githubusercontent.com/huzeyann/BrainDecodesDeepNets/main/assets/weights/clip_factorTopy.pth"
    plmodel.model.load_state_dict(
        torch.hub.load_state_dict_from_url(checkpoint, progress=True)
    )
    plmodel = plmodel.eval()
    log(f"BDDN loaded ({plmodel.n_vertices} vertices)")

    # Predict per frame
    transform = T.Compose([
        T.Resize(cfg.DATASET.RESOLUTION),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    log("Running BDDN on frames...")
    t0 = time.time()
    preds = []
    for i, frame in enumerate(frames):
        x = transform(frame.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            brain_value, _ = plmodel.forward(x)
        preds.append(brain_value.squeeze(0).numpy())
        log(f"  Frame {i}: done")
    preds = np.stack(preds)
    log(f"BDDN done: {preds.shape} in {time.time() - t0:.1f}s")

    # Extract ROI means
    from brainnet.roi import roi_dict, nsdgeneral_indices
    nsd_set = set(nsdgeneral_indices.tolist())
    nsd_to_local = {v: i for i, v in enumerate(nsdgeneral_indices)}

    roi_means = {}
    for roi_name, fsavg_idx in roi_dict.items():
        local = [nsd_to_local[v] for v in fsavg_idx if v in nsd_set]
        if local:
            roi_means[roi_name] = float(preds[:, local].mean())

    # Save results
    np.save(CACHE / "bddn_preds.npy", preds)
    with open(CACHE / "bddn_rois.json", "w") as f:
        json.dump(roi_means, f, indent=2)

    log("Results saved to cache/bddn_preds.npy and cache/bddn_rois.json")

    # Print
    print(f"\n{'=' * 40}")
    print(f"  BDDN ROI activations (frames {CLIP_START}-{CLIP_END}s)")
    print(f"{'=' * 40}")
    for roi, val in sorted(roi_means.items(), key=lambda x: -x[1]):
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"  {roi:8s} {val:8.4f}  {sign}{bar}")

    log("Done! Now run: source tribev2/.venv/bin/activate && python compare_tribe.py")


if __name__ == "__main__":
    main()
