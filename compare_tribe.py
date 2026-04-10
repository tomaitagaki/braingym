"""Step 2: Run Tribe on the same clip, then compare with BDDN results. Use tribev2 venv."""

import time
import json
import numpy as np
import torch
from pathlib import Path

CACHE = Path("./cache")
SINTEL_VIDEO = CACHE / "sintel_trailer.mp4"
CLIP_START, CLIP_END = 20, 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    # --- Load BDDN results ---
    bddn_rois_path = CACHE / "bddn_rois.json"
    if not bddn_rois_path.exists():
        print("ERROR: Run compare_bddn.py first (with bddn venv)")
        return
    with open(bddn_rois_path) as f:
        bddn_rois = json.load(f)
    log("Loaded BDDN results from cache")

    # --- Create clip ---
    clip_path = CACHE / "compare_clip.mp4"
    if not clip_path.exists():
        log(f"Creating {CLIP_START}-{CLIP_END}s clip...")
        from moviepy import VideoFileClip
        clip = VideoFileClip(str(SINTEL_VIDEO)).subclipped(CLIP_START, CLIP_END)
        clip.write_videofile(str(clip_path), logger=None)
        clip.close()

    # --- Load Tribe ---
    log("Loading Tribe v2...")
    from tribev2.demo_utils import TribeModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
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
    log("Tribe loaded")

    # --- Run Tribe ---
    log("Running Tribe on video clip...")
    t0 = time.time()
    df = model.get_events_dataframe(video_path=str(clip_path))
    preds, segments = model.predict(events=df)
    log(f"Tribe done: {preds.shape} in {time.time() - t0:.1f}s")

    # --- Extract Tribe ROIs ---
    from quantify import extract_network_signals
    signals = extract_network_signals(preds)
    tribe_rois = {k: float(np.mean(v)) for k, v in signals.items()}

    # Save
    np.save(CACHE / "tribe_preds.npy", preds)
    with open(CACHE / "tribe_rois.json", "w") as f:
        json.dump(tribe_rois, f, indent=2)

    # ============================================================
    # COMPARISON
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"  BDDN (image→fMRI) vs Tribe (video→fMRI)")
    print(f"  Clip: Sintel trailer {CLIP_START}-{CLIP_END}s")
    print(f"{'=' * 70}")

    # BDDN
    print(f"\n  BDDN — visual ROIs (per-frame avg, 37984 vertices):")
    print(f"  {'ROI':8s} {'Mean':>8s}")
    print(f"  {'-' * 30}")
    for roi, val in sorted(bddn_rois.items(), key=lambda x: -x[1]):
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"  {roi:8s} {val:8.4f}  {sign}{bar}")

    # Tribe
    print(f"\n  Tribe — network ROIs (video+audio+text, 20484 vertices):")
    print(f"  {'Network':8s} {'Mean':>8s}")
    print(f"  {'-' * 30}")
    for roi, val in sorted(tribe_rois.items(), key=lambda x: -x[1]):
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"  {roi:8s} {val:8.4f}  {sign}{bar}")

    # Summary comparison
    bddn_visual = np.mean([bddn_rois.get(r, 0) for r in ["V1", "V2", "V3", "V4"]])
    bddn_face = np.mean([bddn_rois.get(r, 0) for r in ["FFA", "OFA"]])
    bddn_scene = np.mean([bddn_rois.get(r, 0) for r in ["PPA", "OPA"]])
    bddn_body = np.mean([bddn_rois.get(r, 0) for r in ["EBA", "FBA"]])

    print(f"\n  {'=' * 55}")
    print(f"  Cross-model summary")
    print(f"  {'─' * 55}")
    print(f"  {'Measure':<30s} {'BDDN':>10s} {'Tribe':>10s}")
    print(f"  {'─' * 55}")
    print(f"  {'Early visual (V1-V4)':<30s} {bddn_visual:10.4f} {'—':>10s}")
    print(f"  {'Face selective (FFA/OFA)':<30s} {bddn_face:10.4f} {'—':>10s}")
    print(f"  {'Scene selective (PPA/OPA)':<30s} {bddn_scene:10.4f} {'—':>10s}")
    print(f"  {'Body selective (EBA/FBA)':<30s} {bddn_body:10.4f} {'—':>10s}")
    print(f"  {'Visual network':<30s} {'—':>10s} {tribe_rois.get('Vis', 0):10.4f}")
    print(f"  {'Dorsal attention (DAN)':<30s} {'—':>10s} {tribe_rois.get('DAN', 0):10.4f}")
    print(f"  {'Default mode (DMN)':<30s} {'—':>10s} {tribe_rois.get('DMN', 0):10.4f}")
    print(f"  {'Frontoparietal (FPN)':<30s} {'—':>10s} {tribe_rois.get('FPN', 0):10.4f}")
    print(f"  {'Somatomotor':<30s} {'—':>10s} {tribe_rois.get('SomMot', 0):10.4f}")
    print(f"  {'Ventral attention (VAN)':<30s} {'—':>10s} {tribe_rois.get('VAN', 0):10.4f}")
    print(f"  {'Limbic':<30s} {'—':>10s} {tribe_rois.get('Limbic', 0):10.4f}")

    log("Done!")


if __name__ == "__main__":
    main()
