"""
Run TRIBE v2 on a single video and produce brain metrics + timeseries plot.

Uploads the video to Modal volume, runs inference on A10G, saves:
  - JSON results with full timeseries + scalar metrics
  - PNG timeseries plot (attention, cog load, networks)

Usage:
    modal run modal_analyze_video.py --video-path /path/to/video.mov
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-analyze-video")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        "transformers>=4.57,<5",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "huggingface_hub",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "x_transformers==1.27.20",
        "nibabel",
        "scipy",
        "Pillow",
        "matplotlib",
        "moviepy>=2.2.1",
        "gtts",
        "nilearn",
        "mne",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git")
)


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def analyze_video(video_filename: str) -> dict:
    """Run TRIBE on a single video, return metrics + timeseries + plot bytes."""
    import os
    import time
    import numpy as np
    import torch
    import nibabel as nib
    from urllib.request import urlretrieve
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    video_path = Path(CACHE_DIR) / "uploads" / video_filename
    if not video_path.exists():
        return {"error": f"Video not found at {video_path}"}

    print(f"Video: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f}MB)")

    # ---- Load TRIBE ----
    print("Loading TRIBE v2...")
    t0 = time.time()
    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tribe = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
        },
    )
    print(f"  Loaded in {time.time() - t0:.1f}s on {device}")

    # ---- Load parcellation ----
    SCHAEFER_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
    ALIASES = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
    }

    parc_dir = Path(CACHE_DIR) / "schaefer"
    parc_dir.mkdir(exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parc_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        labels, _, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = ALIASES.get(parts[2], parts[2])
                if net not in networks:
                    networks[net] = {"lh": [], "rh": []}
                networks[net][hemi].extend(np.where(labels == idx)[0].tolist())
    for net in networks:
        for hemi in ("lh", "rh"):
            networks[net][hemi] = np.array(networks[net][hemi])

    N = 10242  # fsaverage5 vertices per hemisphere

    # ---- Run TRIBE ----
    print("Running TRIBE inference...")
    t0 = time.time()
    df = tribe.get_events_dataframe(video_path=str(video_path))
    preds, segments = tribe.predict(events=df)
    elapsed = time.time() - t0
    print(f"  Predictions: {preds.shape} in {elapsed:.1f}s")

    # ---- Extract network signals ----
    preds_lh = preds[:, :N]
    preds_rh = preds[:, N:2*N]
    net_signals = {}
    for net_name, masks in networks.items():
        lh_idx, rh_idx = masks["lh"], masks["rh"]
        sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
        sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
        net_signals[net_name] = (sig_lh + sig_rh) / 2

    dan = net_signals["DAN"]
    dmn = net_signals["DMN"]
    fpn = net_signals["FPN"]
    van = net_signals["VAN"]
    vis = net_signals["Vis"]
    limbic = net_signals["Limbic"]
    sommot = net_signals["SomMot"]

    n_tp = len(dan)
    attn = dan - dmn
    cog_load = fpn + dan - dmn

    # ---- Scalar metrics ----
    result = {
        "video": video_filename,
        "n_timepoints": n_tp,
        "n_vertices": int(preds.shape[1]),
        # Summary scalars
        "attention_mean": float(np.mean(attn)),
        "attention_first3s": float(np.mean(attn[:min(3, n_tp)])),
        "attention_peak": float(np.max(attn)),
        "attention_slope": float(np.polyfit(np.arange(n_tp), attn, 1)[0]) if n_tp > 1 else 0,
        "attention_var": float(np.var(attn)),
        "cognitive_load_mean": float(np.mean(cog_load)),
        "cognitive_load_first3s": float(np.mean(cog_load[:min(3, n_tp)])),
        "engagement": float(-np.mean(dmn)),
        "salience_mean": float(np.mean(van)),
        "emotional_mean": float(np.mean(limbic)),
        # Network means
        "net_DAN": float(np.mean(dan)),
        "net_DMN": float(np.mean(dmn)),
        "net_FPN": float(np.mean(fpn)),
        "net_VAN": float(np.mean(van)),
        "net_Vis": float(np.mean(vis)),
        "net_Limbic": float(np.mean(limbic)),
        "net_SomMot": float(np.mean(sommot)),
        # Temporal features
        "dan_slope": float(np.polyfit(np.arange(n_tp), dan, 1)[0]) if n_tp > 1 else 0,
        "dmn_slope": float(np.polyfit(np.arange(n_tp), dmn, 1)[0]) if n_tp > 1 else 0,
        "dan_peak_time": int(np.argmax(dan)),
        "van_peak_time": int(np.argmax(van)),
        # Full timeseries (1 Hz, each value = 1 second)
        "timeseries": {
            net: sig.tolist() for net, sig in net_signals.items()
        },
    }

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS: {video_filename}")
    print(f"{'='*60}")
    print(f"  Duration:            {n_tp}s ({n_tp} timepoints at 1Hz)")
    print(f"  Attention (first 3s): {result['attention_first3s']:.4f}")
    print(f"  Attention (mean):     {result['attention_mean']:.4f}")
    print(f"  Attention (peak):     {result['attention_peak']:.4f}")
    print(f"  Attention (slope):    {result['attention_slope']:.6f}")
    print(f"  Cognitive load:       {result['cognitive_load_mean']:.4f}")
    print(f"  Engagement (-DMN):    {result['engagement']:.4f}")
    print(f"  Salience (VAN):       {result['salience_mean']:.4f}")
    print(f"  Emotional (Limbic):   {result['emotional_mean']:.4f}")
    print(f"\n  Network means:")
    for net in ["DAN", "DMN", "FPN", "VAN", "Vis", "Limbic", "SomMot"]:
        print(f"    {net:8s}: {result[f'net_{net}']:.4f}")

    # ---- Extract video frames + audio ----
    from moviepy import VideoFileClip
    from PIL import Image
    import io

    clip = VideoFileClip(str(video_path))
    video_duration = clip.duration

    # Extract one frame per second (matching 1Hz brain predictions)
    frames_b64 = []
    for sec in range(n_tp):
        # Brain prediction at t accounts for hemodynamic lag internally
        frame_time = min(sec, video_duration - 0.01)
        frame = clip.get_frame(frame_time)
        img = Image.fromarray(frame.astype("uint8"))
        img.thumbnail((256, 256))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        frames_b64.append(list(buf.getvalue()))
    print(f"  Extracted {len(frames_b64)} frames")

    # Extract audio waveform (downsampled for visualization)
    audio_waveform = None
    if clip.audio is not None:
        try:
            audio_fps = 1000  # 1kHz for waveform viz
            samples = np.array(list(clip.audio.iter_frames(fps=audio_fps)))
            if samples.ndim > 1:
                samples = samples.mean(axis=1)  # mono
            audio_waveform = samples.tolist()
            print(f"  Audio waveform: {len(audio_waveform)} samples at {audio_fps}Hz")
        except Exception as e:
            print(f"  Audio extraction failed: {e}")

    clip.close()

    result["_frames"] = frames_b64
    result["audio_waveform"] = audio_waveform
    result["audio_sample_rate"] = 1000 if audio_waveform else 0

    # ---- Build timeline plot ----
    has_audio = audio_waveform is not None and len(audio_waveform) > 0
    n_rows = 4 if has_audio else 3
    height_ratios = [1.2, 0.5, 1.5, 1] if has_audio else [1.2, 1.5, 1]

    fig = plt.figure(figsize=(max(16, n_tp * 0.8), 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

    # Row 0: Video frames
    ax_frames = fig.add_subplot(gs[0])
    frame_strip = []
    for fb in frames_b64:
        img = Image.open(io.BytesIO(bytes(fb)))
        frame_strip.append(np.array(img))
    # Pad to uniform size
    h = max(f.shape[0] for f in frame_strip)
    w = max(f.shape[1] for f in frame_strip)
    padded = []
    for f in frame_strip:
        p = np.zeros((h, w, 3), dtype=np.uint8)
        p[:f.shape[0], :f.shape[1]] = f
        padded.append(p)
    strip = np.concatenate(padded, axis=1)
    ax_frames.imshow(strip)
    # Add second markers
    for i in range(n_tp):
        x = i * w + w // 2
        ax_frames.axvline(i * w, color="white", linewidth=0.5, alpha=0.3)
        ax_frames.text(x, 5, f"{i}s", ha="center", va="top", fontsize=7,
                       color="white", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
    ax_frames.set_xlim(0, strip.shape[1])
    ax_frames.axis("off")
    ax_frames.set_title(f"Video Frames — {video_filename}", fontsize=12, fontweight="bold")

    row_idx = 1

    # Row 1 (optional): Audio waveform
    if has_audio:
        ax_audio = fig.add_subplot(gs[row_idx])
        audio_arr = np.array(audio_waveform)
        audio_t = np.linspace(0, n_tp, len(audio_arr))
        ax_audio.fill_between(audio_t, audio_arr, alpha=0.5, color="#6366f1")
        ax_audio.plot(audio_t, audio_arr, color="#6366f1", linewidth=0.3)
        ax_audio.set_xlim(0, n_tp)
        ax_audio.set_ylabel("Audio", fontsize=10)
        ax_audio.set_xticklabels([])
        ax_audio.grid(alpha=0.2)
        ax_audio.spines["top"].set_visible(False)
        ax_audio.spines["right"].set_visible(False)
        row_idx += 1

    # Row 2: Brain metrics
    ax_brain = fig.add_subplot(gs[row_idx])
    t_arr = np.arange(n_tp)

    # Derived metrics (bold)
    reward = 0.4 * (-dmn) + 0.3 * attn + 0.2 * van + 0.1 * limbic
    ax_brain.plot(t_arr, attn, "-", label="Attention (DAN−DMN)", color="#ef4444", linewidth=2.5)
    ax_brain.plot(t_arr, -dmn, "-", label="Engagement (−DMN)", color="#3b82f6", linewidth=2.5)
    ax_brain.plot(t_arr, cog_load, "-", label="Cog Load", color="#f59e0b", linewidth=2)
    ax_brain.plot(t_arr, reward, "-", label="Brain Reward", color="#8b5cf6", linewidth=2)

    # Raw networks (faded)
    ax_brain.plot(t_arr, dan, "--", color="#ef4444", linewidth=0.8, alpha=0.3, label="DAN")
    ax_brain.plot(t_arr, dmn, "--", color="#3b82f6", linewidth=0.8, alpha=0.3, label="DMN")
    ax_brain.plot(t_arr, van, "--", color="#8b5cf6", linewidth=0.8, alpha=0.3, label="VAN")
    ax_brain.plot(t_arr, vis, "--", color="#22c55e", linewidth=0.8, alpha=0.3, label="Vis")

    ax_brain.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_brain.axvspan(0, min(3, n_tp), alpha=0.08, color="#ef4444")
    ax_brain.set_xlim(0, n_tp - 1)
    ax_brain.set_ylabel("Activation", fontsize=10)
    ax_brain.legend(loc="upper right", fontsize=7, ncol=4)
    ax_brain.grid(alpha=0.2)
    ax_brain.spines["top"].set_visible(False)
    ax_brain.spines["right"].set_visible(False)
    ax_brain.set_xticklabels([])
    row_idx += 1

    # Row 3: Network heatmap
    ax_heat = fig.add_subplot(gs[row_idx])
    net_order = ["DAN", "VAN", "FPN", "Vis", "SomMot", "DMN", "Limbic"]
    heatmap_data = np.array([net_signals[n] for n in net_order])
    vabs = np.abs(heatmap_data).max()
    im = ax_heat.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax_heat.set_yticks(range(len(net_order)))
    ax_heat.set_yticklabels(net_order, fontsize=9)
    ax_heat.set_xlabel("Time (s)", fontsize=10)
    fig.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.02)

    fig.tight_layout()

    # Save
    plot_dir = Path(CACHE_DIR) / "analyze_results"
    plot_dir.mkdir(exist_ok=True)
    stem = Path(video_filename).stem
    plot_path = plot_dir / f"{stem}_timeline.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Timeline saved: {plot_path}")

    # Save JSON to volume
    # Strip frames from JSON (too large), save separately
    result_for_json = {k: v for k, v in result.items() if k != "_frames"}
    json_path = plot_dir / f"{stem}_results.json"
    json_path.write_text(json.dumps(result_for_json, indent=2))
    print(f"  JSON saved: {json_path}")

    # Save raw preds
    preds_path = plot_dir / f"{stem}_preds.npy"
    np.save(str(preds_path), preds)
    print(f"  Preds saved: {preds_path} ({preds.shape})")

    # Save frames as numpy array
    frames_dir = plot_dir / f"{stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    for i, fb in enumerate(frames_b64):
        (frames_dir / f"{i:03d}.jpg").write_bytes(bytes(fb))
    print(f"  Frames saved: {frames_dir}/ ({len(frames_b64)} files)")

    vol.commit()

    # Return timeline plot bytes
    with open(plot_path, "rb") as f:
        result["_plot_bytes"] = list(f.read())

    return result


@app.local_entrypoint()
def main(video_path: str):
    """Upload video to Modal volume and run analysis."""
    local_path = Path(video_path).expanduser().resolve()
    if not local_path.exists():
        print(f"ERROR: Video not found: {local_path}")
        return

    print(f"Video: {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f}MB)")

    # Upload to Modal volume
    filename = local_path.name
    remote_dir = "uploads"
    print(f"Uploading to Modal volume braingym-cache/{remote_dir}/{filename}...")
    vol_ref = modal.Volume.from_name("braingym-cache")
    with vol_ref.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), f"{remote_dir}/{filename}")
    print("  Uploaded.")

    # Run analysis
    print("\nStarting TRIBE analysis on Modal A10G...")
    result = analyze_video.remote(filename)

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    # Save results locally
    cache = Path("./cache/analyze_results")
    cache.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    # Save timeline plot
    plot_bytes = bytes(result.pop("_plot_bytes"))
    plot_path = cache / f"{stem}_timeline.png"
    plot_path.write_bytes(plot_bytes)
    print(f"\nTimeline saved: {plot_path}")

    # Save frames
    frames_data = result.pop("_frames", [])
    if frames_data:
        frames_dir = cache / f"{stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, fb in enumerate(frames_data):
            (frames_dir / f"{i:03d}.jpg").write_bytes(bytes(fb))
        print(f"Frames saved: {frames_dir}/ ({len(frames_data)} files)")

    # Save JSON (without frames/large binary data)
    result.pop("audio_waveform", None)
    json_path = cache / f"{stem}_results.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"JSON saved: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Duration:              {result['n_timepoints']}s")
    print(f"  Attention (first 3s):  {result['attention_first3s']:.4f}")
    print(f"  Attention (mean):      {result['attention_mean']:.4f}")
    print(f"  Attention (peak):      {result['attention_peak']:.4f}")
    print(f"  Attention (slope):     {result['attention_slope']:.6f}")
    print(f"  Cognitive load:        {result['cognitive_load_mean']:.4f}")
    print(f"  Engagement (-DMN):     {result['engagement']:.4f}")
    print(f"  Salience (VAN):        {result['salience_mean']:.4f}")
    print(f"  Emotional (Limbic):    {result['emotional_mean']:.4f}")

    # Show where this video's first-3s attention falls in TikTok distribution
    tiktok_results_path = Path("./cache/tribe_tiktok_train_results.json")
    if tiktok_results_path.exists():
        import numpy as np
        tiktok_results = json.loads(tiktok_results_path.read_text())
        tiktok_attn = []
        for r in tiktok_results:
            if "timeseries" in r:
                ts_dan = np.array(r["timeseries"]["DAN"])
                ts_dmn = np.array(r["timeseries"]["DMN"])
                a = ts_dan - ts_dmn
                tiktok_attn.append(float(np.mean(a[:min(3, len(a))])))
        if tiktok_attn:
            pct = sum(1 for a in tiktok_attn if a < result["attention_first3s"]) / len(tiktok_attn) * 100
            print(f"\n  vs TikTok (n={len(tiktok_attn)}):")
            print(f"    First-3s attention percentile: {pct:.0f}th")
            print(f"    TikTok range: [{min(tiktok_attn):.4f}, {max(tiktok_attn):.4f}]")
