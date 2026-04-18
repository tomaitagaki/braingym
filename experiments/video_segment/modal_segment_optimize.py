"""
Video segment optimization: find the most brain-engaging window of a real video.

Takes a source video (nature documentary, movie clip — in Tribe's training distribution)
and optimizes: which segment, at what speed, with what framing maximizes brain engagement.

Content is FIXED — the optimizer can only change presentation (timing, speed, crop).
Cannot degenerate because every output is a real video segment.

Paradigm:
    source.mp4 → ffmpeg(start, duration, speed, crop) → segment.mp4 → Tribe → reward

Usage:
    # Download a sample clip first (nature documentary, creative commons)
    modal run experiments/video_segment/modal_segment_optimize.py

    # With a specific source video on the Modal volume
    modal run experiments/video_segment/modal_segment_optimize.py --source-video /cache/my_clip.mp4
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-segment-optimize")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

segment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        "transformers>=4.48,<5",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "moviepy>=2.2.1",
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
        "yt-dlp",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)


# ============================================================
# PARAM SPACE
# ============================================================

def make_param_space(source_duration: float):
    """Create param space based on source video duration."""
    max_start = max(0, source_duration - 15)  # at least 15s left
    return {
        "start_time": {
            "type": "float",
            "low": 0.0,
            "high": max_start,
            "desc": "Where in the source to start the segment",
        },
        "duration": {
            "type": "float",
            "low": 10.0,
            "high": min(30.0, source_duration),
            "desc": "Segment length in seconds",
        },
        "playback_speed": {
            "type": "float",
            "low": 0.7,
            "high": 1.5,
            "desc": "Speed multiplier (1.0 = normal)",
        },
        "crop_x": {
            "type": "float",
            "low": 0.0,
            "high": 0.25,
            "desc": "Horizontal crop from each side (0 = no crop, 0.25 = 50% zoom)",
        },
        "crop_y": {
            "type": "float",
            "low": 0.0,
            "high": 0.25,
            "desc": "Vertical crop from each side",
        },
    }


# ============================================================
# GRPO (same as manim experiment)
# ============================================================

class GRPO:
    def __init__(self, param_space, lr=0.4, temperature=0.5):
        import numpy as np
        self.rng = np.random.default_rng()
        self.lr = lr
        self.temperature = temperature
        self.distributions = {}
        for name, spec in param_space.items():
            if spec["type"] == "float":
                lo, hi = spec["low"], spec["high"]
                self.distributions[name] = {
                    "type": "gaussian", "mu": (lo + hi) / 2,
                    "sigma": (hi - lo) / 4, "low": lo, "high": hi,
                }
            elif spec["type"] == "categorical":
                choices = spec["choices"]
                n = len(choices)
                self.distributions[name] = {
                    "type": "categorical",
                    "probs": {c: 1.0 / n for c in choices},
                }

    def suggest(self, n):
        import numpy as np
        samples = []
        for _ in range(n):
            params = {}
            for name, dist in self.distributions.items():
                if dist["type"] == "gaussian":
                    val = self.rng.normal(dist["mu"], dist["sigma"])
                    val = float(np.clip(val, dist["low"], dist["high"]))
                    params[name] = val
                elif dist["type"] == "categorical":
                    choices = list(dist["probs"].keys())
                    probs = np.array([dist["probs"][c] for c in choices])
                    probs = probs / probs.sum()
                    params[name] = self.rng.choice(choices, p=probs)
            samples.append(params)
        return samples

    def update(self, params_list, rewards):
        import numpy as np
        rewards = np.array(rewards)
        if len(rewards) < 2 or rewards.std() < 1e-8:
            return
        advantages = (rewards - rewards.mean()) / rewards.std()
        scaled = advantages / self.temperature
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        for name, dist in self.distributions.items():
            if dist["type"] == "gaussian":
                values = np.array([p[name] for p in params_list])
                new_mu = float(np.sum(weights * values))
                new_sigma = float(np.sqrt(np.sum(weights * (values - new_mu) ** 2)))
                new_sigma = max(new_sigma, (dist["high"] - dist["low"]) * 0.05)
                dist["mu"] = dist["mu"] * (1 - self.lr) + new_mu * self.lr
                dist["sigma"] = dist["sigma"] * (1 - self.lr) + new_sigma * self.lr


# ============================================================
# MAIN
# ============================================================

@app.function(
    image=segment_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=32768,
)
def run_optimization(
    source_video: str = "",
    n_generations: int = 10,
    n_samples: int = 6,
):
    """GRPO loop: segment params → ffmpeg extract → Tribe → reward → update."""
    import os
    import time
    import subprocess
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "segment_optimize"
    results_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = results_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # ---- Get source video ----
    if source_video and Path(source_video).exists():
        src_path = Path(source_video)
    else:
        # Download a creative commons nature clip
        src_path = results_dir / "source.mp4"
        if not src_path.exists():
            print("Downloading sample nature clip...")
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--format", "mp4",
                    "--max-filesize", "50M",
                    "--output", str(src_path),
                    # BBC Earth creative commons sample — coral reef (2 min)
                    "https://www.youtube.com/watch?v=qULsFslUPY8",
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not src_path.exists():
                # Fallback: generate a synthetic nature-like video with ffmpeg
                print("Download failed, generating synthetic source...")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-i", "mandelbrot=size=640x480:maxiter=200:rate=15",
                        "-t", "120",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        str(src_path),
                    ],
                    check=True, capture_output=True,
                )
            vol.commit()

    # Get source duration
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", str(src_path),
        ],
        capture_output=True, text=True,
    )
    src_duration = float(json.loads(probe.stdout)["format"]["duration"])
    print(f"Source: {src_path.name} ({src_duration:.1f}s)")

    # ---- Load Tribe ----
    print("Loading Tribe v2...")
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
    print(f"Loaded on {device}")

    # Load parcellation
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

    N_VERT = 10242

    def score_video(video_path):
        df = tribe.get_events_dataframe(video_path=str(video_path))
        preds, _ = tribe.predict(events=df)
        preds_lh = preds[:, :N_VERT]
        preds_rh = preds[:, N_VERT:2 * N_VERT]
        signals = {}
        for net_name, masks in networks.items():
            lh_idx, rh_idx = masks["lh"], masks["rh"]
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            signals[net_name] = (sig_lh + sig_rh) / 2
        dan, dmn = signals["DAN"], signals["DMN"]
        van, limbic = signals["VAN"], signals["Limbic"]
        attn = dan - dmn
        n3 = min(3, len(dan))
        reward = float(
            0.4 * (-np.mean(dmn)) + 0.3 * np.mean(attn)
            + 0.2 * np.mean(van) + 0.1 * np.mean(limbic)
        )
        metrics = {
            "reward": reward,
            "attention": float(np.mean(attn)),
            "engagement": float(-np.mean(dmn)),
            "cognitive_load": float(np.mean(signals["FPN"]) + np.mean(dan) - np.mean(dmn)),
            "hook_3s": float(np.mean(attn[:n3])),
            "peak_attn": float(np.max(attn)),
            "sustained": float(-np.std(dmn)),
            "n_trs": int(preds.shape[0]),
        }
        return reward, metrics, {k: v.tolist() for k, v in signals.items()}

    def extract_segment(params, output_path):
        """Extract and transform a video segment via ffmpeg."""
        start = params["start_time"]
        dur = params["duration"]
        speed = params["playback_speed"]
        cx = params["crop_x"]
        cy = params["crop_y"]

        # Build crop filter
        # crop=in_w*(1-2*cx):in_h*(1-2*cy):in_w*cx:in_h*cy
        filters = []
        if cx > 0.01 or cy > 0.01:
            filters.append(
                f"crop=in_w*{1-2*cx:.3f}:in_h*{1-2*cy:.3f}:in_w*{cx:.3f}:in_h*{cy:.3f}"
            )
        # Scale back to 480p
        filters.append("scale=640:480")
        # Speed adjustment
        if abs(speed - 1.0) > 0.05:
            filters.append(f"setpts={1/speed:.3f}*PTS")

        vf = ",".join(filters)

        # Audio speed (atempo only supports 0.5-100)
        af_parts = []
        if abs(speed - 1.0) > 0.05:
            af_parts.append(f"atempo={speed:.3f}")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-t", f"{dur:.2f}",
            "-i", str(src_path),
            "-vf", vf,
        ]
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]
        cmd += [
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-300:]}")
        return output_path

    # ---- Optimization loop ----
    param_space = make_param_space(src_duration)
    opt = GRPO(param_space, lr=0.4, temperature=0.5)
    all_results = []

    print(f"\n{'=' * 60}")
    print(f"Segment Optimization: {n_generations} gen x {n_samples} samples")
    print(f"Source: {src_duration:.1f}s, segment range: 10-30s")
    print(f"{'=' * 60}\n")

    for gen in range(n_generations):
        t0 = time.time()
        param_batch = opt.suggest(n_samples)

        gen_rewards = []
        gen_metrics = []
        gen_params = []

        for i, params in enumerate(param_batch):
            run_id = f"g{gen:02d}_s{i:02d}"
            seg_path = segments_dir / f"{run_id}.mp4"

            try:
                extract_segment(params, seg_path)
                reward, metrics, timeseries = score_video(seg_path)
                gen_rewards.append(reward)
                gen_metrics.append(metrics)
                gen_params.append(params)

                print(
                    f"  [{run_id}] reward={reward:+.4f}  "
                    f"start={params['start_time']:.1f}s  "
                    f"dur={params['duration']:.1f}s  "
                    f"speed={params['playback_speed']:.2f}x  "
                    f"attn={metrics['attention']:+.3f}"
                )
            except Exception as e:
                print(f"  [{run_id}] FAILED: {e}")
                gen_rewards.append(0.0)
                gen_metrics.append({"reward": 0.0})
                gen_params.append(params)

        opt.update(gen_params, gen_rewards)
        elapsed = time.time() - t0

        rewards_np = np.array(gen_rewards)
        best_idx = int(rewards_np.argmax())

        gen_result = {
            "generation": gen,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "best_params": gen_params[best_idx],
            "best_metrics": gen_metrics[best_idx],
            "all_rewards": gen_rewards,
            "elapsed_s": elapsed,
            "distributions": {
                name: {k: v for k, v in dist.items() if k not in ("low", "high")}
                for name, dist in opt.distributions.items()
            },
        }
        all_results.append(gen_result)

        print(
            f"\n  Gen {gen:2d}: "
            f"mean={rewards_np.mean():+.4f}  best={rewards_np.max():+.4f}  "
            f"start_mu={opt.distributions['start_time']['mu']:.1f}s  "
            f"speed_mu={opt.distributions['playback_speed']['mu']:.2f}x  "
            f"({elapsed:.0f}s)\n"
        )

        (results_dir / "optimization_log.json").write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        vol.commit()

    # ---- Final ----
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    baseline = all_results[0]
    optimized = all_results[-1]
    print(f"  Baseline (gen 0): mean={baseline['reward_mean']:+.4f}")
    print(f"  Optimized (gen {n_generations-1}): mean={optimized['reward_mean']:+.4f}")
    print(f"  Improvement: {optimized['reward_mean'] - baseline['reward_mean']:+.4f}")

    bp = optimized["best_params"]
    print(f"\n  Best segment: start={bp['start_time']:.1f}s, dur={bp['duration']:.1f}s, "
          f"speed={bp['playback_speed']:.2f}x, crop=({bp['crop_x']:.2f}, {bp['crop_y']:.2f})")

    vol.commit()
    return all_results


@app.local_entrypoint()
def main(
    source_video: str = "",
    generations: int = 10,
    samples: int = 6,
):
    results = run_optimization.remote(
        source_video=source_video,
        n_generations=generations,
        n_samples=samples,
    )

    local_dir = Path("cache/segment_optimize")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "optimization_log.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nSaved: {local_dir / 'optimization_log.json'}")
