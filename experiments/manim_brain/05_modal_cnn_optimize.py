"""
Closed-loop CNN Manim optimization: GRPO tunes presentation params with Tribe reward.

Content: "How a CNN processes an image" — input grid, conv filter, feature map,
pooling, classification. All content blocks are MANDATORY (can't be skipped).
GRPO optimizes HOW the content is presented, not WHETHER it appears.

Usage:
    modal run experiments/manim_brain/05_modal_cnn_optimize.py
    modal run experiments/manim_brain/05_modal_cnn_optimize.py --generations 12 --samples 6
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-cnn-optimize")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

manim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg", "git",
        "texlive-latex-base", "texlive-latex-extra",
        "texlive-fonts-recommended", "texlive-fonts-extra",
        "texlive-science", "dvisvgm",
        "pkg-config", "libcairo2-dev", "libpango1.0-dev",
        "libgirepository1.0-dev", "gir1.2-gtk-3.0",
    )
    .pip_install(
        "manim>=0.18",
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
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)

# ============================================================
# PARAM SPACE
# ============================================================

PARAM_SPACE = {
    "pacing": {"type": "float", "low": 0.5, "high": 2.5},
    "pause_duration": {"type": "float", "low": 0.3, "high": 2.0},
    "grid_detail": {
        "type": "float", "low": 0.3, "high": 1.0,
        "desc": "How much numeric detail in grids (0.3=boxes only, 1.0=all values shown)",
    },
    "filter_animation": {
        "type": "categorical",
        "choices": ["slide_all", "slide_3", "highlight_only"],
        "desc": "How the conv filter is shown: full slide, 3 positions, or just highlight",
    },
    "reveal_style": {
        "type": "categorical",
        "choices": ["sequential", "simultaneous", "progressive"],
    },
    "color_scheme": {
        "type": "categorical",
        "choices": ["mono", "dual", "vibrant"],
    },
    "hook": {
        "type": "categorical",
        "choices": ["none", "question", "surprise"],
    },
}


# ============================================================
# CNN MANIM SCRIPT BUILDER
# ============================================================

def build_cnn_script(params):
    """Generate a Manim CNN explanation from presentation parameters.

    ALL content blocks are mandatory — the optimizer can only change
    pacing, detail level, reveal style, and visual treatment.
    """
    pace = params.get("pacing", 1.0)
    pause = params.get("pause_duration", 0.8)
    detail = params.get("grid_detail", 0.6)
    filter_anim = params.get("filter_animation", "slide_3")
    reveal = params.get("reveal_style", "sequential")
    colors = params.get("color_scheme", "dual")
    hook = params.get("hook", "none")

    color_map = {
        "mono": ("BLUE", "BLUE", "BLUE"),
        "dual": ("BLUE", "GREEN", "RED"),
        "vibrant": ("YELLOW", "TEAL", "PINK"),
    }
    c1, c2, c3 = color_map.get(colors, ("BLUE", "GREEN", "RED"))

    sections = []

    # --- HOOK (always present, style varies) ---
    if hook == "question":
        sections.append(
            f'title = Text("How does a CNN see an image?", font_size=36, color=YELLOW)\n'
            f'self.play(Write(title), run_time={pace * 1.5})\n'
            f'self.wait({pause})\nself.play(FadeOut(title))\n'
        )
    elif hook == "surprise":
        sections.append(
            f'title = Text("Your phone identifies objects\\nin under 10 milliseconds.", font_size=30, color=YELLOW)\n'
            f'self.play(Write(title), run_time={pace * 1.5})\n'
            f'self.wait({pause})\nself.play(FadeOut(title))\n'
            f'title2 = Text("Here is how.", font_size=36)\n'
            f'self.play(Write(title2), run_time={pace * 0.5})\n'
            f'self.wait({pause * 0.5})\nself.play(FadeOut(title2))\n'
        )
    else:
        sections.append(
            f'title = Text("Convolutional Neural Network", font_size=40)\n'
            f'self.play(Write(title), run_time={pace})\n'
            f'self.wait({pause * 0.5})\nself.play(FadeOut(title))\n'
        )

    # --- INPUT GRID (mandatory) ---
    sections.append(
        f'input_label = Text("Input Image (5x5)", font_size=22).to_edge(UP)\n'
        f'self.play(Write(input_label), run_time={pace * 0.5})\n'
        f'pixel_values = [[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]\n'
        f'cell_size = 0.5\n'
        f'input_grid = VGroup()\n'
        f'for r in range(5):\n'
        f'    for c in range(5):\n'
        f'        val = pixel_values[r][c]\n'
        f'        sq = Square(side_length=cell_size)\n'
        f'        sq.set_fill(WHITE if val == 1 else DARK_GRAY, opacity=0.8)\n'
        f'        sq.set_stroke(GRAY, width=1)\n'
        f'        sq.move_to(np.array([c * cell_size, -r * cell_size, 0]))\n'
        f'        input_grid.add(sq)\n'
        f'input_grid.move_to(LEFT * 4)\n'
    )

    if reveal == "simultaneous":
        sections.append(
            f'self.play(Create(input_grid), Write(input_label), run_time={pace})\n'
            f'self.wait({pause * 0.5})\n'
        )
    else:
        sections.append(
            f'self.play(Create(input_grid), run_time={pace * 1.2})\n'
            f'self.wait({pause})\n'
        )

    # Add numbers to grid if high detail
    if detail > 0.6:
        sections.append(
            f'nums = VGroup()\n'
            f'for r in range(5):\n'
            f'    for c in range(5):\n'
            f'        val = pixel_values[r][c]\n'
            f'        t = Text(str(val), font_size=14, color={c1}).move_to(input_grid[r*5+c])\n'
            f'        nums.add(t)\n'
            f'self.play(Write(nums), run_time={pace * 0.8})\n'
            f'self.wait({pause * 0.5})\n'
        )

    # --- CONV FILTER (mandatory) ---
    sections.append(
        f'filter_label = Text("3x3 Filter", font_size=18, color={c1}).next_to(input_grid, DOWN, buff=0.4)\n'
        f'self.play(Write(filter_label), run_time={pace * 0.4})\n'
        f'highlight = Square(side_length=cell_size * 3, color={c1}, stroke_width=3)\n'
        f'top_left = input_grid[0].get_center()\n'
        f'highlight.move_to(top_left + np.array([cell_size, -cell_size, 0]))\n'
        f'self.play(Create(highlight), run_time={pace * 0.5})\n'
    )

    # Filter animation style
    if filter_anim == "slide_all":
        sections.append(
            f'for dx in range(3):\n'
            f'    for dy in range(3):\n'
            f'        target = top_left + np.array([(dx+1)*cell_size, -(dy+1)*cell_size, 0])\n'
            f'        self.play(highlight.animate.move_to(target), run_time={pace * 0.2})\n'
        )
    elif filter_anim == "slide_3":
        sections.append(
            f'for dx in range(3):\n'
            f'    target = top_left + np.array([(dx+1)*cell_size, -cell_size, 0])\n'
            f'    self.play(highlight.animate.move_to(target), run_time={pace * 0.3})\n'
        )
    else:  # highlight_only — already shown, just pause
        sections.append(
            f'self.wait({pause})\n'
        )

    sections.append(
        f'self.play(FadeOut(highlight), FadeOut(filter_label), run_time={pace * 0.3})\n'
    )

    # --- FEATURE MAP (mandatory) ---
    sections.append(
        f'feat_label = Text("Feature Map", font_size=18, color={c2}).move_to(UP * 2.8)\n'
        f'self.play(Write(feat_label), run_time={pace * 0.4})\n'
        f'feat_values = [[2,3,1],[1,3,1],[1,3,2]]\n'
        f'feat_grid = VGroup()\n'
        f'for r in range(3):\n'
        f'    for c in range(3):\n'
        f'        val = feat_values[r][c]\n'
        f'        sq = Square(side_length=cell_size*0.8)\n'
        f'        sq.set_fill({c2}, opacity=val/3.0)\n'
        f'        sq.set_stroke({c2}, width=1)\n'
        f'        cell_grp = VGroup(sq)\n'
    )

    if detail > 0.4:
        sections[-1] += (
            f'        num = Text(str(val), font_size=14).move_to(sq)\n'
            f'        cell_grp.add(num)\n'
        )

    sections[-1] += (
        f'        cell_grp.move_to(np.array([c*cell_size, -r*cell_size, 0]))\n'
        f'        feat_grid.add(cell_grp)\n'
        f'feat_grid.move_to(RIGHT * 0 + DOWN * 0.3)\n'
        f'arrow1 = Arrow(input_grid.get_right(), feat_grid.get_left(), buff=0.3, color={c1})\n'
        f'conv_text = Text("Conv", font_size=14, color={c1}).next_to(arrow1, UP, buff=0.1)\n'
    )

    if reveal == "simultaneous":
        sections.append(
            f'self.play(Create(arrow1), Write(conv_text), Create(feat_grid), run_time={pace})\n'
            f'self.wait({pause * 0.5})\n'
        )
    else:
        sections.append(
            f'self.play(Create(arrow1), Write(conv_text), run_time={pace * 0.6})\n'
            f'self.play(Create(feat_grid), run_time={pace})\n'
            f'self.wait({pause})\n'
        )

    # --- POOLING (mandatory) ---
    sections.append(
        f'pool_label = Text("Max Pool", font_size=18, color={c3}).move_to(UP * 2.8 + RIGHT * 4)\n'
        f'self.play(Write(pool_label), run_time={pace * 0.4})\n'
        f'pool_grid = VGroup()\n'
        f'pool_vals = [[3,3],[3,3]]\n'
        f'for r in range(2):\n'
        f'    for c in range(2):\n'
        f'        val = pool_vals[r][c]\n'
        f'        sq = Square(side_length=cell_size*0.8)\n'
        f'        sq.set_fill({c3}, opacity=val/3.0)\n'
        f'        sq.set_stroke({c3}, width=1)\n'
        f'        cell_grp = VGroup(sq)\n'
    )

    if detail > 0.4:
        sections[-1] += (
            f'        num = Text(str(val), font_size=14).move_to(sq)\n'
            f'        cell_grp.add(num)\n'
        )

    sections[-1] += (
        f'        cell_grp.move_to(np.array([c*cell_size, -r*cell_size, 0]))\n'
        f'        pool_grid.add(cell_grp)\n'
        f'pool_grid.move_to(RIGHT * 4 + DOWN * 0.3)\n'
        f'arrow2 = Arrow(feat_grid.get_right(), pool_grid.get_left(), buff=0.3, color={c3})\n'
        f'pool_text = Text("Pool", font_size=14, color={c3}).next_to(arrow2, UP, buff=0.1)\n'
        f'self.play(Create(arrow2), Write(pool_text), run_time={pace * 0.6})\n'
        f'self.play(Create(pool_grid), run_time={pace * 0.8})\n'
        f'self.wait({pause})\n'
    )

    # --- CLASSIFICATION OUTPUT (mandatory) ---
    sections.append(
        f'self.play(FadeOut(input_label), FadeOut(feat_label), FadeOut(pool_label), run_time={pace * 0.3})\n'
        f'output_label = Text("Classification", font_size=22).to_edge(DOWN, buff=1.5)\n'
        f'classes = VGroup(\n'
        f'    Text("cat: 12%", font_size=16, color=GRAY),\n'
        f'    Text("dog: 8%", font_size=16, color=GRAY),\n'
        f'    Text("digit 1: 80%", font_size=16, color={c2}),\n'
        f').arrange(DOWN, aligned_edge=LEFT).next_to(output_label, UP, buff=0.3)\n'
        f'arrow3 = Arrow(pool_grid.get_bottom(), classes.get_top(), buff=0.2, color=WHITE)\n'
        f'fc_text = Text("FC Layer", font_size=14).next_to(arrow3, RIGHT, buff=0.1)\n'
    )

    if reveal == "sequential":
        sections.append(
            f'self.play(Create(arrow3), Write(fc_text), run_time={pace * 0.6})\n'
            f'self.play(Write(output_label), run_time={pace * 0.4})\n'
            f'self.play(Write(classes), run_time={pace})\n'
            f'self.wait({pause})\n'
        )
    else:
        sections.append(
            f'self.play(Create(arrow3), Write(fc_text), Write(output_label), Write(classes), run_time={pace})\n'
            f'self.wait({pause})\n'
        )

    # --- FINALE (mandatory) ---
    sections.append(
        f'self.play(*[FadeOut(mob) for mob in self.mobjects], run_time={pace * 0.5})\n'
        f'summary = Text("Input > Conv > Pool > Classify", font_size=34, color=YELLOW)\n'
        f'self.play(Write(summary), run_time={pace})\n'
        f'self.wait({pause * 1.5})\n'
    )

    # Assemble script
    indent = "        "
    all_lines = []
    for section in sections:
        for line in section.strip().split("\n"):
            all_lines.append(indent + line)
    body = "\n".join(all_lines)

    return (
        "from manim import *\n"
        "import numpy as np\n\n"
        "class Generated(Scene):\n"
        "    def construct(self):\n"
        f"{body}\n"
    )


# ============================================================
# GRPO
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
            elif dist["type"] == "categorical":
                choice_w = {c: 0.0 for c in dist["probs"]}
                for p, w in zip(params_list, weights):
                    choice_w[p[name]] += w
                for c in dist["probs"]:
                    dist["probs"][c] = (
                        dist["probs"][c] * (1 - self.lr) + choice_w.get(c, 0) * self.lr
                    )
                total = sum(dist["probs"].values())
                dist["probs"] = {c: p / total for c, p in dist["probs"].items()}


# ============================================================
# MAIN
# ============================================================

@app.function(
    image=manim_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=28800,  # 8 hours
    memory=32768,
)
def run_optimization(n_generations: int = 12, n_samples: int = 6):
    import os
    import time
    import subprocess
    import shutil
    import tempfile
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "cnn_optimize"
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = results_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

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

    def render_manim(params, output_path):
        script = build_cnn_script(params)
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "scene.py"
            script_path.write_text(script)
            media_dir = Path(tmpdir) / "media"
            result = subprocess.run(
                [
                    "python", "-m", "manim", "render",
                    "-ql", "--media_dir", str(media_dir),
                    str(script_path), "Generated",
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Manim failed: {result.stderr[-500:]}")
            rendered = list(media_dir.rglob("Generated.mp4"))
            if not rendered:
                raise FileNotFoundError("No Generated.mp4")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rendered[0]), str(output_path))
        return output_path

    # ---- Optimization loop ----
    print(f"\n{'=' * 60}")
    print(f"CNN GRPO: {n_generations} gen x {n_samples} samples")
    print(f"{'=' * 60}\n")

    opt = GRPO(PARAM_SPACE, lr=0.4, temperature=0.5)
    all_results = []

    for gen in range(n_generations):
        t0 = time.time()
        param_batch = opt.suggest(n_samples)
        gen_rewards, gen_metrics, gen_params = [], [], []

        for i, params in enumerate(param_batch):
            run_id = f"g{gen:02d}_s{i:02d}"
            video_path = videos_dir / f"{run_id}.mp4"
            try:
                render_manim(params, video_path)
                reward, metrics, ts = score_video(video_path)
                gen_rewards.append(reward)
                gen_metrics.append(metrics)
                gen_params.append(params)
                print(
                    f"  [{run_id}] reward={reward:+.4f}  "
                    f"attn={metrics['attention']:+.3f}  "
                    f"hook={metrics['hook_3s']:+.3f}  "
                    f"sustained={metrics['sustained']:+.3f}  "
                    f"TRs={metrics['n_trs']}"
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
            "reward_min": float(rewards_np.min()),
            "best_params": {k: v for k, v in gen_params[best_idx].items()},
            "best_metrics": gen_metrics[best_idx],
            "all_rewards": gen_rewards,
            "all_metrics": gen_metrics,
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
            f"({elapsed:.0f}s)"
        )

        if gen % 3 == 2 or gen == n_generations - 1:
            print("  Distributions:")
            for name, dist in opt.distributions.items():
                if dist["type"] == "gaussian":
                    print(f"    {name:>18s}: mu={dist['mu']:.2f}  sigma={dist['sigma']:.2f}")
                else:
                    top = max(dist["probs"], key=dist["probs"].get)
                    print(f"    {name:>18s}: top={top} (p={dist['probs'][top]:.2f})")

        (results_dir / "optimization_log.json").write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        vol.commit()

    # ---- Final ----
    print(f"\n{'=' * 60}")
    print("RESULTS: Baseline vs Optimized")
    print(f"{'=' * 60}")
    baseline = all_results[0]
    optimized = all_results[-1]
    best_ever = max(all_results, key=lambda r: r["reward_max"])
    print(f"\n  Baseline (gen 0):     mean={baseline['reward_mean']:+.4f}  best={baseline['reward_max']:+.4f}")
    print(f"  Optimized (gen {n_generations-1:2d}):  mean={optimized['reward_mean']:+.4f}  best={optimized['reward_max']:+.4f}")
    print(f"  Best ever (gen {best_ever['generation']:2d}):  best={best_ever['reward_max']:+.4f}")
    print(f"\n  Improvement (mean): {optimized['reward_mean'] - baseline['reward_mean']:+.4f}")

    bp = optimized["best_params"]
    print(f"\n  Converged params: pacing={bp.get('pacing', 0):.2f} pause={bp.get('pause_duration', 0):.2f} "
          f"detail={bp.get('grid_detail', 0):.2f} filter={bp.get('filter_animation', '?')} "
          f"reveal={bp.get('reveal_style', '?')} colors={bp.get('color_scheme', '?')} hook={bp.get('hook', '?')}")

    vol.commit()
    return all_results


@app.local_entrypoint()
def main(generations: int = 12, samples: int = 6):
    results = run_optimization.remote(
        n_generations=generations,
        n_samples=samples,
    )
    local_dir = Path("cache/cnn_optimize")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "optimization_log.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nSaved: {local_dir / 'optimization_log.json'}")
