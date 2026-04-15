"""
Closed-loop Manim optimization: GRPO tunes presentation params with Tribe reward.

The experiment:
    Gen 0:  random params → render Manim → Tribe → baseline reward
    Gen N:  GRPO-optimized params → render Manim → Tribe → optimized reward

    Does optimized > baseline? By how much?

Usage:
    modal run experiments/manim_brain/04_modal_optimize.py
    modal run experiments/manim_brain/04_modal_optimize.py --generations 15 --samples 6
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-manim-optimize")
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
# PARAM SPACE — what GRPO optimizes
# ============================================================

PARAM_SPACE = {
    "pacing": {"type": "float", "low": 0.5, "high": 2.5},
    "pause_duration": {"type": "float", "low": 0.3, "high": 2.0},
    # Clamped floors: optimizer must include visuals AND text (no degenerate solutions)
    "visual_density": {"type": "float", "low": 0.4, "high": 1.0},
    "text_density": {"type": "float", "low": 0.3, "high": 1.0},
    "reveal_style": {
        "type": "categorical",
        "choices": ["sequential", "simultaneous", "progressive"],
    },
    "color_scheme": {
        "type": "categorical",
        "choices": ["mono", "dual", "vibrant"],
    },
    "narrative_hook": {
        "type": "categorical",
        "choices": ["none", "question", "story", "surprise"],
    },
}


# ============================================================
# MANIM SCRIPT BUILDER
# ============================================================

def build_manim_script(params):
    """Generate a Manim Python script from presentation parameters."""
    pace = params.get("pacing", 1.0)
    pause = params.get("pause_duration", 1.0)
    vis = params.get("visual_density", 0.5)
    text = params.get("text_density", 0.5)
    reveal = params.get("reveal_style", "sequential")
    colors = params.get("color_scheme", "dual")
    hook = params.get("narrative_hook", "none")

    color_map = {
        "mono": ("BLUE", "BLUE", "BLUE"),
        "dual": ("BLUE", "RED", "GREEN"),
        "vibrant": ("YELLOW", "RED", "TEAL"),
    }
    c1, c2, c3 = color_map.get(colors, ("BLUE", "RED", "GREEN"))

    # Build sections
    sections = []

    # Hook
    if hook == "question":
        sections.append(
            f'q = Text("What if there was a pattern\\nhidden in every right triangle?", font_size=32)\n'
            f'self.play(Write(q), run_time={pace * 1.5})\n'
            f'self.wait({pause})\nself.play(FadeOut(q))\n'
        )
    elif hook == "story":
        sections.append(
            f'hook = Text("2,500 years ago, a mathematician\\nmade a discovery...", font_size=30)\n'
            f'self.play(Write(hook), run_time={pace * 2})\n'
            f'self.wait({pause})\nself.play(FadeOut(hook))\n'
        )
    elif hook == "surprise":
        sections.append(
            f'hook = Text("Every right triangle hides\\na perfect equation.", font_size=36, color=YELLOW)\n'
            f'self.play(Write(hook), run_time={pace})\n'
            f'self.wait({pause * 0.5})\nself.play(FadeOut(hook))\n'
        )
    else:
        sections.append(
            f'title = Text("The Pythagorean Theorem", font_size=48)\n'
            f'self.play(Write(title), run_time={pace})\n'
            f'self.wait({pause * 0.5})\nself.play(FadeOut(title))\n'
        )

    # Visuals (triangle + labels)
    if vis >= 0.2:
        if reveal == "simultaneous":
            sections.append(
                f'triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color={c1})\n'
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)\n'
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)\n'
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)\n'
                f'self.play(Create(triangle), Write(a_label), Write(b_label), Write(c_label), run_time={pace})\n'
                f'self.wait({pause})\n'
            )
        elif reveal == "progressive":
            sections.append(
                f'triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color={c1})\n'
                f'self.play(Create(triangle), run_time={pace})\n'
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)\n'
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)\n'
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)\n'
                f'self.play(Write(a_label), run_time={pace * 0.5})\n'
                f'self.play(Write(b_label), run_time={pace * 0.5})\n'
                f'self.play(Write(c_label), run_time={pace * 0.5})\n'
                f'self.wait({pause})\n'
            )
        else:  # sequential
            sections.append(
                f'triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color={c1})\n'
                f'self.play(Create(triangle), run_time={pace * 1.5})\nself.wait({pause})\n'
                f'a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)\n'
                f'self.play(Write(a_label), run_time={pace})\nself.wait({pause * 0.5})\n'
                f'b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)\n'
                f'self.play(Write(b_label), run_time={pace})\nself.wait({pause * 0.5})\n'
                f'c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)\n'
                f'self.play(Write(c_label), run_time={pace})\nself.wait({pause})\n'
            )

        # Squares on sides (high visual density)
        if vis > 0.6:
            sections.append(
                f'sq_a = Square(side_length=3, color={c2}, fill_opacity=0.3).next_to(triangle, DOWN, buff=0)\n'
                f'sq_b = Square(side_length=4, color={c3}, fill_opacity=0.3).next_to(triangle, RIGHT, buff=0)\n'
                f'self.play(Create(sq_a), run_time={pace})\n'
                f'a_area = MathTex("9", font_size=36, color={c2}).move_to(sq_a)\n'
                f'self.play(Write(a_area), run_time={pace * 0.5})\n'
                f'self.play(Create(sq_b), run_time={pace})\n'
                f'b_area = MathTex("16", font_size=36, color={c3}).move_to(sq_b)\n'
                f'self.play(Write(b_area), run_time={pace * 0.5})\n'
                f'self.wait({pause})\n'
            )

    # Text / equations
    if text >= 0.2:
        sections.append(
            f'eq = MathTex("a^2 + b^2 = c^2", font_size=48).to_edge(UP)\n'
            f'self.play(Write(eq), run_time={pace})\nself.wait({pause})\n'
        )
    if text > 0.5:
        sections.append(
            f'eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=44).next_to(eq, DOWN)\n'
            f'self.play(Write(eq2), run_time={pace})\nself.wait({pause * 0.5})\n'
        )
    if text > 0.8:
        sections.append(
            f'eq3 = MathTex("9 + 16 = 25", font_size=44).next_to(eq2, DOWN)\n'
            f'self.play(Write(eq3), run_time={pace})\nself.wait({pause})\n'
        )

    # Finale
    sections.append(
        f'self.play(*[FadeOut(mob) for mob in self.mobjects], run_time={pace * 0.5})\n'
        f'final = MathTex("a^2 + b^2 = c^2", font_size=72, color={c1})\n'
        f'self.play(Write(final), run_time={pace})\nself.wait({pause})\n'
    )

    # Indent every line in the body to sit inside def construct(self):
    indent = "        "  # 8 spaces
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
# GRPO (inline — avoid import issues on Modal)
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
    timeout=10800,  # 3 hours
    memory=32768,
)
def run_optimization(n_generations: int = 10, n_samples: int = 4):
    """GRPO optimization loop: Manim params → render → Tribe → reward → update."""
    import os
    import time
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "manim_optimize"
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
        """Score a video with Tribe, return (reward, metrics, timeseries)."""
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
        n3 = min(3, len(dan))
        attn = dan - dmn

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

        timeseries = {k: v.tolist() for k, v in signals.items()}
        return reward, metrics, timeseries

    def render_manim(params, output_path):
        """Render a Manim scene from params, return video path."""
        script = build_manim_script(params)
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
                raise RuntimeError(f"Manim failed: {result.stderr[-300:]}")
            rendered = list(media_dir.rglob("Generated.mp4"))
            if not rendered:
                raise FileNotFoundError("No Generated.mp4 found")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(rendered[0]), str(output_path))
        return output_path

    # ---- Optimization loop ----
    print(f"\n{'=' * 60}")
    print(f"GRPO Optimization: {n_generations} generations x {n_samples} samples")
    print(f"{'=' * 60}\n")

    opt = GRPO(PARAM_SPACE, lr=0.4, temperature=0.5)
    all_results = []

    for gen in range(n_generations):
        t0 = time.time()
        param_batch = opt.suggest(n_samples)

        gen_rewards = []
        gen_metrics = []
        gen_params = []

        for i, params in enumerate(param_batch):
            run_id = f"g{gen:02d}_s{i:02d}"
            video_path = videos_dir / f"{run_id}.mp4"

            try:
                # Render
                render_manim(params, video_path)

                # Score
                reward, metrics, timeseries = score_video(video_path)
                gen_rewards.append(reward)
                gen_metrics.append(metrics)
                gen_params.append(params)

                print(
                    f"  [{run_id}] reward={reward:+.4f}  "
                    f"attn={metrics['attention']:+.3f}  "
                    f"hook={metrics['hook_3s']:+.3f}  "
                    f"sustained={metrics['sustained']:+.3f}"
                )

            except Exception as e:
                print(f"  [{run_id}] FAILED: {e}")
                gen_rewards.append(0.0)
                gen_metrics.append({"reward": 0.0})
                gen_params.append(params)

        # Update GRPO
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
            "best_params": gen_params[best_idx],
            "best_metrics": gen_metrics[best_idx],
            "all_rewards": gen_rewards,
            "all_metrics": gen_metrics,
            "elapsed_s": elapsed,
            "distributions": {
                name: {
                    k: v for k, v in dist.items()
                    if k not in ("low", "high")
                }
                for name, dist in opt.distributions.items()
            },
        }
        all_results.append(gen_result)

        print(
            f"\n  Gen {gen:2d}: "
            f"mean={rewards_np.mean():+.4f}  "
            f"best={rewards_np.max():+.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Show distribution convergence every 3 generations
        if gen % 3 == 2 or gen == n_generations - 1:
            print("  Distributions:")
            for name, dist in opt.distributions.items():
                if dist["type"] == "gaussian":
                    print(f"    {name:>18s}: mu={dist['mu']:.2f}  sigma={dist['sigma']:.2f}")
                else:
                    top = max(dist["probs"], key=dist["probs"].get)
                    print(f"    {name:>18s}: top={top} (p={dist['probs'][top]:.2f})")

        # Save checkpoint
        (results_dir / "optimization_log.json").write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        vol.commit()

    # ---- Final comparison ----
    print(f"\n{'=' * 60}")
    print("RESULTS: Baseline vs Optimized")
    print(f"{'=' * 60}")

    baseline = all_results[0]
    optimized = all_results[-1]
    best_ever = max(all_results, key=lambda r: r["reward_max"])

    print(f"\n  Baseline (gen 0):     mean={baseline['reward_mean']:+.4f}  best={baseline['reward_max']:+.4f}")
    print(f"  Optimized (gen {n_generations-1}):  mean={optimized['reward_mean']:+.4f}  best={optimized['reward_max']:+.4f}")
    print(f"  Best ever (gen {best_ever['generation']}):   best={best_ever['reward_max']:+.4f}")
    print(f"\n  Improvement (mean): {optimized['reward_mean'] - baseline['reward_mean']:+.4f}")
    print(f"  Improvement (best): {optimized['reward_max'] - baseline['reward_max']:+.4f}")

    # Compare best baseline vs best optimized metrics
    print(f"\n  Best baseline metrics:")
    for k, v in baseline["best_metrics"].items():
        if isinstance(v, float):
            print(f"    {k:>15s}: {v:+.4f}")
    print(f"\n  Best optimized metrics:")
    for k, v in optimized["best_metrics"].items():
        if isinstance(v, float):
            print(f"    {k:>15s}: {v:+.4f}")

    # Final distributions
    print(f"\n  Converged distributions:")
    for name, dist in opt.distributions.items():
        if dist["type"] == "gaussian":
            print(f"    {name:>18s}: mu={dist['mu']:.2f}  sigma={dist['sigma']:.2f}")
        else:
            ranked = sorted(dist["probs"].items(), key=lambda x: x[1], reverse=True)
            print(f"    {name:>18s}: {' > '.join(f'{c}({p:.2f})' for c, p in ranked)}")

    vol.commit()
    return all_results


@app.local_entrypoint()
def main(generations: int = 10, samples: int = 4):
    results = run_optimization.remote(
        n_generations=generations,
        n_samples=samples,
    )

    local_dir = Path("cache/manim_optimize")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "optimization_log.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nSaved: {local_dir / 'optimization_log.json'}")
