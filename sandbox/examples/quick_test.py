"""
Quick test: verify the sandbox loop converges on a known optimum.

Uses a synthetic reward function instead of Tribe so no GPU or Manim needed.
The "brain reward" is a known function of params — GRPO should find the peak.

    python -m sandbox.examples.quick_test

Hidden optimum:
    pacing=1.2, pause=0.5, visual_density=0.8, text_density=0.3
    reveal=progressive, color=vibrant, hook=surprise
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from sandbox.core import BrainReward, ExplorationTree, RunRecord, SandboxRunner
from sandbox.generators.base import Generator
from sandbox.optimizers.grpo import GRPOOptimizer


# ============================================================
# Synthetic generator + reward (no Manim, no Tribe)
# ============================================================

# The "true" optimal parameters (GRPO doesn't know these)
OPTIMUM = {
    "pacing": 1.2,
    "pause_duration": 0.5,
    "visual_density": 0.8,
    "text_density": 0.3,
    "reveal_style": "progressive",
    "color_scheme": "vibrant",
    "narrative_hook": "surprise",
}

# Reward bonus for matching categorical optimum
CATEGORICAL_BONUS = {
    "reveal_style": {"progressive": 0.3, "sequential": 0.1, "simultaneous": 0.0},
    "color_scheme": {"vibrant": 0.2, "dual": 0.1, "mono": 0.0},
    "narrative_hook": {"surprise": 0.25, "story": 0.15, "question": 0.1, "none": 0.0},
}


class DummyGenerator:
    """Generator that creates empty files (no rendering needed)."""

    name = "dummy"

    def param_space(self) -> dict[str, dict[str, Any]]:
        return {
            "pacing": {"type": "float", "low": 0.3, "high": 3.0},
            "pause_duration": {"type": "float", "low": 0.1, "high": 2.5},
            "visual_density": {"type": "float", "low": 0.0, "high": 1.0},
            "text_density": {"type": "float", "low": 0.0, "high": 1.0},
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

    def generate(self, params: dict[str, Any], output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy_" + uuid.uuid4().hex[:8].encode())
        return output_path


def synthetic_reward(params: dict[str, Any]) -> float:
    """Known reward function. Peaks at OPTIMUM params.

    Continuous params: negative squared distance from optimum.
    Categorical params: bonus for matching best choice.
    Noise added to simulate real scoring variance.
    """
    reward = 0.0

    # Continuous: Gaussian peaks at optimum
    for key in ("pacing", "pause_duration", "visual_density", "text_density"):
        opt = OPTIMUM[key]
        val = params.get(key, opt)
        # Wider range params get more weight
        reward -= (val - opt) ** 2

    # Categorical: bonus for matching
    for key, bonuses in CATEGORICAL_BONUS.items():
        val = params.get(key, "")
        reward += bonuses.get(val, 0.0)

    # Add noise (simulates Tribe scoring variance)
    reward += np.random.normal(0, 0.05)

    return float(reward)


# ============================================================
# Synthetic scorer that bypasses Tribe entirely
# ============================================================

class SyntheticScorer:
    """Replaces TribeScorer — returns fake signals shaped by synthetic_reward."""

    N_VERTICES = 20484

    def predict(self, video_path):
        return np.zeros((10, self.N_VERTICES))

    def extract_signals(self, preds):
        # Return flat signals — the real reward comes from SyntheticReward
        n = preds.shape[0]
        return {
            "DAN": np.zeros(n),
            "DMN": np.zeros(n),
            "VAN": np.zeros(n),
            "FPN": np.zeros(n),
            "Limbic": np.zeros(n),
            "Vis": np.zeros(n),
            "SomMot": np.zeros(n),
        }


class SyntheticReward:
    """Reward based on known param→score function instead of brain signals."""

    def __call__(self, signals, params=None):
        # We'll inject params separately in the patched runner
        return 0.0, {}


# ============================================================
# Run
# ============================================================

def main():
    gen = DummyGenerator()
    scorer = SyntheticScorer()
    reward_fn = BrainReward()  # won't be used directly
    opt = GRPOOptimizer(
        gen.param_space(),
        group_size=8,
        lr=0.5,
        temperature=0.5,  # greedier selection
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = SandboxRunner(gen, scorer, reward_fn, opt, save_dir=tmpdir)

        # Patch step() to use synthetic reward from params
        original_step = runner.step

        def patched_step(n_samples=8):
            param_batch = runner.optimizer.suggest(runner.tree, n=n_samples)
            records = []
            for i, params in enumerate(param_batch):
                run_id = f"gen{runner.generation:03d}_{i:02d}_{uuid.uuid4().hex[:6]}"
                output_path = runner.video_dir / f"{run_id}.mp4"

                video_path = runner.generator.generate(params, output_path)

                # Use synthetic reward directly from params
                reward_val = synthetic_reward(params)
                metrics = {
                    "synthetic_reward": reward_val,
                    "dist_to_optimum": sum(
                        (params.get(k, 0) - OPTIMUM[k]) ** 2
                        for k in ("pacing", "pause_duration", "visual_density", "text_density")
                    ),
                }

                record = RunRecord(
                    id=run_id,
                    params={k: v for k, v in params.items() if not k.startswith("_")},
                    video_path=str(video_path),
                    reward=reward_val,
                    metrics=metrics,
                    generation=runner.generation,
                    generator_name=runner.generator.name,
                    optimizer_name=runner.optimizer.name,
                )
                records.append(record)
                runner.tree.add(record)

            diagnostics = runner.optimizer.update(runner.tree, records)

            rewards = [r.reward for r in records]
            best = max(records, key=lambda r: r.reward)
            print(
                f"  Gen {runner.generation:2d}: "
                f"best={max(rewards):+.3f}  mean={np.mean(rewards):+.3f}  "
                f"dist={best.metrics['dist_to_optimum']:.3f}"
            )

            runner.generation += 1
            return records, diagnostics

        runner.step = patched_step

        # Run optimization
        n_steps = 15
        print(f"GRPO optimizing toward hidden optimum")
        print(f"True optimum: pacing={OPTIMUM['pacing']}, pause={OPTIMUM['pause_duration']}, "
              f"vis={OPTIMUM['visual_density']}, text={OPTIMUM['text_density']}")
        print(f"              reveal={OPTIMUM['reveal_style']}, color={OPTIMUM['color_scheme']}, "
              f"hook={OPTIMUM['narrative_hook']}")
        print(f"\n{'='*60}")
        print(f"Running {n_steps} generations x 8 samples\n")

        for step in range(n_steps):
            runner.step(n_samples=8)

        # Results
        print(f"\n{'='*60}")
        print("CONVERGENCE CHECK")
        print(f"{'='*60}\n")

        # Show distribution convergence
        print("Parameter distributions (should converge to optimum):\n")
        print(f"  {'Param':<18s} {'Optimum':>8s} {'GRPO mu':>8s} {'GRPO sigma':>10s} {'Match?':>8s}")
        print(f"  {'-'*54}")

        for name, dist in opt.distributions.items():
            if dist["type"] == "gaussian":
                opt_val = OPTIMUM[name]
                mu = dist["mu"]
                sigma = dist["sigma"]
                match = "yes" if abs(mu - opt_val) < 0.3 else "close" if abs(mu - opt_val) < 0.6 else "no"
                print(f"  {name:<18s} {opt_val:>8.2f} {mu:>8.3f} {sigma:>10.3f} {match:>8s}")
            elif dist["type"] == "categorical":
                opt_val = OPTIMUM[name]
                top_choice = max(dist["probs"], key=dist["probs"].get)
                top_prob = dist["probs"][top_choice]
                match = "yes" if top_choice == opt_val else "no"
                print(f"  {name:<18s} {opt_val:>8s} {top_choice:>8s} {f'p={top_prob:.2f}':>10s} {match:>8s}")

        # Best run
        best = runner.tree.best(1)[0]
        print(f"\nBest run: reward={best.reward:+.3f}, dist_to_opt={best.metrics['dist_to_optimum']:.4f}")
        print(f"  params: pacing={best.params['pacing']:.2f} pause={best.params['pause_duration']:.2f} "
              f"vis={best.params['visual_density']:.2f} text={best.params['text_density']:.2f} "
              f"reveal={best.params['reveal_style']} color={best.params['color_scheme']} "
              f"hook={best.params['narrative_hook']}")


if __name__ == "__main__":
    main()
