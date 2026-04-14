"""
TTS sandbox test: GRPO optimizing narrated explanations scored by Tribe.

Generates real audio via macOS TTS, scores with Tribe (mock or local).
Content varies across structures, voices, pacing — the optimizer discovers
which combinations produce the strongest predicted brain response.

    python -m sandbox.examples.tts_test          # mock Tribe (instant)
    python -m sandbox.examples.tts_test --local  # real Tribe (needs GPU/MPS)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sandbox import BrainReward, SandboxRunner, TribeScorer
from sandbox.generators import NarrationGenerator
from sandbox.optimizers import GRPOOptimizer


def main():
    parser = argparse.ArgumentParser(description="TTS sandbox test")
    parser.add_argument(
        "--local", action="store_true",
        help="Use real Tribe model instead of mock",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of generations")
    parser.add_argument("--samples", type=int, default=4, help="Samples per generation")
    parser.add_argument(
        "--save-dir", type=str, default="cache/sandbox_tts",
        help="Where to save runs",
    )
    args = parser.parse_args()

    # --- Setup ---
    gen = NarrationGenerator()
    scorer = TribeScorer(mode="local" if args.local else "mock")
    reward = BrainReward()
    opt = GRPOOptimizer(
        gen.param_space(),
        group_size=args.samples,
        lr=0.4,
        temperature=0.8,
    )

    print("Narration Generator — param space:")
    for name, spec in gen.param_space().items():
        if spec["type"] == "categorical":
            print(f"  {name}: {spec['choices']}")
        else:
            print(f"  {name}: [{spec['low']}, {spec['high']}]")
    print()

    # --- Run ---
    runner = SandboxRunner(gen, scorer, reward, opt, save_dir=args.save_dir)
    tree = runner.run(n_steps=args.steps, n_samples=args.samples)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("GRPO DISTRIBUTION CONVERGENCE")
    print("=" * 60)

    for name, dist in opt.distributions.items():
        if dist["type"] == "gaussian":
            print(f"  {name:<18s}  mu={dist['mu']:.1f}  sigma={dist['sigma']:.1f}")
        elif dist["type"] == "categorical":
            ranked = sorted(dist["probs"].items(), key=lambda x: x[1], reverse=True)
            top = ranked[0]
            print(f"  {name:<18s}  top={top[0]} (p={top[1]:.2f})")

    print(f"\nBest run: {tree.best(1)[0].id}")
    best = tree.best(1)[0]
    print(f"  reward={best.reward:.4f}")
    print(f"  params={best.params}")
    print(f"  audio: {best.video_path}")

    return tree


if __name__ == "__main__":
    main()
