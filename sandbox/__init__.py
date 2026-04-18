"""
BrainGym Sandbox: brain-reward optimization loop for content generation.

    Generator(params) → video.mp4 → Tribe → BrainReward → Optimizer → repeat

The generator is a swappable plugin (Manim, diffusion, TTS, etc.).
The optimizer explores the parameter space to maximize brain activation:
  - GRPO: distribution-based, for explicit param spaces
  - GEPA: LLM-guided strategy evolution via DSPy, for open-ended exploration

Usage:
    from sandbox import SandboxRunner, BrainReward, TribeScorer
    from sandbox.generators import ManimGenerator
    from sandbox.optimizers import GRPOOptimizer

    gen = ManimGenerator(concept="Pythagorean Theorem")
    scorer = TribeScorer(mode="mock")
    reward = BrainReward()
    opt = GRPOOptimizer(gen.param_space())

    runner = SandboxRunner(gen, scorer, reward, opt, save_dir="runs/exp1")
    tree = runner.run(n_steps=10, n_samples=8)
"""

from .core import BrainReward, ExplorationTree, RunRecord, SandboxRunner
from .tribe import TribeScorer

__all__ = [
    "BrainReward",
    "ExplorationTree",
    "RunRecord",
    "SandboxRunner",
    "TribeScorer",
]
