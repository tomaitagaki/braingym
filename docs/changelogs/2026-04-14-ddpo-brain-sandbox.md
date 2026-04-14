# Brain-Reward Sandbox + DDPO Image Experiment

**Date:** 2026-04-14
**Branch:** `feat/ddpo-brain-sandbox`

## What changed

Added a generator-agnostic brain-reward optimization framework (`sandbox/`) and ran the first DDPO experiment fine-tuning SD 1.5 with Tribe brain predictions as reward.

## New files

### `sandbox/` — Brain-reward optimization framework
- **core.py**: `RunRecord`, `ExplorationTree`, `BrainReward`, `SandboxRunner`, Pareto utilities
- **tribe.py**: `TribeScorer` wrapper (local + mock modes, supports video and audio)
- **generators/**: `Generator` Protocol, `ManimGenerator` (parameterized video), `NarrationGenerator` (TTS via macOS say)
- **optimizers/**: `Optimizer` Protocol, `GRPOOptimizer` (distribution-based), `GEPAOptimizer` (LLM-guided strategy evolution via DSPy)
- **examples/**: `quick_test.py` (synthetic convergence proof), `tts_test.py` (real TTS + Tribe scoring)

### `experiments/ddpo_brain/` — DDPO image generation on Modal
- **modal_ddpo.py**: Custom DDPO implementation with SD 1.5 + LoRA, Tribe scoring, PPO-clipped updates. 5 content categories (faces, landscapes, diagrams, action, abstract).

## Results

### GRPO synthetic test
- All 7 parameters (4 continuous, 3 categorical) converge to hidden optimum in 15 generations
- Proves the optimization loop works correctly

### DDPO faces (16 epochs, 137 images, Modal A100)
- Reward: +0.035 (epoch 0) oscillating to +0.030 (epoch 15), best single image +0.065
- LoRA checkpoints saved at epochs 4, 9, 14
- **Finding: static images are wrong paradigm for Tribe.** Activations at noise floor (~0.03). V-JEPA2 responds to texture complexity, not cognitive engagement. 2 of 3 encoders (audio, text) unused on static images.

## Key insight

Tribe needs temporal, multimodal stimuli to produce meaningful brain predictions. Static images engage only V-JEPA2's visual features — the fMRI projection adds no value over raw backbone features. Next step: video generation or BDDN (single-image fMRI model).
