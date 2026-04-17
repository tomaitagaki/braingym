---
title: Claude
marimo-version: 0.23.0
---

# BrainGym

fMRI brain encoding models predicting engagement and retention from video content.

## Key Technical Context

### Hemodynamic Lag (CRITICAL)

From the TRIBEv2 paper:

> There is a roughly 5-second delay between stimuli presentation and peak response in the brain. For this reason, we offset the fMRI timeseries by 5 seconds relative to the stimuli, meaning that to predict a window [0,T] of fMRI, TRIBE takes as input a window [-5, T-5] of stimuli.

This means: **Tribe's brain prediction at time t reflects the stimulus at time t-5.** When correlating brain timeseries with behavioral timeseries (retention curves, engagement), always account for this 5-second lag. Brain signal at t=5 predicts behavior at t=0.

### Models

- **TRIBEv2 (Meta)** — video/audio/text -> predicted fMRI on fsaverage5 (~20k vertices). Uses V-JEPA2 (video), LLaMA 3.2-3B (text), Wav2Vec-BERT (audio). Output: 1 Hz (1 prediction per second).
- **BDDN (BrainDecodesDeepNets)** — single image -> predicted fMRI on nsdgeneral (~38k vertices). Uses CLIP backbone.

### Brain Networks (Schaefer 7-Network Parcellation)

| Network | Abbreviation | Function |
|---------|-------------|----------|
| Dorsal Attention | DAN | Focused, top-down attention |
| Default Mode | DMN | Mind-wandering, self-reflection. Suppressed during engagement |
| Frontoparietal Control | FPN | Executive control, cognitive effort |
| Ventral Attention / Salience | VAN | Surprise detection, salience |
| Visual | Vis | Visual processing |
| Somatomotor | SomMot | Motor processing |
| Limbic | Limbic | Emotion |

### Key Findings (as of 2026-04-12)

**TikTok (n=77):**
- attn_first3s -> share_rate: r=+0.26, p=0.02 (early hook predicts sharing)
- attn_peak -> engagement_rate: r=-0.35, p=0.002 (attention-grabbing content gets FEWER likes per view due to reach dilution)

**Tsinghua Retention (n=97):**
- -DMN with 5s lag correction -> retention: median r=0.72, 75% of videos positive
- After detrending: real signal is r~0.10 (modest but real, most of the correlation is shared temporal trend)
- DMN oscillation range predicts half_life (r=+0.30, p<0.003)

## Architecture

- `explore.py` — TikTok engagement analysis (marimo)
- `explore_retention.py` — Tsinghua retention analysis (marimo)
- `modal_tribe_100.py` — TikTok Modal pipeline
- `experiments/retention/03_modal_tribe_parallel.py` — Tsinghua Modal pipeline (10 GPU parallel)
- `quantify.py` — ROI extraction from Tribe predictions
- `recover_results.py` — recover cached results from Modal volume
- `experiments/tts_brain.py` — TTS voice comparison experiment
- `experiments/ddpo_brain/modal_ddpo.py` — DDPO image RL on Modal (SD 1.5 + Tribe)
- `sandbox/` — Brain-reward optimization framework (generators, optimizers, scoring)

### GEPA (Genetic-Pareto Prompt Optimization)

From https://arxiv.org/abs/2507.19457 — optimizes prompts through natural language reflection instead of gradient updates. 35x fewer rollouts than GRPO. The approach for braingym: use GEPA to optimize Manim generation prompts where Tribe brain metrics are the reward signal. Cycle: sample Manim outputs → run Tribe → reflect on which presentations scored best cognitively → evolve the prompt.

### TTS Experiment Finding (2026-04-13)

Same text with different TTS voices produces nearly identical brain predictions (r=0.84-0.99). Different text with the same voice produces much more different patterns (r=0.55-0.90). **Content drives ~85-90% of predicted brain response; voice/prosody drives ~5-15%.** TTS optimization is real but marginal in the brain encoding framework. Exception: emotional content with flat delivery (gTTS on narrative) shows the largest voice effect.

### Preference Experiment Finding (2026-04-14)

TRIBE (DMN + Limbic) scores 75% on forced pairwise-choice preference task (HPD v3, 50% floor). However, raw V-JEPA2 embeddings perform comparably — the brain mapping adds little for static image preference. The JEPA embedding space is higher-dimensional than the cortical fMRI representation; going from JEPA → TRIBE is lossy compression. TRIBE's value-add is for paradigms where behavioral data is hard to collect at scale.

### DDPO Brain-Reward Experiment (2026-04-14)

Ran DDPO (RL fine-tuning) on SD 1.5 with Tribe brain predictions as reward signal. 16 epochs, 137 generated face images on Modal A100.

**Result:** Static images produce near-noise-floor Tribe activations (~0.03). V-JEPA2 responds to texture complexity, not cognitive engagement. 2 of 3 encoders (audio, text) are unused on static images. The LoRA learns to generate high-frequency textures, not compelling portraits. **Static images are the wrong paradigm for Tribe.**

### Key Takeaways (as of 2026-04-14)

**When TRIBE adds value vs when backbone features are better:**
- TRIBE's grounded neuroscientific representation is useful for paradigms where behavioral data is hard to collect at scale
- For "easy" paradigms (e.g., image preference), raw backbone features (V-JEPA2, CLIP) perform comparably — the brain mapping is lossy compression
- The slam-dunk use case remains dynamic attention modeling for naturalistic video

**Training data limitations:**
- TRIBE v2 is trained heavily on naturalistic video (movies, nature scenes, narratives with audio)
- It does NOT extrapolate well outside this distribution: static images, synthetic animations (Manim), UI recordings, TTS-only audio
- Proof: attention dynamics for a static image shown for ~30s do not follow expected patterns. Cannot trust predicted brain response to out-of-distribution stimuli

**The bandwidth problem (what we're excited about):**
- Many paradigms where the brain is a bandwidth bottleneck: communication (brain→computer: keyboard, speech; computer→brain: video, UI, TTS)
- For generative models that interface with humans, there are theoretical gains in having a human-factor reward signal (attention, cognitive load, retention)
- For generative UI, TRIBE v2 is out of scope — would need datasets from companies doing traditional attention proxies (eye-tracking heatmaps, Neurons Inc.)
- Task-specific cost functions matter: personalized learning → retention, coding agents → decision making, generative UI → low cognitive load
- Treating this as a human-data collection problem could be interesting — analogous to FDE companies but for cognitive/behavioral data

**Brain-reward RL sandbox (`sandbox/`):**
- Generator-agnostic: any content source (Manim, TTS, diffusion, video) plugs into the same optimize loop
- GRPO (distribution-based) for parameterized generators with known knobs
- GEPA (DSPy LLM-guided strategy evolution) for open-ended content exploration
- DDPO for diffusion model RL fine-tuning (custom implementation)
- Proven: GRPO converges on synthetic reward; full DDPO + Tribe loop runs on Modal A100

**Manim GRPO Experiments (2026-04-14 through 2026-04-17):**

Ran GRPO optimization on Manim-generated educational animations (Pythagorean theorem, CNN architecture) with Tribe brain predictions as reward. The optimization loop works mechanically — GRPO converges, reward increases (+27-31% over baseline), distributions narrow. But the experiment fundamentally fails for a different reason:

**Tribe's predicted brain response is dominated by the content itself, not the presentation style.** Tweaking pacing, reveal style, colors, and detail level produces marginal variation compared to what the content (the actual visual objects, text, and structure) drives. The presentation parameters we optimize are second-order effects on top of a first-order content signal.

Specific failures:
- **Pythagorean GRPO** degenerated — the optimizer removed the triangle diagram entirely because shorter videos scored higher on attention-per-second. Content was not properly constrained.
- **CNN GRPO** with mandatory content blocks showed +31% improvement, but the "optimized" videos are not meaningfully different to a human viewer. The reward difference between presentation styles is small relative to the noise.
- **Video segment experiment** on a mandelbrot fallback was wasted compute — the source video was synthetic and nearly uniform, producing a flat reward landscape.

**Core insight:** For this paradigm to work, you need to optimize the CONTENT (what is shown/said), not just the PRESENTATION (how fast, what color). But optimizing content requires a generative model (LLM writing scripts, video diffusion) and introduces the problem of controlling for information completeness. The TTS experiment already showed this: content drives ~85-90% of predicted brain response, delivery drives ~5-15%. Presentation optimization is optimizing the 5-15%.