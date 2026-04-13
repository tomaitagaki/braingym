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
