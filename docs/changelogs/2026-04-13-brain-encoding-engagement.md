# Brain encoding models predict short-video engagement and retention

## What

Added two experiments exploring whether fMRI brain encoding models (TRIBEv2) can predict real-world video engagement from predicted brain responses.

## TikTok experiment (n=77)

- Predicted brain attention correlates with share rate at r=0.53 after controlling for duration
- Overall attention level matters, not specifically the first 3 seconds
- Higher stimulation → more reach but lower per-view engagement (depth vs breadth)
- Brain-type classification (high-stim vs high-processing) via Vis+Attention vs FPN+DMN

## Tsinghua retention experiment (n=97)

- DMN suppression tracks retention within videos (r=0.42 pct-normalized, r=0.12 detrended)
- Network dynamics (DMN range, Limbic range) predict half-life (r=0.30-0.42, p<0.003)
- Cross-platform brain responses are similar; composite attention (DAN-DMN) is stable across platforms

## Infrastructure

- Modal GPU pipeline: sequential + 10-GPU parallel for Tribe v2 inference
- Per-video caching on Modal volumes with recovery script
- Marimo notebooks (explore.py, explore_retention.py) for interactive analysis

## Honest assessment

- Duration is the master confound — all correlations must control for it
- The n=9 pilot finding (r=0.85) was spurious — collapsed at scale
- Per-second brain-retention coupling is mostly shared temporal drift
- Real signal: ~25% variance explained for share rate within fixed duration
