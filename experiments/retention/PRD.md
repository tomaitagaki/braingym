# Experiment: Brain Timeseries × Retention Curves

**Created:** 2026-04-10
**Status:** Planning Complete
**Dataset:** Tsinghua ShortVideo (WWW 2025)

## Hypothesis

Predicted brain network dynamics from TRIBEv2 (especially DAN, DMN, and their coupling) correlate with viewer retention at each second of a short-form video. Specifically:

- **DAN activation** at time t predicts what fraction of viewers are still watching at t
- **DAN-DMN anticorrelation** tracks moments where viewers drop off vs. stay engaged
- The **temporal derivative** of brain signals predicts retention slope changes (acceleration of drop-off)

## Prior Work (BrainGym pilot)

- 9 TikTok videos: DAN peak → share_rate at r=0.85, p<0.005
- fc_DAN_DMN anticorrelation → share_rate at r=-0.85, p<0.004
- Scalar brain metrics predict scalar engagement — this experiment extends to **timeseries vs. timeseries**

## Dataset: Tsinghua ShortVideo

| Field | Value |
|-------|-------|
| Source | Kuaishou short videos |
| Videos | 153,561 MP4 files (3.2 TB total) |
| Users | 10,000 |
| Interactions | 1,019,568 |
| Key field | `watch_time` — per-user viewing duration in seconds |
| Other signals | like, comment, follow, collect, forward, hate |
| Pre-extracted | 256-dim visual features, ASR transcripts |
| Download | http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset |
| Paper | https://arxiv.org/abs/2502.05922 |

## Experiment Design

### Phase 1: Data Acquisition & Video Selection

1. Download `interaction_filtered.csv` (~50 MB) and `video_info.csv`
2. Count viewers per video: `viewer_count = interactions.groupby('video_id').size()`
3. Filter to videos with ≥30 viewers (statistical floor for per-second curves)
4. From those, stratified-sample ~100 videos across duration buckets (10-20s, 20-40s, 40-60s)
5. Download only the selected MP4s (~2 GB estimated)

### Phase 2: Retention Curve Construction

For each selected video of duration D seconds:

```
S(t) = |{users with watch_time ≥ t}| / |{users who started video}|
```

Compute S(t) at each integer second t ∈ [1, D]. This gives a per-video empirical survival function.

**Derived metrics per video:**
- `half_life` — time at which S(t) = 0.5
- `early_drop` — S(3) / S(1), fraction surviving first 3 seconds
- `completion_rate` — S(D), fraction who watched to the end
- `rewatch_rate` — fraction with watch_time > D
- `retention_auc` — area under the survival curve (normalized by D)
- `slope_changes` — second derivative of S(t), identifies "cliff" moments

### Phase 3: Brain Metric Timeseries (TRIBEv2 via Modal)

Adapt `modal_tribe_100.py` pipeline:

1. Upload selected MP4s to Modal volume
2. Run TRIBEv2 → fMRI predictions on fsaverage5 (20k vertices)
3. Parcellate with Schaefer 400-parcel 7-network atlas
4. Extract per-second network mean activation: DAN(t), DMN(t), FPN(t), VAN(t), Limbic(t), SomMot(t), Vis(t)
5. Compute derived timeseries:
   - `attention(t)` = DAN(t) - DMN(t)
   - `cognitive_load(t)` = FPN(t) + DAN(t) - DMN(t)
   - `dan_dmn_coupling(t)` = rolling correlation of DAN and DMN (window=3s)

### Phase 4: Correlation Analysis

**4a. Cross-video, per-timepoint:**
At each second t (across all videos of similar duration), correlate brain_metric(t) with S(t).
→ Does higher DAN at second 5 predict more viewers still watching at second 5?

**4b. Within-video temporal alignment:**
For each video, correlate the brain timeseries with the retention curve over time.
→ When DAN spikes, does retention flatten (viewers stay)?
→ When DMN rises, does retention drop (viewers leave)?

**4c. Lagged correlation / Granger causality:**
Does brain state at t predict retention change at t+k?
→ The brain model processes the video; retention is the behavioral response. If they're causally linked, brain signals should slightly lead retention changes.

**4d. Scalar summary correlation (replicates pilot):**
Correlate scalar brain metrics (mean DAN, mean attention, etc.) with scalar retention metrics (completion_rate, half_life, retention_auc).
→ Direct comparison to the pilot's DAN → share_rate finding.

### Phase 5: Visualization (Marimo notebook)

- Dual-axis plots: brain timeseries + retention curve overlaid per video
- Heatmap: videos × time, colored by DAN activation, sorted by retention shape
- Scatter plots: scalar brain metrics vs. retention metrics
- Cluster analysis: group videos by retention curve shape, compare brain signatures per cluster

## File Structure

```
experiments/retention/
├── PRD.md                  ← this file
├── 01_select_videos.py     ← download interactions, select 100 videos
├── 02_retention_curves.py  ← build per-second S(t) for each video
├── 03_modal_tribe.py       ← TRIBEv2 inference on Modal (adapted from modal_tribe_100.py)
├── 04_correlate.py         ← all correlation analyses
└── explore_retention.py    ← marimo notebook for visualization
```

## Risks

### TIGERS (clear threats)
1. **Sparse viewers per video** — 1M interactions / 153K videos ≈ 6.6 avg. Mitigation: filter to ≥30 viewers, accept smaller sample size.
2. **TRIBEv2 temporal resolution mismatch** — TRIBEv2 outputs at ~1 TR (≈1.5s fMRI resolution), retention is per-second. Mitigation: interpolate brain timeseries to 1Hz or bin retention to match TR.
3. **Short video duration** — Many videos are <15s, giving very few timepoints for within-video correlation. Mitigation: stratify sample toward 30-60s videos.
4. **Confound: video duration itself** — Longer videos have lower completion rates mechanically. Mitigation: normalize retention curves by duration, analyze within duration buckets.

### ELEPHANTS (unspoken concerns)
1. **Brain encoding models predict averaged brain responses, not individual viewers** — TRIBEv2 predicts what an "average brain" does, while retention is an aggregate of individual decisions. This is actually fine — both are population-level signals.
2. **Cultural content gap** — Tsinghua videos are from Kuaishou (Chinese market), TRIBEv2 was trained on English-language neuroscience studies. Cross-cultural/language transfer is untested. Mitigation: this IS the experiment — if it works across cultures, the finding is stronger.
