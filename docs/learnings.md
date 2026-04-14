# BrainGym Learnings Document

**Date:** 2026-04-12 to 2026-04-14
**Authors:** Toma Itagaki + Claude

---

## 1. TRIBEv2: What It Is and What It Isn't

### What It Is

TRIBEv2 (Meta FAIR, March 2026) is a tri-modal brain encoding model: video/audio/text in, predicted fMRI out. It predicts how the average healthy adult brain would respond to any stimulus, second by second, across 20,484 cortical vertices (fsaverage5).

- **Backbones:** V-JEPA2 Giant (video), Wav2Vec-BERT 2.0 (audio), LLaMA 3.2-3B (text)
- **Output:** 1Hz predicted fMRI on fsaverage5 cortical surface
- **Training:** 451 hours of fMRI from 25 subjects watching movies/podcasts
- **Context window:** 100 seconds
- **5-second hemodynamic lag:** Baked into the model. TRIBE's prediction at time t reflects stimulus at t-5.

### What It Isn't

- Not a mind reader (encoding, not decoding)
- Not an individual-level predictor (population average only)
- Not capable of subcortical predictions (cortical checkpoint released; subcortical trained but not released)
- Not trained on active viewing (passive scanner viewing only)
- CC BY-NC license: explicitly no commercial use, confirmed by lead author

### Key Caveats from the Paper

- The Algonauts 2025 competition win was by razor-thin margins (0.211 vs 0.2125 for 2nd place). Architecture didn't matter — ensembling strategy did. (Source: Paul Scotti's Algonauts analysis)
- The field appears to have hit a plateau at current data scales.
- Scaling law shows log-linear improvement with more data, no saturation.

---

## 2. Brain Networks and What They Mean

### Schaefer 7-Network Parcellation

| Network | Abbreviation | What it measures | Seminal paper |
|---------|-------------|-----------------|---------------|
| Dorsal Attention | DAN | Focused, top-down attention | Fox et al. 2005, PNAS |
| Default Mode | DMN | Mind-wandering, self-reflection | Raichle et al. 2001, PNAS |
| Frontoparietal Control | FPN | Executive control, cognitive effort | Duncan & Owen 2000, Trends Neurosci |
| Ventral Attention | VAN | Surprise, salience detection | Corbetta & Shulman 2002, Nat Rev Neurosci |
| Visual | Vis | Visual processing | — |
| Somatomotor | SomMot | Motor/physical processing | — |
| Limbic | Limbic | Emotion, reward (OFC + temporal pole) | — |

### Derived Metrics

- **Attention = DAN - DMN** (positive = engaged, negative = mind-wandering)
- **Engagement = -DMN** (higher = more engaged)
- **Cognitive Load = FPN + DAN - DMN**

### The DAN-DMN Anticorrelation

Fox et al. (2005) showed DAN and DMN are intrinsically anticorrelated: when one activates, the other suppresses. This is a fundamental organizational principle of the brain, not a task artifact.

---

## 3. Experiments and What They Proved

### 3.1 TikTok Engagement (n=77)

**Setup:** Ran TRIBE on 77 TikTok videos, correlated brain predictions with engagement metrics.

**Results:**
- Early attention (first 3s DAN-DMN) -> share_rate: **r=+0.26, p=0.02**
- Peak attention -> engagement_rate: **r=-0.35, p=0.002** (attention-grabbing content gets fewer likes per view due to reach dilution)

**Learning:** The brain's initial attention response predicts sharing behavior. But peak attention is negatively correlated with engagement rate — virality dilutes per-viewer engagement.

### 3.2 Tsinghua Retention Curves (n=97)

**Setup:** Ran TRIBE on 97 short videos from the Tsinghua ShortVideo dataset, correlated brain predictions with second-by-second retention curves.

**Results:**
- -DMN with 5s lag -> retention: **median r=0.72** (raw), **r~0.10** after detrending
- DMN oscillation range -> half_life: **r=+0.30, p<0.003**

**Learning:** Most of the raw r=0.72 correlation is shared temporal trend (both brain signal and retention curve drift downward over time). After detrending, the real signal is modest (r~0.10) but consistent across 75% of videos. Content that oscillates between high and low engagement (DMN oscillation range) keeps people watching longer.

### 3.3 Detrending Analysis

**Setup:** Applied scipy.signal.detrend (linear) to both brain timeseries and retention curves independently, then correlated the residuals.

**Learning:** Linear detrending is conservative but reasonable. The r~0.10 after detrending is real signal (survives shuffle tests) but modest. The question of how much detrending is "right" remains open — linear detrending removes the dominant shared trend but not nonlinear shared patterns.

### 3.4 Habituation Test (30-second static image)

**Setup:** Fed a single static image as a 30-second silent video through TRIBE to test whether the model predicts realistic temporal dynamics.

**Results (corrected for 5s baked-in lag):**
- t=0-5 (pre-stimulus): Baseline prediction, not a response to the image
- t=5-10 (image onset): Weak initial response, attention drops toward zero
- t=10-20: Attention **rebounds** — the transformer hallucinates a "something new happened" response
- t=20-30: Mixed signals, DMN rises at end

**Learning:** TRIBE does NOT model realistic habituation for static images. Beyond ~5 seconds, the temporal dynamics are artifacts of movie-trained patterns, not genuine brain predictions. The model "expects" scene changes and generates corresponding brain responses even when nothing changes.

**Implication:** For static images, only trust the initial response (t=5 per the paper's flash protocol). The temporal dynamics are hallucinated.

### 3.5 Image Preference Experiment (HPD v3, n=100 pairs)

**Setup:** Sampled 100 high-confidence pairwise preference comparisons from HPD v3 (AI-generated images). Ran both images through TRIBE (as 2-second silent videos), averaged across timepoints. Tested whether any brain signal distinguishes preferred from non-preferred.

**Results (paired t-test, preferred - non-preferred):**

| Signal | Cohen's d | Accuracy | p-value | Direction |
|--------|-----------|----------|---------|-----------|
| DMN | 0.594 | 70% | <0.000001 | preferred > non-preferred |
| Limbic | 0.426 | 70% | 0.00005 | preferred > non-preferred |
| **DMN + Limbic** | **0.535** | **75%** | **0.000001** | **preferred > non-preferred** |
| Vis | 0.336 | 60% | 0.001 | preferred > non-preferred |
| Attention (DAN-DMN) | -0.314 | — | 0.002 | non-preferred > preferred |
| Engagement (-DMN) | -0.594 | — | <0.000001 | non-preferred > preferred |
| CLIP score | 0.325 | 50% | 0.002 | preferred > non-preferred |

**Surprising finding:** Preferred images activate DMN MORE, not less. This means preferred images trigger more self-referential/contemplative processing, not more focused attention. The "engagement = -DMN" metric goes the wrong way for preference — it's an engagement metric, not a preference metric.

**DMN + Limbic at 75% accuracy** correctly identifies which image humans prefer in 3 out of 4 pairs — substantially better than CLIP score (50% on individual pairs).

### 3.6 V-JEPA2 Control Experiment (the critical test)

**Setup:** Extracted raw V-JEPA2 embeddings (1408-dim) for the same 100 image pairs. Tested whether V-JEPA2 features alone predict preference without any brain mapping.

**Results:**

| Method | Accuracy | Training? |
|--------|----------|-----------|
| V-JEPA2 LOO-CV logistic regression | 99% | Yes — **BUT BOGUS** (1408 dims >> 200 samples, curse of dimensionality) |
| V-JEPA2 PCA-5 mean direction (LOO) | **82%** | No (honest) |
| V-JEPA2 PCA-2/10/50 mean direction | **79%** | No (honest) |
| V-JEPA2 permutation test | p<0.0001 | Real=83%, null=70% |
| TRIBE DMN+Limbic | **75%** | No (honest) |
| CLIP score | 50% | No (honest) |

**The 99% was a curse-of-dimensionality artifact.** With 1408 features and only 200 samples (n/d = 0.14), logistic regression trivially finds a separating hyperplane in high-dimensional space. After PCA reduction to d=5 (n/d = 20), the honest accuracy is 79-82%.

**KEY LEARNING: V-JEPA2 alone beats TRIBE for preference (82% vs 75%).** The brain mapping is a lossy projection of V-JEPA2 features. It preserves most of the preference signal (75/82 = 91%) but cannot exceed its input features.

### 3.7 The n << d Problem (Curse of Dimensionality)

**What happened:** Logistic regression on 1408-dim V-JEPA2 embeddings with only 200 samples reported 99% accuracy.

**Why it was wrong:** In d dimensions, you need at least d+1 samples to ensure linear inseparability of random data. With n < d, a linear classifier can almost always find a perfect separator — even on noise.

**Rule of thumb:** Need n/d >= 10 for reliable results. We had n/d = 0.14 (70x too few samples).

**Fix:** PCA reduction to d=5 gives n/d = 20, yielding honest 79-82% accuracy.

---

## 4. The Central Insight: When TRIBE Adds Value

### TRIBE does NOT add value when:
**Human labels exist at scale for the target metric.**

Preference has millions of human labels (HPD v3, Pick-a-Pic, ImageReward). Any good visual backbone (V-JEPA2, CLIP) can be directly trained/fine-tuned on these labels. Going through the brain is a lossy detour — the brain mapping can't beat direct supervision on a well-labeled task.

### TRIBE DOES add value when:
**The brain IS the ground truth — no human label exists.**

| Signal | Why TRIBE is irreplaceable |
|--------|---------------------------|
| Per-second attention (DAN-DMN) | No dataset of "attention at second 7" exists. Can't crowdsource this. |
| Cognitive load per second (FPN+DAN-DMN) | Can't self-report without disrupting the task |
| Mind-wandering onset (DMN activation) | People don't know when they zone out |
| Surprise/salience timing (VAN spikes) | Subjective surprise != neural surprise |
| Emotional engagement (Limbic) | Self-report is post-hoc and reconstructed |

**The competitive moat isn't "brain beats features."** It's **"brain gives you metrics that don't exist anywhere else."** No amount of V-JEPA2 fine-tuning gives you per-second DAN-DMN because there's no label to fine-tune on.

### Framework for evaluating TRIBE for any new task:

```
1. Does a large-scale human label dataset exist for this task?
   ├── Yes → Skip TRIBE, use backbone + labels directly
   └── No → Does the brain define the ground truth?
       ├── Yes → TRIBE is the best (possibly only) option
       └── No → TRIBE probably doesn't help
```

---

## 5. Technical Findings

### 5.1 Hemodynamic Lag Handling

TRIBE's 5-second lag is baked into the model. TRIBE at t=5 reflects stimulus at t=0. The paper's protocol for static images: flash image for 1 second, read prediction at t=5 (the hemodynamic peak). We confirmed this with the habituation test — the true image response starts at output t=5.

### 5.2 Static Images Through TRIBE

TRIBE expects video. For static images:
- **2-second silent video (naive):** Works for getting an initial brain response. Average across 2 timepoints.
- **1s flash + 7s gray (paper protocol):** More rigorous. Read at t=5 for hemodynamic peak.
- **Beyond 5-10 seconds:** Temporal dynamics are hallucinated. The model generates movie-like temporal patterns even with no stimulus change.

### 5.3 TRIBE vs BDDN

BDDN (BrainDecodesDeepNets) is an image-to-fMRI model using CLIP backbone, outputting 37,984 nsdgeneral vertices.

| Property | TRIBE | BDDN |
|----------|-------|------|
| Input | Video/audio/text | Single image |
| Output space | fsaverage5 (20,484 cortical) | nsdgeneral (37,984 visual cortex) |
| Backbone | V-JEPA2 | CLIP |
| Covers reward regions (vmPFC, OFC) | Partially (via Limbic parcels) | No (nsdgeneral = visual cortex only) |
| Covers subcortical (NAcc, amygdala) | No (released model is cortical only) | No |
| Temporal dynamics | Yes (1Hz timeseries) | No (single prediction per image) |

We attempted to run BDDN on Modal but the brainnet package has a deep dependency chain (pycortex, dinov2, SAM, matplotlib, torchmetrics) that required 8 attempts to resolve. We eventually reimplemented the FactorTopy architecture from scratch without the brainnet package — but it still failed due to open_clip API differences. BDDN was dropped from the preference experiment.

### 5.4 Multimodal Integration

From the TRIBEv2 paper: multimodal predictions exceed best-unimodal by up to 50% at the temporal-parietal-occipital junction. Text dominates in language/prefrontal cortex, video in occipital/parietal, audio in temporal cortex.

### 5.5 TTS Voice Experiment

Same text with different TTS voices produces nearly identical brain predictions (r=0.84-0.99). Different text with the same voice produces much more different patterns (r=0.55-0.90). **Content drives ~85-90% of predicted brain response; voice/prosody drives ~5-15%.** (Documented in CLAUDE.md)

---

## 6. The Landscape

### 6.1 Community Reception

TRIBE was largely ignored by the ML community (no Reddit threads, minimal HN engagement). The neuromarketing community is excited but blocked by CC BY-NC. GitHub issues show strong commercial demand (3 separate licensing requests). The most technically rigorous external analysis (Scotti's Algonauts paper) found the winning margin was razor-thin and architecture-independent.

### 6.2 Commercial Players

| Company | Approach | Signal | Limitation |
|---------|----------|--------|------------|
| Neurons Inc | Trained on EEG/eye-tracking, deploy as pure CV | Attention heatmaps | No longer measures brain — predicts what brain would do |
| Neurensics | Actual fMRI scanning for ad testing | 13 neural networks | Still requires scanner ($800/hr) |
| Immersion Neuroscience | Smartwatch HRV → "oxytocin" | Engagement score | Oxytocin claims are questionable |
| Neuro-Insight | SST (refined EEG) | Memory encoding | "86% correlation to sales" from one study |

### 6.3 Cross-Modal Brain Foundation Models

The "train on fMRI, deploy on EEG" paradigm has been tried by 6+ groups (Brain-OF, BrainOmni, LaBraM, MCSP, NOBEL, etc.). The honest results:

- **EEG-to-fMRI regression: r=0.175 maximum** (~3% shared variance)
- **LLMs for EEG-to-fMRI: complete failure** (Llama 0.5% significant)
- **Clinical classification:** Cross-modal pretraining adds 2-10% accuracy (real)
- **Fundamental limit:** Volume conduction blurs EEG to 5-9cm resolution. This is physics, not data.

**Key insight:** TRIBE already IS the "deploy cheap" solution. It goes stimulus → brain directly, no scanner or wearable needed. The cross-modal question is irrelevant for TRIBE's use case.

---

## 7. Available Data for Future Work

### What TRIBE Trained On
451h fMRI, 25 subjects, 3T only, all Western adults, movies/podcasts

### What's Available Beyond TRIBE

| Dataset | Why it matters | Subjects | Field |
|---------|---------------|----------|-------|
| NSD | 7T gold standard, 10K images/subject | 8 | 7T |
| Spacetop | 101 subjects, movies + physiology | 101 | 3T |
| CamCAN | 700 subjects ages 18-88 (age diversity) | 700 | 3T |
| HAD | 21,600 action video clips | 30 | 3T |
| StudyForrest | 7T, rich annotations | 20 | 7T |
| Emo-FilM | Emotion-rated films, heart rate | 30 | 3T |
| 101 Dalmatians | Multimodal (AV/A-only/V-only) | 50 | 3T |

### Gaps Nobody Has Filled
- No TikTok/short-form fMRI data exists publicly
- No ad-viewing fMRI data is public (proprietary neuromarketing)
- No UI/interface viewing fMRI data exists at all
- No non-WEIRD population data beyond LPP multilingual
- Subcortical checkpoint trained but not released by Meta

---

## 8. Products Built

### 8.1 BrainBiz Chat
RAG chat over 653 neuromarketing papers. ChromaDB vectorstore with OpenAlex-sourced abstracts. Streamlit + Claude API. Marketing/lead-gen play.

### 8.2 NeuroScript
Content script optimizer for creators. Scores on 5 brain-based dimensions. Consumer-facing.

### 8.3 NeuroTribe
Interpretable TRIBE interface. Upload video/audio/text → encode on Modal A10G → visualize network timeseries + brain surfaces → chat with citations.

### 8.4 OpenAlex Scraper
Discovered 922 papers (70% OA) across neuromarketing/neuroforecasting. The key improvement: use keyword ID filters instead of free-text search to avoid irrelevant results.

---

## 9. Open Questions

1. **Does the flash protocol (1s image + 7s gray, read at t=5) improve preference prediction over the naive 2s static approach?** The flash run completed on Modal but results haven't been analyzed yet.

2. **Would subcortical predictions (nucleus accumbens, amygdala) add preference signal that cortical predictions miss?** Likely yes — reward circuitry is subcortical. But weights aren't released.

3. **Can we retrain TRIBE with subcortical targets?** Yes — all training data, code, and backbone models are public. Feature extraction costs ~$3-5K, training costs ~$30. The bottleneck is fMRI data preprocessing.

4. **Is TRIBE's attention signal genuinely better than V-JEPA2 features for retention prediction?** We tested this for preference (V-JEPA2 wins). Haven't tested for temporal attention/retention — this is where TRIBE should have the advantage because temporal dynamics are the value-add.

5. **What's the right detrending approach for brain-retention correlations?** Linear detrending gives r~0.10 (real but modest). No detrending gives r~0.72 (inflated by shared trend). The truth is somewhere in between.

---

## 10. Key Takeaways

1. **TRIBE's value is temporal attention dynamics in video, not static image analysis.** For static comparisons, use the backbone directly.

2. **The "brain as reward model" thesis is weakened** by the V-JEPA2 control experiment showing backbone features beat brain predictions for preference.

3. **The "brain as attention model" thesis remains strong** because per-second attention labels don't exist — the brain IS the ground truth.

4. **DMN + Limbic predicts image preference at 75%** — a real finding, but V-JEPA2 gets 82% without the brain mapping.

5. **The commercial moat is not "brain beats features"** but "brain gives you metrics that don't exist elsewhere."

6. **CC BY-NC blocks all commercial use of TRIBE.** Retraining from public data with your own weights is the path to commercialization.

7. **The field is at a plateau.** Top Algonauts teams converged on the same approach. More data (not architecture innovation) is the path to improvement.
