# Changelog

## 2026-04-14

### Preference Experiment Results
- **HPD v3 image preference experiment (n=100 pairs):** DMN + Limbic predicts human image preference at 75% pairwise accuracy (Cohen's d=0.535, p<0.000001)
- **V-JEPA2 control experiment:** Raw V-JEPA2 embeddings predict preference at 79-82% after PCA reduction — backbone features beat brain mapping (75%)
- **Key insight:** TRIBE adds value for temporal attention dynamics (where brain IS ground truth) but not for preference (where human labels exist at scale)
- Habituation test: TRIBE hallucinates temporal dynamics beyond ~5s on static images — only initial response (t=5) is trustworthy
- Flash protocol (1s image + 7s gray, paper's method) running on Modal for comparison with naive 2s static approach

### Infrastructure
- Standalone BDDN FactorTopy reimplementation (no brainnet dependency) for Modal deployment
- V-JEPA2 embedding extraction pipeline on Modal
- HPD v3 data pipeline: partial shard extraction (84,596 images from first 10GB shard)

### Documentation
- `docs/learnings.md`: Comprehensive learnings document covering all experiments, findings, and the central TRIBE value framework
- Updated CLAUDE.md with TTS experiment finding and GEPA reference

## 2026-04-13

### Apps
- **BrainBiz Chat:** RAG chat over 653 neuromarketing papers (ChromaDB + Claude API + Streamlit)
- **NeuroScript:** Content script optimizer scoring on 5 brain-based dimensions for creators
- **NeuroTribe:** Interpretable TRIBEv2 interface — upload content, encode on Modal A10G, visualize brain response, chat with Perplexity-style citations
- Modal endpoint deployed for TRIBE inference (`neurotribe-api`)

### Research
- OpenAlex corpus discovery: 922 neuromarketing papers found, 70% open access
- Key signals sidebar with seminal paper citations (Fox 2005, Raichle 2001, Corbetta & Shulman 2002, Duncan & Owen 2000)
- Extensive research on fMRI-to-commercial applications, neuroforecasting literature

### Infrastructure
- Query tracking (JSONL) for NeuroTribe usage analytics
- `neurotribe-analytics` Claude skill for viewing tracking data
- PR #2 opened on braingym repo

### Findings
- TTS voice experiment: content drives 85-90% of brain response, voice/prosody drives 5-15%
- TRIBEv2 paper deep-dive: 5s hemodynamic lag is baked in, model uses unseen-subject mode for zero-shot, CC BY-NC license confirmed no commercial use

## 2026-04-12

### Prior Work (documented in CLAUDE.md)
- TikTok (n=77): early attention predicts sharing (r=0.26), peak attention negatively correlates with engagement rate (r=-0.35)
- Tsinghua retention (n=97): -DMN tracks retention (median r=0.72 raw, r~0.10 detrended), DMN oscillation range predicts half_life (r=0.30)
