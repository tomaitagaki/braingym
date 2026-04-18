# TRIBE for UI Design Evaluation: Experiment Results

**Date:** 2026-04-15
**Question:** Can TRIBE (a video-based brain encoding model) evaluate UI design quality?

## Background

TRIBE v2 predicts fMRI brain responses from video using V-JEPA2 (video), Wav2Vec (audio), and LLaMA (text). It was trained on naturalistic video (movies, nature scenes, narratives). We tested whether it could distinguish good UI design from bad UI design by converting UI screenshots into scroll videos.

## Experiment 1: V-JEPA2 OOD Check

**Goal:** Are UI screen recordings out-of-distribution for V-JEPA2?

**Method:** Extracted V-JEPA2 frame embeddings from naturalistic videos (Sintel trailer, TikTok, Tsinghua movie clips), synthetic animations (Manim), and real UI screen recordings. Compared distributions via L2 norms, cosine similarity, Mahalanobis distance, and PCA.

**Result:** UI embeddings have normal L2 norms (96.8 vs 93.9 naturalistic) and in-range Mahalanobis distance (6.5 vs 6.6 naturalistic), but cluster separately in PCA space (PC1 ~60-80 vs naturalistic at -20 to +20). **Verdict: borderline.** Not pathologically OOD, but occupying a distinct region.

**Files:** `experiments/jepa_ood/modal_jepa_ood.py`, `cache/jepa_ood/`

## Experiment 2: TikTok vs UI Screen Recordings

**Goal:** Does TRIBE produce directionally correct predictions for UI?

**Method:** Ran TRIBE on 2 TikTok videos and 2 real UI screen recordings. Compared network activations against neuroscientific hypotheses.

**Result:** 6/6 hypotheses confirmed:

| Metric | TikTok | UI | Hypothesis | Result |
|--------|--------|-----|-----------|--------|
| DAN | +0.127 | +0.031 | TikTok higher (captures attention) | PASS |
| DMN | -0.048 | -0.046 | TikTok lower (suppresses mind-wandering) | PASS (marginal) |
| VAN | +0.036 | +0.025 | TikTok higher (novelty/surprise) | PASS |
| FPN | -0.009 | +0.031 | UI higher (more cognitive effort) | PASS |
| Engagement | +0.048 | +0.046 | TikTok higher | PASS (marginal) |
| Brain reward | +0.077 | +0.045 | TikTok higher | PASS |

**Interpretation:** TRIBE distinguishes gross content categories. TikTok (designed for engagement) vs UI (functional) produces the expected pattern. However, this may just reflect V-JEPA2 seeing fundamentally different visual content, not cognitive processing differences.

**Files:** `experiments/jepa_ood/modal_tiktok_vs_ui.py`, `cache/tiktok_vs_ui/`

## Experiment 3: RLDF Ranking Pairs (Subtle Design Differences)

**Goal:** Can TRIBE distinguish designer-preferred UI from rejected UI?

**Method:** Used Apple's RLDF dataset (ranking split). Took 3 chosen/rejected pairs (chatbot, live chat, dashboard). Rendered HTML in Playwright at 512×512 viewport, recorded standardized 10-second scroll videos, ran TRIBE.

**Result:** Chosen wins 13/21 comparisons (62%). Dashboard pair was strongest (5/7), but overall marginally above chance.

**Interpretation:** The ranking pairs have very subtle design differences (both are LLM-generated, both decent). TRIBE can't reliably distinguish between them.

**Files:** `experiments/rldf_ui/02_modal_tribe_rldf.py`, `cache/rldf/rldf_tribe_results.json`

## Experiment 4: RLDF Comment-Improved Pairs (Designer Critique → Fix)

**Goal:** Can TRIBE distinguish before/after when designers explicitly identified and fixed issues?

**Method:** Used 3 pairs from RLDF comment-improved split. Each had 5-7 designer-identified issues (misaligned carets, empty sections, broken typography). Recorded scroll videos of before and after.

**Result:** Improved wins 10/21 comparisons (48%) — exactly chance.

**Interpretation:** Even with explicit designer fixes, the visual changes in scroll videos are too subtle for V-JEPA2 to encode differently.

**Files:** `experiments/rldf_ui/03_modal_tribe_improved.py`, `cache/rldf/improved_tribe_results.json`

## Experiment 5: Vibe-Coded vs Professional (Extreme Contrast)

**Goal:** With maximum visual contrast (same content), can TRIBE tell them apart?

**Method:** Hand-crafted two chatbot UIs with identical conversation. "Vibe-coded": rainbow gradient, Comic Sans, neon green/blue chat bubbles, blinking ad banners, red SEND button. "Professional": Apple HIG-style, SF Pro font, clean blue bubbles, proper spacing.

**Result:** Clear separation. Network profile correlation r=0.57 (genuinely different brain patterns).

| Metric | Vibe | Pro | Diff |
|--------|------|-----|------|
| Attention (DAN-DMN) | +0.014 | +0.068 | +0.054 |
| DAN | -0.045 | +0.010 | +0.055 |
| Brain Reward | +0.021 | +0.038 | +0.018 |
| Visual (Vis) | +0.034 | +0.012 | -0.022 |

**Interpretation:** TRIBE sees the difference, but this is trivially obvious — the visual content (pixel values) is completely different. The vibe-coded version has rainbow gradients and bright neon colors; the professional version has white/gray/blue. V-JEPA2 encodes these as different visual features regardless of design quality.

**Files:** `experiments/rldf_ui/04_modal_vibe_vs_pro.py`, `cache/rldf/vibe_vs_pro_results.json`

## Experiment 6: CRAP Jitter (Medium Severity Design Defects)

**Goal:** UIClip-style jitter functions (color swap, contrast reduction, font scramble, spacing noise, alignment break) — medium severity, realistic "junior dev" mistakes.

**Method:** Created 3 clean UIs (dashboard, settings, product page). Applied jitter functions that modify CSS properties (swap colors, reduce contrast, randomize font sizes, add spacing noise). Recorded scroll videos.

**Result:** Clean wins 12/18 comparisons (67%). Engagement and brain reward favor clean in all 3 pairs.

**Critical flaw identified:** The jitter changes pixel content (colors, contrast, font sizes), not just design quality. V-JEPA2 responds to the pixel differences, not the design quality differences. When user watched the actual videos side-by-side, the jitter was barely noticeable during scroll.

**Files:** `experiments/rldf_ui/05_jitter_and_run.py`, `experiments/rldf_ui/06_modal_jitter.py`, `cache/rldf/jitter/`

## Experiment 7: Layout-Only (Same Pixels, Different Section Order)

**Goal:** The definitive test. Keep all CSS, colors, fonts, images identical. Only change the order sections appear during scroll. Does TRIBE respond to information architecture?

**Method:** Created 4 variants of a product page with identical styling:
- **good_hierarchy:** image → price → description → specs → CTA → reviews
- **specs_first:** image → specs → description → price → CTA → reviews
- **cta_first:** image → CTA → price → reviews → description → specs
- **reversed:** reviews → specs → description → CTA → price → image

All have identical page height (1211px) and scroll distance (699px).

**Result:** DAN timeseries correlation between layouts: r = 0.79–0.98. The trajectories are nearly identical. Total metric spread across all 4 layouts: 1.3%.

`brain_reward` happens to perfectly rank the layouts (rho = -1.0), but inverted (lower = better) and with negligible magnitude differences (0.045 vs 0.051). This is likely coincidence with n=4.

**Interpretation:** When pixel content is controlled, TRIBE produces the same output regardless of layout. The model responds to aggregate visual features across the scroll, not the temporal order of information.

**Files:** `experiments/rldf_ui/07_layout_experiment.py`, `experiments/rldf_ui/08_modal_layout.py`, `cache/rldf/layout/`

## Cross-Experiment Metric Analysis

Pooled all 10 pairs across experiments (ranking, improved, vibe/pro, jitter) to find which metrics best predict preference.

**Best single metrics (80% win rate):** engagement (-DMN), brain_reward, engage_consistency, fpn_slope

**Best composite (90% win rate):** `engagement + attention` (= DAN - 2×DMN)

**However:** These metrics win because they're driven by pixel-level visual differences (experiments 5 and 6), not design quality. When those experiments are removed, the win rates drop to chance.

## Conclusions

1. **TRIBE cannot evaluate UI design quality.** The model was trained on naturalistic video (movies, nature) and learned to predict brain responses to narrative, faces, and scene dynamics. UI scrolling videos lack these features.

2. **Every positive result was driven by pixel differences, not design understanding.** When we controlled for pixel content (layout experiment), the signal vanished.

3. **V-JEPA2 processes visual texture, not spatial organization.** It sees "blue pixels vs red pixels," not "good hierarchy vs bad hierarchy." The 4-second clip window averages over spatial structure.

4. **The sensitivity gradient:**
   - Cross-category (TikTok vs UI): Strong signal, but trivially different content
   - Extreme visual contrast (vibe vs pro): Moderate signal, driven by color/contrast differences
   - Medium jitter (CRAP violations): Weak signal, driven by pixel changes
   - Subtle fixes (RLDF improved): No signal
   - Layout-only (same pixels): No signal

5. **For UI design evaluation, a different approach is needed:** image-based models (CLIP, UIClip), eye-tracking proxies, or brain encoding models trained on UI-specific fMRI data.
