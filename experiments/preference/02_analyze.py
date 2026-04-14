"""
Phase 2: Analyze brain encoding × preference.

Tests whether TRIBE/BDDN predicted fMRI distinguishes
preferred from non-preferred images in HPD v3.

Usage: marimo edit experiments/preference/02_analyze.py
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy import stats

    CACHE = Path("cache/preference")
    mo.md("## Brain-Predicts-Preference Analysis")
    return CACHE, Path, json, mo, np, pd, stats


@app.cell
def _(CACHE, json, mo, pd):
    """Load encoding results."""
    results_path = CACHE / "encoding_results.json"
    data = json.loads(results_path.read_text())
    df = pd.DataFrame(data)

    n_pairs = df["pair_id"].nunique()
    n_with_tribe = df["tribe_attention"].notna().sum()
    n_with_bddn = df.apply(lambda r: bool(r.get("bddn_roi_means")), axis=1).sum()
    n_with_clip = df["clip_score"].notna().sum()

    mo.md(f"""### Data loaded
- **{n_pairs}** pairs, **{len(df)}** images
- TRIBE encoded: **{n_with_tribe}**
- BDDN encoded: **{n_with_bddn}**
- CLIP scored: **{n_with_clip}**
""")
    return (df,)


@app.cell
def _(df, mo, np, pd, stats):
    """Analysis 1: Paired t-test per brain region — does preferred ≠ non-preferred?"""

    # Pivot to preferred vs non-preferred per pair
    pref = df[df["is_preferred"] == True].set_index("pair_id")
    nonpref = df[df["is_preferred"] == False].set_index("pair_id")
    shared_pairs = sorted(set(pref.index) & set(nonpref.index))

    # ── TRIBE network means ──
    tribe_results = []
    tribe_networks = ["DAN", "DMN", "FPN", "VAN", "Vis", "SomMot", "Limbic"]

    for net in tribe_networks:
        deltas = []
        for pid in shared_pairs:
            p_nets = pref.loc[pid, "tribe_network_means"]
            n_nets = nonpref.loc[pid, "tribe_network_means"]
            if isinstance(p_nets, dict) and isinstance(n_nets, dict) and net in p_nets and net in n_nets:
                deltas.append(p_nets[net] - n_nets[net])

        if len(deltas) >= 10:
            deltas = np.array(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = deltas.mean() / max(deltas.std(), 1e-10)  # Cohen's d
            tribe_results.append({
                "region": f"TRIBE {net}",
                "mean_delta": round(deltas.mean(), 6),
                "std_delta": round(deltas.std(), 6),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 5),
                "cohens_d": round(d, 3),
                "n_pairs": len(deltas),
                "direction": "preferred > non-preferred" if deltas.mean() > 0 else "non-preferred > preferred",
            })

    # Derived TRIBE metrics
    for metric_name, metric_col in [("attention", "tribe_attention"), ("engagement", "tribe_engagement"), ("cognitive_load", "tribe_cognitive_load")]:
        deltas = []
        for pid in shared_pairs:
            p_val_m = pref.loc[pid, metric_col]
            n_val_m = nonpref.loc[pid, metric_col]
            if pd.notna(p_val_m) and pd.notna(n_val_m):
                deltas.append(p_val_m - n_val_m)
        if len(deltas) >= 10:
            deltas = np.array(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = deltas.mean() / max(deltas.std(), 1e-10)
            tribe_results.append({
                "region": f"TRIBE {metric_name}",
                "mean_delta": round(deltas.mean(), 6),
                "std_delta": round(deltas.std(), 6),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 5),
                "cohens_d": round(d, 3),
                "n_pairs": len(deltas),
                "direction": "preferred > non-preferred" if deltas.mean() > 0 else "non-preferred > preferred",
            })

    # ── BDDN ROI means ──
    bddn_results = []
    bddn_rois = ["V1", "V2", "V3", "V4", "EBA", "FBA", "OFA", "FFA", "OPA", "PPA", "OWFA", "VWFA"]

    for roi in bddn_rois:
        deltas = []
        for pid in shared_pairs:
            p_rois = pref.loc[pid, "bddn_roi_means"]
            n_rois = nonpref.loc[pid, "bddn_roi_means"]
            if isinstance(p_rois, dict) and isinstance(n_rois, dict) and roi in p_rois and roi in n_rois:
                deltas.append(p_rois[roi] - n_rois[roi])

        if len(deltas) >= 10:
            deltas = np.array(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = deltas.mean() / max(deltas.std(), 1e-10)
            bddn_results.append({
                "region": f"BDDN {roi}",
                "mean_delta": round(deltas.mean(), 6),
                "std_delta": round(deltas.std(), 6),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 5),
                "cohens_d": round(d, 3),
                "n_pairs": len(deltas),
                "direction": "preferred > non-preferred" if deltas.mean() > 0 else "non-preferred > preferred",
            })

    # ── CLIP control ──
    clip_deltas = []
    for pid in shared_pairs:
        p_clip = pref.loc[pid, "clip_score"]
        n_clip = nonpref.loc[pid, "clip_score"]
        if pd.notna(p_clip) and pd.notna(n_clip):
            clip_deltas.append(p_clip - n_clip)

    clip_result = None
    if len(clip_deltas) >= 10:
        clip_deltas = np.array(clip_deltas)
        t_stat, p_val = stats.ttest_1samp(clip_deltas, 0)
        d = clip_deltas.mean() / max(clip_deltas.std(), 1e-10)
        clip_result = {
            "region": "CLIP score",
            "mean_delta": round(clip_deltas.mean(), 6),
            "std_delta": round(clip_deltas.std(), 6),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 5),
            "cohens_d": round(d, 3),
            "n_pairs": len(clip_deltas),
            "direction": "preferred > non-preferred" if clip_deltas.mean() > 0 else "non-preferred > preferred",
        }

    # Combine all results
    all_results = tribe_results + bddn_results
    if clip_result:
        all_results.append(clip_result)

    results_df = pd.DataFrame(all_results).sort_values("p_value")

    # Bonferroni correction
    n_tests = len(results_df)
    results_df["p_bonferroni"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
    results_df["significant"] = results_df["p_bonferroni"] < 0.05

    mo.vstack([
        mo.md(f"""### Paired t-tests: preferred vs non-preferred ({len(shared_pairs)} pairs)
Does any brain region systematically differ between preferred and non-preferred images?

`*` = p < 0.05 (Bonferroni corrected across {n_tests} tests)"""),
        mo.ui.table(results_df.round(5)),
    ])
    return all_results, bddn_results, clip_deltas, clip_result, nonpref, pref, results_df, shared_pairs, tribe_results


@app.cell
def _(results_df):
    """Plot effect sizes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_df = results_df.sort_values("cohens_d")
    colors = ["#22c55e" if d > 0 else "#ef4444" for d in sorted_df["cohens_d"]]
    bars = ax.barh(range(len(sorted_df)), sorted_df["cohens_d"], color=colors, edgecolor="white")

    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["region"], fontsize=9)
    ax.set_xlabel("Cohen's d (positive = preferred > non-preferred)", fontsize=11)
    ax.set_title("Effect sizes: brain response difference (preferred - non-preferred)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.5)

    # Mark significant
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        if row.get("significant"):
            ax.text(row["cohens_d"], i, " *", fontsize=14, fontweight="bold", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.close("all")
    return (fig,)


@app.cell
def _(fig, mo):
    mo.vstack([mo.md("### Effect sizes"), fig])
    return


@app.cell
def _(clip_deltas, mo, np, stats):
    """Analysis 2: CLIP null hypothesis."""
    if len(clip_deltas) >= 10:
        t, p = stats.ttest_1samp(clip_deltas, 0)
        pct_correct = (clip_deltas > 0).mean() * 100

        mo.md(f"""### CLIP null hypothesis control

Does CLIP score alone predict preference?

- **CLIP score delta** (preferred - non-preferred): mean = {clip_deltas.mean():.4f}, std = {clip_deltas.std():.4f}
- **t-test**: t = {t:.3f}, p = {p:.5f}
- **Accuracy** (% pairs where CLIP prefers the human-preferred image): **{pct_correct:.1f}%**

{"CLIP score significantly predicts preference. BDDN results must be interpreted with caution — the CLIP backbone may drive any preference signal." if p < 0.05 else "CLIP score does NOT significantly predict preference. Any BDDN brain signal that does would be genuinely novel."}

Note: TRIBE uses V-JEPA2 (not CLIP), so TRIBE preference signals are independent of this confound.
""")
    else:
        mo.md("### CLIP null hypothesis: insufficient data")
    return


@app.cell
def _(df, mo, np, pd, shared_pairs, stats):
    """Analysis 3: Multivariate classification — can the full brain pattern predict preference?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    pref = df[df["is_preferred"] == True].set_index("pair_id")
    nonpref = df[df["is_preferred"] == False].set_index("pair_id")

    # Build feature matrices from network/ROI means
    def build_features(row, model="tribe"):
        if model == "tribe":
            nets = row.get("tribe_network_means", {})
            if not isinstance(nets, dict):
                return None
            return [nets.get(n, 0) for n in ["DAN", "DMN", "FPN", "VAN", "Vis", "SomMot", "Limbic"]]
        elif model == "bddn":
            rois = row.get("bddn_roi_means", {})
            if not isinstance(rois, dict):
                return None
            return [rois.get(r, 0) for r in ["V1", "V2", "V3", "V4", "EBA", "FBA", "OFA", "FFA", "OPA", "PPA", "OWFA", "VWFA"]]
        return None

    results = {}
    for model_name in ["tribe", "bddn"]:
        X, y = [], []
        for pid in shared_pairs:
            feat_p = build_features(pref.loc[pid], model_name)
            feat_n = build_features(nonpref.loc[pid], model_name)
            if feat_p is not None and feat_n is not None:
                # Difference vector: preferred - non-preferred
                diff = np.array(feat_p) - np.array(feat_n)
                X.append(diff)
                y.append(1)  # label: is this the correct direction?

        if len(X) < 20:
            results[model_name] = {"accuracy": None, "n": len(X)}
            continue

        X = np.array(X)
        y = np.array(y)

        # LOO CV: predict if delta is positive (preferred > non-preferred)
        loo = LeaveOneOut()
        correct = 0
        for train_idx, test_idx in loo.split(X):
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            correct += (pred == y[test_idx]).sum()

        accuracy = correct / len(X)
        # Binomial test: is accuracy > chance (50%)?
        binom_p = stats.binom_test(correct, len(X), 0.5)

        results[model_name] = {
            "accuracy": round(accuracy * 100, 1),
            "n": len(X),
            "correct": correct,
            "p_value": round(binom_p, 5),
        }

    tribe_r = results.get("tribe", {})
    bddn_r = results.get("bddn", {})

    mo.md(f"""### Multivariate classification (LOO-CV)

Can the full brain pattern (not just individual ROIs) distinguish preferred from non-preferred?

| Model | Features | Accuracy | p-value (vs chance) | n pairs |
|-------|----------|----------|-------|---------|
| TRIBE | 7 network means | {tribe_r.get('accuracy', 'N/A')}% | {tribe_r.get('p_value', 'N/A')} | {tribe_r.get('n', 0)} |
| BDDN | 12 ROI means | {bddn_r.get('accuracy', 'N/A')}% | {bddn_r.get('p_value', 'N/A')} | {bddn_r.get('n', 0)} |

Chance = 50%. p < 0.05 means the brain pattern reliably encodes preference.
""")
    return


@app.cell
def _(df, mo, pd, shared_pairs):
    """Summary: top findings."""
    pref = df[df["is_preferred"] == True].set_index("pair_id")
    nonpref = df[df["is_preferred"] == False].set_index("pair_id")

    # Quick check: mean attention for preferred vs non-preferred
    pref_attn = pref.loc[pref.index.isin(shared_pairs), "tribe_attention"].dropna()
    nonpref_attn = nonpref.loc[nonpref.index.isin(shared_pairs), "tribe_attention"].dropna()

    mo.md(f"""### Quick summary

**TRIBE attention (DAN-DMN):**
- Preferred images: mean = {pref_attn.mean():.4f}
- Non-preferred images: mean = {nonpref_attn.mean():.4f}
- Delta: {(pref_attn.mean() - nonpref_attn.mean()):.4f}

**Interpretation:** {"Preferred images evoke higher predicted attention (DAN-DMN)" if pref_attn.mean() > nonpref_attn.mean() else "Non-preferred images actually evoke higher predicted attention — the signal may be more complex than simple 'more activation = more preferred'"}

**Key caveats:**
1. TRIBE was trained on passive video viewing, not aesthetic judgment
2. BDDN only covers visual cortex — reward regions (vmPFC, NAcc) are not predicted
3. With 100 pairs, we have good power for medium effects (d>0.3) but may miss small ones
4. CLIP confound applies to BDDN but not TRIBE
""")
    return
