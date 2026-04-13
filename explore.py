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

    mo.md("### BrainGym: TikTok Engagement × Brain Encoding")
    return Path, json, mo, np, pd


@app.cell
def _(Path, json, mo, pd):
    CACHE = Path("./cache")

    # Load test set (9 videos) + training set (100 videos)
    _test = json.loads((CACHE / "tiktok_valid.json").read_text())
    _train_meta_path = CACHE / "tiktok_train_100_meta.json"
    _train = json.loads(_train_meta_path.read_text()) if _train_meta_path.exists() else []

    # Combine, deduplicate by id
    _all = {str(s["id"]): s for s in _test + _train}
    _samples = list(_all.values())

    df = pd.DataFrame(_samples)
    _cols = ["id", "desc", "duration", "play_count", "digg_count", "comment_count",
             "share_count", "vq_score"]
    df_display = df[_cols].copy()
    df_display["desc"] = df_display["desc"].str[:60]
    df_display["engagement_rate"] = (df["digg_count"] / df["play_count"] * 100).round(2)
    df_display["share_rate"] = (df["share_count"] / df["play_count"] * 100).round(2)

    mo.md(f"""
    ## Dataset: {len(_test)} test + {len(_train)} train = {len(_samples)} total videos
    Engagement range: **{min(s['play_count'] for s in _samples):,}** to **{max(s['play_count'] for s in _samples):,}** plays
    """)
    return CACHE, df, df_display


@app.cell
def _(df_display, mo):
    mo.ui.table(df_display)
    return


@app.cell
def _(df):
    import matplotlib.pyplot as plt_corr

    corr_cols = ["duration", "play_count", "digg_count", "comment_count",
                 "share_count", "vq_score"]
    corr_df = df[corr_cols].copy()
    corr_df["engagement_rate"] = df["digg_count"] / df["play_count"] * 100
    corr_df["share_rate"] = df["share_count"] / df["play_count"] * 100
    corr_df["comment_rate"] = df["comment_count"] / df["play_count"] * 100

    corr = corr_df.corr()

    fig_corr, ax_corr = plt_corr.subplots(figsize=(9, 7))
    im = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    labels = corr.columns.tolist()
    ax_corr.set_xticks(range(len(labels)))
    ax_corr.set_yticks(range(len(labels)))
    ax_corr.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticklabels(labels, fontsize=9)

    for _i in range(len(labels)):
        for _j in range(len(labels)):
            _val = corr.values[_i, _j]
            _color = "white" if abs(_val) > 0.6 else "black"
            ax_corr.text(_j, _i, f"{_val:.2f}", ha="center", va="center",
                         fontsize=8, color=_color)

    fig_corr.colorbar(im, ax=ax_corr, shrink=0.8, label="Pearson r")
    ax_corr.set_title("Engagement Metric Correlations", fontsize=12, fontweight="bold")
    plt_corr.tight_layout()
    plt_corr.close("all")
    return (fig_corr,)


@app.cell
def _(fig_corr, mo):
    mo.vstack([mo.md("## Correlation matrix"), fig_corr])
    return


@app.cell
def _(df):
    import matplotlib.pyplot as plt_dist

    fig_dist, axes_dist = plt_dist.subplots(1, 3, figsize=(14, 4))

    axes_dist[0].bar(range(len(df)), df["vq_score"].values, color="#4a9eed")
    axes_dist[0].set_ylabel("vq_score")
    axes_dist[0].set_title("Video Quality Score")
    axes_dist[0].set_xticks(range(len(df)))
    axes_dist[0].set_xticklabels(range(len(df)), fontsize=8)

    eng_rate = df["digg_count"] / df["play_count"] * 100
    axes_dist[1].bar(range(len(df)), eng_rate.values, color="#22c55e")
    axes_dist[1].set_ylabel("Like rate (%)")
    axes_dist[1].set_title("Engagement Rate (likes/views)")

    share_rate = df["share_count"] / df["play_count"] * 100
    axes_dist[2].bar(range(len(df)), share_rate.values, color="#f59e0b")
    axes_dist[2].set_ylabel("Share rate (%)")
    axes_dist[2].set_title("Share Rate (shares/views)")

    fig_dist.tight_layout()
    plt_dist.close("all")
    return (fig_dist,)


@app.cell
def _(fig_dist, mo):
    mo.vstack([mo.md("## Engagement distributions"), fig_dist])
    return


@app.cell
def _(CACHE, json, pd):
    # Load brain results from both test (9 videos) and train (75 videos)
    _all_brain = []
    for _path in [CACHE / "tribe_tiktok_results.json", CACHE / "tribe_tiktok_train_results.json"]:
        if _path.exists():
            _all_brain.extend(json.loads(_path.read_text()))

    if _all_brain:
        # Deduplicate by video_id
        _seen = {}
        for _r in _all_brain:
            if "error" not in _r:
                _seen[_r["video_id"]] = _r
        brain_df = pd.DataFrame([
            {
                "id": int(_r["video_id"]),
                "cognitive_load": _r["cognitive_load"],
                "attention": _r["attention"],
                "engagement_brain": _r["engagement"],
                "net_Vis": _r["net_Vis"],
                "net_DAN": _r["net_DAN"],
                "net_DMN": _r["net_DMN"],
                "net_FPN": _r["net_FPN"],
                "net_VAN": _r["net_VAN"],
                "net_SomMot": _r["net_SomMot"],
                "net_Limbic": _r["net_Limbic"],
                "n_timepoints": _r["n_timepoints"],
            }
            for _r in _seen.values()
        ])
    else:
        brain_df = None
    return (brain_df,)


@app.cell
def _(brain_df, mo):
    _n = len(brain_df) if brain_df is not None else 0
    mo.md(f"## Brain x Engagement correlation\n**{_n} videos** with brain predictions loaded." if brain_df is not None else "## Brain x Engagement\n**Waiting for Modal results...**")
    return


@app.cell
def _(brain_df, df, pd):
    if brain_df is not None:
        merged = pd.merge(df, brain_df, on="id", how="inner")
        merged["engagement_rate"] = merged["digg_count"] / merged["play_count"] * 100
        merged["share_rate"] = merged["share_count"] / merged["play_count"] * 100
        merged["comment_rate"] = merged["comment_count"] / merged["play_count"] * 100
    else:
        merged = None
    return (merged,)


@app.cell
def _(merged, mo):
    mo.md(f"**{len(merged)} videos** matched.") if merged is not None else mo.md("_No matches yet._")
    return


@app.cell
def _(merged):
    """Full correlation matrix: brain metrics + engagement metrics."""
    import matplotlib.pyplot as plt_fullcorr

    if merged is not None:
        _brain_cols = ["cognitive_load", "attention", "engagement_brain",
                       "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
        _eng_cols = ["engagement_rate", "share_rate", "comment_rate",
                     "vq_score", "duration"]
        _all_cols = _brain_cols + _eng_cols

        _corr_data = merged[_all_cols].copy()
        _corr_data.columns = [
            "cog_load", "attention", "engage_brain",
            "Vis", "DAN", "DMN", "FPN",
            "eng_rate", "share_rate", "comment_rate",
            "vq_score", "duration",
        ]
        _corr = _corr_data.corr()

        fig_fullcorr, _ax = plt_fullcorr.subplots(figsize=(12, 10))
        _im = _ax.imshow(_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

        _labels = _corr.columns.tolist()
        _ax.set_xticks(range(len(_labels)))
        _ax.set_yticks(range(len(_labels)))
        _ax.set_xticklabels(_labels, rotation=45, ha="right", fontsize=9)
        _ax.set_yticklabels(_labels, fontsize=9)

        for _i in range(len(_labels)):
            for _j in range(len(_labels)):
                _val = _corr.values[_i, _j]
                _clr = "white" if abs(_val) > 0.6 else "black"
                _ax.text(_j, _i, f"{_val:.2f}", ha="center", va="center",
                         fontsize=7, color=_clr)

        # Draw separator line between brain and engagement metrics
        _sep = len(_brain_cols) - 0.5
        _ax.axhline(_sep, color="black", linewidth=2)
        _ax.axvline(_sep, color="black", linewidth=2)

        fig_fullcorr.colorbar(_im, ax=_ax, shrink=0.8, label="Pearson r")
        _ax.set_title(f"Brain + Engagement Correlations (n={len(merged)})", fontsize=12, fontweight="bold")
        plt_fullcorr.tight_layout()
        plt_fullcorr.close("all")
    else:
        fig_fullcorr = None
    return (fig_fullcorr,)


@app.cell
def _(fig_fullcorr, mo):
    mo.vstack([mo.md("## Full correlation matrix (brain + engagement)"), fig_fullcorr]) if fig_fullcorr is not None else mo.md("")
    return


@app.cell
def _(brain_df):
    import matplotlib.pyplot as plt_brain

    if brain_df is not None:
        brain_cols = ["cognitive_load", "attention", "engagement_brain",
                      "net_Vis", "net_DAN", "net_DMN", "net_FPN", "net_VAN"]
        fig_brain, axes_brain = plt_brain.subplots(2, 4, figsize=(16, 6))
        axes_flat = axes_brain.flatten()

        for _i, _col in enumerate(brain_cols):
            axes_flat[_i].bar(range(len(brain_df)), brain_df[_col].values,
                              color="#8b5cf6" if "net_" in _col else "#4a9eed")
            axes_flat[_i].set_title(_col.replace("net_", "").replace("engagement_brain", "engagement"), fontsize=10)
            axes_flat[_i].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axes_flat[_i].set_xticks(range(len(brain_df)))
            axes_flat[_i].tick_params(labelsize=7)

        fig_brain.suptitle("Brain metrics per TikTok video", fontweight="bold")
        fig_brain.tight_layout()
        plt_brain.close("all")
    else:
        fig_brain = None
    return (fig_brain,)


@app.cell
def _(fig_brain, mo):
    mo.vstack([mo.md("### Predicted brain activations per video"), fig_brain]) if fig_brain is not None else mo.md("")
    return


@app.cell
def _(merged, np):
    import matplotlib.pyplot as plt_scatter
    from scipy import stats as sp_stats

    if merged is not None:
        _bm = ["cognitive_load", "attention", "engagement_brain",
                "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
        _em = ["engagement_rate", "share_rate", "comment_rate", "vq_score"]

        fig_scatter, axes_sc = plt_scatter.subplots(len(_em), len(_bm), figsize=(20, 12))

        for _i, _eng in enumerate(_em):
            for _j, _brain in enumerate(_bm):
                _ax = axes_sc[_i][_j]
                _x = merged[_brain].values
                _y = merged[_eng].values

                _ax.scatter(_x, _y, s=40, alpha=0.7, c="#4a9eed", edgecolors="white", linewidth=0.5)

                if len(_x) > 2:
                    _r, _p = sp_stats.pearsonr(_x, _y)
                    _z = np.polyfit(_x, _y, 1)
                    _xline = np.linspace(_x.min(), _x.max(), 50)
                    _ax.plot(_xline, np.polyval(_z, _xline), "r-", alpha=0.5, linewidth=1.5)
                    _title_color = "#22c55e" if _p < 0.1 else "#888888"
                    _ax.set_title(f"r={_r:.2f} p={_p:.2f}", fontsize=8, color=_title_color)

                if _i == len(_em) - 1:
                    _ax.set_xlabel(_brain.replace("net_", "").replace("engagement_brain", "engage"), fontsize=8)
                if _j == 0:
                    _ax.set_ylabel(_eng, fontsize=8)
                _ax.tick_params(labelsize=6)

        fig_scatter.suptitle(
            "Brain predictions x Engagement metrics\n(green title = p < 0.1)",
            fontweight="bold", fontsize=12
        )
        fig_scatter.tight_layout()
        plt_scatter.close("all")
    else:
        fig_scatter = None
    return (fig_scatter,)


@app.cell
def _(fig_scatter, mo):
    mo.vstack([mo.md("### Correlation scatterplots (rates)"), fig_scatter]) if fig_scatter is not None else mo.md("")
    return


@app.cell
def _(merged, np):
    import matplotlib.pyplot as plt_scatter2
    from scipy import stats as sp_stats_raw

    if merged is not None:
        _bm_raw = ["cognitive_load", "attention", "engagement_brain",
                    "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
        _em_raw = ["play_count", "digg_count", "comment_count", "share_count", "vq_score", "duration"]

        fig_scatter2, _axes_raw = plt_scatter2.subplots(len(_em_raw), len(_bm_raw), figsize=(20, 16))

        for _i, _eng in enumerate(_em_raw):
            for _j, _brain in enumerate(_bm_raw):
                _ax = _axes_raw[_i][_j]
                _x = merged[_brain].values
                _y = merged[_eng].values

                _ax.scatter(_x, _y, s=40, alpha=0.7, c="#f59e0b", edgecolors="white", linewidth=0.5)

                if len(_x) > 2:
                    _r, _p = sp_stats_raw.pearsonr(_x, _y)
                    _z = np.polyfit(_x, _y, 1)
                    _xline = np.linspace(_x.min(), _x.max(), 50)
                    _ax.plot(_xline, np.polyval(_z, _xline), "r-", alpha=0.5, linewidth=1.5)
                    _title_color = "#22c55e" if _p < 0.1 else "#888888"
                    _ax.set_title(f"r={_r:.2f} p={_p:.2f}", fontsize=8, color=_title_color)

                if _i == len(_em_raw) - 1:
                    _ax.set_xlabel(_brain.replace("net_", "").replace("engagement_brain", "engage"), fontsize=8)
                if _j == 0:
                    _ax.set_ylabel(_eng, fontsize=8)
                _ax.tick_params(labelsize=6)

        fig_scatter2.suptitle(
            "Brain predictions x Raw TikTok metrics\n(green title = p < 0.1)",
            fontweight="bold", fontsize=12
        )
        fig_scatter2.tight_layout()
        plt_scatter2.close("all")
    else:
        fig_scatter2 = None
    return (fig_scatter2,)


@app.cell
def _(fig_scatter2, mo):
    mo.vstack([mo.md("### Correlation scatterplots (raw counts)"), fig_scatter2]) if fig_scatter2 is not None else mo.md("")
    return


@app.cell
def _(merged, mo, pd):
    from scipy import stats as sp_stats2

    if merged is not None:
        _bm2 = ["cognitive_load", "attention", "engagement_brain",
                 "net_Vis", "net_DAN", "net_DMN", "net_FPN", "net_VAN", "net_Limbic"]
        _em2 = ["engagement_rate", "share_rate", "comment_rate", "vq_score",
                 "play_count", "digg_count", "share_count"]

        _rows = []
        for _eng in _em2:
            _row = {"metric": _eng}
            for _brain in _bm2:
                _x = merged[_brain].values
                _y = merged[_eng].values
                if len(_x) > 2:
                    _r, _p = sp_stats2.pearsonr(_x, _y)
                    _stars = ""
                    if _p < 0.05:
                        _stars = "**"
                    elif _p < 0.1:
                        _stars = "*"
                    _row[_brain.replace("net_", "").replace("engagement_brain", "engage")] = f"{_r:+.2f}{_stars}"
                else:
                    _row[_brain.replace("net_", "").replace("engagement_brain", "engage")] = "—"
            _rows.append(_row)

        _corr_table = pd.DataFrame(_rows)
        _out = mo.vstack([
            mo.md("### Correlation table (Pearson r)\n`*` = p < 0.1, `**` = p < 0.05"),
            mo.ui.table(_corr_table),
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Correlation hypotheses

    | Brain metric | Engagement metric | Hypothesis |
    |---|---|---|
    | Visual cortex (Vis) | play_count | Higher visual activation -> more views |
    | Face areas (FFA) | digg_count | Face-heavy content -> more likes |
    | DMN suppression | vq_score | More engaging -> lower DMN -> higher quality score |
    | Cognitive load (FPN+DAN-DMN) | share_count | Optimal load -> more shares |
    | Attention (DAN) | comment_count | High attention -> more comments |
    | Scene areas (PPA/OPA) | share_rate | Scenic/travel content -> higher share rate |

    **Key question:** Is there a "sweet spot" of cognitive load that maximizes engagement,
    or is it monotonic (more stimulating = more engagement)?
    """)
    return


@app.cell
def _(CACHE, json, np):
    import matplotlib.pyplot as plt_ts

    # Load from both test and train
    _all_ts = []
    for _rp in [CACHE / "tribe_tiktok_results.json", CACHE / "tribe_tiktok_train_results.json"]:
        if _rp.exists():
            _all_ts.extend(json.loads(_rp.read_text()))
    _seen_ts = {}
    for _r in _all_ts:
        if "error" not in _r:
            _seen_ts[_r["video_id"]] = _r
    _results = list(_seen_ts.values())
    if _results:
        # Only plot first 12 for readability
        _results = _results[:12]

        _lines = {
            "Attention (DAN-DMN)": {"color": "#ef4444", "lw": 2.5},
            "Cog Load (FPN+DAN-DMN)": {"color": "#f59e0b", "lw": 2.5},
            "DAN": {"color": "#ef4444", "lw": 1, "alpha": 0.4, "ls": "--"},
            "DMN": {"color": "#3b82f6", "lw": 1, "alpha": 0.4, "ls": "--"},
            "Vis": {"color": "#22c55e", "lw": 1.2},
            "FPN": {"color": "#f59e0b", "lw": 1, "alpha": 0.4, "ls": "--"},
        }
        _n = len(_results)
        _cols = 3
        _rows_grid = (_n + _cols - 1) // _cols

        fig_ts, _axes_ts = plt_ts.subplots(_rows_grid, _cols, figsize=(18, 4.5 * _rows_grid))
        _axes_ts = _axes_ts.flatten() if _n > _cols else [_axes_ts] if _n == 1 else _axes_ts.flatten()

        for _idx, _r in enumerate(_results):
            _ax = _axes_ts[_idx]
            _ts = _r["timeseries"]
            _vid = _r["video_id"]

            # Compute derived metrics per timepoint
            _dan = np.array(_ts["DAN"])
            _dmn = np.array(_ts["DMN"])
            _fpn = np.array(_ts["FPN"])
            _vis = np.array(_ts["Vis"])
            _attention = _dan - _dmn
            _cog_load = _fpn + _dan - _dmn
            _t = np.arange(len(_dan))

            # Plot derived (bold)
            _ax.plot(_t, _attention, "-", label="Attention (DAN-DMN)", color="#ef4444", linewidth=2.5)
            _ax.plot(_t, _cog_load, "-", label="Cog Load (FPN+DAN-DMN)", color="#f59e0b", linewidth=2.5)
            _ax.plot(_t, _vis, "-", label="Vis", color="#22c55e", linewidth=1.2)

            # Plot raw networks (faded)
            _ax.plot(_t, _dan, "--", label="DAN", color="#ef4444", linewidth=1, alpha=0.4)
            _ax.plot(_t, _dmn, "--", label="DMN", color="#3b82f6", linewidth=1, alpha=0.4)
            _ax.plot(_t, _fpn, "--", label="FPN", color="#f59e0b", linewidth=1, alpha=0.4)

            _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            _ax.set_title(f"Video ...{_vid[-6:]}", fontsize=9)
            _ax.set_xlabel("Time (s)", fontsize=8)
            _ax.set_ylabel("Activation", fontsize=8)
            _ax.tick_params(labelsize=7)
            if _idx == 0:
                _ax.legend(fontsize=6, loc="upper right", ncol=2)

        # Hide unused subplots
        for _idx in range(_n, len(_axes_ts)):
            _axes_ts[_idx].set_visible(False)

        fig_ts.suptitle("Brain network activations over time per TikTok video", fontweight="bold")
        fig_ts.tight_layout()
        plt_ts.close("all")
    else:
        fig_ts = None
    return (fig_ts,)


@app.cell
def _(fig_ts, mo):
    mo.vstack([mo.md("### Per-video brain timeseries"), fig_ts]) if fig_ts is not None else mo.md("")
    return


@app.cell
def _(CACHE, json, np, pd):
    """Compute functional connectivity + temporal features per video."""
    from scipy.stats import pearsonr as _pearsonr

    # Load from both test and train result files
    _all_results = []
    for _rp in [CACHE / "tribe_tiktok_results.json", CACHE / "tribe_tiktok_train_results.json"]:
        if _rp.exists():
            _all_results.extend(json.loads(_rp.read_text()))
    # Deduplicate
    _seen_fc = {}
    for _r in _all_results:
        if "error" not in _r:
            _seen_fc[_r["video_id"]] = _r
    _results = list(_seen_fc.values())
    if _results:

        _fc_rows = []
        for _r in _results:
            _ts = _r["timeseries"]
            _dan = np.array(_ts["DAN"])
            _dmn = np.array(_ts["DMN"])
            _fpn = np.array(_ts["FPN"])
            _vis = np.array(_ts["Vis"])
            _van = np.array(_ts["VAN"])
            _attn = _dan - _dmn
            _n_tp = len(_dan)

            _row = {"id": int(_r["video_id"])}

            # Functional connectivity: linear (Pearson) + non-linear (Spearman, MI)
            from scipy.stats import spearmanr as _spearmanr
            from sklearn.metrics import mutual_info_score as _mi_score

            def _discretize(_arr, _bins=10):
                return np.digitize(_arr, np.linspace(_arr.min(), _arr.max(), _bins))

            if _n_tp > 3:
                # Pearson (linear)
                _row["fc_DAN_DMN"], _ = _pearsonr(_dan, _dmn)
                _row["fc_DAN_VIS"], _ = _pearsonr(_dan, _vis)
                _row["fc_FPN_DMN"], _ = _pearsonr(_fpn, _dmn)
                _row["fc_VAN_DMN"], _ = _pearsonr(_van, _dmn)
                _row["fc_FPN_DAN"], _ = _pearsonr(_fpn, _dan)

                # Spearman (monotonic, rank-based)
                _row["sp_DAN_DMN"], _ = _spearmanr(_dan, _dmn)
                _row["sp_DAN_VIS"], _ = _spearmanr(_dan, _vis)

                # Mutual Information (any dependency)
                _row["mi_DAN_DMN"] = float(_mi_score(_discretize(_dan), _discretize(_dmn)))
                _row["mi_DAN_VIS"] = float(_mi_score(_discretize(_dan), _discretize(_vis)))
            else:
                _row["fc_DAN_DMN"] = _row["fc_DAN_VIS"] = _row["fc_FPN_DMN"] = 0
                _row["fc_VAN_DMN"] = _row["fc_FPN_DAN"] = 0
                _row["sp_DAN_DMN"] = _row["sp_DAN_VIS"] = 0
                _row["mi_DAN_DMN"] = _row["mi_DAN_VIS"] = 0

            # Temporal features
            _row["attn_peak"] = float(np.max(_attn))
            _row["attn_mean"] = float(np.mean(_attn))
            _row["attn_slope"] = float(np.polyfit(np.arange(_n_tp), _attn, 1)[0]) if _n_tp > 1 else 0
            _row["attn_first3s"] = float(np.mean(_attn[:min(3, _n_tp)]))
            _row["attn_var"] = float(np.var(_attn))
            _row["vis_peak"] = float(np.max(_vis))
            _row["vis_mean"] = float(np.mean(_vis))
            _row["dan_peak"] = float(np.max(_dan))
            _row["dmn_min"] = float(np.min(_dmn))  # most suppressed

            _fc_rows.append(_row)

        fc_df = pd.DataFrame(_fc_rows)
    else:
        fc_df = None
    return (fc_df,)


@app.cell
def _(df, fc_df, mo, np, pd):
    """Correlate FC + temporal features with TikTok engagement."""
    from scipy.stats import pearsonr as _pearsonr2

    if fc_df is not None:
        _merged_fc = pd.merge(df, fc_df, on="id", how="inner")
        _merged_fc["engagement_rate"] = _merged_fc["digg_count"] / _merged_fc["play_count"] * 100
        _merged_fc["share_rate"] = _merged_fc["share_count"] / _merged_fc["play_count"] * 100
        _merged_fc["comment_rate"] = _merged_fc["comment_count"] / _merged_fc["play_count"] * 100

        _brain_feats = [
            "fc_DAN_DMN", "sp_DAN_DMN", "mi_DAN_DMN",
            "fc_DAN_VIS", "sp_DAN_VIS", "mi_DAN_VIS",
            "fc_FPN_DMN", "fc_VAN_DMN", "fc_FPN_DAN",
            "attn_peak", "attn_mean", "attn_slope", "attn_first3s", "attn_var",
            "vis_peak", "vis_mean", "dan_peak", "dmn_min",
        ]
        _eng_feats = ["engagement_rate", "share_rate", "comment_rate", "vq_score"]

        _rows = []
        for _eng in _eng_feats:
            _row = {"metric": _eng}
            for _brain in _brain_feats:
                _x = _merged_fc[_brain].values
                _y = _merged_fc[_eng].values
                if len(_x) > 2 and np.std(_x) > 1e-10:
                    _r, _p = _pearsonr2(_x, _y)
                    _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                    _row[_brain] = f"{_r:+.2f}{_stars}"
                else:
                    _row[_brain] = "—"
            _rows.append(_row)

        _fc_table = pd.DataFrame(_rows)
        _out = mo.vstack([
            mo.md("""### Functional connectivity + temporal features
    **FC (Pearson):** Linear correlation between network timeseries. `fc_DAN_DMN` = -1 means perfectly anticorrelated.

    **SP (Spearman):** Rank-based monotonic correlation. Catches non-linear monotonic coupling.

    **MI (Mutual Information):** Any statistical dependency — catches U-shaped, threshold, and non-monotonic relationships.

    **Temporal features:** Peak, mean, slope, first 3s (hook strength), variance (stability).
            """),
            mo.ui.table(_fc_table),
        ])
    else:
        _out = mo.md("_Waiting for brain predictions..._")
    _out
    return


@app.cell
def _(df, fc_df, np, pd):
    """Scatter plots for top FC predictors."""
    import matplotlib.pyplot as plt_fc
    from scipy.stats import pearsonr as _pearsonr3

    if fc_df is not None:
        _merged_fc2 = pd.merge(df, fc_df, on="id", how="inner")
        _merged_fc2["share_rate"] = _merged_fc2["share_count"] / _merged_fc2["play_count"] * 100
        _merged_fc2["engagement_rate"] = _merged_fc2["digg_count"] / _merged_fc2["play_count"] * 100
        _merged_fc2["comment_rate"] = _merged_fc2["comment_count"] / _merged_fc2["play_count"] * 100

        _top_feats = ["fc_DAN_DMN", "mi_DAN_DMN", "attn_peak", "attn_first3s", "attn_slope", "attn_var"]
        _targets = ["share_rate", "engagement_rate", "comment_rate", "vq_score"]

        fig_fc, _axes_fc = plt_fc.subplots(len(_targets), len(_top_feats), figsize=(20, 12))

        for _i, _tgt in enumerate(_targets):
            for _j, _feat in enumerate(_top_feats):
                _ax = _axes_fc[_i][_j]
                _x = _merged_fc2[_feat].values
                _y = _merged_fc2[_tgt].values

                _ax.scatter(_x, _y, s=50, alpha=0.7, c="#4a9eed", edgecolors="white")

                if len(_x) > 2 and np.std(_x) > 1e-10:
                    _r, _p = _pearsonr3(_x, _y)
                    _z = np.polyfit(_x, _y, 1)
                    _xl = np.linspace(_x.min(), _x.max(), 50)
                    _ax.plot(_xl, np.polyval(_z, _xl), "r-", alpha=0.6, linewidth=2)
                    _tc = "#22c55e" if _p < 0.1 else "#888888"
                    _ax.set_title(f"r={_r:.2f} p={_p:.2f}", fontsize=9, color=_tc, fontweight="bold")

                if _i == len(_targets) - 1:
                    _ax.set_xlabel(_feat, fontsize=8)
                if _j == 0:
                    _ax.set_ylabel(_tgt, fontsize=8)
                _ax.tick_params(labelsize=7)

        fig_fc.suptitle("Functional Connectivity + Temporal Features vs Engagement\n(green = p < 0.1)", fontweight="bold")
        fig_fc.tight_layout()
        plt_fc.close("all")
    else:
        fig_fc = None
    return (fig_fc,)


@app.cell
def _(fig_fc, mo):
    mo.vstack([mo.md("### FC + temporal feature scatterplots"), fig_fc]) if fig_fc is not None else mo.md("")
    return


@app.cell
def _(df, fc_df, mo, np, pd):
    """Ranked predictor table: all brain features vs all engagement metrics, sorted by |r|."""
    from scipy.stats import pearsonr as _pearsonr4

    if fc_df is not None:
        _merged_all = pd.merge(df, fc_df, on="id", how="inner")
        _merged_all["engagement_rate"] = _merged_all["digg_count"] / _merged_all["play_count"] * 100
        _merged_all["share_rate"] = _merged_all["share_count"] / _merged_all["play_count"] * 100
        _merged_all["comment_rate"] = _merged_all["comment_count"] / _merged_all["play_count"] * 100

        _brain_feats = [
            "fc_DAN_DMN", "fc_DAN_VIS", "fc_FPN_DMN", "fc_VAN_DMN", "fc_FPN_DAN",
            "attn_peak", "attn_mean", "attn_slope", "attn_first3s", "attn_var",
            "vis_peak", "vis_mean", "dan_peak", "dmn_min",
        ]
        _eng_feats = ["share_rate", "engagement_rate", "comment_rate", "vq_score"]

        _rows = []
        for _eng in _eng_feats:
            for _brain in _brain_feats:
                _x = _merged_all[_brain].values
                _y = _merged_all[_eng].values
                if len(_x) > 2 and np.std(_x) > 1e-10:
                    _r, _p = _pearsonr4(_x, _y)
                    _sig = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                    _rows.append({
                        "engagement_metric": _eng,
                        "brain_feature": _brain,
                        "r": round(_r, 3),
                        "p": round(_p, 4),
                        "abs_r": round(abs(_r), 3),
                        "sig": _sig,
                    })

        _ranked = pd.DataFrame(_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
        _ranked.index = _ranked.index + 1
        _ranked.index.name = "rank"

        _out = mo.vstack([
            mo.md("### All brain-engagement correlations ranked by |r|\n`**` = p < 0.05, `*` = p < 0.1"),
            mo.ui.table(_ranked),
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(df, fc_df, np, pd):
    """Hero plot: attn_first3s vs share_rate."""
    import matplotlib.pyplot as plt_hero
    from scipy.stats import pearsonr as _pearsonr_hero

    if fc_df is not None:
        _m = pd.merge(df, fc_df, on="id", how="inner")
        _m["share_rate"] = _m["share_count"] / _m["play_count"] * 100

        _x = _m["attn_first3s"].values
        _y = _m["share_rate"].values
        _r, _p = _pearsonr_hero(_x, _y)

        fig_hero, _ax = plt_hero.subplots(figsize=(8, 6))

        _ax.scatter(_x, _y, s=60, alpha=0.6, c="#4a9eed", edgecolors="white", linewidth=0.8, zorder=3)

        # Regression line
        _z = np.polyfit(_x, _y, 1)
        _xl = np.linspace(_x.min(), _x.max(), 100)
        _ax.plot(_xl, np.polyval(_z, _xl), "-", color="#ef4444", linewidth=2.5, alpha=0.8, zorder=2)

        # Confidence band
        _y_pred = np.polyval(_z, _x)
        _resid_std = np.std(_y - _y_pred)
        _ax.fill_between(_xl, np.polyval(_z, _xl) - _resid_std, np.polyval(_z, _xl) + _resid_std,
                         color="#ef4444", alpha=0.1, zorder=1)

        _ax.set_xlabel("Predicted brain attention (first 3 seconds)", fontsize=13, fontweight="bold")
        _ax.set_ylabel("TikTok share rate (%)", fontsize=13, fontweight="bold")
        _ax.set_title(
            f"Early attention predicts sharing (n={len(_x)}, r={_r:.2f}, p={_p:.3f})",
            fontsize=14, fontweight="bold", pad=15
        )

        # Annotation
        _ax.text(0.97, 0.95,
                 f"Pearson r = {_r:.3f}\np = {_p:.4f}",
                 transform=_ax.transAxes, fontsize=11,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))

        _ax.text(0.97, 0.05,
                 "Brain response predicted by TRIBEv2 (Meta)\nfMRI encoding model on TikTok videos",
                 transform=_ax.transAxes, fontsize=8, color="#888888",
                 verticalalignment="bottom", horizontalalignment="right")

        _ax.grid(alpha=0.2)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        fig_hero.tight_layout()
        fig_hero.savefig("cache/attn_first3s_vs_share_rate.png", dpi=200, bbox_inches="tight")
        plt_hero.close("all")
    else:
        fig_hero = None
    return (fig_hero,)


@app.cell
def _(fig_hero, mo):
    mo.vstack([mo.md("### Early attention predicts TikTok sharing"), fig_hero]) if fig_hero is not None else mo.md("")
    return


@app.cell
def _(brain_df, df, pd):
    """Classify TikTok videos into brain types: high-stim vs high-processing."""

    if brain_df is not None:
        _merged = pd.merge(df, brain_df, on="id", how="inner")
        _merged["engagement_rate"] = _merged["digg_count"] / _merged["play_count"] * 100
        _merged["share_rate"] = _merged["share_count"] / _merged["play_count"] * 100

        # Scores
        _merged["stim_score"] = _merged["net_Vis"] + _merged["attention"]
        _merged["process_score"] = _merged["net_FPN"] + (-_merged["net_DMN"])

        _stim_med = _merged["stim_score"].median()
        _proc_med = _merged["process_score"].median()

        def _classify(_row):
            _hi_s = _row["stim_score"] > _stim_med
            _hi_p = _row["process_score"] > _proc_med
            if _hi_s and not _hi_p: return "high-stim"
            if _hi_p and not _hi_s: return "high-processing"
            if _hi_s and _hi_p: return "both"
            return "low-both"

        _merged["brain_type"] = _merged.apply(_classify, axis=1)

        # Build URL column
        _merged["link"] = _merged["id"].apply(lambda _x: f"https://tiktok.com/@/video/{_x}")

        classified_df = _merged
    else:
        classified_df = None
    return (classified_df,)


@app.cell
def _(classified_df, mo):
    """Summary stats per brain type."""
    if classified_df is not None:
        _summary = classified_df.groupby("brain_type").agg(
            n=("id", "count"),
            mean_attention=("attention", "mean"),
            mean_Vis=("net_Vis", "mean"),
            mean_FPN=("net_FPN", "mean"),
            mean_DMN=("net_DMN", "mean"),
            mean_eng_rate=("engagement_rate", "mean"),
            mean_share_rate=("share_rate", "mean"),
        ).round(3)

        _out = mo.vstack([
            mo.md("## Brain-type classification\n"
                   "**high-stim:** high Vis + attention, low FPN. Visually grabbing.\n"
                   "**high-processing:** high FPN + DMN suppression. Cognitively engaging.\n"
                   "**both:** high on everything.\n"
                   "**low-both:** minimal brain activation. Simple/relatable."),
            mo.ui.table(_summary.reset_index()),
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(classified_df):
    """Scatter plot: stim_score vs process_score colored by type."""
    import matplotlib.pyplot as plt_classify

    if classified_df is not None:
        fig_classify, _ax = plt_classify.subplots(figsize=(10, 8))

        _type_colors = {
            "high-stim": "#ef4444",
            "high-processing": "#3b82f6",
            "both": "#8b5cf6",
            "low-both": "#22c55e",
        }

        for _type, _color in _type_colors.items():
            _sub = classified_df[classified_df["brain_type"] == _type]
            _ax.scatter(_sub["stim_score"], _sub["process_score"],
                        s=60, alpha=0.7, c=_color, label=_type, edgecolors="white", linewidth=0.5)

        _ax.axhline(classified_df["process_score"].median(), color="gray", linewidth=0.5, linestyle="--")
        _ax.axvline(classified_df["stim_score"].median(), color="gray", linewidth=0.5, linestyle="--")
        _ax.set_xlabel("Stimulation score (Vis + Attention)", fontsize=11)
        _ax.set_ylabel("Processing score (FPN + (-DMN))", fontsize=11)
        _ax.set_title("TikTok videos classified by brain response type", fontsize=12, fontweight="bold")
        _ax.legend(fontsize=10)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        fig_classify.tight_layout()
        plt_classify.close("all")
    else:
        fig_classify = None
    return (fig_classify,)


@app.cell
def _(fig_classify, mo):
    mo.vstack([mo.md("### Brain-type quadrant plot"), fig_classify]) if fig_classify is not None else mo.md("")
    return


@app.cell
def _(classified_df, mo):
    """Per-type video examples with clickable links."""
    if classified_df is not None:
        _tables = []
        for _type in ["high-stim", "high-processing", "both", "low-both"]:
            _sub = classified_df[classified_df["brain_type"] == _type].sort_values("play_count", ascending=False)
            _display = _sub[["link", "desc", "attention", "net_Vis", "net_FPN", "net_DMN",
                             "engagement_rate", "share_rate"]].head(5).copy()
            _display["desc"] = _display["desc"].str[:60]
            _display = _display.round(3)
            _tables.append(mo.md(f"#### {_type} (n={len(_sub)})"))
            _tables.append(mo.ui.table(_display))

        _out = mo.vstack(_tables)
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(classified_df, mo, np, pd):
    """Does the brain-engagement relationship differ by brain type?"""
    from scipy.stats import pearsonr as _pearsonr_type

    if classified_df is not None:
        _types = ["high-stim", "high-processing", "both", "low-both"]
        _brain_cols = ["attention", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
        _eng_cols = ["engagement_rate", "share_rate"]

        _rows = []
        for _type in _types:
            _sub = classified_df[classified_df["brain_type"] == _type]
            _n = len(_sub)
            for _ec in _eng_cols:
                _row = {"brain_type": _type, "n": _n, "target": _ec}
                for _bc in _brain_cols:
                    if _n >= 5 and np.std(_sub[_bc]) > 1e-10:
                        _r, _p = _pearsonr_type(_sub[_bc].values, _sub[_ec].values)
                        _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                        _row[_bc.replace("net_", "")] = f"{_r:+.2f}{_stars}"
                    else:
                        _row[_bc.replace("net_", "")] = "—"
                _rows.append(_row)

        _type_corr = pd.DataFrame(_rows)
        _out = mo.vstack([
            mo.md("### Does brain-type change the brain-engagement relationship?\n"
                   "Same brain metric can predict engagement **positively** in one type and **negatively** in another."),
            mo.ui.table(_type_corr),
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(classified_df, np):
    """Scatter: attention vs engagement_rate, colored by brain type."""
    import matplotlib.pyplot as plt_typeeng
    from scipy.stats import pearsonr as _pearsonr_te

    if classified_df is not None:
        _type_colors = {
            "high-stim": "#ef4444",
            "high-processing": "#3b82f6",
            "both": "#8b5cf6",
            "low-both": "#22c55e",
        }

        fig_typeeng, _axes = plt_typeeng.subplots(1, 2, figsize=(16, 6))

        # Left: attention vs engagement_rate by type
        for _type, _color in _type_colors.items():
            _sub = classified_df[classified_df["brain_type"] == _type]
            _axes[0].scatter(_sub["attention"], _sub["engagement_rate"],
                             s=50, alpha=0.7, c=_color, label=_type, edgecolors="white", linewidth=0.5)
            # Per-type regression line
            if len(_sub) >= 5:
                _z = np.polyfit(_sub["attention"], _sub["engagement_rate"], 1)
                _xl = np.linspace(_sub["attention"].min(), _sub["attention"].max(), 50)
                _axes[0].plot(_xl, np.polyval(_z, _xl), "-", color=_color, alpha=0.5, linewidth=1.5)

        _axes[0].set_xlabel("Attention (DAN-DMN)", fontsize=11)
        _axes[0].set_ylabel("Engagement rate (%)", fontsize=11)
        _axes[0].set_title("Attention vs Engagement by brain type", fontsize=11, fontweight="bold")
        _axes[0].legend(fontsize=8)

        # Right: DMN vs share_rate by type
        for _type, _color in _type_colors.items():
            _sub = classified_df[classified_df["brain_type"] == _type]
            _axes[1].scatter(_sub["net_DMN"], _sub["share_rate"],
                             s=50, alpha=0.7, c=_color, label=_type, edgecolors="white", linewidth=0.5)
            if len(_sub) >= 5:
                _z = np.polyfit(_sub["net_DMN"], _sub["share_rate"], 1)
                _xl = np.linspace(_sub["net_DMN"].min(), _sub["net_DMN"].max(), 50)
                _axes[1].plot(_xl, np.polyval(_z, _xl), "-", color=_color, alpha=0.5, linewidth=1.5)

        _axes[1].set_xlabel("DMN activation", fontsize=11)
        _axes[1].set_ylabel("Share rate (%)", fontsize=11)
        _axes[1].set_title("DMN vs Share rate by brain type", fontsize=11, fontweight="bold")

        fig_typeeng.suptitle("Brain-engagement relationship differs by content type", fontweight="bold")
        fig_typeeng.tight_layout()
        plt_typeeng.close("all")
    else:
        fig_typeeng = None
    return (fig_typeeng,)


@app.cell
def _(fig_typeeng, mo):
    mo.vstack([mo.md("### Brain-type moderated scatterplots"), fig_typeeng]) if fig_typeeng is not None else mo.md("")
    return


@app.cell
def _(classified_df, mo, np):
    """Engagement comparison across brain types with significance test."""
    from scipy.stats import kruskal as _kruskal

    if classified_df is not None:
        _types = ["high-stim", "high-processing", "both", "low-both"]

        import matplotlib.pyplot as plt_typebar
        fig_typebar, _axes = plt_typebar.subplots(1, 2, figsize=(14, 5))

        for _i, _metric in enumerate(["engagement_rate", "share_rate"]):
            _groups = [classified_df[classified_df["brain_type"] == _t][_metric].values for _t in _types]
            _means = [_g.mean() for _g in _groups]
            _sems = [_g.std() / max(np.sqrt(len(_g)), 1) for _g in _groups]

            _colors = ["#ef4444", "#3b82f6", "#8b5cf6", "#22c55e"]
            _axes[_i].bar(range(4), _means, yerr=_sems, color=_colors, alpha=0.7, capsize=4, edgecolor="white")
            _axes[_i].set_xticks(range(4))
            _axes[_i].set_xticklabels(_types, fontsize=9, rotation=15)
            _axes[_i].set_ylabel(_metric, fontsize=10)

            # Kruskal-Wallis test
            if all(len(_g) >= 3 for _g in _groups):
                _h, _p = _kruskal(*_groups)
                _axes[_i].set_title(f"{_metric}\nKruskal-Wallis p={_p:.3f}{'**' if _p < 0.05 else '*' if _p < 0.1 else ''}", fontsize=11, fontweight="bold")
            else:
                _axes[_i].set_title(_metric, fontsize=11, fontweight="bold")

        fig_typebar.suptitle("Do brain types perform differently?", fontweight="bold")
        fig_typebar.tight_layout()
        plt_typebar.close("all")
        _out = mo.vstack([mo.md("### Engagement by brain type"), fig_typebar])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(df, fc_df, np, pd):
    """Duration-controlled analysis: the real signal."""
    import matplotlib.pyplot as plt_durctrl
    from scipy.stats import pearsonr as _pearsonr_dur
    from numpy.linalg import lstsq as _lstsq

    if fc_df is not None:
        _m = pd.merge(df, fc_df, on="id", how="inner")
        _m["share_rate"] = _m["share_count"] / _m["play_count"] * 100
        _m["log_plays"] = np.log10(_m["play_count"].clip(1))

        def _partial(_x, _y, _covs):
            _C = np.column_stack(_covs)
            _xr = _x - _C @ _lstsq(_C, _x, rcond=None)[0]
            _yr = _y - _C @ _lstsq(_C, _y, rcond=None)[0]
            return _pearsonr_dur(_xr, _yr)

        # Brain features to test
        _feats = ["attn_mean", "attn_first3s", "attn_peak", "dan_peak",
                  "vis_mean", "vis_peak", "fc_DAN_DMN"]
        _dur = _m["duration"].values

        fig_durctrl, _axes = plt_durctrl.subplots(2, 4, figsize=(20, 9))
        _axes = _axes.flatten()

        for _i, _feat in enumerate(_feats):
            _ax = _axes[_i]
            _x = _m[_feat].values
            _y = _m["share_rate"].values
            _d = _m["duration"].values

            # Color by duration
            _scatter = _ax.scatter(_x, _y, s=50, alpha=0.7, c=_d, cmap="coolwarm",
                                   edgecolors="white", linewidth=0.5)

            # Raw correlation
            _r_raw, _p_raw = _pearsonr_dur(_x, _y)

            # Duration-controlled
            _r_ctrl, _p_ctrl = _partial(_x, _y, [_dur])

            # Regression line (raw)
            if np.std(_x) > 1e-10:
                _z = np.polyfit(_x, _y, 1)
                _xl = np.linspace(_x.min(), _x.max(), 50)
                _ax.plot(_xl, np.polyval(_z, _xl), "r-", alpha=0.5, linewidth=2)

            _ax.set_xlabel(_feat, fontsize=9)
            if _i % 4 == 0:
                _ax.set_ylabel("Share rate (%)", fontsize=9)
            _ax.set_title(f"raw r={_r_raw:+.2f} p={_p_raw:.3f}\nctrl|dur r={_r_ctrl:+.2f} p={_p_ctrl:.3f}",
                          fontsize=9, color="#22c55e" if _p_ctrl < 0.05 else "#f59e0b" if _p_ctrl < 0.1 else "#888")
            _ax.tick_params(labelsize=7)

        # Hide last subplot, add colorbar there
        _axes[7].set_visible(False)
        _cb = fig_durctrl.colorbar(_scatter, ax=_axes[7], fraction=0.5)
        _cb.set_label("Duration (s)", fontsize=10)
        _axes[7].set_visible(True)
        _axes[7].axis("off")

        fig_durctrl.suptitle(
            "Brain features vs Share rate (colored by duration)\n"
            "Top line = raw, bottom line = controlling for duration",
            fontweight="bold", fontsize=12
        )
        fig_durctrl.tight_layout()
        plt_durctrl.close("all")
    else:
        fig_durctrl = None
    return (fig_durctrl,)


@app.cell
def _(fig_durctrl, mo):
    mo.vstack([
        mo.md("### Duration-controlled: the real signal\n"
              "Dots colored by duration (blue=short, red=long). "
              "Green title = significant after controlling for duration."),
        fig_durctrl,
    ]) if fig_durctrl is not None else mo.md("")
    return


@app.cell
def _(df, fc_df, np, pd):
    """DAN network correlations with all engagement metrics, duration-controlled."""
    import matplotlib.pyplot as plt_dan
    from scipy.stats import pearsonr as _pearsonr_dan
    from numpy.linalg import lstsq as _lstsq_dan

    if fc_df is not None:
        _m = pd.merge(df, fc_df, on="id", how="inner")
        _m["share_rate"] = _m["share_count"] / _m["play_count"] * 100
        _m["engagement_rate"] = _m["digg_count"] / _m["play_count"] * 100
        _m["comment_rate"] = _m["comment_count"] / _m["play_count"] * 100

        def _partial2(_x, _y, _covs):
            _C = np.column_stack(_covs)
            _xr = _x - _C @ _lstsq_dan(_C, _x, rcond=None)[0]
            _yr = _y - _C @ _lstsq_dan(_C, _y, rcond=None)[0]
            return _pearsonr_dan(_xr, _yr)

        _dan_feats = ["dan_peak", "attn_mean", "attn_peak", "fc_DAN_DMN"]
        _eng_targets = ["share_rate", "engagement_rate", "comment_rate", "vq_score"]
        _dur = _m["duration"].values

        fig_dan, _axes = plt_dan.subplots(len(_eng_targets), len(_dan_feats), figsize=(18, 14))

        for _i, _eng in enumerate(_eng_targets):
            for _j, _feat in enumerate(_dan_feats):
                _ax = _axes[_i][_j]
                _x = _m[_feat].values
                _y = _m[_eng].values

                _ax.scatter(_x, _y, s=40, alpha=0.6, c=_dur, cmap="coolwarm",
                            edgecolors="white", linewidth=0.5)

                _r_raw, _p_raw = _pearsonr_dan(_x, _y)
                _r_ctrl, _p_ctrl = _partial2(_x, _y, [_dur])

                if np.std(_x) > 1e-10:
                    _z = np.polyfit(_x, _y, 1)
                    _xl = np.linspace(_x.min(), _x.max(), 50)
                    _ax.plot(_xl, np.polyval(_z, _xl), "r-", alpha=0.5, linewidth=1.5)

                _best_p = min(_p_raw, _p_ctrl)
                _tc = "#22c55e" if _best_p < 0.05 else "#f59e0b" if _best_p < 0.1 else "#888"
                _ax.set_title(f"r={_r_raw:+.2f}  |dur r={_r_ctrl:+.2f}", fontsize=8, color=_tc)

                if _i == len(_eng_targets) - 1:
                    _ax.set_xlabel(_feat, fontsize=9)
                if _j == 0:
                    _ax.set_ylabel(_eng, fontsize=9)
                _ax.tick_params(labelsize=6)

        fig_dan.suptitle(
            "DAN-related features vs Engagement (colored by duration)\n"
            "Left r = raw, right r = controlling for duration",
            fontweight="bold", fontsize=12
        )
        fig_dan.tight_layout()
        plt_dan.close("all")
    else:
        fig_dan = None
    return (fig_dan,)


@app.cell
def _(fig_dan, mo):
    mo.vstack([mo.md("### DAN correlations with all engagement metrics"), fig_dan]) if fig_dan is not None else mo.md("")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
