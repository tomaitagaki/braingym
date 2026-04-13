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

    mo.md("### BrainGym: Retention Curves x Brain Timeseries (n=72)")
    return Path, json, mo, np, pd


@app.cell
def _(Path, json, mo, pd):
    CACHE = Path("cache/tsinghua/selected")

    retention = json.loads((CACHE / "retention_curves.json").read_text())
    ret_metrics = pd.read_csv(CACHE / "retention_metrics.csv")

    _brain_raw = json.loads((CACHE / "tribe_results.json").read_text())
    brain_results = {str(_r["video_id"]): _r for _r in _brain_raw if "error" not in _r}

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]
    mo.md(f"## Data\n**{len(retention)}** retention curves, **{len(brain_results)}** brain predictions, **{len(_shared)}** matched")
    return brain_results, ret_metrics, retention


@app.cell
def _(mo, ret_metrics):
    mo.vstack([mo.md("## Retention metric distributions"), mo.ui.table(ret_metrics.round(3))])
    return


@app.cell
def _(ret_metrics):
    """Retention metric histograms."""
    import matplotlib.pyplot as plt_hist

    fig_hist, _axes = plt_hist.subplots(1, 4, figsize=(16, 4))
    for _i, _col in enumerate(["half_life", "completion_rate", "retention_auc", "early_drop_3s"]):
        _axes[_i].hist(ret_metrics[_col].dropna(), bins=15, color="#4a9eed", edgecolor="white")
        _axes[_i].set_title(_col, fontsize=10)
        _axes[_i].set_ylabel("Count")
    fig_hist.suptitle("Retention metric distributions", fontweight="bold")
    fig_hist.tight_layout()
    plt_hist.close("all")
    return (fig_hist,)


@app.cell
def _(fig_hist, mo):
    mo.vstack([mo.md("## Histograms"), fig_hist]) if fig_hist is not None else mo.md("")
    return


@app.cell
def _(brain_results, np, retention):
    """Overlay brain timeseries and retention curves for individual videos."""
    import matplotlib.pyplot as plt_overlay

    _shared = [_vid for _vid in retention if str(_vid) in brain_results][:12]

    _cols = 3
    _nrows = (len(_shared) + _cols - 1) // _cols
    fig_overlay, _axes = plt_overlay.subplots(_nrows, _cols, figsize=(18, 4.5 * _nrows))
    _axes = _axes.flatten()

    for _idx, _vid in enumerate(_shared):
        _ax = _axes[_idx]
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]

        _t_ret = np.array(_ret["seconds"])
        _s = np.array(_ret["survival"])
        _ax.plot(_t_ret, _s, "k-", linewidth=2.5, label="Retention")
        _ax.set_ylabel("Retention S(t)", fontsize=9)
        _ax.set_ylim(0, 1.05)

        _ax2 = _ax.twinx()
        _t_brain = np.arange(len(_brain["attention_ts"]))
        _ax2.plot(_t_brain, _brain["attention_ts"], "-", color="#ef4444", linewidth=1.5, alpha=0.8, label="Attention")
        _ax2.plot(_t_brain, _brain["timeseries"]["Vis"], "-", color="#22c55e", linewidth=1, alpha=0.6, label="Vis")
        _ax2.plot(_t_brain, _brain["timeseries"]["DMN"], "--", color="#3b82f6", linewidth=1, alpha=0.6, label="DMN")
        _ax2.set_ylabel("Brain activation", fontsize=9, color="#ef4444")

        _dur = _ret["duration"]
        _cr = _ret["completion_rate"]
        _hl = _ret["half_life"]
        _ax.set_title(f"Video {_vid} ({_dur:.0f}s, CR={_cr:.0%}, HL={_hl:.0f}s)", fontsize=9)
        _ax.set_xlabel("Time (s)", fontsize=8)

        if _idx == 0:
            _l1, _lb1 = _ax.get_legend_handles_labels()
            _l2, _lb2 = _ax2.get_legend_handles_labels()
            _ax.legend(_l1 + _l2, _lb1 + _lb2, fontsize=7, loc="upper right")

    for _idx in range(len(_shared), len(_axes)):
        _axes[_idx].set_visible(False)

    fig_overlay.suptitle("Retention curve (black) vs Brain timeseries (colored)", fontweight="bold")
    fig_overlay.tight_layout()
    plt_overlay.close("all")
    return (fig_overlay,)


@app.cell
def _(fig_overlay, mo):
    mo.vstack([mo.md("### Per-video: Retention x Brain overlay"), fig_overlay])
    return


@app.cell
def _(brain_results, mo, np, pd, retention):
    """Scalar brain vs scalar retention correlations."""
    from scipy.stats import pearsonr as _pearsonr

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _rows = []
    for _vid in _shared:
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _ts = _brain["timeseries"]
        _dan = np.array(_ts["DAN"])
        _dmn = np.array(_ts["DMN"])
        _fpn = np.array(_ts["FPN"])
        _vis = np.array(_ts["Vis"])
        _attn = _dan - _dmn
        _n = len(_dan)

        _row = {
            "video_id": _vid,
            "duration": _ret["duration"],
            "half_life": _ret["half_life"],
            "completion_rate": _ret["completion_rate"],
            "early_drop_3s": _ret["early_drop_3s"],
            "retention_auc": _ret["retention_auc"],
            "rewatch_rate": _ret["rewatch_rate"],
            "cognitive_load": _brain["cognitive_load"],
            "attention": _brain["attention"],
            "engagement_brain": _brain["engagement"],
            "net_Vis": _brain["net_Vis"],
            "net_DAN": _brain["net_DAN"],
            "net_DMN": _brain["net_DMN"],
            "net_FPN": _brain["net_FPN"],
            "attn_peak": float(np.max(_attn)),
            "attn_first3s": float(np.mean(_attn[:min(3, _n)])),
            "attn_slope": float(np.polyfit(np.arange(_n), _attn, 1)[0]) if _n > 1 else 0,
            "attn_var": float(np.var(_attn)),
            "dan_peak": float(np.max(_dan)),
            "dmn_min": float(np.min(_dmn)),
        }
        if _n > 3:
            _row["fc_DAN_DMN"], _ = _pearsonr(_dan, _dmn)
        else:
            _row["fc_DAN_DMN"] = 0
        _rows.append(_row)

    combined = pd.DataFrame(_rows)

    _bc = ["cognitive_load", "attention", "net_Vis", "net_DAN", "net_DMN", "net_FPN",
           "attn_peak", "attn_first3s", "attn_slope", "fc_DAN_DMN"]
    _rc = ["half_life", "completion_rate", "early_drop_3s", "retention_auc", "rewatch_rate"]

    _corr_rows = []
    for _r_col in _rc:
        _row = {"retention_metric": _r_col}
        for _b_col in _bc:
            _x = combined[_b_col].values
            _y = combined[_r_col].values
            if len(_x) > 3 and np.std(_x) > 1e-10:
                _r, _p = _pearsonr(_x, _y)
                _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                _row[_b_col.replace("net_", "")] = f"{_r:+.2f}{_stars}"
            else:
                _row[_b_col.replace("net_", "")] = "—"
        _corr_rows.append(_row)

    _corr_df = pd.DataFrame(_corr_rows)
    mo.vstack([
        mo.md(f"### Scalar correlations ({len(combined)} videos)\n`**` p<0.05, `*` p<0.1"),
        mo.ui.table(_corr_df),
    ])
    return (combined,)


@app.cell
def _(combined):
    """Full correlation matrix: brain + retention."""
    import matplotlib.pyplot as plt_fullcorr

    _brain_cols = ["cognitive_load", "attention", "net_Vis", "net_DAN", "net_DMN",
                   "net_FPN", "attn_peak", "attn_first3s", "fc_DAN_DMN"]
    _ret_cols = ["half_life", "completion_rate", "early_drop_3s", "retention_auc", "rewatch_rate"]
    _all = _brain_cols + _ret_cols

    _data = combined[_all].copy()
    _data.columns = [_c.replace("net_", "").replace("early_drop_3s", "early_drop") for _c in _all]
    _corr = _data.corr()

    fig_fullcorr, _ax = plt_fullcorr.subplots(figsize=(13, 11))
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
            _ax.text(_j, _i, f"{_val:.2f}", ha="center", va="center", fontsize=7, color=_clr)

    _sep = len(_brain_cols) - 0.5
    _ax.axhline(_sep, color="black", linewidth=2)
    _ax.axvline(_sep, color="black", linewidth=2)

    fig_fullcorr.colorbar(_im, ax=_ax, shrink=0.8, label="Pearson r")
    _ax.set_title(f"Brain + Retention Correlations (n={len(combined)})", fontsize=12, fontweight="bold")
    plt_fullcorr.tight_layout()
    plt_fullcorr.close("all")
    return (fig_fullcorr,)


@app.cell
def _(fig_fullcorr, mo):
    mo.vstack([mo.md("## Full correlation matrix (brain + retention)"), fig_fullcorr])
    return


@app.cell
def _(brain_results, np, retention):
    """Per-timepoint: does attention at time t predict retention at time t?"""
    from scipy.stats import pearsonr as _pearsonr2

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _n_bins = 20
    _surv_binned = []
    _attn_binned = []

    for _vid in _shared:
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _surv = np.array(_ret["survival"])
        _attn = np.array(_brain["attention_ts"])

        if len(_surv) < 3 or len(_attn) < 3:
            continue

        _surv_r = np.interp(np.linspace(0, 1, _n_bins), np.linspace(0, 1, len(_surv)), _surv)
        _attn_r = np.interp(np.linspace(0, 1, _n_bins), np.linspace(0, 1, len(_attn)), _attn)
        _surv_binned.append(_surv_r)
        _attn_binned.append(_attn_r)

    _surv_mat = np.array(_surv_binned)
    _attn_mat = np.array(_attn_binned)

    _rs = []
    _ps = []
    for _b in range(_n_bins):
        if np.std(_attn_mat[:, _b]) > 1e-10:
            _r, _p = _pearsonr2(_attn_mat[:, _b], _surv_mat[:, _b])
        else:
            _r, _p = 0, 1
        _rs.append(_r)
        _ps.append(_p)

    import matplotlib.pyplot as plt_persec
    fig_persec, (_ax1, _ax2) = plt_persec.subplots(2, 1, figsize=(12, 6), sharex=True)

    _pct = np.linspace(0, 100, _n_bins)
    _colors = ["#22c55e" if _p < 0.05 else "#f59e0b" if _p < 0.1 else "#94a3b8" for _p in _ps]
    _ax1.bar(_pct, _rs, width=4, color=_colors)
    _ax1.set_ylabel("Pearson r")
    _ax1.set_title("Attention vs Retention at each % of video\n(green=p<0.05, yellow=p<0.1, gray=n.s.)")
    _ax1.axhline(0, color="gray", linewidth=0.5)

    _ax2.bar(_pct, [-np.log10(max(_p, 1e-10)) for _p in _ps], width=4, color=_colors)
    _ax2.axhline(-np.log10(0.05), color="red", linewidth=0.5, linestyle="--", label="p=0.05")
    _ax2.set_ylabel("-log10(p)")
    _ax2.set_xlabel("% of video duration")
    _ax2.legend(fontsize=8)

    fig_persec.tight_layout()
    plt_persec.close("all")
    return (fig_persec,)


@app.cell
def _(fig_persec, mo):
    mo.vstack([mo.md("### Per-timepoint: Does attention predict retention at each moment?"), fig_persec])
    return


@app.cell
def _(combined, mo, np, pd):
    """Ranked predictor table."""
    from scipy.stats import pearsonr as _pearsonr3

    _bf = ["cognitive_load", "attention", "attn_peak", "attn_first3s", "attn_slope",
           "attn_var", "dan_peak", "dmn_min", "fc_DAN_DMN", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
    _rf = ["half_life", "completion_rate", "retention_auc", "early_drop_3s", "rewatch_rate"]

    _ranked_rows = []
    for _rc in _rf:
        for _bc in _bf:
            _x = combined[_bc].values
            _y = combined[_rc].values
            if len(_x) > 3 and np.std(_x) > 1e-10:
                _r, _p = _pearsonr3(_x, _y)
                _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                _ranked_rows.append({
                    "retention_metric": _rc,
                    "brain_feature": _bc.replace("net_", ""),
                    "r": round(_r, 3),
                    "p": round(_p, 4),
                    "abs_r": round(abs(_r), 3),
                    "sig": _stars,
                })

    _ranked = pd.DataFrame(_ranked_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    _ranked.index = _ranked.index + 1

    mo.vstack([
        mo.md(f"### Ranked brain-retention correlations ({len(combined)} videos)\n`**` p<0.05, `*` p<0.1"),
        mo.ui.table(_ranked),
    ])
    return


@app.cell
def _(combined, np):
    """Hero plot: best brain predictor vs retention."""
    import matplotlib.pyplot as plt_hero
    from scipy.stats import pearsonr as _pearsonr4

    # Find best predictor of completion_rate
    _bf = ["attention", "attn_peak", "attn_first3s", "attn_slope", "fc_DAN_DMN", "dan_peak"]
    _best_r, _best_feat = 0, ""
    for _f in _bf:
        _r, _p = _pearsonr4(combined[_f].values, combined["completion_rate"].values)
        if abs(_r) > abs(_best_r):
            _best_r, _best_feat = _r, _f

    _x = combined[_best_feat].values
    _y = combined["completion_rate"].values
    _r, _p = _pearsonr4(_x, _y)

    fig_hero, _ax = plt_hero.subplots(figsize=(8, 6))
    _ax.scatter(_x, _y, s=60, alpha=0.6, c="#4a9eed", edgecolors="white", linewidth=0.8, zorder=3)

    _z = np.polyfit(_x, _y, 1)
    _xl = np.linspace(_x.min(), _x.max(), 100)
    _ax.plot(_xl, np.polyval(_z, _xl), "-", color="#ef4444", linewidth=2.5, alpha=0.8, zorder=2)

    _y_pred = np.polyval(_z, _x)
    _resid_std = np.std(_y - _y_pred)
    _ax.fill_between(_xl, np.polyval(_z, _xl) - _resid_std, np.polyval(_z, _xl) + _resid_std,
                     color="#ef4444", alpha=0.1, zorder=1)

    _ax.set_xlabel(f"Predicted brain: {_best_feat}", fontsize=13, fontweight="bold")
    _ax.set_ylabel("Video completion rate", fontsize=13, fontweight="bold")
    _ax.set_title(f"Brain {_best_feat} predicts retention (n={len(_x)}, r={_r:.2f}, p={_p:.3f})",
                  fontsize=14, fontweight="bold", pad=15)

    _ax.text(0.97, 0.95, f"Pearson r = {_r:.3f}\np = {_p:.4f}",
             transform=_ax.transAxes, fontsize=11,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    _ax.text(0.97, 0.05,
             "Brain response predicted by TRIBEv2 (Meta)\nTsinghua ShortVideo dataset (WWW 2025)",
             transform=_ax.transAxes, fontsize=8, color="#888888",
             verticalalignment="bottom", horizontalalignment="right")

    _ax.grid(alpha=0.2)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    fig_hero.tight_layout()
    fig_hero.savefig("cache/brain_vs_retention_hero.png", dpi=200, bbox_inches="tight")
    plt_hero.close("all")
    return (fig_hero,)


@app.cell
def _(fig_hero, mo):
    mo.vstack([mo.md("### Best brain predictor of video completion"), fig_hero])
    return


@app.cell
def _(brain_results, np, pd, retention):
    """Per-video: correlate attention timeseries with retention curve."""
    from scipy.stats import pearsonr as _pearsonr5

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _coupling_rows = []
    for _vid in _shared:
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _surv = np.array(_ret["survival"])
        _attn = np.array(_brain["attention_ts"])

        # Align to same length via interpolation
        _n_pts = min(len(_surv), len(_attn))
        if _n_pts < 4:
            continue
        _surv_aligned = np.interp(np.linspace(0, 1, _n_pts), np.linspace(0, 1, len(_surv)), _surv)
        _attn_aligned = np.interp(np.linspace(0, 1, _n_pts), np.linspace(0, 1, len(_attn)), _attn)

        _r, _p = _pearsonr5(_attn_aligned, _surv_aligned)

        _coupling_rows.append({
            "video_id": _vid,
            "duration": _ret["duration"],
            "completion_rate": _ret["completion_rate"],
            "half_life": _ret["half_life"],
            "attn_retention_r": round(_r, 3),
            "attn_retention_p": round(_p, 4),
            "n_timepoints": _n_pts,
        })

    coupling_df = pd.DataFrame(_coupling_rows).sort_values("attn_retention_r", ascending=False)
    return (coupling_df,)


@app.cell
def _(coupling_df, mo):
    mo.vstack([
        mo.md(f"### Per-video: Attention x Retention coupling (n={len(coupling_df)})\nPositive r = attention tracks retention. Negative r = they diverge."),
        mo.ui.table(coupling_df.round(3)),
    ])
    return


@app.cell
def _(coupling_df):
    """Histogram of per-video attention-retention correlations."""
    import matplotlib.pyplot as plt_coupling

    fig_coupling, _ax = plt_coupling.subplots(figsize=(10, 5))

    _rs = coupling_df["attn_retention_r"].values
    _colors = ["#22c55e" if _r > 0 else "#ef4444" for _r in _rs]
    _ax.bar(range(len(_rs)), _rs, color=_colors, edgecolor="white", linewidth=0.5)
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.set_xlabel("Videos (sorted by coupling strength)", fontsize=11)
    _ax.set_ylabel("Pearson r (attention vs retention)", fontsize=11)
    _ax.set_title(f"Per-video attention-retention coupling\n"
                  f"Mean r = {_rs.mean():.3f}, "
                  f"{(_rs > 0).sum()}/{len(_rs)} positive, "
                  f"{(coupling_df['attn_retention_p'].values < 0.05).sum()} significant (p<0.05)",
                  fontsize=12, fontweight="bold")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    fig_coupling.tight_layout()
    plt_coupling.close("all")
    return (fig_coupling,)


@app.cell
def _(fig_coupling, mo):
    mo.vstack([mo.md("### Attention-retention coupling distribution"), fig_coupling])
    return


@app.cell
def _(brain_results, coupling_df, np, retention):
    """Plot the top 3 best-coupled and top 3 worst-coupled videos side by side."""
    import matplotlib.pyplot as plt_bestworst

    _best = coupling_df.head(3)["video_id"].tolist()
    _worst = coupling_df.tail(3)["video_id"].tolist()
    _vids = _best + _worst
    _titles = ["BEST"] * 3 + ["WORST"] * 3

    fig_bw, _axes = plt_bestworst.subplots(2, 3, figsize=(18, 8))
    _axes = _axes.flatten()

    for _idx, (_vid, _label) in enumerate(zip(_vids, _titles)):
        _ax = _axes[_idx]
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _surv = np.array(_ret["survival"])
        _attn = np.array(_brain["attention_ts"])

        _t_ret = np.linspace(0, 100, len(_surv))
        _t_attn = np.linspace(0, 100, len(_attn))

        _ax.plot(_t_ret, _surv, "k-", linewidth=2.5, label="Retention")
        _ax.set_ylabel("Retention", fontsize=9)
        _ax.set_ylim(0, 1.05)

        _ax2 = _ax.twinx()
        _ax2.plot(_t_attn, _attn, "-", color="#ef4444", linewidth=1.5, alpha=0.8, label="Attention")
        _ax2.set_ylabel("Attention", fontsize=9, color="#ef4444")

        _r_val = coupling_df[coupling_df["video_id"] == _vid]["attn_retention_r"].values[0]
        _clr = "#22c55e" if _label == "BEST" else "#ef4444"
        _ax.set_title(f"{_label}: Video {_vid}\nr={_r_val:.2f}", fontsize=10, color=_clr, fontweight="bold")
        _ax.set_xlabel("% of video", fontsize=8)

        if _idx == 0:
            _l1, _lb1 = _ax.get_legend_handles_labels()
            _l2, _lb2 = _ax2.get_legend_handles_labels()
            _ax.legend(_l1 + _l2, _lb1 + _lb2, fontsize=7)

    fig_bw.suptitle("Best vs Worst attention-retention coupling", fontweight="bold", fontsize=13)
    fig_bw.tight_layout()
    plt_bestworst.close("all")
    return (fig_bw,)


@app.cell
def _(fig_bw, mo):
    mo.vstack([mo.md("### Best vs Worst coupled videos"), fig_bw])
    return


@app.cell
def _(brain_results, mo, np, pd, retention):
    """Grid search: all brain metrics x preprocessing x normalization for per-video coupling."""
    from scipy.stats import pearsonr as _pearsonr_grid
    from scipy.signal import detrend as _detrend_grid

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _grid_rows = []
    for _vid in _shared:
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _surv = np.array(_ret["survival"])
        _ts = _brain["timeseries"]
        _dan = np.array(_ts["DAN"])
        _dmn = np.array(_ts["DMN"])
        _fpn = np.array(_ts["FPN"])
        _vis = np.array(_ts["Vis"])
        _van = np.array(_ts["VAN"])
        _limbic = np.array(_ts["Limbic"])
        _dur = _ret["duration"]
        _n_brain = len(_dan)

        _signals = {
            "-DMN": -_dmn,
            "Attention": _dan - _dmn,
            "CogLoad": _fpn + _dan - _dmn,
            "DAN": _dan,
            "Vis": _vis,
            "VAN": _van,
            "Limbic": _limbic,
            "FPN": _fpn,
        }

        # Window options
        _windows = {
            "full_pct": {"mode": "pct", "n_bins": 20},
            "first_10s": {"mode": "crop", "seconds": 10},
            "first_15s": {"mode": "crop", "seconds": 15},
            "first_20s": {"mode": "crop", "seconds": 20},
        }

        for _wname, _wcfg in _windows.items():
            if _wcfg["mode"] == "pct":
                _s = np.interp(np.linspace(0, 1, _wcfg["n_bins"]),
                               np.linspace(0, 1, len(_surv)), _surv)
            else:
                _sec = _wcfg["seconds"]
                if _dur < _sec or _n_brain < _sec:
                    continue
                _s = _surv[:_sec]

            for _mname, _sig in _signals.items():
                if _wcfg["mode"] == "pct":
                    _b = np.interp(np.linspace(0, 1, _wcfg["n_bins"]),
                                   np.linspace(0, 1, len(_sig)), _sig)
                else:
                    _b = _sig[:_wcfg["seconds"]]

                _n = min(len(_s), len(_b))
                if _n < 4:
                    continue

                _sv, _bv = _s[:_n].astype(float), _b[:_n].astype(float)

                # Raw
                if np.std(_bv) > 1e-10:
                    _r, _p = _pearsonr_grid(_bv, _sv)
                    _grid_rows.append({
                        "video_id": _vid, "metric": _mname, "window": _wname,
                        "preprocess": "raw", "r": _r, "p": _p, "n_pts": _n,
                    })

                # Z-scored
                _bz = (_bv - _bv.mean()) / max(_bv.std(), 1e-10)
                _sz = (_sv - _sv.mean()) / max(_sv.std(), 1e-10)
                if np.std(_bz) > 1e-10:
                    _r, _p = _pearsonr_grid(_bz, _sz)
                    _grid_rows.append({
                        "video_id": _vid, "metric": _mname, "window": _wname,
                        "preprocess": "z-scored", "r": _r, "p": _p, "n_pts": _n,
                    })

                # Detrended
                if _n >= 4:
                    _bd = _detrend_grid(_bv)
                    _sd = _detrend_grid(_sv)
                    if np.std(_bd) > 1e-10 and np.std(_sd) > 1e-10:
                        _r, _p = _pearsonr_grid(_bd, _sd)
                        _grid_rows.append({
                            "video_id": _vid, "metric": _mname, "window": _wname,
                            "preprocess": "detrended", "r": _r, "p": _p, "n_pts": _n,
                        })

    grid_df = pd.DataFrame(_grid_rows)

    # Summarize: best combos
    _summary = grid_df.groupby(["metric", "window", "preprocess"]).agg(
        mean_r=("r", "mean"),
        median_r=("r", "median"),
        pct_positive=("r", lambda _x: (_x > 0).mean() * 100),
        pct_sig=("p", lambda _x: (_x < 0.05).mean() * 100),
        n_videos=("r", "count"),
    ).round(3).sort_values("mean_r", ascending=False).reset_index()

    mo.vstack([
        mo.md(f"### Grid search: {len(_summary)} combinations (metric x window x preprocess)\nTop 20 by mean r:"),
        mo.ui.table(_summary.head(20)),
    ])
    return (grid_df,)


@app.cell
def _(grid_df):
    """Heatmap: mean r for each metric x window, using raw preprocessing."""
    import matplotlib.pyplot as plt_heatmap

    _raw = grid_df[grid_df["preprocess"] == "raw"]
    _pivot = _raw.groupby(["metric", "window"])["r"].mean().unstack(fill_value=0)

    # Sort metrics by mean across windows
    _order = _pivot.mean(axis=1).sort_values(ascending=False).index
    _pivot = _pivot.loc[_order]

    fig_heatmap, _ax = plt_heatmap.subplots(figsize=(10, 7))
    _im = _ax.imshow(_pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

    _ax.set_xticks(range(len(_pivot.columns)))
    _ax.set_yticks(range(len(_pivot.index)))
    _ax.set_xticklabels(_pivot.columns, rotation=30, ha="right", fontsize=10)
    _ax.set_yticklabels(_pivot.index, fontsize=10)

    for _i in range(len(_pivot.index)):
        for _j in range(len(_pivot.columns)):
            _val = _pivot.values[_i, _j]
            _clr = "white" if abs(_val) > 0.3 else "black"
            _ax.text(_j, _i, f"{_val:.2f}", ha="center", va="center", fontsize=9, color=_clr)

    fig_heatmap.colorbar(_im, ax=_ax, shrink=0.8, label="Mean Pearson r")
    _ax.set_title("Brain metric x Window size: mean per-video coupling (raw)", fontsize=12, fontweight="bold")
    fig_heatmap.tight_layout()
    plt_heatmap.close("all")
    return (fig_heatmap,)


@app.cell
def _(fig_heatmap, mo):
    mo.vstack([mo.md("### Grid search heatmap (raw preprocessing)"), fig_heatmap])
    return


@app.cell
def _(grid_df):
    """Same heatmap but detrended — shows what's real vs trend artifact."""
    import matplotlib.pyplot as plt_heatmap2

    _det = grid_df[grid_df["preprocess"] == "detrended"]
    if len(_det) > 0:
        _pivot = _det.groupby(["metric", "window"])["r"].mean().unstack(fill_value=0)
        _order = _pivot.mean(axis=1).sort_values(ascending=False).index
        _pivot = _pivot.loc[_order]

        fig_heatmap2, _ax = plt_heatmap2.subplots(figsize=(10, 7))
        _im = _ax.imshow(_pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

        _ax.set_xticks(range(len(_pivot.columns)))
        _ax.set_yticks(range(len(_pivot.index)))
        _ax.set_xticklabels(_pivot.columns, rotation=30, ha="right", fontsize=10)
        _ax.set_yticklabels(_pivot.index, fontsize=10)

        for _i in range(len(_pivot.index)):
            for _j in range(len(_pivot.columns)):
                _val = _pivot.values[_i, _j]
                _clr = "white" if abs(_val) > 0.3 else "black"
                _ax.text(_j, _i, f"{_val:.2f}", ha="center", va="center", fontsize=9, color=_clr)

        fig_heatmap2.colorbar(_im, ax=_ax, shrink=0.8, label="Mean Pearson r")
        _ax.set_title("Brain metric x Window size: mean per-video coupling (DETRENDED)\nThis is what's real beyond shared drift", fontsize=11, fontweight="bold")
        fig_heatmap2.tight_layout()
        plt_heatmap2.close("all")
    else:
        fig_heatmap2 = None
    return (fig_heatmap2,)


@app.cell
def _(fig_heatmap2, mo):
    mo.vstack([mo.md("### Grid search heatmap (detrended — honest signal)"), fig_heatmap2]) if fig_heatmap2 is not None else mo.md("")
    return


@app.cell
def _(brain_results, np, pd, retention):
    """Per-video coupling for multiple brain metrics vs retention."""
    from scipy.stats import pearsonr as _pearsonr6

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _all_rows = []
    for _vid in _shared:
        _ret = retention[str(_vid)]
        _brain = brain_results[str(_vid)]
        _surv = np.array(_ret["survival"])
        _ts = _brain["timeseries"]
        _dan = np.array(_ts["DAN"])
        _dmn = np.array(_ts["DMN"])
        _fpn = np.array(_ts["FPN"])
        _vis = np.array(_ts["Vis"])
        _van = np.array(_ts["VAN"])
        _limbic = np.array(_ts["Limbic"])
        _attn = _dan - _dmn
        _cogload = _fpn + _dan - _dmn

        # All brain timeseries to correlate with retention
        _metrics = {
            "Attention (DAN-DMN)": _attn,
            "Cog Load (FPN+DAN-DMN)": _cogload,
            "DAN": _dan,
            "DMN": _dmn,
            "FPN": _fpn,
            "Vis": _vis,
            "VAN": _van,
            "-DMN (engagement)": -_dmn,
        }

        for _name, _ts_arr in _metrics.items():
            _n_pts = min(len(_surv), len(_ts_arr))
            if _n_pts < 4:
                continue
            _s_aligned = np.interp(np.linspace(0, 1, _n_pts), np.linspace(0, 1, len(_surv)), _surv)
            _b_aligned = np.interp(np.linspace(0, 1, _n_pts), np.linspace(0, 1, len(_ts_arr)), _ts_arr)

            if np.std(_b_aligned) < 1e-10:
                continue

            _r, _p = _pearsonr6(_b_aligned, _s_aligned)
            _all_rows.append({
                "video_id": _vid,
                "brain_metric": _name,
                "r": _r,
                "p": _p,
            })

    multi_coupling = pd.DataFrame(_all_rows)
    return (multi_coupling,)


@app.cell
def _(mo, multi_coupling):
    """Summary: mean coupling per brain metric across all videos."""

    _summary = multi_coupling.groupby("brain_metric").agg(
        mean_r=("r", "mean"),
        median_r=("r", "median"),
        pct_positive=("r", lambda _x: (_x > 0).mean() * 100),
        pct_sig=("p", lambda _x: (_x < 0.05).mean() * 100),
        n_videos=("r", "count"),
    ).round(3).sort_values("mean_r", ascending=False)

    mo.vstack([
        mo.md("### Which brain metric best tracks retention? (per-video timeseries correlation)\n"
               "Each row = average across all videos of the within-video r(brain_ts, retention_ts)"),
        mo.ui.table(_summary.reset_index()),
    ])
    return


@app.cell
def _(multi_coupling, np):
    """Box plot: distribution of per-video r for each brain metric."""
    import matplotlib.pyplot as plt_box

    _metrics = multi_coupling["brain_metric"].unique()
    _order = multi_coupling.groupby("brain_metric")["r"].mean().sort_values(ascending=False).index.tolist()

    fig_box, _ax = plt_box.subplots(figsize=(12, 6))

    _data = [multi_coupling[multi_coupling["brain_metric"] == _m]["r"].values for _m in _order]
    _bp = _ax.boxplot(_data, labels=_order, patch_artist=True, vert=True)

    for _i, _box in enumerate(_bp["boxes"]):
        _median = np.median(_data[_i])
        _color = "#22c55e" if _median > 0 else "#ef4444"
        _box.set_facecolor(_color)
        _box.set_alpha(0.5)

    _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _ax.set_ylabel("Pearson r (brain timeseries vs retention)", fontsize=11)
    _ax.set_title("Per-video brain-retention coupling by metric\n(green = positive median, red = negative median)",
                  fontsize=12, fontweight="bold")
    _ax.tick_params(axis="x", rotation=30, labelsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    fig_box.tight_layout()
    plt_box.close("all")
    return (fig_box,)


@app.cell
def _(fig_box, mo):
    mo.vstack([mo.md("### Brain-retention coupling distributions"), fig_box])
    return


@app.cell
def _(multi_coupling):
    """Scatter: mean attention-retention r vs mean DAN-retention r per video."""
    import matplotlib.pyplot as plt_compare
    from scipy.stats import pearsonr as _pearsonr7

    _attn = multi_coupling[multi_coupling["brain_metric"] == "Attention (DAN-DMN)"].set_index("video_id")["r"]
    _vis = multi_coupling[multi_coupling["brain_metric"] == "Vis"].set_index("video_id")["r"]
    _dmn = multi_coupling[multi_coupling["brain_metric"] == "DMN"].set_index("video_id")["r"]

    _common = list(set(_attn.index) & set(_vis.index) & set(_dmn.index))

    fig_compare, _axes = plt_compare.subplots(1, 2, figsize=(14, 5))

    # Attention coupling vs Vis coupling
    _x1, _y1 = _attn[_common].values, _vis[_common].values
    _axes[0].scatter(_x1, _y1, s=40, alpha=0.6, c="#4a9eed", edgecolors="white")
    if len(_x1) > 3:
        _r1, _p1 = _pearsonr7(_x1, _y1)
        _axes[0].set_title(f"Attention vs Vis coupling (r={_r1:.2f}, p={_p1:.3f})", fontsize=10)
    _axes[0].set_xlabel("Attention-Retention r", fontsize=10)
    _axes[0].set_ylabel("Vis-Retention r", fontsize=10)
    _axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _axes[0].axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Attention coupling vs DMN coupling
    _x2, _y2 = _attn[_common].values, _dmn[_common].values
    _axes[1].scatter(_x2, _y2, s=40, alpha=0.6, c="#ef4444", edgecolors="white")
    if len(_x2) > 3:
        _r2, _p2 = _pearsonr7(_x2, _y2)
        _axes[1].set_title(f"Attention vs DMN coupling (r={_r2:.2f}, p={_p2:.3f})", fontsize=10)
    _axes[1].set_xlabel("Attention-Retention r", fontsize=10)
    _axes[1].set_ylabel("DMN-Retention r", fontsize=10)
    _axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _axes[1].axvline(0, color="gray", linewidth=0.5, linestyle="--")

    fig_compare.suptitle("Do different brain metrics agree on which videos couple with retention?", fontweight="bold")
    fig_compare.tight_layout()
    plt_compare.close("all")
    return (fig_compare,)


@app.cell
def _(fig_compare, mo):
    mo.vstack([mo.md("### Cross-metric coupling comparison"), fig_compare])
    return


@app.cell
def _(Path, json, mo, pd):
    """Cross-dataset comparison: TikTok vs Tsinghua brain responses."""
    from scipy.stats import mannwhitneyu as _mwu, ks_2samp as _ks

    # Load TikTok brain results
    _tt_path = Path("cache/tribe_tiktok_train_results.json")
    if not _tt_path.exists():
        cross_df = None
        _out_cross = mo.md("_TikTok results not found._")
    else:
        _tt_raw = json.loads(_tt_path.read_text())
        _tt = {str(r["video_id"]): r for r in _tt_raw if "error" not in r}

        _ts_path = Path("cache/tsinghua/selected/tribe_results.json")
        _ts_raw = json.loads(_ts_path.read_text())
        _ts = {str(r["video_id"]): r for r in _ts_raw if "error" not in r}

        _metrics = ["attention", "cognitive_load", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]

        _rows = []
        for _vid, _r in _tt.items():
            _row = {"dataset": "TikTok", "video_id": _vid}
            for _m in _metrics:
                _row[_m] = _r.get(_m, _r.get(f"net_{_m.replace('net_', '')}", 0))
            _rows.append(_row)
        for _vid, _r in _ts.items():
            _row = {"dataset": "Tsinghua", "video_id": _vid}
            for _m in _metrics:
                _row[_m] = _r.get(_m, _r.get(f"net_{_m.replace('net_', '')}", 0))
            _rows.append(_row)

        cross_df = pd.DataFrame(_rows)

        _test_rows = []
        for _m in _metrics:
            _tt_vals = cross_df[cross_df["dataset"] == "TikTok"][_m].values
            _ts_vals = cross_df[cross_df["dataset"] == "Tsinghua"][_m].values
            _u_stat, _u_p = _mwu(_tt_vals, _ts_vals, alternative="two-sided")
            _ks_stat, _ks_p = _ks(_tt_vals, _ts_vals)
            _test_rows.append({
                "metric": _m,
                "tiktok_mean": f"{_tt_vals.mean():+.3f}",
                "tsinghua_mean": f"{_ts_vals.mean():+.3f}",
                "diff": f"{_ts_vals.mean() - _tt_vals.mean():+.3f}",
                "mann_whitney_p": round(_u_p, 4),
                "ks_p": round(_ks_p, 4),
                "sig": "**" if min(_u_p, _ks_p) < 0.05 else "*" if min(_u_p, _ks_p) < 0.1 else "",
            })

        _test_df = pd.DataFrame(_test_rows)
        _out_cross = mo.vstack([
            mo.md(f"## Cross-dataset: TikTok (n={len(_tt)}) vs Tsinghua (n={len(_ts)})\n"
                   "Are the brain responses to these two platforms statistically different?"),
            mo.ui.table(_test_df),
        ])
    _out_cross
    return (cross_df,)


@app.cell
def _(cross_df):
    """Side-by-side violin plots of brain metrics by dataset."""
    import matplotlib.pyplot as plt_cross

    if cross_df is not None:
        _metrics = ["attention", "cognitive_load", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]

        fig_cross, _axes = plt_cross.subplots(2, 3, figsize=(16, 8))
        _axes = _axes.flatten()

        for _i, _m in enumerate(_metrics):
            _ax = _axes[_i]
            _tt = cross_df[cross_df["dataset"] == "TikTok"][_m].values
            _ts = cross_df[cross_df["dataset"] == "Tsinghua"][_m].values

            _parts = _ax.violinplot([_tt, _ts], positions=[0, 1], showmeans=True, showmedians=True)
            _parts["bodies"][0].set_facecolor("#4a9eed")
            _parts["bodies"][1].set_facecolor("#f59e0b")
            for _pc in _parts["bodies"]:
                _pc.set_alpha(0.5)

            _ax.set_xticks([0, 1])
            _ax.set_xticklabels(["TikTok", "Tsinghua"], fontsize=10)
            _ax.set_title(_m.replace("net_", ""), fontsize=11, fontweight="bold")
            _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        fig_cross.suptitle("Brain metric distributions: TikTok (blue) vs Tsinghua (orange)", fontweight="bold")
        fig_cross.tight_layout()
        plt_cross.close("all")
    else:
        fig_cross = None
    return (fig_cross,)


@app.cell
def _(fig_cross, mo):
    mo.vstack([mo.md("### Brain response distributions by platform"), fig_cross]) if fig_cross is not None else mo.md("")
    return


@app.cell
def _(Path, cross_df, json, mo, np, pd):
    """Percentile-matched comparison."""
    import matplotlib.pyplot as plt_pct

    if cross_df is not None:
        _tt_meta = json.loads(Path("cache/tiktok_train_100_meta.json").read_text())
        _tt_meta.extend(json.loads(Path("cache/tiktok_valid.json").read_text()))
        _tt_eng = {str(_m["id"]): _m for _m in _tt_meta}

        _ts_ret = json.loads(Path("cache/tsinghua/selected/retention_curves.json").read_text())

        _tt_brain_raw = json.loads(Path("cache/tribe_tiktok_train_results.json").read_text())
        _tt_brain_raw.extend(json.loads(Path("cache/tribe_tiktok_results.json").read_text()))
        _tt_brain = {str(_r["video_id"]): _r for _r in _tt_brain_raw if "error" not in _r}

        _ts_brain_raw = json.loads(Path("cache/tsinghua/selected/tribe_results.json").read_text())
        _ts_brain = {str(_r["video_id"]): _r for _r in _ts_brain_raw if "error" not in _r}

        _bcols = ["attention", "cognitive_load", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]

        _tt_rows = []
        for _vid, _br in _tt_brain.items():
            if _vid in _tt_eng:
                _em = _tt_eng[_vid]
                _er = _em["digg_count"] / max(_em["play_count"], 1) * 100
                _tt_rows.append({"video_id": _vid, "performance": _er, **{_k: _br[_k] for _k in _bcols}})
        _tt_df = pd.DataFrame(_tt_rows)

        _ts_rows = []
        for _vid, _br in _ts_brain.items():
            if _vid in _ts_ret:
                _cr = _ts_ret[_vid]["completion_rate"]
                _ts_rows.append({"video_id": _vid, "performance": _cr * 100, **{_k: _br[_k] for _k in _bcols}})
        _ts_df = pd.DataFrame(_ts_rows)

        _tt_df["pct_bin"] = pd.qcut(_tt_df["performance"], q=4, labels=["bottom 25%", "25-50%", "50-75%", "top 25%"])
        _ts_df["pct_bin"] = pd.qcut(_ts_df["performance"], q=4, labels=["bottom 25%", "25-50%", "50-75%", "top 25%"])

        _plot_metrics = ["attention", "cognitive_load", "net_DAN", "net_DMN"]
        _bins = ["bottom 25%", "25-50%", "50-75%", "top 25%"]

        fig_pct, _axes = plt_pct.subplots(2, 2, figsize=(14, 10))
        _axes = _axes.flatten()

        for _i, _pm in enumerate(_plot_metrics):
            _ax = _axes[_i]
            _x = np.arange(len(_bins))
            _w = 0.35
            _tt_means = [_tt_df[_tt_df["pct_bin"] == _b][_pm].mean() for _b in _bins]
            _ts_means = [_ts_df[_ts_df["pct_bin"] == _b][_pm].mean() for _b in _bins]
            _tt_sems = [_tt_df[_tt_df["pct_bin"] == _b][_pm].sem() for _b in _bins]
            _ts_sems = [_ts_df[_ts_df["pct_bin"] == _b][_pm].sem() for _b in _bins]
            _ax.bar(_x - _w/2, _tt_means, _w, yerr=_tt_sems, label="TikTok", color="#4a9eed", alpha=0.7, capsize=3)
            _ax.bar(_x + _w/2, _ts_means, _w, yerr=_ts_sems, label="Tsinghua", color="#f59e0b", alpha=0.7, capsize=3)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_bins, fontsize=9)
            _ax.set_title(_pm.replace("net_", ""), fontsize=11, fontweight="bold")
            _ax.set_ylabel("Mean brain activation", fontsize=9)
            _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            if _i == 0:
                _ax.legend(fontsize=9)

        fig_pct.suptitle(
            "Brain responses by engagement percentile\n"
            "TikTok: engagement_rate | Tsinghua: completion_rate",
            fontweight="bold", fontsize=12
        )
        fig_pct.tight_layout()
        plt_pct.close("all")
        _out = mo.vstack([
            mo.md("### Percentile-matched: do similar-performing videos produce similar brain responses?"),
            fig_pct,
        ])
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(Path, brain_results, np, pd, retention):
    """Category analysis: do brain-retention relationships differ by content type?"""
    from scipy.stats import pearsonr as _pearsonr_cat

    _interactions = pd.read_csv(Path("cache/tsinghua/interaction_sampled.csv"))
    _cats = pd.read_csv(Path("cache/tsinghua/categories_cn_en.csv"))

    _vid_cats = _interactions.groupby("pid").agg(root_id=("root_id", "first")).reset_index()
    _root_cats = _cats[_cats["category_level"] == 1][["category_id", "category_name_en"]].rename(
        columns={"category_id": "root_id", "category_name_en": "root_category"}
    )
    _vid_cats = _vid_cats.merge(_root_cats, on="root_id", how="left")
    _vid_cats["root_category"] = _vid_cats["root_category"].str.strip()

    _shared = [_vid for _vid in retention if str(_vid) in brain_results]

    _rows = []
    for _vid in _shared:
        _br = brain_results[str(_vid)]
        _ret = retention[str(_vid)]
        _cat_row = _vid_cats[_vid_cats["pid"] == int(_vid)]
        _cat = _cat_row["root_category"].values[0] if len(_cat_row) > 0 else "unknown"
        _ts = _br["timeseries"]
        _dan = np.array(_ts["DAN"])
        _dmn = np.array(_ts["DMN"])
        _attn = _dan - _dmn

        _rows.append({
            "video_id": _vid,
            "category": _cat,
            "half_life": _ret["half_life"],
            "completion_rate": _ret["completion_rate"],
            "retention_auc": _ret["retention_auc"],
            "attention": _br["attention"],
            "net_DMN": _br["net_DMN"],
            "net_FPN": _br["net_FPN"],
            "net_Vis": _br["net_Vis"],
            "attn_peak": float(np.max(_attn)),
        })

    cat_df = pd.DataFrame(_rows)

    # Categories with >=5 videos
    _good_cats = cat_df["category"].value_counts()
    _good_cats = _good_cats[_good_cats >= 5].index.tolist()
    cat_df_filtered = cat_df[cat_df["category"].isin(_good_cats)].copy()
    return (cat_df_filtered,)


@app.cell
def _(cat_df_filtered, np):
    """Brain metric profiles by category."""
    import matplotlib.pyplot as plt_catprofile

    _cats = cat_df_filtered["category"].unique()
    _metrics = ["attention", "net_DMN", "net_FPN", "net_Vis"]

    fig_catprofile, _ax = plt_catprofile.subplots(figsize=(14, 6))

    _x = np.arange(len(_cats))
    _w = 0.2
    _colors = {"attention": "#ef4444", "net_DMN": "#3b82f6", "net_FPN": "#f59e0b", "net_Vis": "#22c55e"}

    for _i, _m in enumerate(_metrics):
        _means = [cat_df_filtered[cat_df_filtered["category"] == _c][_m].mean() for _c in _cats]
        _sems = [cat_df_filtered[cat_df_filtered["category"] == _c][_m].sem() for _c in _cats]
        _ax.bar(_x + _i * _w, _means, _w, yerr=_sems, label=_m.replace("net_", ""),
                color=_colors[_m], alpha=0.7, capsize=2)

    _ax.set_xticks(_x + 1.5 * _w)
    _ax.set_xticklabels(_cats, rotation=30, ha="right", fontsize=9)
    _ax.set_ylabel("Mean brain activation", fontsize=10)
    _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    _ax.legend(fontsize=9)
    _ax.set_title("Brain response profiles by content category", fontsize=12, fontweight="bold")

    fig_catprofile.tight_layout()
    plt_catprofile.close("all")
    return (fig_catprofile,)


@app.cell
def _(fig_catprofile, mo):
    mo.vstack([mo.md("## Category analysis"), fig_catprofile])
    return


@app.cell
def _(cat_df_filtered, mo, pd):
    """Per-category: does the brain-retention relationship differ?"""
    from scipy.stats import pearsonr as _pearsonr_cat2

    _cats = cat_df_filtered["category"].unique()
    _brain_cols = ["attention", "net_DMN", "net_FPN", "net_Vis"]

    _cat_corr_rows = []
    for _cat in _cats:
        _sub = cat_df_filtered[cat_df_filtered["category"] == _cat]
        _n = len(_sub)
        _row = {"category": _cat, "n": _n}
        for _bc in _brain_cols:
            if _n >= 5:
                _r, _p = _pearsonr_cat2(_sub[_bc].values, _sub["half_life"].values)
                _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                _row[f"{_bc.replace('net_', '')}->HL"] = f"{_r:+.2f}{_stars}"
            else:
                _row[f"{_bc.replace('net_', '')}->HL"] = "—"
        # Also completion rate
        for _bc in _brain_cols:
            if _n >= 5:
                _r, _p = _pearsonr_cat2(_sub[_bc].values, _sub["completion_rate"].values)
                _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                _row[f"{_bc.replace('net_', '')}->CR"] = f"{_r:+.2f}{_stars}"
            else:
                _row[f"{_bc.replace('net_', '')}->CR"] = "—"
        _cat_corr_rows.append(_row)

    _cat_corr_df = pd.DataFrame(_cat_corr_rows)

    mo.vstack([
        mo.md("### Per-category brain-retention correlations\n"
              "HL = half_life, CR = completion_rate. Do different content types have different brain signatures for success?"),
        mo.ui.table(_cat_corr_df),
    ])
    return


@app.cell
def _(cat_df_filtered):
    """Scatter: attention vs half_life colored by category."""
    import matplotlib.pyplot as plt_catscat

    _cats = cat_df_filtered["category"].unique()
    _colors_map = dict(zip(_cats, [
        "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
        "#ec4899", "#14b8a6", "#f97316", "#6366f1",
    ][:len(_cats)]))

    fig_catscat, _axes = plt_catscat.subplots(1, 2, figsize=(16, 6))

    # Left: attention vs half_life
    for _cat in _cats:
        _sub = cat_df_filtered[cat_df_filtered["category"] == _cat]
        _axes[0].scatter(_sub["attention"], _sub["half_life"], s=50, alpha=0.7,
                         color=_colors_map[_cat], label=_cat, edgecolors="white", linewidth=0.5)
    _axes[0].set_xlabel("Attention (DAN-DMN)", fontsize=10)
    _axes[0].set_ylabel("Half-life (seconds)", fontsize=10)
    _axes[0].set_title("Attention vs Retention by category", fontsize=11, fontweight="bold")
    _axes[0].legend(fontsize=7, loc="upper right", ncol=2)

    # Right: DMN vs half_life
    for _cat in _cats:
        _sub = cat_df_filtered[cat_df_filtered["category"] == _cat]
        _axes[1].scatter(_sub["net_DMN"], _sub["half_life"], s=50, alpha=0.7,
                         color=_colors_map[_cat], label=_cat, edgecolors="white", linewidth=0.5)
    _axes[1].set_xlabel("DMN activation", fontsize=10)
    _axes[1].set_ylabel("Half-life (seconds)", fontsize=10)
    _axes[1].set_title("DMN vs Retention by category", fontsize=11, fontweight="bold")

    fig_catscat.suptitle("Does the brain-retention relationship differ by content type?", fontweight="bold")
    fig_catscat.tight_layout()
    plt_catscat.close("all")
    return (fig_catscat,)


@app.cell
def _(fig_catscat, mo):
    mo.vstack([mo.md("### Category-colored scatterplots"), fig_catscat])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
