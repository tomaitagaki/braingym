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

    mo.md("### BrainGym: Retention Curves x Brain Timeseries")
    return Path, json, mo, np, pd


@app.cell
def _(Path, json, mo, pd):
    CACHE = Path("../../cache/tsinghua/selected")

    _curves_path = CACHE / "retention_curves.json"
    _tribe_path = CACHE / "tribe_results.json"

    if _curves_path.exists():
        retention = json.loads(_curves_path.read_text())
        ret_metrics = pd.read_csv(CACHE / "retention_metrics.csv")
        _msg = f"**{len(retention)} videos** with retention curves"
    else:
        retention = None
        ret_metrics = None
        _msg = "Waiting for retention curves..."

    if _tribe_path.exists():
        _brain_raw = json.loads(_tribe_path.read_text())
        brain_results = {str(r["video_id"]): r for r in _brain_raw if "error" not in r}
        _msg += f", **{len(brain_results)} videos** with brain predictions"
    else:
        brain_results = None
        _msg += ", waiting for brain predictions..."

    mo.md(f"## Data\n{_msg}")
    return CACHE, brain_results, ret_metrics, retention


@app.cell
def _(mo, ret_metrics):
    if ret_metrics is not None:
        mo.vstack([mo.md("## Retention metrics"), mo.ui.table(ret_metrics.round(3))])
    return


@app.cell
def _(brain_results, mo, np, retention):
    """Overlay brain timeseries and retention curves."""
    import matplotlib.pyplot as plt_overlay

    if retention and brain_results:
        _shared = [_vid for _vid in retention if str(_vid) in brain_results][:9]

        if _shared:
            _cols = 3
            _nrows = (len(_shared) + _cols - 1) // _cols
            fig_overlay, _axes = plt_overlay.subplots(_nrows, _cols, figsize=(18, 5 * _nrows))
            if len(_shared) > 1:
                _axes = _axes.flatten()
            else:
                _axes = [_axes]

            for _idx, _vid in enumerate(_shared):
                _ax = _axes[_idx]
                _ret = retention[str(_vid)]
                _brain = brain_results[str(_vid)]

                _t_ret = np.array(_ret["seconds"])
                _s = np.array(_ret["survival"])
                _ax.plot(_t_ret, _s, "k-", linewidth=2, label="Retention")
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
                _ax.set_title(f"Video {_vid} ({_dur:.0f}s, CR={_cr:.0%})", fontsize=9)
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
        else:
            fig_overlay = None
    else:
        fig_overlay = None
    return (fig_overlay,)


@app.cell
def _(fig_overlay, mo):
    mo.vstack([mo.md("### Per-video: Retention x Brain overlay"), fig_overlay]) if fig_overlay is not None else mo.md("")
    return


@app.cell
def _(brain_results, mo, np, pd, retention):
    """Scalar brain vs scalar retention correlations."""
    from scipy.stats import pearsonr as _pearsonr

    if retention and brain_results:
        _shared = [_vid for _vid in retention if str(_vid) in brain_results]

        _rows = []
        for _vid in _shared:
            _ret = retention[str(_vid)]
            _brain = brain_results[str(_vid)]
            _rows.append({
                "video_id": _vid,
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
            })

        combined = pd.DataFrame(_rows)

        _bc = ["cognitive_load", "attention", "engagement_brain", "net_Vis", "net_DAN", "net_DMN", "net_FPN"]
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
                    _row[_b_col.replace("net_", "").replace("engagement_brain", "engage")] = f"{_r:+.2f}{_stars}"
                else:
                    _row[_b_col.replace("net_", "").replace("engagement_brain", "engage")] = "—"
            _corr_rows.append(_row)

        _corr_df = pd.DataFrame(_corr_rows)
        mo.vstack([
            mo.md(f"### Scalar correlations ({len(combined)} videos)\n`**` p<0.05, `*` p<0.1"),
            mo.ui.table(_corr_df),
        ])
    else:
        combined = None
    return (combined,)


@app.cell
def _(brain_results, mo, np, retention):
    """Per-timepoint: does attention at time t predict retention at time t?"""
    from scipy.stats import pearsonr as _pearsonr2

    if retention and brain_results:
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

        if _surv_binned:
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
            _ax1.set_title("Attention vs Retention at each % of video\n(green=p<0.05, yellow=p<0.1)")
            _ax1.axhline(0, color="gray", linewidth=0.5)

            _ax2.bar(_pct, [-np.log10(max(_p, 1e-10)) for _p in _ps], width=4, color=_colors)
            _ax2.axhline(-np.log10(0.05), color="red", linewidth=0.5, linestyle="--", label="p=0.05")
            _ax2.set_ylabel("-log10(p)")
            _ax2.set_xlabel("% of video duration")
            _ax2.legend(fontsize=8)

            fig_persec.tight_layout()
            plt_persec.close("all")
        else:
            fig_persec = None
    else:
        fig_persec = None
    return (fig_persec,)


@app.cell
def _(fig_persec, mo):
    mo.vstack([mo.md("### Per-timepoint: Attention predicts retention?"), fig_persec]) if fig_persec is not None else mo.md("")
    return


@app.cell
def _(brain_results, mo, np, pd, retention):
    """Ranked predictor table."""
    from scipy.stats import pearsonr as _pearsonr3

    if retention and brain_results:
        _shared = [_vid for _vid in retention if str(_vid) in brain_results]

        _feat_rows = []
        for _vid in _shared:
            _brain = brain_results[str(_vid)]
            _ret = retention[str(_vid)]
            _ts = _brain["timeseries"]
            _dan = np.array(_ts["DAN"])
            _dmn = np.array(_ts["DMN"])
            _fpn = np.array(_ts["FPN"])
            _vis = np.array(_ts["Vis"])
            _attn = _dan - _dmn
            _n_tp = len(_dan)

            _row = {
                "video_id": _vid,
                "half_life": _ret["half_life"],
                "completion_rate": _ret["completion_rate"],
                "retention_auc": _ret["retention_auc"],
                "early_drop_3s": _ret["early_drop_3s"],
                "attn_mean": float(np.mean(_attn)),
                "attn_peak": float(np.max(_attn)),
                "attn_first3s": float(np.mean(_attn[:min(3, _n_tp)])),
                "attn_slope": float(np.polyfit(np.arange(_n_tp), _attn, 1)[0]) if _n_tp > 1 else 0,
                "dan_peak": float(np.max(_dan)),
                "vis_mean": float(np.mean(_vis)),
                "dmn_min": float(np.min(_dmn)),
            }
            if _n_tp > 3:
                _row["fc_DAN_DMN"], _ = _pearsonr3(_dan, _dmn)
            else:
                _row["fc_DAN_DMN"] = 0
            _feat_rows.append(_row)

        _feat_df = pd.DataFrame(_feat_rows)

        _bf = ["attn_mean", "attn_peak", "attn_first3s", "attn_slope",
               "dan_peak", "vis_mean", "dmn_min", "fc_DAN_DMN"]
        _rf = ["half_life", "completion_rate", "retention_auc", "early_drop_3s"]

        _ranked_rows = []
        for _rc in _rf:
            for _bc in _bf:
                _x = _feat_df[_bc].values
                _y = _feat_df[_rc].values
                if len(_x) > 3 and np.std(_x) > 1e-10:
                    _r, _p = _pearsonr3(_x, _y)
                    _stars = "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                    _ranked_rows.append({
                        "retention_metric": _rc,
                        "brain_feature": _bc,
                        "r": round(_r, 3),
                        "p": round(_p, 4),
                        "abs_r": round(abs(_r), 3),
                        "sig": _stars,
                    })

        _ranked = pd.DataFrame(_ranked_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
        _ranked.index = _ranked.index + 1

        mo.vstack([
            mo.md(f"### Ranked brain-retention correlations ({len(_feat_df)} videos)\n`**` p<0.05, `*` p<0.1"),
            mo.ui.table(_ranked),
        ])
    return


if __name__ == "__main__":
    app.run()
