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

    mo.md("# Video Brain Comparison\nCompare TRIBE v2 brain metrics across uploaded videos.")
    return Path, json, mo, np, pd


@app.cell
def _(Path, json, mo):
    RESULTS_DIR = Path("./cache/analyze_results")
    _result_files = sorted(RESULTS_DIR.glob("*_results.json"))

    videos = []
    for _f in _result_files:
        _r = json.loads(_f.read_text())
        videos.append(_r)

    mo.md(f"**{len(videos)} videos loaded** from `{RESULTS_DIR}`")
    return (videos,)


@app.cell
def _(mo, pd, videos):
    """Summary table."""
    if not videos:
        mo.output.replace(mo.md("_No videos yet. Run `modal run modal_analyze_video.py --video-path <path>`_"))
    else:
        _rows = []
        for _v in videos:
            _rows.append({
                "video": _v["video"][:40],
                "dur (s)": _v["n_timepoints"],
                "attn_3s": round(_v["attention_first3s"], 4),
                "attn_mean": round(_v["attention_mean"], 4),
                "attn_peak": round(_v["attention_peak"], 4),
                "attn_slope": round(_v["attention_slope"], 5),
                "cog_load": round(_v["cognitive_load_mean"], 4),
                "engage": round(_v["engagement"], 4),
                "salience": round(_v["salience_mean"], 4),
                "vis": round(_v["net_Vis"], 4),
            })
        _summary_df = pd.DataFrame(_rows)
        mo.vstack([
            mo.md("## Scalar metrics"),
            mo.ui.table(_summary_df),
        ])
    return


@app.cell
def _(mo, np, videos):
    """Attention curves overlay."""
    import matplotlib.pyplot as plt_ts

    if not videos:
        mo.output.replace(mo.md(""))
    else:
        _colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
                    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]

        fig_ts, _axes = plt_ts.subplots(2, 2, figsize=(16, 10))

        # --- Top-left: Attention (DAN - DMN) ---
        _ax = _axes[0][0]
        for _i, _v in enumerate(videos):
            _ts = _v["timeseries"]
            _attn = np.array(_ts["DAN"]) - np.array(_ts["DMN"])
            _t = np.arange(len(_attn))
            _c = _colors[_i % len(_colors)]
            _ax.plot(_t, _attn, "-", color=_c, linewidth=2, label=_v["video"][:30])
            _ax.scatter([1.5], [_v["attention_first3s"]], color=_c, s=80, zorder=5, marker="D")
        _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax.axvspan(0, 3, alpha=0.08, color="#ef4444")
        _ax.set_title("Attention (DAN − DMN)", fontweight="bold")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Activation")
        _ax.legend(fontsize=7, loc="upper right")
        _ax.grid(alpha=0.2)

        # --- Top-right: Cognitive Load (FPN + DAN - DMN) ---
        _ax = _axes[0][1]
        for _i, _v in enumerate(videos):
            _ts = _v["timeseries"]
            _cl = np.array(_ts["FPN"]) + np.array(_ts["DAN"]) - np.array(_ts["DMN"])
            _t = np.arange(len(_cl))
            _c = _colors[_i % len(_colors)]
            _ax.plot(_t, _cl, "-", color=_c, linewidth=2, label=_v["video"][:30])
        _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax.set_title("Cognitive Load (FPN + DAN − DMN)", fontweight="bold")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Activation")
        _ax.legend(fontsize=7, loc="upper right")
        _ax.grid(alpha=0.2)

        # --- Bottom-left: DMN (mind wandering, lower = more engaged) ---
        _ax = _axes[1][0]
        for _i, _v in enumerate(videos):
            _ts = _v["timeseries"]
            _dmn = np.array(_ts["DMN"])
            _t = np.arange(len(_dmn))
            _c = _colors[_i % len(_colors)]
            _ax.plot(_t, _dmn, "-", color=_c, linewidth=2, label=_v["video"][:30])
        _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax.set_title("DMN (lower = more engaged)", fontweight="bold")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Activation")
        _ax.legend(fontsize=7, loc="upper right")
        _ax.grid(alpha=0.2)

        # --- Bottom-right: Visual cortex ---
        _ax = _axes[1][1]
        for _i, _v in enumerate(videos):
            _ts = _v["timeseries"]
            _vis = np.array(_ts["Vis"])
            _t = np.arange(len(_vis))
            _c = _colors[_i % len(_colors)]
            _ax.plot(_t, _vis, "-", color=_c, linewidth=2, label=_v["video"][:30])
        _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax.set_title("Visual Cortex", fontweight="bold")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Activation")
        _ax.legend(fontsize=7, loc="upper right")
        _ax.grid(alpha=0.2)

        fig_ts.suptitle("Brain Metric Timeseries", fontsize=14, fontweight="bold")
        fig_ts.tight_layout()
        plt_ts.close("all")
        mo.output.replace(mo.vstack([mo.md("## Timeseries comparison"), fig_ts]))
    return


@app.cell
def _(mo, np, videos):
    """Per-video heatmaps side by side."""
    import matplotlib.pyplot as plt_hm

    if not videos:
        mo.output.replace(mo.md(""))
    else:
        _n = len(videos)
        fig_hm, _axes = plt_hm.subplots(_n, 1, figsize=(14, 2.5 * _n + 1), squeeze=False)
        _net_order = ["DAN", "VAN", "FPN", "Vis", "SomMot", "DMN", "Limbic"]

        _vmax = max(
            np.abs(np.array([_v["timeseries"][_net] for _net in _net_order])).max()
            for _v in videos
        )

        for _i, _v in enumerate(videos):
            _ax = _axes[_i][0]
            _data = np.array([_v["timeseries"][_net] for _net in _net_order])
            _im = _ax.imshow(_data, aspect="auto", cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            _ax.set_yticks(range(len(_net_order)))
            _ax.set_yticklabels(_net_order, fontsize=9)
            _ax.set_title(_v["video"][:50], fontsize=10, fontweight="bold")
            if _i == _n - 1:
                _ax.set_xlabel("Time (s)")
            fig_hm.colorbar(_im, ax=_ax, shrink=0.8)

        fig_hm.suptitle("Network Heatmaps", fontsize=13, fontweight="bold")
        fig_hm.tight_layout()
        plt_hm.close("all")
        mo.output.replace(mo.vstack([mo.md("## Network heatmaps"), fig_hm]))
    return


@app.cell
def _(Path, mo, np, videos):
    """Per-video timeline: video frames + brain metrics + heatmap."""
    import matplotlib.pyplot as plt_tl
    import matplotlib.image as mpimg
    from PIL import Image as PILImage
    import io

    _results_dir = Path("./cache/analyze_results")
    _figs = []

    for _v in videos:
        _stem = Path(_v["video"]).stem
        _frames_dir = _results_dir / f"{_stem}_frames"
        _ts = _v["timeseries"]
        _n_tp = _v["n_timepoints"]
        _dan = np.array(_ts["DAN"]); _dmn = np.array(_ts["DMN"])
        _fpn = np.array(_ts["FPN"]); _van = np.array(_ts["VAN"])
        _vis = np.array(_ts["Vis"]); _limbic = np.array(_ts["Limbic"])
        _attn = _dan - _dmn
        _cog = _fpn + _dan - _dmn
        _reward = 0.4 * (-_dmn) + 0.3 * _attn + 0.2 * _van + 0.1 * _limbic
        _t = np.arange(_n_tp)

        _has_frames = _frames_dir.exists() and len(list(_frames_dir.glob("*.jpg"))) > 0
        _n_rows = 3 if _has_frames else 2
        _ratios = [1, 1.5, 0.8] if _has_frames else [1.5, 0.8]

        _fig = plt_tl.figure(figsize=(max(14, _n_tp * 0.7), 2.5 * _n_rows))
        _gs = _fig.add_gridspec(_n_rows, 1, height_ratios=_ratios, hspace=0.25)
        _row = 0

        # Row: Video frames
        if _has_frames:
            _ax_f = _fig.add_subplot(_gs[_row])
            _frame_imgs = []
            for _i in range(_n_tp):
                _fp = _frames_dir / f"{_i:03d}.jpg"
                if _fp.exists():
                    _img = np.array(PILImage.open(str(_fp)))
                    _frame_imgs.append(_img)
            if _frame_imgs:
                _h = max(_f.shape[0] for _f in _frame_imgs)
                _w = max(_f.shape[1] for _f in _frame_imgs)
                _padded = []
                for _f in _frame_imgs:
                    _p = np.zeros((_h, _w, 3), dtype=np.uint8)
                    _p[:_f.shape[0], :_f.shape[1]] = _f
                    _padded.append(_p)
                _strip = np.concatenate(_padded, axis=1)
                _ax_f.imshow(_strip)
                for _i in range(len(_frame_imgs)):
                    _ax_f.axvline(_i * _w, color="white", linewidth=0.3, alpha=0.3)
                    _ax_f.text(_i * _w + _w // 2, 4, f"{_i}s", ha="center", va="top",
                               fontsize=6, color="white", fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
                _ax_f.set_xlim(0, _strip.shape[1])
            _ax_f.axis("off")
            _row += 1

        # Row: Brain metrics
        _ax_b = _fig.add_subplot(_gs[_row])
        _ax_b.plot(_t, _attn, "-", label="Attention (DAN−DMN)", color="#ef4444", linewidth=2.5)
        _ax_b.plot(_t, -_dmn, "-", label="Engagement (−DMN)", color="#3b82f6", linewidth=2.5)
        _ax_b.plot(_t, _cog, "-", label="Cog Load", color="#f59e0b", linewidth=2)
        _ax_b.plot(_t, _reward, "-", label="Brain Reward", color="#8b5cf6", linewidth=2)
        _ax_b.plot(_t, _dan, "--", color="#ef4444", linewidth=0.7, alpha=0.3, label="DAN")
        _ax_b.plot(_t, _dmn, "--", color="#3b82f6", linewidth=0.7, alpha=0.3, label="DMN")
        _ax_b.plot(_t, _van, "--", color="#a855f7", linewidth=0.7, alpha=0.3, label="VAN")
        _ax_b.plot(_t, _vis, "--", color="#22c55e", linewidth=0.7, alpha=0.3, label="Vis")
        _ax_b.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax_b.axvspan(0, min(3, _n_tp), alpha=0.06, color="#ef4444")
        _ax_b.set_xlim(0, _n_tp - 1)
        _ax_b.set_ylabel("Activation", fontsize=10)
        _ax_b.legend(loc="upper right", fontsize=7, ncol=4)
        _ax_b.grid(alpha=0.15)
        _ax_b.spines["top"].set_visible(False)
        _ax_b.spines["right"].set_visible(False)
        _row += 1

        # Row: Network heatmap
        _ax_h = _fig.add_subplot(_gs[_row])
        _net_order = ["DAN", "VAN", "FPN", "Vis", "SomMot", "DMN", "Limbic"]
        _hdata = np.array([_ts[_n] for _n in _net_order])
        _vabs = np.abs(_hdata).max()
        _im = _ax_h.imshow(_hdata, aspect="auto", cmap="RdBu_r", vmin=-_vabs, vmax=_vabs)
        _ax_h.set_yticks(range(len(_net_order)))
        _ax_h.set_yticklabels(_net_order, fontsize=8)
        _ax_h.set_xlabel("Time (s)", fontsize=10)
        _fig.colorbar(_im, ax=_ax_h, shrink=0.6, pad=0.02)

        _fig.suptitle(_v["video"][:60], fontsize=12, fontweight="bold")
        _fig.tight_layout()
        _figs.append(_fig)
        plt_tl.close("all")

    if not _figs:
        mo.output.replace(mo.md(""))
    else:
        mo.output.replace(mo.vstack([mo.md("## Video + Brain Timelines")] + _figs))
    return


@app.cell
def _(mo, np, videos):
    """Bar chart comparison of key scalars."""
    import matplotlib.pyplot as plt_bar

    if len(videos) < 2:
        mo.output.replace(mo.md("_Upload more videos to see comparison bars._"))
    else:
        _metrics = ["attention_first3s", "attention_mean", "attention_peak",
                     "cognitive_load_mean", "engagement", "salience_mean", "net_Vis"]
        _labels = ["Attn 3s", "Attn mean", "Attn peak", "Cog load", "Engage", "Salience", "Visual"]
        _colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
                    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]

        _x = np.arange(len(_metrics))
        _width = 0.8 / len(videos)

        fig_bar, _ax = plt_bar.subplots(figsize=(12, 5))
        for _i, _v in enumerate(videos):
            _vals = [_v[_m] for _m in _metrics]
            _ax.bar(_x + _i * _width, _vals, _width, label=_v["video"][:30],
                    color=_colors[_i % len(_colors)], alpha=0.85)

        _ax.set_xticks(_x + _width * (len(videos) - 1) / 2)
        _ax.set_xticklabels(_labels, fontsize=10)
        _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        _ax.set_ylabel("Activation")
        _ax.set_title("Metric Comparison", fontweight="bold")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.2, axis="y")
        fig_bar.tight_layout()
        plt_bar.close("all")
        mo.output.replace(mo.vstack([mo.md("## Head-to-head"), fig_bar]))
    return


@app.cell
def _(Path, json, mo, np, pd, videos):
    """TikTok percentile context."""
    _tiktok_path = Path("./cache/tribe_tiktok_train_results.json")
    if not _tiktok_path.exists() or not videos:
        mo.output.replace(mo.md(""))
    else:
        _tk = json.loads(_tiktok_path.read_text())
        _tk_attn3s = []
        _tk_attn_mean = []
        for _r in _tk:
            if "timeseries" in _r:
                _dan = np.array(_r["timeseries"]["DAN"])
                _dmn = np.array(_r["timeseries"]["DMN"])
                _a = _dan - _dmn
                _tk_attn3s.append(float(np.mean(_a[:min(3, len(_a))])))
                _tk_attn_mean.append(float(np.mean(_a)))

        _rows = []
        for _v in videos:
            _pct_3s = sum(1 for _a in _tk_attn3s if _a < _v["attention_first3s"]) / len(_tk_attn3s) * 100
            _pct_mean = sum(1 for _a in _tk_attn_mean if _a < _v["attention_mean"]) / len(_tk_attn_mean) * 100
            _rows.append({
                "video": _v["video"][:40],
                "attn_3s": round(_v["attention_first3s"], 4),
                "pctl_3s": f"{_pct_3s:.0f}th",
                "attn_mean": round(_v["attention_mean"], 4),
                "pctl_mean": f"{_pct_mean:.0f}th",
            })

        _pct_df = pd.DataFrame(_rows)
        mo.vstack([
            mo.md(f"## vs TikTok (n={len(_tk_attn3s)})"),
            mo.ui.table(_pct_df),
            mo.md(f"TikTok attn_3s range: [{min(_tk_attn3s):.4f}, {max(_tk_attn3s):.4f}]"),
        ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
