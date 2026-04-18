import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.size": 11})
    return json, mo, np, plt


@app.cell
def _(json):
    with open("cache/ddpo_results/faces/training_log.json") as f:
        ddpo_data = json.load(f)
    with open("cache/manim_experiment/jepa_vs_tribe_results.json") as f:
        style_data = json.load(f)
    with open("cache/manim_optimize/optimization_log.json") as f:
        opt_data = json.load(f)
    return ddpo_data, opt_data, style_data


@app.cell
def _(mo):
    mo.md("""
    # Brain-Reward Optimization: Full Results

    Three experiments testing whether Tribe (fMRI encoding model) can serve as a reward signal
    for content generation optimization.

    | Experiment | Generator | Optimizer | Result |
    |---|---|---|---|
    | **DDPO Faces** | SD 1.5 (static images) | DDPO (RL fine-tuning) | Flat — noise floor |
    | **Manim Styles** | 5 hand-crafted variations | None (comparison only) | Meaningful variation |
    | **Manim GRPO** | Parameterized Manim | GRPO (10 generations) | +27% reward improvement |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 1: DDPO + SD 1.5 (Static Images)

    Fine-tuned SD 1.5 LoRA with Tribe brain predictions as reward. 16 epochs, 8 images/epoch,
    137 total face images on Modal A100.

    **Result:** Reward flat at noise floor (~0.03). Static images are out-of-distribution for Tribe
    (trained on video/audio). 2 of 3 encoders unused.
    """)
    return


@app.cell
def _(ddpo_data, mo, np, plt):
    _fig_ddpo, _axes_ddpo = plt.subplots(1, 3, figsize=(15, 4))

    _epochs = [r["epoch"] for r in ddpo_data]
    _means = [r["reward_mean"] for r in ddpo_data]
    _stds = [r["reward_std"] for r in ddpo_data]
    _maxes = [r["reward_max"] for r in ddpo_data]

    _ax = _axes_ddpo[0]
    _ax.fill_between(_epochs, np.array(_means) - np.array(_stds),
                     np.array(_means) + np.array(_stds), alpha=0.2, color="#4a9eed")
    _ax.plot(_epochs, _means, "o-", color="#4a9eed", label="Mean +/- std", markersize=4)
    _ax.plot(_epochs, _maxes, "s--", color="#22c55e", label="Best", markersize=4, alpha=0.7)
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Reward")
    _ax.set_title("DDPO Reward (Static Images)")
    _ax.legend(fontsize=8)
    _ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    _ax = _axes_ddpo[1]
    _best_ep = max(ddpo_data, key=lambda r: r["reward_max"])
    _best_i = _best_ep["rewards"].index(max(_best_ep["rewards"]))
    _m = _best_ep["all_metrics"][_best_i]
    _nets = ["net_DAN", "net_DMN", "net_VAN", "net_FPN", "net_Vis", "net_Limbic"]
    _net_labels = ["DAN", "DMN", "VAN", "FPN", "Vis", "Limbic"]
    _net_vals = [_m[n] for n in _nets]
    _bar_colors = ["#22c55e" if v > 0 else "#ef4444" for v in _net_vals]
    _ax.barh(_net_labels, _net_vals, color=_bar_colors)
    _ax.set_xlabel("Mean Activation")
    _ax.set_title(f"Best Image Networks (e{_best_ep['epoch']:03d})")
    _ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

    _ax = _axes_ddpo[2]
    _ddpo_best = max(r["reward_max"] for r in ddpo_data)
    _ddpo_mean = np.mean(_means)
    _manim_mean = 0.100
    _manim_best = 0.135
    _cat_x = np.arange(2)
    _cat_w = 0.35
    _ax.bar(_cat_x - _cat_w / 2, [_ddpo_mean, _manim_mean], _cat_w, label="Mean", color="#4a9eed")
    _ax.bar(_cat_x + _cat_w / 2, [_ddpo_best, _manim_best], _cat_w, label="Best", color="#22c55e")
    _ax.set_xticks(_cat_x)
    _ax.set_xticklabels(["DDPO\n(images)", "Manim\n(video)"])
    _ax.set_ylabel("Reward")
    _ax.set_title("Static Image vs Video")
    _ax.legend(fontsize=8)

    plt.tight_layout()
    mo.output.replace(_fig_ddpo)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 2: Manim Style Comparison (5 Presentation Styles)

    Same concept (Pythagorean theorem), 5 different presentation styles scored by Tribe.

    **Result:** Reward range of 0.057 (V2Fast: +0.135 vs V4TextHeavy: +0.077).
    V3Visual has the best hook AND the best sustained engagement.
    V2Fast grabs attention but drops it.
    """)
    return


@app.cell
def _(mo, np, plt, style_data):
    _scenes = style_data["scene_names"]
    _results = style_data["results"]
    _scene_colors = ["#4a9eed", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]

    _fig_style, _axes_style = plt.subplots(1, 3, figsize=(15, 4.5))

    # Reward comparison
    _ax = _axes_style[0]
    _rewards = [_results[s]["reward"] for s in _scenes]
    _r_colors = ["#22c55e" if r == max(_rewards) else "#ef4444" if r == min(_rewards) else "#4a9eed" for r in _rewards]
    _bars = _ax.bar(range(len(_scenes)), _rewards, color=_r_colors)
    _ax.set_xticks(range(len(_scenes)))
    _ax.set_xticklabels(_scenes, rotation=20, ha="right", fontsize=9)
    _ax.set_ylabel("Composite Reward")
    _ax.set_title("Reward by Style")
    for _bar, _r in zip(_bars, _rewards):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.002,
                f"{_r:.3f}", ha="center", fontsize=8)

    # Network profiles
    _ax = _axes_style[1]
    _nn = ["DAN", "DMN", "VAN", "FPN", "Vis", "Limbic"]
    _nx = np.arange(len(_nn))
    _nw = 0.15
    for _i, _scene in enumerate(_scenes):
        _nv = [_results[_scene][f"net_{n}"] for n in _nn]
        _ax.bar(_nx + _i * _nw, _nv, _nw, label=_scene, color=_scene_colors[_i], alpha=0.8)
    _ax.set_xticks(_nx + _nw * 2)
    _ax.set_xticklabels(_nn)
    _ax.set_ylabel("Activation")
    _ax.set_title("Network Profiles")
    _ax.legend(fontsize=7, loc="lower left")
    _ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Hook vs sustained scatter
    _ax = _axes_style[2]
    for _i, _scene in enumerate(_scenes):
        _ts = _results[_scene]["timeseries"]
        _dan = np.array(_ts["DAN"])
        _dmn = np.array(_ts["DMN"])
        _attn = _dan - _dmn
        _n3 = min(3, len(_attn))
        _hook = float(np.mean(_attn[:_n3]))
        _sust = float(-np.std(_dmn))
        _ax.scatter(_hook, _sust, s=100, color=_scene_colors[_i], label=_scene, zorder=5)
        _ax.annotate(_scene, (_hook, _sust), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")
    _ax.set_xlabel("Hook (attention in first 3s)")
    _ax.set_ylabel("Sustained (-DMN variance)")
    _ax.set_title("Hook vs Sustained Engagement")
    _ax.legend(fontsize=7)

    plt.tight_layout()
    mo.output.replace(_fig_style)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 3: GRPO Closed-Loop Optimization

    GRPO tunes 7 Manim presentation parameters using Tribe reward signal.
    10 generations x 4 samples = 40 videos generated and brain-scored.

    **Result:** +27% improvement in mean reward. Sustained engagement improved 2.4x.
    Converged on: fast pacing, skip the hook, sequential reveal, dual colors, minimal visuals.
    """)
    return


@app.cell
def _(mo, np, opt_data, plt):
    _fig_opt, _axes_opt = plt.subplots(2, 2, figsize=(14, 10))

    _gens = [r["generation"] for r in opt_data]
    _o_means = [r["reward_mean"] for r in opt_data]
    _o_stds = [r["reward_std"] for r in opt_data]
    _o_maxes = [r["reward_max"] for r in opt_data]
    _o_mins = [r["reward_min"] for r in opt_data]

    # (A) Reward convergence
    _ax = _axes_opt[0, 0]
    _ax.fill_between(_gens, _o_mins, _o_maxes, alpha=0.15, color="#4a9eed", label="Min-Max range")
    _ax.fill_between(_gens, np.array(_o_means) - np.array(_o_stds),
                     np.array(_o_means) + np.array(_o_stds), alpha=0.3, color="#4a9eed")
    _ax.plot(_gens, _o_means, "o-", color="#4a9eed", label="Mean +/- std", markersize=6)
    _ax.plot(_gens, _o_maxes, "s--", color="#22c55e", label="Best", markersize=5, alpha=0.7)
    _ax.axhline(_o_means[0], color="gray", linestyle=":", alpha=0.5, label=f"Baseline ({_o_means[0]:.3f})")
    _ax.set_xlabel("Generation")
    _ax.set_ylabel("Reward")
    _ax.set_title("GRPO Reward Convergence")
    _ax.legend(fontsize=8)

    # (B) Per-metric improvement
    _ax = _axes_opt[0, 1]
    _bm = opt_data[0]["best_metrics"]
    _fm = opt_data[-1]["best_metrics"]
    _mk = ["reward", "attention", "engagement", "cognitive_load", "hook_3s", "sustained"]
    _ml = ["Reward", "Attention", "Engagement", "Cog Load", "Hook (3s)", "Sustained"]
    _bv = [_bm[k] for k in _mk]
    _fv = [_fm[k] for k in _mk]
    _mx = np.arange(len(_mk))
    _mw = 0.35
    _ax.bar(_mx - _mw / 2, _bv, _mw, label="Baseline (gen 0)", color="#94a3b8")
    _ax.bar(_mx + _mw / 2, _fv, _mw, label="Optimized (gen 9)", color="#22c55e")
    _ax.set_xticks(_mx)
    _ax.set_xticklabels(_ml, rotation=25, ha="right", fontsize=9)
    _ax.set_ylabel("Value")
    _ax.set_title("Baseline vs Optimized Metrics")
    _ax.legend(fontsize=8)
    _ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

    # (C) Continuous param convergence
    _ax = _axes_opt[1, 0]
    _cp = ["pacing", "pause_duration", "visual_density", "text_density"]
    _pc = ["#4a9eed", "#22c55e", "#f59e0b", "#ef4444"]
    for _param, _color in zip(_cp, _pc):
        _mus = [r["distributions"][_param]["mu"] for r in opt_data]
        _sigs = [r["distributions"][_param]["sigma"] for r in opt_data]
        _ax.plot(_gens, _mus, "o-", color=_color, label=_param, markersize=4)
        _ax.fill_between(_gens, np.array(_mus) - np.array(_sigs),
                         np.array(_mus) + np.array(_sigs), alpha=0.1, color=_color)
    _ax.set_xlabel("Generation")
    _ax.set_ylabel("Parameter Value")
    _ax.set_title("Continuous Param Convergence")
    _ax.legend(fontsize=8)

    # (D) Categorical param convergence — one subplot per parameter
    _ax = _axes_opt[1, 1]
    _cat_params = {
        "reveal_style": {
            "choices": ["sequential", "simultaneous", "progressive"],
            "colors": ["#22c55e", "#94a3b8", "#d4d4d4"],
        },
        "color_scheme": {
            "choices": ["mono", "dual", "vibrant"],
            "colors": ["#d4d4d4", "#4a9eed", "#94a3b8"],
        },
        "narrative_hook": {
            "choices": ["none", "question", "story", "surprise"],
            "colors": ["#f59e0b", "#d4d4d4", "#d4d4d4", "#d4d4d4"],
        },
    }
    _styles = ["-", "--", ":"]
    for _pi, (_pn, _spec) in enumerate(_cat_params.items()):
        for _ci, _ch in enumerate(_spec["choices"]):
            _probs = [r["distributions"][_pn]["probs"][_ch] for r in opt_data]
            _is_winner = _probs[-1] > 0.5
            _ax.plot(
                _gens, _probs,
                linestyle=_styles[_pi],
                color=_spec["colors"][_ci],
                markersize=0,
                linewidth=2.5 if _is_winner else 1,
                alpha=1.0 if _is_winner else 0.3,
                label=f"{_ch}" if _is_winner else None,
            )
    _ax.set_xlabel("Generation")
    _ax.set_ylabel("Probability")
    _ax.set_title("Categorical Winners")
    _ax.set_ylim(-0.05, 1.05)
    _ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    _ax.legend(fontsize=9, title="Converged to:", title_fontsize=9, loc="center right")

    plt.tight_layout()
    mo.output.replace(_fig_opt)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 3b: All Samples + Improvement Breakdown
    """)
    return


@app.cell
def _(mo, opt_data, plt):
    _fig_ts, _axes_ts = plt.subplots(1, 2, figsize=(14, 4))

    # All individual sample rewards
    _ax = _axes_ts[0]
    for _gi, _r in enumerate(opt_data):
        for _rw in _r["all_rewards"]:
            _ax.scatter(_gi, _rw, s=20, color="#4a9eed", alpha=0.4, zorder=3)
    _ml2 = [r["reward_mean"] for r in opt_data]
    _ax.plot(range(len(opt_data)), _ml2, "o-", color="#ef4444", markersize=5, linewidth=2,
            label="Generation mean", zorder=5)
    _ax.set_xlabel("Generation")
    _ax.set_ylabel("Reward")
    _ax.set_title("All Samples (dots) + Generation Mean (line)")
    _ax.legend(fontsize=8)

    # Improvement waterfall
    _ax = _axes_ts[1]
    _bm2 = opt_data[0]["best_metrics"]
    _fm2 = opt_data[-1]["best_metrics"]
    _mk2 = ["attention", "engagement", "cognitive_load", "hook_3s", "sustained"]
    _ml3 = ["Attention", "Engagement", "Cog Load", "Hook 3s", "Sustained"]
    _pct = [
        (_fm2[k] - _bm2[k]) / abs(_bm2[k]) * 100 if _bm2[k] != 0 else 0
        for k in _mk2
    ]
    _wc = ["#22c55e" if v > 0 else "#ef4444" for v in _pct]
    _wbars = _ax.bar(range(len(_mk2)), _pct, color=_wc)
    _ax.set_xticks(range(len(_mk2)))
    _ax.set_xticklabels(_ml3, rotation=20, ha="right", fontsize=9)
    _ax.set_ylabel("% Improvement")
    _ax.set_title("Optimized vs Baseline (% change)")
    _ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    for _wb, _p in zip(_wbars, _pct):
        _ax.text(_wb.get_x() + _wb.get_width() / 2,
                _wb.get_height() + (2 if _p > 0 else -5),
                f"{_p:+.0f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    mo.output.replace(_fig_ts)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    | | DDPO (Static Images) | Manim GRPO (Video) |
    |---|---|---|
    | **Generator** | SD 1.5 + LoRA | Parameterized Manim |
    | **Optimizer** | DDPO (policy gradient) | GRPO (distribution-based) |
    | **Tribe encoders active** | 1/3 (V-JEPA2 only) | 1/3 (V-JEPA2 only) |
    | **Baseline reward** | +0.035 | +0.111 |
    | **Optimized reward** | +0.030 (declined) | +0.140 (+27%) |
    | **Signal quality** | Noise floor | Meaningful |
    | **Convergence** | No | Yes |
    | **Interpretation** | Texture detector | Presentation optimizer |

    **Key insight:** Temporal dynamics (video) produce 3x stronger Tribe activations than
    static images, and the variation across presentation styles is large enough for GRPO
    to exploit. The closed loop works for video content.

    **Open question:** Does Tribe-optimized content actually engage real humans more?
    The optimization found that "fast pacing, skip the hook, jump to equations" maximizes
    predicted brain engagement. But Tribe is a model trained on fMRI data from movie-watching --
    does its notion of "engagement" transfer to educational content? Needs human validation.

    **Converged "optimal" Manim style:**
    - Pacing: 0.88 (fast)
    - Pause: 0.97s (short)
    - Visual density: 0.12 (minimal geometry)
    - Text density: 0.51 (medium equations)
    - Reveal: sequential (98%)
    - Colors: dual (99%)
    - Hook: none (99%) -- jump straight in
    """)
    return


if __name__ == "__main__":
    app.run()
