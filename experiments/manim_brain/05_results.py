import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # Brain-Reward Optimization: Full Results

    Three experiments testing whether Tribe (fMRI encoding model) can serve as a reward signal
    for content generation optimization.

    | Experiment | Generator | Optimizer | Result |
    |---|---|---|---|
    | **DDPO Faces** | SD 1.5 (static images) | DDPO (RL fine-tuning) | ❌ Flat — noise floor |
    | **Manim Styles** | 5 hand-crafted variations | None (comparison only) | ✅ Meaningful variation |
    | **Manim GRPO** | Parameterized Manim | GRPO (10 generations) | ✅ +27% reward improvement |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    return json, mo, np


@app.cell
def _(json):
    # Load all data
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
    ## Experiment 1: DDPO + SD 1.5 (Static Images → Tribe)

    Fine-tuned SD 1.5 LoRA with Tribe brain predictions as reward. 16 epochs, 8 images/epoch,
    137 total face images on Modal A100.

    **Hypothesis:** DDPO can learn to generate faces that maximize predicted brain engagement.

    **Result:** Reward flat at noise floor (~0.03). Static images are out-of-distribution for Tribe
    (trained on video/audio). 2 of 3 encoders unused. The LoRA learned to generate high-frequency
    textures, not compelling portraits.
    """)
    return


@app.cell
def _(ddpo_data, mo, np, plt):
    fig_ddpo, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = [r["epoch"] for r in ddpo_data]
    means = [r["reward_mean"] for r in ddpo_data]
    stds = [r["reward_std"] for r in ddpo_data]
    maxes = [r["reward_max"] for r in ddpo_data]

    # Reward over epochs
    ax = axes[0]
    ax.fill_between(epochs, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.2, color="#4a9eed")
    ax.plot(epochs, means, "o-", color="#4a9eed", label="Mean ± std", markersize=4)
    ax.plot(epochs, maxes, "s--", color="#22c55e", label="Best", markersize=4, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("DDPO Reward (Static Images)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Network activations for best image
    ax = axes[1]
    best_epoch = max(ddpo_data, key=lambda r: r["reward_max"])
    best_idx = best_epoch["rewards"].index(max(best_epoch["rewards"]))
    m = best_epoch["all_metrics"][best_idx]
    nets = ["net_DAN", "net_DMN", "net_VAN", "net_FPN", "net_Vis", "net_Limbic"]
    labels = ["DAN", "DMN", "VAN", "FPN", "Vis", "Limbic"]
    vals = [m[n] for n in nets]
    colors_bar = ["#22c55e" if v > 0 else "#ef4444" for v in vals]
    ax.barh(labels, vals, color=colors_bar)
    ax.set_xlabel("Mean Activation")
    ax.set_title(f"Best Image Networks (e{best_epoch['epoch']:03d})")
    ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

    # Comparison: DDPO vs Manim activations
    ax = axes[2]
    ddpo_best = max(r["reward_max"] for r in ddpo_data)
    ddpo_mean = np.mean(means)
    categories = ["DDPO\n(images)", "Manim\n(video)"]
    ddpo_vals = [ddpo_mean, ddpo_best]
    # Manim values from style comparison
    manim_mean = 0.100  # approximate from style data
    manim_best = 0.135
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [ddpo_mean, manim_mean], width, label="Mean", color="#4a9eed")
    ax.bar(x + width / 2, [ddpo_best, manim_best], width, label="Best", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Reward")
    ax.set_title("Static Image vs Video")
    ax.legend(fontsize=8)

    plt.tight_layout()
    mo.output.replace(fig_ddpo)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.size": 11})
    return matplotlib, plt


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 2: Manim Style Comparison (5 Presentation Styles)

    Same concept (Pythagorean theorem), 5 different presentation styles scored by Tribe.

    **Hypothesis:** Different presentation styles produce meaningfully different brain predictions.

    **Result:** Yes — reward range of 0.057 (V2Fast: +0.135 vs V4TextHeavy: +0.077).
    Temporal dynamics reveal even more: V3Visual has the best hook AND the best sustained engagement,
    while V2Fast grabs attention but drops it.
    """)
    return


@app.cell
def _(mo, np, plt, style_data):
    scenes = style_data["scene_names"]
    results = style_data["results"]

    fig_style, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Reward comparison
    ax = axes[0]
    rewards = [results[s]["reward"] for s in scenes]
    colors_r = ["#22c55e" if r == max(rewards) else "#ef4444" if r == min(rewards) else "#4a9eed" for r in rewards]
    bars = ax.bar(range(len(scenes)), rewards, color=colors_r)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([s.replace("V", "V") for s in scenes], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Composite Reward")
    ax.set_title("Reward by Style")
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{r:.3f}", ha="center", fontsize=8)

    # Network profiles
    ax = axes[1]
    net_names = ["DAN", "DMN", "VAN", "FPN", "Vis", "Limbic"]
    x = np.arange(len(net_names))
    width = 0.15
    scene_colors = ["#4a9eed", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]
    for i, scene in enumerate(scenes):
        vals = [results[scene][f"net_{n}"] for n in net_names]
        ax.bar(x + i * width, vals, width, label=scene, color=scene_colors[i], alpha=0.8)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(net_names)
    ax.set_ylabel("Activation")
    ax.set_title("Network Profiles")
    ax.legend(fontsize=7, loc="lower left")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Temporal dynamics: hook vs sustained
    ax = axes[2]
    for i, scene in enumerate(scenes):
        ts = results[scene]["timeseries"]
        dan = np.array(ts["DAN"])
        dmn = np.array(ts["DMN"])
        attn = dan - dmn
        n3 = min(3, len(attn))
        hook = float(np.mean(attn[:n3]))
        sustained = float(-np.std(dmn))
        ax.scatter(hook, sustained, s=100, color=scene_colors[i], label=scene, zorder=5)
        ax.annotate(scene, (hook, sustained), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Hook (attention in first 3s)")
    ax.set_ylabel("Sustained (-DMN variance)")
    ax.set_title("Hook vs Sustained Engagement")
    ax.legend(fontsize=7)

    plt.tight_layout()
    mo.output.replace(fig_style)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 3: GRPO Closed-Loop Optimization

    GRPO tunes 7 Manim presentation parameters using Tribe reward signal.
    10 generations × 4 samples = 40 videos generated and brain-scored.

    **Hypothesis:** GRPO can optimize presentation style to maximize predicted brain engagement.

    **Result:** +27% improvement in mean reward. Sustained engagement improved 2.4×.
    GRPO converged on: fast pacing, skip the hook, sequential reveal, dual colors, minimal visuals,
    medium text density.
    """)
    return


@app.cell
def _(mo, np, opt_data, plt):
    fig_opt, axes = plt.subplots(2, 2, figsize=(14, 10))

    gens = [r["generation"] for r in opt_data]
    means = [r["reward_mean"] for r in opt_data]
    stds = [r["reward_std"] for r in opt_data]
    maxes = [r["reward_max"] for r in opt_data]
    mins = [r["reward_min"] for r in opt_data]

    # (A) Reward convergence
    ax = axes[0, 0]
    ax.fill_between(gens, mins, maxes, alpha=0.15, color="#4a9eed", label="Min–Max range")
    ax.fill_between(gens, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.3, color="#4a9eed")
    ax.plot(gens, means, "o-", color="#4a9eed", label="Mean ± std", markersize=6)
    ax.plot(gens, maxes, "s--", color="#22c55e", label="Best", markersize=5, alpha=0.7)
    ax.axhline(means[0], color="gray", linestyle=":", alpha=0.5, label=f"Baseline ({means[0]:.3f})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Reward Convergence")
    ax.legend(fontsize=8)

    # (B) Per-metric improvement
    ax = axes[0, 1]
    baseline_m = opt_data[0]["best_metrics"]
    final_m = opt_data[-1]["best_metrics"]
    metric_names = ["reward", "attention", "engagement", "cognitive_load", "hook_3s", "sustained"]
    metric_labels = ["Reward", "Attention", "Engagement", "Cog Load", "Hook (3s)", "Sustained"]
    b_vals = [baseline_m[k] for k in metric_names]
    f_vals = [final_m[k] for k in metric_names]
    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width / 2, b_vals, width, label="Baseline (gen 0)", color="#94a3b8")
    ax.bar(x + width / 2, f_vals, width, label="Optimized (gen 9)", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("Baseline vs Optimized Metrics")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

    # (C) Continuous param convergence
    ax = axes[1, 0]
    cont_params = ["pacing", "pause_duration", "visual_density", "text_density"]
    param_colors = ["#4a9eed", "#22c55e", "#f59e0b", "#ef4444"]
    for param, color in zip(cont_params, param_colors):
        mus = [r["distributions"][param]["mu"] for r in opt_data]
        sigmas = [r["distributions"][param]["sigma"] for r in opt_data]
        ax.plot(gens, mus, "o-", color=color, label=param, markersize=4)
        ax.fill_between(gens, np.array(mus) - np.array(sigmas),
                         np.array(mus) + np.array(sigmas), alpha=0.1, color=color)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Parameter Value (μ ± σ)")
    ax.set_title("Continuous Param Convergence")
    ax.legend(fontsize=8)

    # (D) Categorical param convergence
    ax = axes[1, 1]
    cat_params = {
        "reveal_style": ["sequential", "simultaneous", "progressive"],
        "color_scheme": ["mono", "dual", "vibrant"],
        "narrative_hook": ["none", "question", "story", "surprise"],
    }
    y_offset = 0
    yticks = []
    ytick_labels = []
    for param_name, choices in cat_params.items():
        for choice in choices:
            probs = [r["distributions"][param_name]["probs"][choice] for r in opt_data]
            color = "#22c55e" if probs[-1] > 0.5 else "#94a3b8"
            ax.plot(gens, probs, "o-", color=color, markersize=3, alpha=0.8)
            ax.text(len(gens) - 1 + 0.3, probs[-1],
                    f"{choice} ({probs[-1]:.0%})", fontsize=7, va="center",
                    color=color, fontweight="bold" if probs[-1] > 0.5 else "normal")
        y_offset += 1
    ax.set_xlabel("Generation")
    ax.set_ylabel("Probability")
    ax.set_title("Categorical Param Convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    mo.output.replace(fig_opt)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 3b: Network Timeseries — Baseline vs Optimized

    The temporal profile matters more than the mean. Here we compare the DAN-DMN (attention)
    timeseries of the best baseline video vs the best optimized video.
    """)
    return


@app.cell
def _(mo, np, opt_data, plt):
    fig_ts, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Best baseline (gen 0) — find the best sample
    gen0 = opt_data[0]
    best_idx_0 = gen0["all_rewards"].index(max(gen0["all_rewards"]))

    # Best optimized (gen 9)
    gen9 = opt_data[-1]
    best_idx_9 = gen9["all_rewards"].index(max(gen9["all_rewards"]))

    # All individual sample rewards across generations
    ax = axes[0]
    for gen_idx, r in enumerate(opt_data):
        for reward in r["all_rewards"]:
            ax.scatter(gen_idx, reward, s=20, color="#4a9eed", alpha=0.4, zorder=3)
    means_line = [r["reward_mean"] for r in opt_data]
    ax.plot(range(len(opt_data)), means_line, "o-", color="#ef4444", markersize=5, linewidth=2,
            label="Generation mean", zorder=5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.set_title("All Samples (dots) + Generation Mean (line)")
    ax.legend(fontsize=8)

    # Improvement waterfall
    ax = axes[1]
    baseline_m = opt_data[0]["best_metrics"]
    final_m = opt_data[-1]["best_metrics"]
    metrics = ["attention", "engagement", "cognitive_load", "hook_3s", "sustained"]
    labels = ["Attention", "Engagement", "Cog Load", "Hook 3s", "Sustained"]
    improvements = [final_m[k] - baseline_m[k] for k in metrics]
    pct_improvements = [
        (final_m[k] - baseline_m[k]) / abs(baseline_m[k]) * 100 if baseline_m[k] != 0 else 0
        for k in metrics
    ]
    colors_w = ["#22c55e" if v > 0 else "#ef4444" for v in improvements]
    bars = ax.bar(range(len(metrics)), pct_improvements, color=colors_w)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% Improvement")
    ax.set_title("Optimized vs Baseline (% change)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    for bar, pct in zip(bars, pct_improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (2 if pct > 0 else -5),
                f"{pct:+.0f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    mo.output.replace(fig_ts)
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
    | **Convergence** | ❌ No | ✅ Yes |
    | **Interpretation** | Texture detector | Presentation optimizer |

    **Key insight:** Temporal dynamics (video) produce 3× stronger Tribe activations than
    static images, and the variation across presentation styles is large enough for GRPO
    to exploit. The closed loop works for video content.

    **Open question:** Does Tribe-optimized content actually engage real humans more?
    The optimization found that "fast pacing, skip the hook, jump to equations" maximizes
    predicted brain engagement. But Tribe is a model trained on fMRI data from movie-watching —
    does its notion of "engagement" transfer to educational content? Needs human validation.

    **Converged "optimal" Manim style:**
    - Pacing: 0.88 (fast)
    - Pause: 0.97s (short)
    - Visual density: 0.12 (minimal geometry)
    - Text density: 0.51 (medium equations)
    - Reveal: sequential (98%)
    - Colors: dual (99%)
    - Hook: none (99%) — jump straight in
    """)
    return


if __name__ == "__main__":
    app.run()
