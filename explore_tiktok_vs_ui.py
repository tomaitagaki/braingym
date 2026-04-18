import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # TikTok vs UI: Brain Predictions

    Comparing V-JEPA2 embeddings and TRIBE brain predictions for TikTok videos
    (designed for engagement) vs UI screen recordings (functional, mundane).

    **Hypothesis:** TikTok should show higher attention (DAN), more DMN suppression,
    more novelty detection (VAN), while UI should require more cognitive effort (FPN).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    from pathlib import Path

    return Path, json, mo, np


@app.cell
def _(Path, json):
    # Load results
    base = Path("cache/tiktok_vs_ui")
    tribe_data = json.loads((base / "phase2_tribe.json").read_text())
    jepa_data = json.loads((base / "phase1_jepa.json").read_text())

    results = tribe_data["results"]
    timeseries = tribe_data["timeseries"]
    return base, jepa_data, results, timeseries


@app.cell
def _(mo):
    mo.md("""
    ## Phase 1: V-JEPA2 Embedding Space
    """)
    return


@app.cell
def _(base, mo):
    pca_img = base / "phase1_jepa_pca.png"
    mo.image(src=pca_img, width=800)
    return


@app.cell
def _(jepa_data, mo):
    # JEPA embedding similarity table
    cat_sim = jepa_data.get("category_sim", {})

    rows = []
    for sim_key, sim_val in sorted(cat_sim.items()):
        parts = sim_key.split("_vs_")
        if len(parts) == 2:
            rows.append({"From": parts[0], "To": parts[1], "Cosine Sim": f"{sim_val:.4f}"})

    mo.md("### Embedding Similarity (Cosine)")
    return (rows,)


@app.cell
def _(mo, rows):
    mo.ui.table(rows, label="V-JEPA2 category-level cosine similarity")
    return


@app.cell
def _(mo):
    mo.md("""
    **Takeaway:** TikTok-to-UI similarity (0.37) is lower than within-category similarity
    for either group (~0.52 TikTok, ~0.77 UI). V-JEPA2 clearly differentiates the content types,
    but UI embeddings are not pathologically out-of-distribution (norms and Mahalanobis distances
    from the OOD experiment confirm this).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Phase 2: TRIBE Brain Predictions
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Cognitive Metrics Comparison
    """)
    return


@app.cell
def _(np, results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Category averages
    cats = {"tiktok": [], "ui": []}
    for _name, _r in results.items():
        cats[_r["category"]].append(_r)

    def cat_avg(cat, key):
        return float(np.mean([r[key] for r in cats[cat]]))

    metrics_to_show = [
        ("engagement", "Engagement\n(-DMN)"),
        ("attention", "Attention\n(DAN-DMN)"),
        ("salience", "Salience\n(VAN)"),
        ("cognitive_load", "Cognitive Load\n(FPN+DAN-DMN)"),
        ("emotional", "Emotional\n(Limbic)"),
        ("brain_reward", "Brain\nReward"),
    ]

    metric_keys = [m[0] for m in metrics_to_show]
    metric_labels = [m[1] for m in metrics_to_show]
    tiktok_vals = [cat_avg("tiktok", k) for k in metric_keys]
    ui_vals = [cat_avg("ui", k) for k in metric_keys]

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        name="TikTok", x=metric_labels, y=tiktok_vals,
        marker_color="#FF0050", text=[f"{v:+.4f}" for v in tiktok_vals],
        textposition="outside",
    ))
    fig_metrics.add_trace(go.Bar(
        name="UI", x=metric_labels, y=ui_vals,
        marker_color="#4CAF50", text=[f"{v:+.4f}" for v in ui_vals],
        textposition="outside",
    ))
    fig_metrics.update_layout(
        barmode="group", title="Cognitive Metrics: TikTok vs UI (Category Averages)",
        yaxis_title="Activation", height=450,
        legend=dict(x=0.01, y=0.99),
    )
    fig_metrics.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_metrics
    return cat_avg, go, make_subplots


@app.cell
def _(mo):
    mo.md("""
    ### Network Activations (7 Yeo Networks)
    """)
    return


@app.cell
def _(cat_avg, go):
    networks = ["DAN", "DMN", "VAN", "FPN", "Limbic", "Vis", "SomMot"]
    net_keys = [f"net_{n}" for n in networks]

    tiktok_net = [cat_avg("tiktok", k) for k in net_keys]
    ui_net = [cat_avg("ui", k) for k in net_keys]

    fig_nets = go.Figure()
    fig_nets.add_trace(go.Bar(
        name="TikTok", x=networks, y=tiktok_net,
        marker_color="#FF0050",
        text=[f"{v:+.4f}" for v in tiktok_net], textposition="outside",
    ))
    fig_nets.add_trace(go.Bar(
        name="UI", x=networks, y=ui_net,
        marker_color="#4CAF50",
        text=[f"{v:+.4f}" for v in ui_net], textposition="outside",
    ))
    fig_nets.update_layout(
        barmode="group", title="Network Activations: TikTok vs UI",
        yaxis_title="Mean Activation", height=420,
        legend=dict(x=0.01, y=0.99),
    )
    fig_nets.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_nets
    return


@app.cell
def _(mo):
    mo.md("""
    ### Hypothesis Check
    """)
    return


@app.cell
def _(cat_avg, mo):
    checks = [
        ("DAN higher for TikTok", "net_DAN", "tiktok", "ui", ">",
         "TikTok designed to capture top-down attention"),
        ("DMN lower for TikTok (more suppressed)", "net_DMN", "tiktok", "ui", "<",
         "TikTok suppresses mind-wandering"),
        ("VAN higher for TikTok (more novelty)", "net_VAN", "tiktok", "ui", ">",
         "TikTok has surprise/novelty elements"),
        ("FPN higher for UI (more effortful)", "net_FPN", "tiktok", "ui", "<",
         "UI requires active cognitive control"),
        ("Engagement higher for TikTok", "engagement", "tiktok", "ui", ">",
         "TikTok keeps attention locked in"),
        ("Brain reward higher for TikTok", "brain_reward", "tiktok", "ui", ">",
         "TikTok optimized for engagement reward"),
    ]

    check_rows = []
    n_pass = 0
    for desc, metric_key, cat_a, cat_b, direction, rationale in checks:
        val_a = cat_avg(cat_a, metric_key)
        val_b = cat_avg(cat_b, metric_key)
        diff = val_a - val_b
        passed = (diff > 0) if direction == ">" else (diff < 0)
        n_pass += int(passed)
        check_rows.append({
            "Hypothesis": desc,
            "TikTok": f"{val_a:+.4f}",
            "UI": f"{val_b:+.4f}",
            "Diff": f"{diff:+.4f}",
            "Result": "PASS" if passed else "FAIL",
            "Rationale": rationale,
        })

    mo.md(f"**Result: {n_pass}/{len(checks)} hypotheses confirmed**")
    return (check_rows,)


@app.cell
def _(check_rows, mo):
    mo.ui.table(check_rows, label="Hypothesis check: directional predictions from 7-network model")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Per-Video Results

    Note: `ui_recording_1` is only 2.6 seconds (3 TRIBE timepoints) -- too short
    for reliable dynamics. `ui_recording_2` (12s, 13 timepoints) is the better UI comparison.
    """)
    return


@app.cell
def _(mo, results):
    vid_rows = []
    for vid_name in sorted(results.keys(), key=lambda n: (0 if results[n]["category"] == "tiktok" else 1, n)):
        vid = results[vid_name]
        vid_rows.append({
            "Video": vid_name,
            "Category": vid["category"],
            "Duration": f"{vid['duration_sec']}s",
            "Engagement": f"{vid['engagement']:+.4f}",
            "Attention": f"{vid['attention']:+.4f}",
            "Salience": f"{vid['salience']:+.4f}",
            "Cog Load": f"{vid['cognitive_load']:+.4f}",
            "Reward": f"{vid['brain_reward']:+.4f}",
            "DAN": f"{vid['net_DAN']:+.4f}",
            "DMN": f"{vid['net_DMN']:+.4f}",
            "VAN": f"{vid['net_VAN']:+.4f}",
            "FPN": f"{vid['net_FPN']:+.4f}",
        })
    mo.ui.table(vid_rows, label="Per-video brain metrics")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Timeseries: Network Dynamics Over Time
    """)
    return


@app.cell
def _(mo):
    video_selector = mo.ui.dropdown(
        options=["all", "tiktok_1", "tiktok_2", "ui_recording_1", "ui_recording_2"],
        value="all",
        label="Select video",
    )
    video_selector
    return (video_selector,)


@app.cell
def _(go, make_subplots, np, timeseries, video_selector):
    net_colors = {
        "DAN": "#e63946", "DMN": "#457b9d", "VAN": "#f4a261",
        "FPN": "#2a9d8f", "Limbic": "#9b59b6",
    }

    selected = video_selector.value
    if selected == "all":
        vid_names = sorted(timeseries.keys(), key=lambda n: (0 if timeseries[n]["_category"] == "tiktok" else 1, n))
    else:
        vid_names = [selected]

    fig_ts = make_subplots(
        rows=len(vid_names), cols=1,
        subplot_titles=[f"{n} ({timeseries[n]['_category']})" for n in vid_names],
        vertical_spacing=0.08,
    )

    for row_idx, ts_name in enumerate(vid_names, 1):
        ts = timeseries[ts_name]
        ts_t = np.arange(len(ts["DAN"]))
        for ts_net in ["DAN", "DMN", "VAN", "FPN", "Limbic"]:
            fig_ts.add_trace(
                go.Scatter(
                    x=ts_t, y=ts[ts_net], name=ts_net,
                    line=dict(color=net_colors[ts_net], width=2),
                    showlegend=(row_idx == 1),
                    legendgroup=ts_net,
                ),
                row=row_idx, col=1,
            )
        fig_ts.add_hline(y=0, line_dash="dot", line_color="gray",
                         row=row_idx, col=1)
        fig_ts.update_yaxes(title_text="Activation", row=row_idx, col=1)
        fig_ts.update_xaxes(title_text="Time (seconds)", row=row_idx, col=1)

    fig_ts.update_layout(
        height=300 * len(vid_names),
        title="Network Timeseries by Video",
        legend=dict(x=1.02, y=1),
    )
    fig_ts
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    **The big signal is DAN (Dorsal Attention), not DMN.**
    TikTok captures 4x more top-down attention than UI (DAN diff = +0.096).
    DMN suppression is nearly identical between the two (diff = 0.002) --
    both content types suppress mind-wandering, but at very different attentional levels.

    **FPN tells the UI story.**
    FPN (frontoparietal control) is positive for UI (+0.031) and negative for TikTok (-0.009).
    This is the cognitive effort signature -- UI requires active control and navigation,
    TikTok is effortless consumption.

    **All 6 directional hypotheses pass.**
    TRIBE produces interpretable, neuroscientifically-grounded predictions for UI content
    that go in the expected direction. Combined with the OOD analysis (UI embeddings have
    normal norms, in-range Mahalanobis distance), this supports the argument that
    **relative TRIBE comparisons between UI variations could be meaningful.**

    ### Caveats
    - n=2 per category -- directional, not statistical
    - `ui_recording_1` is too short (2.6s) for reliable TRIBE dynamics
    - TRIBE was trained on naturalistic video, not UI -- absolute magnitudes may be miscalibrated
    - The relative ordering and directions are the signal, not the raw numbers
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Raw Data: Phase 1 (JEPA) Similarity
    """)
    return


@app.cell
def _(jepa_data, mo):
    inter = jepa_data.get("inter_video_sim", {})
    inter_rows = [
        {"Pair": k, "Cosine Similarity": f"{v:.4f}"}
        for k, v in sorted(inter.items(), key=lambda x: -x[1])
    ]
    mo.ui.table(inter_rows, label="V-JEPA2 inter-video cosine similarity (mean embeddings)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
