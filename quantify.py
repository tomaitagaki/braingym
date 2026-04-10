"""
BrainGym: Quantify TRIBEv2 surface predictions into cognitive load metrics.

Stack: nibabel (parcellation) + brainiak (RSA, event segmentation) + numpy (ROI extraction)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve

CACHE_FOLDER = Path("./cache")
PARCELLATION_DIR = CACHE_FOLDER / "schaefer"
PARCELLATION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DOWNLOAD SCHAEFER 400-PARCEL 7-NETWORK ATLAS (fsaverage5)
# ============================================================

SCHAEFER_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label"
)
ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"

def fetch_schaefer_fsaverage5():
    """Download Schaefer 7-network parcellation for fsaverage5."""
    paths = {}
    for hemi in ("lh", "rh"):
        local = PARCELLATION_DIR / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            url = f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}"
            print(f"Downloading {hemi} parcellation...")
            urlretrieve(url, local)
        paths[hemi] = local
    return paths

annot_paths = fetch_schaefer_fsaverage5()

# ============================================================
# 2. PARSE PARCELLATION → NETWORK MASKS
# ============================================================

# Schaefer parcel names encode their network:
#   "7Networks_LH_DorsAttn_Post_1" → DorsAttn
#   "7Networks_RH_Default_PFC_2"   → Default

NETWORK_ALIASES = {
    "DorsAttn":  "DAN",       # dorsal attention → focused attention
    "Default":   "DMN",       # default mode → mind wandering
    "Cont":      "FPN",       # control / frontoparietal → cognitive control
    "SalVentAttn": "VAN",     # salience / ventral attention
    "Limbic":    "Limbic",    # limbic → emotion
    "SomMot":    "SomMot",    # somatomotor
    "Vis":       "Vis",       # visual
}


def parse_parcellation(annot_path):
    """Load .annot file, return {network_name: vertex_indices}."""
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]

    networks = {}
    for idx, name in enumerate(names):
        # Extract network from parcel name: "7Networks_LH_DorsAttn_Post_1"
        parts = name.split("_")
        if len(parts) >= 3 and parts[0] == "7Networks":
            raw_network = parts[2]
            network = NETWORK_ALIASES.get(raw_network, raw_network)
            if network not in networks:
                networks[network] = []
            networks[network].extend(np.where(labels == idx)[0].tolist())

    # Convert to numpy arrays
    return {k: np.array(v) for k, v in networks.items()}


networks_lh = parse_parcellation(annot_paths["lh"])
networks_rh = parse_parcellation(annot_paths["rh"])

print("Networks found:")
for name in networks_lh:
    n_lh = len(networks_lh[name])
    n_rh = len(networks_rh[name])
    print(f"  {name:8s}: {n_lh + n_rh:5d} vertices ({n_lh} LH + {n_rh} RH)")


# ============================================================
# 3. EXTRACT NETWORK SIGNALS FROM TRIBEV2 PREDICTIONS
# ============================================================

N_VERTICES_HEMI = 10242  # fsaverage5


def extract_network_signals(preds):
    """
    Extract mean activation per network from TRIBEv2 surface predictions.

    Args:
        preds: np.ndarray, shape (n_timesteps, 20484) — TRIBEv2 output
               First 10242 = LH, next 10242 = RH

    Returns:
        dict[str, np.ndarray]: network_name → (n_timesteps,) signal
    """
    preds_lh = preds[:, :N_VERTICES_HEMI]
    preds_rh = preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]

    signals = {}
    for network in networks_lh:
        lh_idx = networks_lh[network]
        rh_idx = networks_rh.get(network, np.array([], dtype=int))

        sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else 0
        sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else 0

        signals[network] = (sig_lh + sig_rh) / 2  # bilateral average

    return signals


def compute_cognitive_metrics(signals):
    """Compute cognitive load, attention, and engagement from network signals."""
    return {
        # Cognitive load: high control + high attention - low DMN
        "cognitive_load": (
            np.mean(signals["FPN"]) +
            np.mean(signals["DAN"]) -
            np.mean(signals["DMN"])
        ),
        # Attention: DAN - DMN anticorrelation
        "attention": np.mean(signals["DAN"]) - np.mean(signals["DMN"]),
        # Engagement: inverse of DMN (less mind-wandering = more engaged)
        "engagement": -np.mean(signals["DMN"]),
        # Per-network means
        **{f"net_{k}": np.mean(v) for k, v in signals.items()},
    }


# ============================================================
# 3b. SPATIAL REGION EXTRACTION (finer than 7 networks)
# ============================================================

# Schaefer parcel names encode spatial location:
#   "7Networks_LH_Vis_1"              → Vis, region=generic
#   "7Networks_LH_DorsAttn_Post_1"    → DAN, region=Post (posterior parietal)
#   "7Networks_LH_Default_Temp_1"     → DMN, region=Temp (temporal)
#   "7Networks_LH_Cont_PFCl_1"       → FPN, region=PFCl (lateral PFC)

# Map Schaefer sub-region tags to anatomical areas
SPATIAL_REGIONS = {
    # Visual cortex
    "visual_cortex":    {"network": "Vis",    "tags": None},  # all Vis parcels
    # Temporal cortex (language areas — superior/middle temporal)
    "temporal_cortex":  {"network": None,     "tags": ["Temp", "TempPar", "TempOcc"]},
    # Prefrontal cortex (cognitive control)
    "prefrontal":       {"network": None,     "tags": ["PFCl", "PFCd", "PFCv", "PFCm", "PFCmp"]},
    # Parietal (attention, spatial)
    "parietal":         {"network": None,     "tags": ["Post", "IPL", "SPL", "Par", "ParOper", "ParMed"]},
    # Cingulate (salience, conflict monitoring)
    "cingulate":        {"network": None,     "tags": ["Cing", "pCun"]},
    # Insular (interoception, salience)
    "insular":          {"network": None,     "tags": ["FrOper", "Ins"]},
}


def _build_region_masks():
    """Parse Schaefer parcel names into spatial region vertex masks."""
    region_vertices = {r: {"lh": [], "rh": []} for r in SPATIAL_REGIONS}

    for hemi_key, hemi_networks, annot_path in [
        ("lh", networks_lh, annot_paths["lh"]),
        ("rh", networks_rh, annot_paths["rh"]),
    ]:
        labels, ctab, names = nib.freesurfer.read_annot(annot_path)
        names = [n.decode() if isinstance(n, bytes) else n for n in names]

        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) < 3 or parts[0] != "7Networks":
                continue

            raw_network = parts[2]
            network = NETWORK_ALIASES.get(raw_network, raw_network)
            # Sub-region tag is parts[3] if it exists
            sub_tag = parts[3] if len(parts) > 3 else ""
            verts = np.where(labels == idx)[0].tolist()

            for region_name, spec in SPATIAL_REGIONS.items():
                match = False
                if spec["tags"] is None and spec["network"] == network:
                    match = True  # grab all parcels in this network
                elif spec["tags"] is not None:
                    if sub_tag in spec["tags"]:
                        match = True
                if match:
                    region_vertices[region_name][hemi_key].extend(verts)

    return {r: {"lh": np.array(v["lh"]), "rh": np.array(v["rh"])}
            for r, v in region_vertices.items()}


_region_masks = _build_region_masks()


def extract_spatial_signals(preds):
    """
    Extract mean activation per spatial region (temporal, visual, PFC, etc.).

    Args:
        preds: (n_timesteps, 20484) TRIBEv2 output

    Returns:
        dict[str, np.ndarray]: region_name → (n_timesteps,) signal
    """
    preds_lh = preds[:, :N_VERTICES_HEMI]
    preds_rh = preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]

    signals = {}
    for region, masks in _region_masks.items():
        lh_idx, rh_idx = masks["lh"], masks["rh"]
        sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(len(preds))
        sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(len(preds))
        signals[region] = (sig_lh + sig_rh) / 2
    return signals


def compute_full_table(preds_dict):
    """
    Compute comprehensive table: cognitive metrics + network + spatial regions.

    Args:
        preds_dict: {stimulus_name: preds array (n_timesteps, n_vertices)}

    Returns:
        rows: list of dicts (one per stimulus)
    """
    rows = []
    for name, preds in preds_dict.items():
        net_signals = extract_network_signals(preds)
        cog_metrics = compute_cognitive_metrics(net_signals)
        spatial_signals = extract_spatial_signals(preds)

        row = {
            "stimulus": name,
            # Cognitive metrics
            "cog_load": cog_metrics["cognitive_load"],
            "attention": cog_metrics["attention"],
            "engagement": cog_metrics["engagement"],
            # Network activations
            "FPN": np.mean(net_signals["FPN"]),
            "DAN": np.mean(net_signals["DAN"]),
            "DMN": np.mean(net_signals["DMN"]),
            "VAN": np.mean(net_signals["VAN"]),
            "Vis": np.mean(net_signals["Vis"]),
            "Limbic": np.mean(net_signals["Limbic"]),
            "SomMot": np.mean(net_signals["SomMot"]),
            # Spatial regions
            "visual_ctx": np.mean(spatial_signals["visual_cortex"]),
            "temporal_ctx": np.mean(spatial_signals["temporal_cortex"]),
            "prefrontal": np.mean(spatial_signals["prefrontal"]),
            "parietal": np.mean(spatial_signals["parietal"]),
            "cingulate": np.mean(spatial_signals["cingulate"]),
            "insular": np.mean(spatial_signals["insular"]),
            # Timeseries for per-TR analysis
            "_net_signals": net_signals,
            "_spatial_signals": spatial_signals,
        }
        rows.append(row)
    return rows


def print_full_table(rows):
    """Pretty-print the comprehensive analysis table."""
    # --- Cognitive metrics ---
    print("\n" + "=" * 80)
    print("COGNITIVE METRICS")
    print("=" * 80)
    print(f"{'Stimulus':15s} {'Cog Load':>10s} {'Attention':>10s} {'Engage':>10s} "
          f"{'DMN':>8s} {'FPN':>8s} {'DAN':>8s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['stimulus']:15s} {r['cog_load']:10.4f} {r['attention']:10.4f} "
              f"{r['engagement']:10.4f} {r['DMN']:8.4f} {r['FPN']:8.4f} {r['DAN']:8.4f}")

    # --- Network activations ---
    print("\n" + "=" * 80)
    print("NETWORK ACTIVATIONS (7 Yeo Networks)")
    print("=" * 80)
    nets = ["Vis", "SomMot", "DAN", "VAN", "Limbic", "FPN", "DMN"]
    print(f"{'Stimulus':15s}", end="")
    for n in nets:
        print(f" {n:>8s}", end="")
    print()
    print("-" * 80)
    for r in rows:
        print(f"{r['stimulus']:15s}", end="")
        for n in nets:
            print(f" {r[n]:8.4f}", end="")
        print()

    # --- Spatial regions ---
    print("\n" + "=" * 80)
    print("SPATIAL REGION ACTIVATIONS")
    print("=" * 80)
    regions = ["visual_ctx", "temporal_ctx", "prefrontal", "parietal", "cingulate", "insular"]
    labels = ["Visual", "Temporal", "PFC", "Parietal", "Cingulate", "Insular"]
    print(f"{'Stimulus':15s}", end="")
    for l in labels:
        print(f" {l:>10s}", end="")
    print()
    print("-" * 80)
    for r in rows:
        print(f"{r['stimulus']:15s}", end="")
        for reg in regions:
            print(f" {r[reg]:10.4f}", end="")
        print()

    # --- Cross-modal: temporal vs visual ratio ---
    print("\n" + "=" * 80)
    print("TEMPORAL vs VISUAL DOMINANCE")
    print("=" * 80)
    print(f"{'Stimulus':15s} {'Temporal':>10s} {'Visual':>10s} {'T/V Ratio':>10s} {'Dominant':>10s}")
    print("-" * 80)
    for r in rows:
        t, v = r["temporal_ctx"], r["visual_ctx"]
        ratio = t / v if abs(v) > 1e-8 else float("inf")
        dom = "Temporal" if t > v else "Visual"
        print(f"{r['stimulus']:15s} {t:10.4f} {v:10.4f} {ratio:10.2f} {dom:>10s}")


# ============================================================
# 4. BRAINIAK: REPRESENTATIONAL SIMILARITY ANALYSIS
# ============================================================

def run_brsa(predictions_dict, n_events=None):
    """
    Bayesian RSA across stimulus conditions using BrainIAK.

    Args:
        predictions_dict: {condition_name: preds array (n_timesteps, n_vertices)}

    Returns:
        similarity_matrix: (n_conditions, n_conditions) correlation matrix
        condition_names: list of condition names
    """
    from brainiak.reprsimil.brsa import BRSA

    conditions = list(predictions_dict.keys())
    all_preds = []
    design_rows = []

    for i, (name, preds) in enumerate(predictions_dict.items()):
        all_preds.append(preds)
        # One-hot design: which condition is active at each timepoint
        col = np.zeros((preds.shape[0], len(conditions)))
        col[:, i] = 1.0
        design_rows.append(col)

    # Stack all conditions into one timeseries
    X = np.vstack(all_preds)          # (total_timesteps, n_vertices)
    design = np.vstack(design_rows)   # (total_timesteps, n_conditions)

    # Subsample vertices for speed (BRSA on 20k vertices is slow)
    n_verts_sample = min(500, X.shape[1])
    rng = np.random.default_rng(42)
    vert_idx = rng.choice(X.shape[1], n_verts_sample, replace=False)
    X_sub = X[:, vert_idx]

    print(f"Running BRSA: {X_sub.shape[0]} timepoints × {n_verts_sample} vertices, "
          f"{len(conditions)} conditions...")

    brsa = BRSA(n_iter=200, auto_nuisance=True)
    brsa.fit(X_sub, design)

    # brsa.U_ is the shared covariance (n_conditions × n_conditions)
    # Normalize to correlation matrix
    U = brsa.U_
    d = np.sqrt(np.diag(U))
    d[d == 0] = 1
    similarity = U / np.outer(d, d)

    return similarity, conditions


# ============================================================
# 5. BRAINIAK: EVENT SEGMENTATION
# ============================================================

def find_cognitive_state_boundaries(preds, n_events=5):
    """
    Use BrainIAK's HMM-based event segmentation to find
    cognitive state transitions in the predicted brain response.

    Args:
        preds: (n_timesteps, n_vertices) — TRIBEv2 output
        n_events: number of discrete cognitive states to find

    Returns:
        event_labels: (n_timesteps,) — which state each timepoint belongs to
        boundaries: list of timepoint indices where state transitions occur
    """
    from brainiak.eventseg.event import EventSegment

    # Subsample vertices for speed
    n_verts_sample = min(500, preds.shape[1])
    rng = np.random.default_rng(42)
    vert_idx = rng.choice(preds.shape[1], n_verts_sample, replace=False)
    X = preds[:, vert_idx]

    print(f"Running EventSegment: {X.shape[0]} timepoints × {n_verts_sample} vertices, "
          f"finding {n_events} states...")

    ev = EventSegment(n_events=n_events)
    ev.fit(X)

    # ev.segments_[0] is (n_timesteps, n_events) probability matrix
    event_probs = ev.segments_[0]
    event_labels = np.argmax(event_probs, axis=1)

    # Find boundaries (where label changes)
    boundaries = np.where(np.diff(event_labels) != 0)[0] + 1

    return event_labels, boundaries


# ============================================================
# 6. FULL PIPELINE: STIMULI → TRIBEV2 → QUANTIFY → PLOT
# ============================================================

def run_pipeline(model, stimuli_dict):
    """
    End-to-end: text stimuli → TRIBEv2 → cognitive metrics + RSA + event segmentation.

    Args:
        model: TribeModel instance (already loaded)
        stimuli_dict: {level_name: text_string}
    """
    # --- Run TRIBEv2 predictions ---
    predictions = {}
    for level, text in stimuli_dict.items():
        text_path = CACHE_FOLDER / f"{level}.txt"
        text_path.write_text(text.strip())
        df = model.get_events_dataframe(text_path=text_path)
        preds, segments = model.predict(events=df)
        predictions[level] = preds
        print(f"  {level:15s} → {preds.shape}")

    # --- Extract network signals & cognitive metrics ---
    all_metrics = {}
    all_signals = {}
    for level, preds in predictions.items():
        signals = extract_network_signals(preds)
        metrics = compute_cognitive_metrics(signals)
        all_signals[level] = signals
        all_metrics[level] = metrics

    # --- Print metrics table ---
    levels = list(stimuli_dict.keys())
    print("\n" + "=" * 75)
    print(f"{'Stimulus':15s} {'Cog Load':>10s} {'Attention':>10s} "
          f"{'Engage':>10s} {'FPN':>8s} {'DAN':>8s} {'DMN':>8s}")
    print("=" * 75)
    for level in levels:
        m = all_metrics[level]
        print(f"{level:15s} {m['cognitive_load']:10.4f} {m['attention']:10.4f} "
              f"{m['engagement']:10.4f} {m['net_FPN']:8.4f} {m['net_DAN']:8.4f} "
              f"{m['net_DMN']:8.4f}")

    # --- BRSA: how similar are the brain response patterns? ---
    print("\n--- Bayesian RSA ---")
    sim_matrix, cond_names = run_brsa(predictions)

    # --- Event segmentation on the most complex stimulus ---
    most_complex = levels[-1]
    print(f"\n--- Event Segmentation ({most_complex}) ---")
    event_labels, boundaries = find_cognitive_state_boundaries(
        predictions[most_complex], n_events=5
    )
    print(f"  Found {len(boundaries)} state transitions at timepoints: {boundaries}")

    # --- Plot everything ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Cognitive load bar chart
    ax = axes[0, 0]
    colors = ["#22c55e", "#4a9eed", "#f59e0b", "#ef4444"][:len(levels)]
    vals = [all_metrics[l]["cognitive_load"] for l in levels]
    ax.bar(levels, vals, color=colors)
    ax.set_title("Cognitive Load Index (FPN + DAN - DMN)")
    ax.set_ylabel("Load")
    ax.tick_params(axis="x", rotation=15)

    # (B) Network breakdown
    ax = axes[0, 1]
    x = np.arange(len(levels))
    width = 0.15
    for i, net in enumerate(["FPN", "DAN", "DMN", "VAN", "Vis"]):
        vals = [all_metrics[l].get(f"net_{net}", 0) for l in levels]
        ax.bar(x + i * width, vals, width, label=net)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(levels, rotation=15)
    ax.set_title("Network Activation Breakdown")
    ax.legend(fontsize=8)

    # (C) RSA similarity matrix
    ax = axes[1, 0]
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cond_names)))
    ax.set_yticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cond_names, fontsize=9)
    ax.set_title("Bayesian RSA (Pattern Similarity)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (D) Event segmentation timeline
    ax = axes[1, 1]
    t = np.arange(len(event_labels))
    ax.fill_between(t, event_labels, alpha=0.6, step="mid", color="#8b5cf6")
    for b in boundaries:
        ax.axvline(b, color="#ef4444", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel("Timepoint (TR)")
    ax.set_ylabel("Cognitive State")
    ax.set_title(f"Event Segmentation — {most_complex}")
    ax.set_yticks(range(5))

    plt.tight_layout()
    plt.savefig(CACHE_FOLDER / "braingym_quantification.png", dpi=150)
    plt.show()

    return all_metrics, sim_matrix, event_labels, boundaries


# ============================================================
# 7. EXAMPLE STIMULI (varying cognitive load)
# ============================================================

STIMULI = {
    "simple": """
        The sun is warm. The sky is blue. Birds fly in the air.
        Dogs like to run. Cats like to sleep. Fish swim in water.
        Trees grow tall. Flowers smell nice. Rain falls from clouds.
    """,
    "narrative": """
        To be or not to be, that is the question.
        Whether 'tis nobler in the mind to suffer
        the slings and arrows of outrageous fortune,
        or to take arms against a sea of troubles
        and by opposing end them. To die, to sleep,
        no more; and by a sleep to say we end
        the heartache and the thousand natural shocks
        that flesh is heir to.
    """,
    "technical": """
        The mitochondrial electron transport chain consists of four
        protein complexes embedded in the inner mitochondrial membrane.
        Complex I, NADH dehydrogenase, catalyzes the transfer of electrons
        from NADH to ubiquinone, coupled with the translocation of four
        protons across the membrane, establishing the electrochemical
        gradient essential for oxidative phosphorylation via ATP synthase.
    """,
    "dense": """
        The renormalization group analysis of quantum chromodynamics
        reveals asymptotic freedom at high energies, where the running
        coupling constant alpha_s decreases logarithmically. The beta
        function coefficient b_0 equals 33 minus 2n_f over 12 pi,
        governing one-loop evolution and implying confinement at low
        momentum transfer through infrared slavery of the gluon
        propagator in the Landau gauge.
    """,
}

# ============================================================
# USAGE (uncomment when model is loaded):
#
#   metrics, rsa, events, bounds = run_pipeline(model, STIMULI)
#
# ============================================================

if __name__ == "__main__":
    # Just test the parcellation loading (no model needed)
    print("\n--- Testing parcellation ---")
    dummy_preds = np.random.randn(20, 2 * N_VERTICES_HEMI)
    signals = extract_network_signals(dummy_preds)
    metrics = compute_cognitive_metrics(signals)
    print(f"Dummy cognitive load: {metrics['cognitive_load']:.4f}")
    print(f"Dummy attention:      {metrics['attention']:.4f}")
    print("Parcellation and extraction pipeline works.")
