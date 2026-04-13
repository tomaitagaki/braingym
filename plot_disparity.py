"""Plot per-sample ROI activations for BDDN and Tribe side by side."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path("bddn")))

CACHE = Path("./cache")

# Load raw predictions
bddn_preds = np.load(CACHE / "bddn_preds.npy")  # (10, 37984)
tribe_preds = np.load(CACHE / "tribe_preds.npy")  # (6, 20484)

print(f"BDDN: {bddn_preds.shape[0]} frames (2fps)")
print(f"Tribe: {tribe_preds.shape[0]} timepoints (1Hz)")

# --- BDDN per-frame ROIs ---
from brainnet.roi import roi_dict, nsdgeneral_indices

nsd_set = set(nsdgeneral_indices.tolist())
nsd_to_local = {v: i for i, v in enumerate(nsdgeneral_indices)}

bddn_roi_names = ["V1", "V2", "V3", "V4", "FFA", "OFA", "PPA", "OPA", "EBA", "FBA"]
bddn_per_frame = {}
for roi_name in bddn_roi_names:
    fsavg_idx = roi_dict[roi_name]
    local = [nsd_to_local[v] for v in fsavg_idx if v in nsd_set]
    if local:
        bddn_per_frame[roi_name] = bddn_preds[:, local].mean(axis=1)  # (10,)

# --- Tribe per-timepoint ROIs ---
N_VERTICES_HEMI = 10242

# Load Schaefer parcellation
from quantify import networks_lh, networks_rh

tribe_roi_names = ["Vis", "DAN", "DMN", "FPN", "VAN", "SomMot", "Limbic"]
tribe_per_tp = {}
preds_lh = tribe_preds[:, :N_VERTICES_HEMI]
preds_rh = tribe_preds[:, N_VERTICES_HEMI:2 * N_VERTICES_HEMI]
for net in tribe_roi_names:
    lh_idx = networks_lh[net]
    rh_idx = networks_rh.get(net, np.array([], dtype=int))
    sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else 0
    sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else 0
    tribe_per_tp[net] = (sig_lh + sig_rh) / 2  # (6,)

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (A) BDDN visual ROIs per frame
ax = axes[0, 0]
t_bddn = np.arange(bddn_preds.shape[0]) * 0.5  # 2fps → 0.5s intervals
for roi in ["V1", "PPA", "OPA", "FFA", "EBA"]:
    ax.plot(t_bddn, bddn_per_frame[roi], "o-", label=roi, markersize=4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Activation")
ax.set_title("BDDN — per-frame ROI activations")
ax.legend(fontsize=8)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(alpha=0.3)

# (B) Tribe networks per timepoint
ax = axes[0, 1]
t_tribe = np.arange(tribe_preds.shape[0])  # 1Hz
for net in ["Vis", "DAN", "DMN", "FPN"]:
    ax.plot(t_tribe, tribe_per_tp[net], "s-", label=net, markersize=5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Activation")
ax.set_title("Tribe — per-timepoint network activations")
ax.legend(fontsize=8)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(alpha=0.3)

# (C) BDDN frame-to-frame variance
ax = axes[1, 0]
roi_names_sorted = sorted(bddn_per_frame.keys(), key=lambda r: -np.std(bddn_per_frame[r]))
means = [bddn_per_frame[r].mean() for r in roi_names_sorted]
stds = [bddn_per_frame[r].std() for r in roi_names_sorted]
colors = ["#ef4444" if m < 0 else "#22c55e" for m in means]
bars = ax.barh(roi_names_sorted, means, xerr=stds, color=colors, alpha=0.7, capsize=3)
ax.set_xlabel("Mean ± Std across frames")
ax.set_title("BDDN — ROI variability across frames")
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(alpha=0.3, axis="x")

# (D) Tribe timepoint-to-timepoint variance
ax = axes[1, 1]
net_names_sorted = sorted(tribe_per_tp.keys(), key=lambda n: -np.std(tribe_per_tp[n]))
means = [tribe_per_tp[n].mean() for n in net_names_sorted]
stds = [tribe_per_tp[n].std() for n in net_names_sorted]
colors = ["#ef4444" if m < 0 else "#22c55e" for m in means]
bars = ax.barh(net_names_sorted, means, xerr=stds, color=colors, alpha=0.7, capsize=3)
ax.set_xlabel("Mean ± Std across timepoints")
ax.set_title("Tribe — network variability across timepoints")
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(alpha=0.3, axis="x")

plt.suptitle("BDDN vs Tribe — Sintel 20-25s per-sample disparity", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(CACHE / "disparity.png", dpi=150)
plt.show()
print(f"\nSaved to {CACHE / 'disparity.png'}")

# --- Print raw values ---
print(f"\n{'=' * 60}")
print("BDDN per-frame (rows=frames, cols=ROIs)")
print(f"{'=' * 60}")
header = f"{'Frame':>6s}" + "".join(f"{r:>8s}" for r in bddn_roi_names)
print(header)
for i in range(bddn_preds.shape[0]):
    row = f"{i:>6d}"
    for r in bddn_roi_names:
        row += f"{bddn_per_frame[r][i]:8.3f}"
    print(row)

print(f"\n{'=' * 60}")
print("Tribe per-timepoint (rows=TRs, cols=networks)")
print(f"{'=' * 60}")
header = f"{'TR':>6s}" + "".join(f"{n:>8s}" for n in tribe_roi_names)
print(header)
for i in range(tribe_preds.shape[0]):
    row = f"{i:>6d}"
    for n in tribe_roi_names:
        row += f"{tribe_per_tp[n][i]:8.3f}"
    print(row)
