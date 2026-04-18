"""
Phase 1b: Score Manim variations with Tribe brain predictions.

For each rendered video, extract brain metrics and compare.
This tells us: do different presentation styles produce meaningfully
different brain predictions? If not, GEPA optimization won't work.

Usage:
    source tribev2/.venv/bin/activate
    python experiments/manim_brain/02_score.py
"""

import json
import time
import numpy as np
import torch
from pathlib import Path

OUTPUT_DIR = Path("cache/manim_experiment")
CACHE_DIR = Path("cache/manim_experiment/tribe_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    videos = sorted(OUTPUT_DIR.glob("V*.mp4"))
    if not videos:
        print("No videos found. Run 01_generate.py first.")
        return

    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  {v.name} ({v.stat().st_size / 1024:.0f} KB)")

    # Load Tribe
    print("\nLoading Tribe v2...")
    from tribev2.demo_utils import TribeModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(CACHE_DIR),
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )
    print(f"Loaded on {device}")

    # Process each video
    results = {}
    for video_path in videos:
        name = video_path.stem
        result_path = OUTPUT_DIR / f"{name}_brain.json"

        if result_path.exists():
            print(f"\n{name}: cached")
            results[name] = json.loads(result_path.read_text())
            continue

        print(f"\n{name}: processing...")
        t0 = time.time()

        df = model.get_events_dataframe(video_path=str(video_path))
        preds, segments = model.predict(events=df)

        elapsed = time.time() - t0
        print(f"  {preds.shape} in {elapsed:.1f}s")

        # Extract network means
        N = 10242
        preds_lh = preds[:, :N]
        preds_rh = preds[:, N:2*N]

        # Schaefer parcellation
        import nibabel as nib
        from urllib.request import urlretrieve

        SCHAEFER_BASE = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/FreeSurfer5.3/fsaverage5/label"
        )
        ANNOT = "Schaefer2018_400Parcels_7Networks_order.annot"
        ALIASES = {
            "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
            "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
        }

        parc_dir = CACHE_DIR / "schaefer"
        parc_dir.mkdir(exist_ok=True)
        networks = {}
        for hemi in ("lh", "rh"):
            local = parc_dir / f"{hemi}.{ANNOT}"
            if not local.exists():
                urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT}", local)
            labels, ctab, names = nib.freesurfer.read_annot(local)
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            nets = {}
            for idx, nm in enumerate(names):
                parts = nm.split("_")
                if len(parts) >= 3 and parts[0] == "7Networks":
                    net = ALIASES.get(parts[2], parts[2])
                    if net not in nets:
                        nets[net] = []
                    nets[net].extend(np.where(labels == idx)[0].tolist())
            networks[hemi] = {k: np.array(v) for k, v in nets.items()}

        # Extract signals
        net_signals = {}
        for net_name in ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}
        dan = np.array(net_signals["DAN"])
        dmn = np.array(net_signals["DMN"])
        attn = dan - dmn

        result = {
            "name": name,
            "n_timepoints": int(preds.shape[0]),
            **{f"net_{k}": v for k, v in net_means.items()},
            "attention": float(np.mean(attn)),
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "attn_peak": float(np.max(attn)),
            "attn_first3s": float(np.mean(attn[:min(3, len(attn))])),
            "timeseries": net_signals,
        }

        result_path.write_text(json.dumps(result, indent=2))
        results[name] = result

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Same concept, different presentations")
    print("=" * 70)
    print(f"\n{'Variation':>15s} {'TRs':>4s} {'Attention':>10s} {'CogLoad':>10s} {'Engage':>10s} {'Vis':>8s} {'DAN':>8s} {'DMN':>8s} {'FPN':>8s}")
    print("-" * 95)
    for name in sorted(results):
        r = results[name]
        print(f"{name:>15s} {r['n_timepoints']:>4d} {r['attention']:>10.4f} {r['cognitive_load']:>10.4f} "
              f"{r['engagement']:>10.4f} {r['net_Vis']:>8.4f} {r['net_DAN']:>8.4f} {r['net_DMN']:>8.4f} {r['net_FPN']:>8.4f}")

    # Pairwise spatial correlation
    print("\nPairwise brain pattern correlation:")
    names = sorted(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ts1 = np.array(results[names[i]]["timeseries"]["DAN"])
            ts2 = np.array(results[names[j]]["timeseries"]["DAN"])
            # Use mean vertex pattern instead (more informative)
            # But we don't have per-vertex means saved, so use network profile
            p1 = np.array([results[names[i]][f"net_{n}"] for n in ALIASES.values()])
            p2 = np.array([results[names[j]][f"net_{n}"] for n in ALIASES.values()])
            r = np.corrcoef(p1, p2)[0, 1]
            print(f"  {names[i]} vs {names[j]}: r={r:.4f}")

    # Save summary
    summary_path = OUTPUT_DIR / "brain_comparison.json"
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "timeseries"}
                    for k, v in results.items()}
    summary_path.write_text(json.dumps(save_results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
