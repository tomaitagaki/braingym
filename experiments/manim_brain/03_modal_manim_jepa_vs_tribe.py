"""
Manim experiment on Modal: generate 5 presentation styles, score with Tribe,
and compare V-JEPA2 features vs final TRIBE predictions.

Key question: does the brain mapping reorganize the representation space,
or is TRIBE just a lossy linear transform of V-JEPA2?

Test:
  1. Render 5 Manim variations of the same concept
  2. Run Tribe → get final fMRI predictions (20484 vertices)
  3. Extract intermediate V-JEPA2 features (before brain projection)
  4. Compare pairwise similarity matrices: JEPA space vs TRIBE space
  5. If they're identical → brain mapping adds nothing for this paradigm

Note: Manim animations are OUT OF DISTRIBUTION for Tribe (trained on movies/nature).
This experiment tests how Tribe behaves on synthetic stimuli.

Usage:
    modal run experiments/manim_brain/03_modal_manim_jepa_vs_tribe.py
"""

import json
import modal
from pathlib import Path

app = modal.App("braingym-manim-jepa-vs-tribe")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

manim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg", "git",
        # LaTeX for MathTex
        "texlive-latex-base", "texlive-latex-extra",
        "texlive-fonts-recommended", "texlive-fonts-extra",
        "texlive-science", "dvisvgm",
        # Manim native deps (cairo, pango, pkg-config)
        "pkg-config", "libcairo2-dev", "libpango1.0-dev",
        "libgirepository1.0-dev", "gir1.2-gtk-3.0",
    )
    .pip_install(
        # Manim
        "manim>=0.18",
        # PyTorch + Tribe
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        "transformers>=4.48,<5",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "moviepy>=2.2.1",
        "huggingface_hub",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "x_transformers==1.27.20",
        "nibabel",
        "scipy",
        "Pillow",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)


# ============================================================
# MANIM SCENES (inline so Modal can serialize them)
# ============================================================

SCENE_SCRIPT = r'''
from manim import *
import numpy as np

class V1Slow(Scene):
    """Slow pacing, one element at a time, long pauses."""
    def construct(self):
        title = Text("The Pythagorean Theorem", font_size=48)
        self.play(Write(title), run_time=2)
        self.wait(2)
        self.play(FadeOut(title))
        self.wait(1)
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        self.play(Create(triangle), run_time=3)
        self.wait(2)
        a_label = MathTex("a = 3", font_size=36).next_to(triangle, DOWN)
        self.play(Write(a_label), run_time=1.5)
        self.wait(1.5)
        b_label = MathTex("b = 4", font_size=36).next_to(triangle, RIGHT)
        self.play(Write(b_label), run_time=1.5)
        self.wait(1.5)
        c_label = MathTex("c = 5", font_size=36).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)
        self.play(Write(c_label), run_time=1.5)
        self.wait(2)
        eq = MathTex("a^2 + b^2 = c^2", font_size=48).to_edge(UP)
        self.play(Write(eq), run_time=2)
        self.wait(2)
        eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=48).next_to(eq, DOWN)
        self.play(Write(eq2), run_time=2)
        self.wait(1.5)
        eq3 = MathTex("9 + 16 = 25", font_size=48).next_to(eq2, DOWN)
        self.play(Write(eq3), run_time=2)
        self.wait(3)

class V2Fast(Scene):
    """Fast pacing, everything appears quickly."""
    def construct(self):
        title = Text("Pythagorean Theorem", font_size=48)
        self.play(Write(title), run_time=0.5)
        self.wait(0.3)
        self.play(FadeOut(title), run_time=0.3)
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        a_label = MathTex("a=3", font_size=32).next_to(triangle, DOWN)
        b_label = MathTex("b=4", font_size=32).next_to(triangle, RIGHT)
        c_label = MathTex("c=5", font_size=32).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)
        self.play(Create(triangle), Write(a_label), Write(b_label), Write(c_label), run_time=1)
        self.wait(0.5)
        eq1 = MathTex("a^2 + b^2 = c^2", font_size=44).to_edge(UP)
        eq2 = MathTex("3^2 + 4^2 = 5^2", font_size=44).next_to(eq1, DOWN)
        eq3 = MathTex(r"9 + 16 = 25 \checkmark", font_size=44).next_to(eq2, DOWN)
        self.play(Write(eq1), run_time=0.5)
        self.play(Write(eq2), run_time=0.5)
        self.play(Write(eq3), run_time=0.5)
        self.wait(1)

class V3Visual(Scene):
    """Heavy geometric visualization, minimal text."""
    def construct(self):
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=WHITE)
        self.play(Create(triangle), run_time=1.5)
        self.wait(0.5)
        sq_a = Square(side_length=3, color=RED, fill_opacity=0.3).next_to(triangle, DOWN, buff=0)
        sq_b = Square(side_length=4, color=GREEN, fill_opacity=0.3).next_to(triangle, RIGHT, buff=0)
        sq_c = Square(side_length=5, color=BLUE, fill_opacity=0.3)
        sq_c.rotate(np.arctan(4 / 3))
        sq_c.move_to(triangle.get_center() + LEFT * 1.5 + UP * 1)
        self.play(Create(sq_a), run_time=1)
        self.wait(0.5)
        a_area = MathTex("9", font_size=36, color=RED).move_to(sq_a)
        self.play(Write(a_area), run_time=0.5)
        self.play(Create(sq_b), run_time=1)
        self.wait(0.5)
        b_area = MathTex("16", font_size=36, color=GREEN).move_to(sq_b)
        self.play(Write(b_area), run_time=0.5)
        self.play(Create(sq_c), run_time=1)
        self.wait(0.5)
        c_area = MathTex("25", font_size=36, color=BLUE).move_to(sq_c)
        self.play(Write(c_area), run_time=0.5)
        self.wait(1)
        eq = MathTex("9", "+", "16", "=", "25", font_size=48).to_edge(UP)
        eq[0].set_color(RED)
        eq[2].set_color(GREEN)
        eq[4].set_color(BLUE)
        self.play(Write(eq), run_time=1)
        self.wait(2)

class V4TextHeavy(Scene):
    """Emphasis on equations and text, minimal geometry."""
    def construct(self):
        title = Text("The Pythagorean Theorem", font_size=42)
        subtitle = Text("For any right triangle:", font_size=28, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), Write(subtitle), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))
        eq = MathTex("a^2 + b^2 = c^2", font_size=72).to_edge(UP, buff=1)
        self.play(Write(eq), run_time=1.5)
        self.wait(1)
        lines = VGroup(
            Text("where:", font_size=28),
            MathTex("a", r"\text{ and }", "b", r"\text{ are the legs}", font_size=28),
            MathTex("c", r"\text{ is the hypotenuse}", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(eq, DOWN, buff=0.8)
        for line in lines:
            self.play(Write(line), run_time=1)
            self.wait(0.5)
        self.wait(1)
        example_title = Text("Example:", font_size=32, color=YELLOW).to_edge(LEFT).shift(DOWN)
        self.play(Write(example_title), run_time=0.5)
        steps = VGroup(
            MathTex(r"a = 3, \quad b = 4, \quad c = ?", font_size=32),
            MathTex("3^2 + 4^2 = c^2", font_size=32),
            MathTex("9 + 16 = c^2", font_size=32),
            MathTex("25 = c^2", font_size=32),
            MathTex("c = 5", font_size=32, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(example_title, DOWN, aligned_edge=LEFT)
        for step in steps:
            self.play(Write(step), run_time=0.8)
            self.wait(0.5)
        self.wait(2)

class V5Narrative(Scene):
    """Storytelling approach with narration-style text."""
    def construct(self):
        hook = Text("2,500 years ago, a Greek mathematician\nmade a discovery...", font_size=32)
        self.play(Write(hook), run_time=2)
        self.wait(2)
        self.play(FadeOut(hook))
        name = Text("Pythagoras", font_size=56, color=YELLOW)
        self.play(Write(name), run_time=1)
        self.wait(1)
        self.play(FadeOut(name))
        q = Text("He noticed something about right triangles...", font_size=28)
        self.play(Write(q), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(q))
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4, color=BLUE)
        self.play(Create(triangle), run_time=1.5)
        a_label = MathTex("3", font_size=36).next_to(triangle, DOWN)
        b_label = MathTex("4", font_size=36).next_to(triangle, RIGHT)
        c_label = MathTex("5", font_size=36).move_to(triangle.get_center() + LEFT * 0.8 + UP * 0.3)
        self.play(Write(a_label), Write(b_label), Write(c_label), run_time=1)
        self.wait(1)
        reveal = Text("If you square the two short sides\nand add them together...", font_size=24).to_edge(UP)
        self.play(Write(reveal), run_time=1.5)
        self.wait(1)
        eq1 = MathTex("3^2 + 4^2 = 9 + 16 = 25", font_size=36).next_to(reveal, DOWN)
        self.play(Write(eq1), run_time=1)
        self.wait(1)
        reveal2 = Text("...you always get the square of the long side!", font_size=24, color=GREEN).next_to(eq1, DOWN)
        self.play(Write(reveal2), run_time=1.5)
        eq2 = MathTex(r"5^2 = 25 \checkmark", font_size=36, color=GREEN).next_to(reveal2, DOWN)
        self.play(Write(eq2), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        final = MathTex("a^2 + b^2 = c^2", font_size=72, color=YELLOW)
        self.play(Write(final), run_time=1.5)
        self.wait(2)
'''


# ============================================================
# MAIN FUNCTION
# ============================================================

@app.function(
    image=manim_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=32768,
)
def run_experiment():
    """Render Manim scenes, score with Tribe, extract JEPA features, compare."""
    import os
    import time
    import subprocess
    import numpy as np
    import torch
    from pathlib import Path
    from urllib.request import urlretrieve
    import nibabel as nib

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    results_dir = Path(CACHE_DIR) / "manim_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Render Manim scenes ----
    print("=" * 60)
    print("STEP 1: Rendering Manim scenes")
    print("=" * 60)

    script_path = results_dir / "scenes.py"
    script_path.write_text(SCENE_SCRIPT)

    scenes = ["V1Slow", "V2Fast", "V3Visual", "V4TextHeavy", "V5Narrative"]
    video_paths = {}

    for scene_name in scenes:
        out_path = results_dir / f"{scene_name}.mp4"
        if out_path.exists():
            print(f"  Cached: {scene_name}")
            video_paths[scene_name] = out_path
            continue

        print(f"  Rendering: {scene_name}...")
        t0 = time.time()
        media_dir = results_dir / "media"
        result = subprocess.run(
            [
                "python", "-m", "manim", "render",
                "-ql",  # 480p15 for speed
                "--media_dir", str(media_dir),
                str(script_path),
                scene_name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[-300:]}")
            continue

        # Find rendered file
        rendered = list(media_dir.rglob(f"{scene_name}.mp4"))
        if rendered:
            rendered[0].rename(out_path)
            video_paths[scene_name] = out_path
            print(f"    Done ({time.time() - t0:.1f}s)")
        else:
            print(f"    No output found")

    print(f"\nRendered {len(video_paths)}/{len(scenes)} scenes")

    # ---- Step 2: Load Tribe ----
    print("\n" + "=" * 60)
    print("STEP 2: Loading Tribe v2")
    print("=" * 60)

    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tribe = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
        },
    )
    print(f"Loaded on {device}")

    # Load parcellation
    SCHAEFER_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
    ALIASES = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
    }

    parc_dir = Path(CACHE_DIR) / "schaefer"
    parc_dir.mkdir(exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parc_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        labels, _, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = ALIASES.get(parts[2], parts[2])
                if net not in networks:
                    networks[net] = {"lh": [], "rh": []}
                networks[net][hemi].extend(np.where(labels == idx)[0].tolist())
    for net in networks:
        for hemi in ("lh", "rh"):
            networks[net][hemi] = np.array(networks[net][hemi])

    # ---- Step 3: Score each video ----
    print("\n" + "=" * 60)
    print("STEP 3: Scoring with Tribe + extracting V-JEPA2 features")
    print("=" * 60)

    N = 10242
    all_results = {}
    all_preds = {}        # full vertex predictions (TRIBE space)
    all_jepa = {}         # V-JEPA2 features (backbone space)

    for scene_name, video_path in video_paths.items():
        print(f"\n  Processing: {scene_name}")
        t0 = time.time()

        # Get events
        df = tribe.get_events_dataframe(video_path=str(video_path))

        # --- Extract V-JEPA2 features (before brain projection) ---
        # Run feature extraction step separately to capture intermediate features
        try:
            # Tribe internally extracts features then projects to brain space.
            # We can access the video features through the model's internal pipeline.
            # The events dataframe tells us what extractors are active.
            features_dict = tribe.extract_features(events=df)
            # features_dict should contain per-modality features
            # Look for video features
            jepa_feats = None
            if hasattr(features_dict, 'items'):
                for key, val in features_dict.items():
                    if 'video' in str(key).lower():
                        jepa_feats = val.cpu().numpy() if torch.is_tensor(val) else np.array(val)
                        print(f"    V-JEPA2 features: {jepa_feats.shape}")
                        break
            if jepa_feats is None:
                # Fallback: try to get features from the model's extractor directly
                print(f"    Could not extract V-JEPA2 features separately, using predict")
                jepa_feats = np.array([])
            all_jepa[scene_name] = jepa_feats
        except (AttributeError, TypeError) as e:
            print(f"    V-JEPA2 extraction not available: {e}")
            all_jepa[scene_name] = np.array([])

        # --- Full TRIBE prediction ---
        preds, segments = tribe.predict(events=df)
        elapsed = time.time() - t0
        print(f"    TRIBE predictions: {preds.shape} in {elapsed:.1f}s")

        all_preds[scene_name] = preds

        # Extract network signals
        preds_lh = preds[:, :N]
        preds_rh = preds[:, N:2*N]
        net_signals = {}
        for net_name, masks in networks.items():
            lh_idx, rh_idx = masks["lh"], masks["rh"]
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(preds.shape[0])
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(preds.shape[0])
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}
        dan = np.array(net_signals["DAN"])
        dmn = np.array(net_signals["DMN"])

        result = {
            "name": scene_name,
            "n_timepoints": int(preds.shape[0]),
            **{f"net_{k}": v for k, v in net_means.items()},
            "attention": float(np.mean(dan - dmn)),
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "reward": float(
                0.4 * (-net_means["DMN"])
                + 0.3 * (net_means["DAN"] - net_means["DMN"])
                + 0.2 * net_means["VAN"]
                + 0.1 * net_means["Limbic"]
            ),
            "timeseries": net_signals,
        }
        all_results[scene_name] = result

    # ---- Step 4: Compare TRIBE pairwise similarity ----
    print("\n" + "=" * 60)
    print("STEP 4: TRIBE pairwise similarity (brain space)")
    print("=" * 60)

    scene_names = sorted(all_preds.keys())
    n_scenes = len(scene_names)

    # TRIBE similarity: correlation of mean vertex patterns
    tribe_patterns = []
    for name in scene_names:
        tribe_patterns.append(all_preds[name].mean(axis=0))  # mean over time
    tribe_patterns = np.array(tribe_patterns)

    tribe_sim = np.corrcoef(tribe_patterns)
    print("\nTRIBE pairwise correlation (mean vertex pattern):")
    print(f"{'':>15s}", end="")
    for name in scene_names:
        print(f" {name:>10s}", end="")
    print()
    for i, name_i in enumerate(scene_names):
        print(f"{name_i:>15s}", end="")
        for j in range(n_scenes):
            print(f" {tribe_sim[i, j]:>10.4f}", end="")
        print()

    # TRIBE similarity: network profile correlation
    print("\nTRIBE pairwise correlation (network profile):")
    net_profiles = []
    for name in scene_names:
        r = all_results[name]
        profile = [r[f"net_{n}"] for n in ALIASES.values()]
        net_profiles.append(profile)
    net_profiles = np.array(net_profiles)

    net_sim = np.corrcoef(net_profiles)
    print(f"{'':>15s}", end="")
    for name in scene_names:
        print(f" {name:>10s}", end="")
    print()
    for i, name_i in enumerate(scene_names):
        print(f"{name_i:>15s}", end="")
        for j in range(n_scenes):
            print(f" {net_sim[i, j]:>10.4f}", end="")
        print()

    # ---- Step 5: Compare V-JEPA2 similarity (if available) ----
    has_jepa = all(len(v) > 0 for v in all_jepa.values()) if all_jepa else False

    if has_jepa:
        print("\n" + "=" * 60)
        print("STEP 5: V-JEPA2 pairwise similarity (backbone space)")
        print("=" * 60)

        jepa_patterns = []
        for name in scene_names:
            feats = all_jepa[name]
            jepa_patterns.append(feats.flatten())
        jepa_patterns = np.array(jepa_patterns)

        jepa_sim = np.corrcoef(jepa_patterns)
        print("\nJEPA pairwise correlation:")
        print(f"{'':>15s}", end="")
        for name in scene_names:
            print(f" {name:>10s}", end="")
        print()
        for i, name_i in enumerate(scene_names):
            print(f"{name_i:>15s}", end="")
            for j in range(n_scenes):
                print(f" {jepa_sim[i, j]:>10.4f}", end="")
            print()

        # Compare JEPA vs TRIBE distance matrices
        # Extract upper triangle
        triu_idx = np.triu_indices(n_scenes, k=1)
        jepa_dists = 1 - jepa_sim[triu_idx]
        tribe_dists = 1 - tribe_sim[triu_idx]

        # Correlation between distance matrices (Mantel-like test)
        dist_corr = np.corrcoef(jepa_dists, tribe_dists)[0, 1]
        print(f"\nJEPA vs TRIBE distance correlation: r={dist_corr:.4f}")
        if dist_corr > 0.9:
            print("→ TRIBE is essentially a linear transform of JEPA (brain mapping adds little)")
        elif dist_corr > 0.5:
            print("→ TRIBE partially reorganizes JEPA space (brain mapping adds some structure)")
        else:
            print("→ TRIBE substantially reorganizes JEPA space (brain mapping matters)")
    else:
        print("\n" + "=" * 60)
        print("STEP 5: V-JEPA2 features not extracted (API limitation)")
        print("=" * 60)
        print("Cannot compare JEPA vs TRIBE directly.")
        print("Alternative: run standalone V-JEPA2 on same videos to get backbone features.")

    # ---- Step 6: Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: Brain metrics by presentation style")
    print("=" * 60)

    print(f"\n{'Scene':>15s} {'Reward':>8s} {'Attn':>8s} {'Engage':>8s} {'CogLoad':>8s} {'Vis':>8s} {'DAN':>8s} {'DMN':>8s}")
    print("-" * 85)
    for name in scene_names:
        r = all_results[name]
        print(f"{name:>15s} {r['reward']:>+8.4f} {r['attention']:>+8.4f} {r['engagement']:>+8.4f} "
              f"{r['cognitive_load']:>+8.4f} {r['net_Vis']:>+8.4f} {r['net_DAN']:>+8.4f} {r['net_DMN']:>+8.4f}")

    # Are the differences meaningful?
    rewards = [all_results[n]["reward"] for n in scene_names]
    reward_range = max(rewards) - min(rewards)
    print(f"\nReward range: {reward_range:.4f}")
    print(f"Reward std:   {np.std(rewards):.4f}")

    if reward_range > 0.01:
        print("→ Meaningful variation across styles — optimization could work")
    else:
        print("→ Small variation — Tribe may not distinguish presentation styles for synthetic content")

    # Save results
    save_data = {
        "results": all_results,
        "tribe_similarity": tribe_sim.tolist(),
        "network_similarity": net_sim.tolist(),
        "scene_names": scene_names,
    }
    if has_jepa:
        save_data["jepa_similarity"] = jepa_sim.tolist()
        save_data["jepa_vs_tribe_dist_corr"] = float(dist_corr)

    out_path = results_dir / "jepa_vs_tribe_results.json"
    out_path.write_text(json.dumps(save_data, indent=2))
    vol.commit()
    print(f"\nSaved: {out_path}")

    return save_data


@app.local_entrypoint()
def main():
    results = run_experiment.remote()

    # Save locally
    local_dir = Path("cache/manim_experiment")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "jepa_vs_tribe_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nSaved local: {local_dir / 'jepa_vs_tribe_results.json'}")
