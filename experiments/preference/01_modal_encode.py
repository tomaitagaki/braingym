"""
Phase 1: Run TRIBE + BDDN + CLIP on 200 images from HPD v3.

Encodes each image through both brain models and extracts
full vertex predictions + ROI summaries.

Usage:
    modal run experiments/preference/01_modal_encode.py
"""

import json
import modal

app = modal.App("braingym-preference")

# ── Modal images ───────────────────────────────────────────────────────

tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        "numpy==2.2.6",
        "einops",
        "pyyaml",
        "moviepy>=2.2.1",
        "huggingface_hub",
        "gtts",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "transformers",
        "x_transformers==1.27.20",
        "nibabel",
        "scipy",
        "Pillow",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)

bddn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1,<2.7",
        "torchvision",
        "open_clip_torch==2.20.0",
        "einops",
        "numpy",
        "nilearn",
        "scipy",
        "Pillow",
    )
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)
CACHE_DIR = "/cache"
IMAGE_DIR = "/cache/preference_images"
RESULTS_DIR = "/cache/preference_results"


# ── TRIBE encoding ─────────────────────────────────────────────────────

@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def encode_tribe_batch(image_names: list[str], batch_idx: int) -> list[dict]:
    """Encode images through TRIBE by creating 2-second silent videos."""
    import os
    import time
    import subprocess
    import tempfile
    import numpy as np
    import torch
    import nibabel as nib
    from pathlib import Path
    from urllib.request import urlretrieve

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # Load model
    print(f"[TRIBE Batch {batch_idx}] Loading model...")
    t0 = time.time()
    from tribev2.demo_utils import TribeModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device=device,
        config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
    )
    print(f"[TRIBE Batch {batch_idx}] Loaded in {time.time() - t0:.1f}s")

    # Load Schaefer parcellation
    SCHAEFER_BASE = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
    NETWORK_ALIASES = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
    }

    parcellation_dir = Path(CACHE_DIR) / "schaefer"
    parcellation_dir.mkdir(parents=True, exist_ok=True)
    networks = {}
    for hemi in ("lh", "rh"):
        local = parcellation_dir / f"{hemi}.{ANNOT_NAME}"
        if not local.exists():
            urlretrieve(f"{SCHAEFER_BASE}/{hemi}.{ANNOT_NAME}", local)
        labels, ctab, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        nets = {}
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = NETWORK_ALIASES.get(parts[2], parts[2])
                if net not in nets:
                    nets[net] = []
                nets[net].extend(np.where(labels == idx)[0].tolist())
        networks[hemi] = {k: np.array(v) for k, v in nets.items()}

    N_HEMI = 10242

    results = []
    for img_name in image_names:
        img_path = Path(IMAGE_DIR) / img_name
        result_path = Path(RESULTS_DIR) / f"tribe_{img_name.rsplit('.', 1)[0]}.json"

        # Check cache
        if result_path.exists():
            print(f"  [TRIBE {batch_idx}] {img_name}: cached")
            results.append(json.loads(result_path.read_text()))
            continue

        if not img_path.exists():
            results.append({"image": img_name, "error": f"not found: {img_path}"})
            continue

        print(f"  [TRIBE {batch_idx}] {img_name}: encoding...")
        t0 = time.time()

        try:
            # Create 2-second silent MP4 from image
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_video = tmp.name

            subprocess.run([
                "ffmpeg", "-y", "-loop", "1",
                "-i", str(img_path),
                "-t", "2",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", "25", "-an",
                tmp_video,
            ], capture_output=True, check=True)

            # Run TRIBE
            df = model.get_events_dataframe(video_path=tmp_video)
            preds, segments = model.predict(events=df)

            os.unlink(tmp_video)
        except Exception as e:
            results.append({"image": img_name, "error": str(e)[:500]})
            continue

        elapsed = time.time() - t0
        print(f"  [TRIBE {batch_idx}] {img_name}: {preds.shape} in {elapsed:.1f}s")

        # Average across timepoints
        avg_preds = preds.mean(axis=0)  # (20484,)

        # Save full vertex array
        npy_path = Path(RESULTS_DIR) / f"tribe_{img_name.rsplit('.', 1)[0]}.npy"
        np.save(str(npy_path), avg_preds)

        # Extract network means
        preds_lh = avg_preds[:N_HEMI]
        preds_rh = avg_preds[N_HEMI:2 * N_HEMI]

        net_means = {}
        for net_name in NETWORK_ALIASES.values():
            lh_idx = networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[lh_idx].mean() if len(lh_idx) > 0 else 0.0
            sig_rh = preds_rh[rh_idx].mean() if len(rh_idx) > 0 else 0.0
            net_means[net_name] = float((sig_lh + sig_rh) / 2)

        result = {
            "image": img_name,
            "model": "tribe",
            "n_timepoints": int(preds.shape[0]),
            "n_vertices": int(preds.shape[1]),
            "elapsed": round(elapsed, 1),
            "preds_path": str(npy_path),
            "network_means": net_means,
            "attention": net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)

    return results


# ── BDDN encoding ──────────────────────────────────────────────────────

@app.function(
    image=bddn_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    timeout=1800,
    memory=16384,
)
def encode_bddn_batch(image_names: list[str], batch_idx: int) -> list[dict]:
    """
    Encode images through BDDN (CLIP → nsdgeneral fMRI).

    Reimplemented without the brainnet package to avoid its
    heavy dependency chain (pycortex, dinov2, SAM, etc.).
    Uses the same CLIP backbone + FactorTopy weights directly.
    """
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from pathlib import Path
    from PIL import Image
    from einops import rearrange, repeat

    os.makedirs(RESULTS_DIR, exist_ok=True)
    nilearn_dir = os.path.join(CACHE_DIR, "nilearn_data")
    os.makedirs(nilearn_dir, exist_ok=True)

    print(f"[BDDN Batch {batch_idx}] Loading model...")
    t0 = time.time()

    # ── Load CLIP backbone (same as brainnet.backbone.ModifiedCLIP) ────
    import open_clip
    from open_clip.transformer import VisionTransformer

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_l_s1b_b8k"
    )
    # open_clip API varies: .visual.trunk (newer) or .visual (older)
    vision = getattr(clip_model.visual, "trunk", clip_model.visual)
    vision.requires_grad_(False)
    vision.eval()

    n_layers = 12
    layer_width = 768

    def extract_clip_tokens(pixel_values):
        """Extract per-layer local and global tokens from CLIP ViT.
        Returns local as (B,D,H,W) and global as (B,D) per layer."""
        x = vision.patch_embed(pixel_values)
        x = vision._pos_embed(x)
        x = vision.patch_drop(x)
        x = vision.norm_pre(x)

        # Infer grid size from number of patch tokens
        n_patches = vision.patch_embed.num_patches
        h = w = int(n_patches ** 0.5)

        local_tokens = {}
        global_tokens = {}
        for i, block in enumerate(vision.blocks):
            x = block(x)
            patches = x[:, 1:, :]  # (B, N, D) — skip CLS
            # Reshape to (B, D, H, W) for grid_sample
            local_tokens[str(i)] = rearrange(patches, "b (h w) d -> b d h w", h=h, w=w)
            global_tokens[str(i)] = x[:, 0, :]  # CLS token (B, D)

        return local_tokens, global_tokens

    # ── Load coordinates (same as brainnet.coords.load_coords) ─────────
    from nilearn import datasets as nilearn_datasets, surface as nilearn_surface
    fsaverage = nilearn_datasets.fetch_surf_fsaverage("fsaverage7", data_dir=nilearn_dir)
    lh_coords, _ = nilearn_surface.load_surf_mesh(fsaverage["sphere_left"])
    rh_coords, _ = nilearn_surface.load_surf_mesh(fsaverage["sphere_right"])
    lh_xmax = np.min(lh_coords[:, 0]) + (np.max(lh_coords[:, 0]) - np.min(lh_coords[:, 0])) * 1.5
    if np.min(rh_coords[:, 0]) < lh_xmax:
        rh_coords[:, 0] += lh_xmax - np.min(rh_coords[:, 0])
    all_coords = np.concatenate((lh_coords, rh_coords), axis=0)

    # ── Load nsdgeneral ROI indices (embedded in brainnet package) ─────
    # Download roi_data.py and roi.py from the repo
    import urllib.request

    roi_data_cache = Path(CACHE_DIR) / "bddn_roi_data.py"
    if not roi_data_cache.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/huzeyann/BrainDecodesDeepNets/main/brainnet/roi_data.py",
            roi_data_cache,
        )

    import importlib.util
    spec = importlib.util.spec_from_file_location("roi_data", roi_data_cache)
    roi_data_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roi_data_mod)
    ROI_DATA = roi_data_mod.ROI_DATA

    nsdgeneral_indices = np.array(ROI_DATA["nsdgeneral"])

    # Build roi_dict (same logic as brainnet/roi.py)
    roi_names = ["V1", "V2", "V3", "V4", "EBA", "FBA", "OFA", "FFA", "OPA", "PPA", "OWFA", "VWFA"]
    _inner_roi_names = [
        ["lh.V1v", "lh.V1d", "rh.V1v", "rh.V1d"],
        ["lh.V2v", "lh.V2d", "rh.V2v", "rh.V2d"],
        ["lh.V3v", "lh.V3d", "rh.V3v", "rh.V3d"],
        ["lh.hV4", "rh.hV4"],
        ["lh.EBA", "rh.EBA"], ["lh.FBA-1", "rh.FBA-1"],
        ["lh.OFA", "rh.OFA"], ["lh.FFA-1", "lh.FFA-2", "rh.FFA-1", "rh.FFA-2"],
        ["lh.OPA", "rh.OPA"], ["lh.PPA", "rh.PPA"],
        ["lh.OWFA", "rh.OWFA"], ["lh.VWFA-1", "lh.VWFA-2", "rh.VWFA-1", "rh.VWFA-2"],
    ]
    roi_dict = {}
    for big, small in zip(roi_names, _inner_roi_names):
        indices = []
        for roi in small:
            if roi in ROI_DATA:
                indices.extend(ROI_DATA[roi])
        roi_dict[big] = np.array(indices)

    coords = all_coords[nsdgeneral_indices]
    n_vertices = len(nsdgeneral_indices)
    coords_tensor = torch.from_numpy(coords).float().cuda()

    # ── Build FactorTopy model (exact replica of brainnet.model) ────────
    from functools import partial as _partial

    def _sinusoidal(positions, features=16, periods=10000):
        dtype = positions.dtype if positions.is_floating_point() else None
        omega = torch.logspace(0, 1 / features - 1, features, periods,
                               device=positions.device, dtype=dtype)
        fraction = omega * positions.unsqueeze(-1)
        return torch.stack((fraction.sin(), fraction.cos()), dim=-1)

    def _point_position_encoding(points, max_steps=100, features=16, periods=10000):
        low = points.min(0).values
        high = points.max(0).values
        steps = high - low
        steps *= max_steps / steps.max()
        positions = (points - low) * (steps / (high - low))
        return _sinusoidal(positions, features, periods).flatten(-3)

    class _PosEnc(nn.Module):
        def __init__(self, max_steps=100, features=32, periods=1000):
            super().__init__()
            self._fn = _partial(_point_position_encoding, max_steps=max_steps,
                                features=features, periods=periods)
        @torch.no_grad()
        def forward(self, x):
            return self._fn(x)

    class FactorTopy(nn.Module):
        def __init__(self, n_verts, layers, layer_widths, bottleneck_dim=128):
            super().__init__()
            self.layers = [str(l) for l in layers]
            self.local_token_bottleneck = nn.ModuleDict({
                str(l): nn.Conv2d(w, bottleneck_dim, 1, bias=False)
                for l, w in zip(layers, layer_widths)
            })
            self.global_token_bottleneck = nn.ModuleDict({
                str(l): nn.Linear(w, bottleneck_dim, bias=False)
                for l, w in zip(layers, layer_widths)
            })
            self.space_selector_mlp = nn.Sequential(
                _PosEnc(features=32), nn.Linear(192, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2, bias=False), nn.Tanh(),
            )
            self.layer_selector_mlp = nn.Sequential(
                _PosEnc(features=32), nn.Linear(192, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(), nn.Linear(128, len(layers), bias=False), nn.Softmax(dim=-1),
            )
            self.scale_selector_mlp = nn.Sequential(
                _PosEnc(features=32), nn.Linear(192, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1, bias=False), nn.Sigmoid(),
            )
            dummy = nn.Linear(bottleneck_dim, n_verts)
            self.weight = nn.Parameter(dummy.weight)
            self.bias = nn.Parameter(dummy.bias)

        def forward(self, local_tokens, global_tokens, coords):
            from einops import repeat
            for layer in self.layers:
                local_tokens[layer] = self.local_token_bottleneck[layer](local_tokens[layer])
                global_tokens[layer] = self.global_token_bottleneck[layer](global_tokens[layer])

            sel_space = self.space_selector_mlp(coords)
            sel_layer = self.layer_selector_mlp(coords)
            sel_scale = self.scale_selector_mlp(coords)

            gt = torch.stack(list(global_tokens.values()), dim=-1)
            gt = repeat(gt, "b d l -> b n d l", n=1)
            _sl = repeat(sel_layer, "n l -> b n d l", b=1, d=1)
            v_global = (gt * _sl).sum(dim=-1)

            bsz = local_tokens[self.layers[0]].shape[0]
            _ss = repeat(sel_space, "n d -> b n c d", b=bsz, c=1)
            _sl2 = repeat(sel_layer, "n l -> b n l", b=1)
            v_local = None
            for i, layer in enumerate(self.layers):
                lt = local_tokens[layer]
                vl = nn.functional.grid_sample(lt, _ss, align_corners=False,
                                               mode="bilinear", padding_mode="zeros")
                vl = vl.squeeze(-1).permute(0, 2, 1)
                vl = vl * _sl2[:, :, i].unsqueeze(-1)
                v_local = vl if v_local is None else v_local + vl

            _sc = repeat(sel_scale, "n d -> b n d", b=1)
            v = (1 - _sc) * v_local + _sc * v_global

            w = self.weight.unsqueeze(0)
            b = self.bias.unsqueeze(0)
            y = (v * w).mean(dim=-1) + b
            return y, None

    checkpoint_url = "https://raw.githubusercontent.com/huzeyann/BrainDecodesDeepNets/main/assets/weights/clip_factorTopy.pth"
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)

    ft_model = FactorTopy(n_vertices, list(range(n_layers)), [layer_width] * n_layers, 128)
    ft_model.load_state_dict(state_dict)
    ft_model = ft_model.eval().cuda()
    vision = vision.cuda()

    print(f"[BDDN Batch {batch_idx}] Loaded in {time.time() - t0:.1f}s ({n_vertices} vertices)")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    nsd_set = set(nsdgeneral_indices.tolist())
    nsd_to_local = {v: i for i, v in enumerate(nsdgeneral_indices)}

    results = []
    for img_name in image_names:
        img_path = Path(IMAGE_DIR) / img_name
        result_path = Path(RESULTS_DIR) / f"bddn_{img_name.rsplit('.', 1)[0]}.json"

        if result_path.exists():
            print(f"  [BDDN {batch_idx}] {img_name}: cached")
            results.append(json.loads(result_path.read_text()))
            continue

        if not img_path.exists():
            results.append({"image": img_name, "error": f"not found: {img_path}"})
            continue

        print(f"  [BDDN {batch_idx}] {img_name}: encoding...")
        t1 = time.time()

        try:
            img = Image.open(str(img_path)).convert("RGB")
            x = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                local_tokens, global_tokens = extract_clip_tokens(x)
                preds, _ = ft_model(local_tokens, global_tokens, coords_tensor)
            preds_np = preds.squeeze(0).cpu().numpy()  # (37984,)
        except Exception as e:
            results.append({"image": img_name, "error": str(e)[:500]})
            continue

        elapsed = time.time() - t1

        # Save full vertex array
        npy_path = Path(RESULTS_DIR) / f"bddn_{img_name.rsplit('.', 1)[0]}.npy"
        np.save(str(npy_path), preds_np)

        # Extract ROI means
        roi_means = {}
        for roi_name, fsavg_idx in roi_dict.items():
            local = [nsd_to_local[v] for v in fsavg_idx if v in nsd_set]
            if local:
                roi_means[roi_name] = float(preds_np[local].mean())

        result = {
            "image": img_name,
            "model": "bddn",
            "n_vertices": len(preds_np),
            "elapsed": round(elapsed, 2),
            "preds_path": str(npy_path),
            "roi_means": roi_means,
        }

        result_path.write_text(json.dumps(result, indent=2))
        vol.commit()
        results.append(result)

    return results


# ── CLIP control features ──────────────────────────────────────────────

@app.function(
    image=bddn_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    timeout=1800,
    memory=16384,
)
def extract_clip_batch(pairs: list[dict], batch_idx: int) -> list[dict]:
    """Extract CLIP embeddings and compute CLIP score vs prompt."""
    import time
    import numpy as np
    import torch
    from pathlib import Path
    from PIL import Image

    print(f"[CLIP Batch {batch_idx}] Loading CLIP...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_l_s1b_b8k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.eval().cuda()

    results = []
    for pair in pairs:
        for side in ["preferred_image", "nonpreferred_image"]:
            img_name = pair[side]
            img_path = Path(IMAGE_DIR) / img_name

            if not img_path.exists():
                results.append({"image": img_name, "error": "not found"})
                continue

            try:
                img = Image.open(str(img_path)).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).cuda()
                text_tokens = tokenizer([pair["prompt"][:77]]).cuda()

                with torch.no_grad():
                    img_feat = model.encode_image(img_tensor)
                    txt_feat = model.encode_text(text_tokens)

                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                    clip_score = float((img_feat @ txt_feat.T).squeeze())

                results.append({
                    "image": img_name,
                    "clip_score": clip_score,
                    "is_preferred": side == "preferred_image",
                    "pair_id": pair["pair_id"],
                })
            except Exception as e:
                results.append({"image": img_name, "error": str(e)[:300]})

    return results


# ── Local entrypoint ───────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    from pathlib import Path

    cache = Path("cache/preference")
    pairs_path = cache / "selected_pairs.json"

    if not pairs_path.exists():
        print("Run 00_sample_pairs.py first!")
        return

    pairs = json.loads(pairs_path.read_text())
    print(f"Loaded {len(pairs)} pairs")

    # Collect all unique image names
    all_images = set()
    for p in pairs:
        all_images.add(p["preferred_image"])
        all_images.add(p["nonpreferred_image"])
    all_images = sorted(all_images)
    print(f"Unique images: {len(all_images)}")

    # Upload images to Modal volume
    print("Uploading images to Modal volume...")
    _vol = modal.Volume.from_name("braingym-cache")
    with _vol.batch_upload(force=True) as batch:
        for i, img_name in enumerate(all_images):
            local_path = cache / "images" / img_name
            if local_path.exists():
                batch.put_file(str(local_path), f"preference_images/{img_name}")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_images)}")
    print(f"Uploaded {len(all_images)} images")

    # Batch images
    BATCH_SIZE = 20
    batches = [all_images[i:i + BATCH_SIZE] for i in range(0, len(all_images), BATCH_SIZE)]
    print(f"\n--- TRIBE: {len(all_images)} images in {len(batches)} batches ---")

    # Run TRIBE
    tribe_results = []
    for result_batch in encode_tribe_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(batches)],
        order_outputs=False,
    ):
        tribe_results.extend(result_batch)
        successes = len([r for r in tribe_results if "error" not in r])
        errors = len([r for r in tribe_results if "error" in r])
        print(f"  TRIBE progress: {successes} ok, {errors} errors")

    # Run BDDN
    print(f"\n--- BDDN: {len(all_images)} images in {len(batches)} batches ---")
    bddn_results = []
    for result_batch in encode_bddn_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(batches)],
        order_outputs=False,
    ):
        bddn_results.extend(result_batch)
        successes = len([r for r in bddn_results if "error" not in r])
        errors = len([r for r in bddn_results if "error" in r])
        print(f"  BDDN progress: {successes} ok, {errors} errors")

    # Run CLIP
    print(f"\n--- CLIP: {len(pairs)} pairs ---")
    CLIP_BATCH = 25
    pair_batches = [pairs[i:i + CLIP_BATCH] for i in range(0, len(pairs), CLIP_BATCH)]
    clip_results = []
    for result_batch in extract_clip_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(pair_batches)],
        order_outputs=False,
    ):
        clip_results.extend(result_batch)
    print(f"  CLIP: {len(clip_results)} results")

    # Save combined results
    tribe_by_image = {r["image"]: r for r in tribe_results if "error" not in r}
    bddn_by_image = {r["image"]: r for r in bddn_results if "error" not in r}
    clip_by_image = {r["image"]: r for r in clip_results if "error" not in r}

    combined = []
    for pair in pairs:
        for side, is_pref in [("preferred_image", True), ("nonpreferred_image", False)]:
            img = pair[side]
            entry = {
                "pair_id": pair["pair_id"],
                "image": img,
                "is_preferred": is_pref,
                "prompt": pair["prompt"],
                "model": pair[f"model_{'preferred' if is_pref else 'nonpreferred'}"],
                "confidence": pair["confidence"],
            }
            if img in tribe_by_image:
                t = tribe_by_image[img]
                entry["tribe_network_means"] = t.get("network_means", {})
                entry["tribe_attention"] = t.get("attention")
                entry["tribe_engagement"] = t.get("engagement")
                entry["tribe_cognitive_load"] = t.get("cognitive_load")
            if img in bddn_by_image:
                entry["bddn_roi_means"] = bddn_by_image[img].get("roi_means", {})
            if img in clip_by_image:
                entry["clip_score"] = clip_by_image[img].get("clip_score")
            combined.append(entry)

    out_path = cache / "encoding_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Summary
    tribe_ok = len([r for r in tribe_results if "error" not in r])
    bddn_ok = len([r for r in bddn_results if "error" not in r])
    clip_ok = len([r for r in clip_results if "error" not in r])
    print(f"\n{'='*50}")
    print(f"TRIBE: {tribe_ok}/{len(all_images)} encoded")
    print(f"BDDN:  {bddn_ok}/{len(all_images)} encoded")
    print(f"CLIP:  {clip_ok}/{len(all_images)*2} scores")
    print(f"Combined: {len(combined)} entries saved to {out_path}")

    # Print errors
    for r in tribe_results + bddn_results:
        if "error" in r:
            print(f"  ERROR [{r.get('model', '?')}] {r.get('image', '?')}: {r['error'][:80]}")
