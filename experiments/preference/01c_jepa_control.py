"""
Control experiment: Extract raw V-JEPA2 embeddings for preference prediction.

Tests whether V-JEPA2 features alone predict preference,
or if TRIBE's brain mapping adds signal.

Usage: modal run experiments/preference/01c_jepa_control.py
"""

import json
import modal

app = modal.App("braingym-preference-jepa")

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
        "huggingface_hub",
        "Pillow",
        "moviepy>=2.2.1",
        "transformers",
    )
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)
CACHE_DIR = "/cache"
IMAGE_DIR = "/cache/preference_images"


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    memory=32768,
)
def extract_jepa_batch(image_names: list[str], batch_idx: int) -> list[dict]:
    """Extract raw V-JEPA2 embeddings for each image."""
    import os
    import time
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from pathlib import Path
    from PIL import Image

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")

    print(f"[JEPA Batch {batch_idx}] Loading V-JEPA2...")
    t0 = time.time()

    from transformers import AutoModel, AutoProcessor

    # V-JEPA2 Giant — same model TRIBE uses
    model_name = "facebook/vjepa2-vitg-fpc64-256"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.eval().cuda()
    print(f"[JEPA Batch {batch_idx}] Loaded in {time.time() - t0:.1f}s")

    results = []
    for img_name in image_names:
        img_path = Path(IMAGE_DIR) / img_name

        if not img_path.exists():
            results.append({"image": img_name, "error": "not found"})
            continue

        print(f"  [JEPA {batch_idx}] {img_name}...")
        t1 = time.time()

        try:
            # Create a minimal video from the image (V-JEPA2 expects video)
            # 16 frames of the same image
            img = Image.open(str(img_path)).convert("RGB")
            img = img.resize((256, 256))

            # Stack as fake video frames
            frames = [np.array(img)] * 16  # 16 identical frames
            video = np.stack(frames)  # (16, 256, 256, 3)

            inputs = processor(video, return_tensors="pt")
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Get the pooled/CLS representation
            if hasattr(outputs, 'last_hidden_state'):
                # Average over spatial and temporal tokens
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            elif hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output.squeeze(0)
            else:
                # Fallback: use whatever the model outputs
                embedding = outputs[0].mean(dim=1).squeeze(0)

            emb_np = embedding.float().cpu().numpy()
            elapsed = time.time() - t1

            results.append({
                "image": img_name,
                "embedding": emb_np.tolist(),
                "embedding_dim": len(emb_np),
                "elapsed": round(elapsed, 2),
            })
            print(f"    -> {emb_np.shape} in {elapsed:.1f}s")

        except Exception as e:
            print(f"    -> ERROR: {e}")
            results.append({"image": img_name, "error": str(e)[:500]})

    return results


@app.local_entrypoint()
def main():
    from pathlib import Path

    cache = Path("cache/preference")
    pairs = json.loads((cache / "selected_pairs.json").read_text())

    all_images = set()
    for p in pairs:
        all_images.add(p["preferred_image"])
        all_images.add(p["nonpreferred_image"])
    all_images = sorted(all_images)
    print(f"{len(all_images)} images")

    BATCH_SIZE = 20
    batches = [all_images[i:i + BATCH_SIZE] for i in range(0, len(all_images), BATCH_SIZE)]
    print(f"Extracting V-JEPA2 in {len(batches)} batches...")

    all_results = []
    for result_batch in extract_jepa_batch.starmap(
        [(batch, idx + 1) for idx, batch in enumerate(batches)],
        order_outputs=False,
    ):
        all_results.extend(result_batch)
        ok = len([r for r in all_results if "error" not in r])
        err = len([r for r in all_results if "error" in r])
        print(f"  Progress: {ok} ok, {err} errors")

    # Save embeddings as JSON (analysis runs separately with system python)
    out_path = cache / "jepa_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f)

    ok = len([r for r in all_results if "error" not in r])
    err = len([r for r in all_results if "error" in r])
    print(f"\nDone: {ok} ok, {err} errors")
    print(f"Saved to {out_path}")
    print(f"Run analysis: python3 experiments/preference/01c_analyze_jepa.py")
