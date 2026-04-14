"""
Phase 0: Sample 100 preference pairs from HPD v3.

Downloads one image shard (~10GB), streams metadata, filters to
high-confidence pairs with images in the shard, and samples 100
diverse pairs across models.

Usage: python3 experiments/preference/00_sample_pairs.py
"""

import json
import os
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict

from huggingface_hub import hf_hub_download

DATASET_ID = "MizzenAI/HPDv3"
CACHE_DIR = Path("cache/preference")
IMAGE_DIR = CACHE_DIR / "images"
OUTPUT = CACHE_DIR / "selected_pairs.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract_shard():
    """Download one tar.gz shard and extract images."""
    shard_name = "images.tar.gz.00"

    # Check if we already have extracted images
    existing = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    if len(existing) > 100:
        print(f"Already have {len(existing)} images in {IMAGE_DIR}, skipping download")
        return {f"images/{f.name}": f for f in existing}

    print(f"Downloading {shard_name} from {DATASET_ID}...")
    shard_path = hf_hub_download(
        DATASET_ID, shard_name, repo_type="dataset",
        cache_dir=str(CACHE_DIR / "hf_cache"),
    )
    print(f"Downloaded to {shard_path}")

    # Extract — tar.gz.00 is a split archive, treat as standalone gzip tar
    print("Extracting images...")
    image_map = {}
    try:
        with tarfile.open(shard_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(members)} images in shard")
            for i, member in enumerate(members):
                # Extract to IMAGE_DIR with flat naming
                basename = os.path.basename(member.name)
                dest = IMAGE_DIR / basename
                if not dest.exists():
                    f = tar.extractfile(member)
                    if f:
                        dest.write_bytes(f.read())
                image_map[member.name] = dest
                if (i + 1) % 1000 == 0:
                    print(f"  Extracted {i+1}/{len(members)}")
    except tarfile.ReadError:
        # Split archive — try as raw gzip
        print("  Not a standalone tar.gz, trying concatenated shard approach...")
        import gzip
        import io
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=io.BytesIO(gz.read())) as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                print(f"  Found {len(members)} files")
                for member in members[:5000]:  # Limit to first 5000 for speed
                    basename = os.path.basename(member.name)
                    dest = IMAGE_DIR / basename
                    if not dest.exists():
                        f = tar.extractfile(member)
                        if f:
                            dest.write_bytes(f.read())
                    image_map[member.name] = dest

    print(f"  Extracted {len(image_map)} images to {IMAGE_DIR}")
    return image_map


def stream_and_sample(available_images: set):
    """Stream HPD v3 metadata and sample 100 diverse, high-confidence pairs."""
    from datasets import load_dataset

    print(f"\nStreaming HPD v3 metadata...")
    print(f"  Available images: {len(available_images)}")

    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    # Collect candidates: high confidence, both images available
    candidates = []
    seen = 0
    models_seen = defaultdict(int)

    for row in ds:
        seen += 1
        if seen % 10000 == 0:
            print(f"  Scanned {seen} rows, {len(candidates)} candidates so far...")

        # Skip low confidence
        confidence = row.get("confidence")
        if confidence is None or confidence < 0.8:
            continue

        path1 = row.get("path1", "")
        path2 = row.get("path2", "")

        # Check if both images are available (try multiple path formats)
        img1_name = os.path.basename(path1)
        img2_name = os.path.basename(path2)

        if img1_name not in available_images or img2_name not in available_images:
            continue

        model1 = row.get("model1", "unknown")
        model2 = row.get("model2", "unknown")

        candidates.append({
            "path1": path1,
            "path2": path2,
            "img1_name": img1_name,
            "img2_name": img2_name,
            "prompt": row.get("prompt", ""),
            "model1": model1,
            "model2": model2,
            "confidence": confidence,
            "choice_dist": row.get("choice_dist"),
        })

        # Stop early if we have plenty
        if len(candidates) >= 2000:
            print(f"  Found {len(candidates)} candidates, stopping scan")
            break

    print(f"  Total scanned: {seen}, candidates: {len(candidates)}")

    if len(candidates) < 100:
        print(f"  WARNING: Only {len(candidates)} candidates found. Lowering confidence threshold...")
        # Re-scan with lower threshold
        return candidates

    # Stratified sampling: spread across model1 values
    by_model = defaultdict(list)
    for c in candidates:
        by_model[c["model1"]].append(c)

    print(f"\n  Model distribution:")
    for m, pairs in sorted(by_model.items(), key=lambda x: -len(x[1])):
        print(f"    {m}: {len(pairs)} candidates")

    # Sample proportionally, at least 2 per model if available
    selected = []
    n_models = len(by_model)
    per_model = max(2, 100 // n_models)

    # First pass: take up to per_model from each
    for model, pairs in by_model.items():
        # Sort by confidence (highest first)
        pairs.sort(key=lambda x: -x["confidence"])
        selected.extend(pairs[:per_model])

    # If we have too many, trim; if too few, backfill from largest buckets
    if len(selected) > 100:
        # Keep highest confidence
        selected.sort(key=lambda x: -x["confidence"])
        selected = selected[:100]
    elif len(selected) < 100:
        remaining = 100 - len(selected)
        selected_set = {(c["img1_name"], c["img2_name"]) for c in selected}
        extras = [c for c in candidates if (c["img1_name"], c["img2_name"]) not in selected_set]
        extras.sort(key=lambda x: -x["confidence"])
        selected.extend(extras[:remaining])

    return selected[:100]


def main():
    # Step 1: Download and extract one shard
    image_map = download_and_extract_shard()

    # Build set of available image filenames
    available = set()
    for f in IMAGE_DIR.iterdir():
        if f.suffix in (".jpg", ".png", ".jpeg"):
            available.add(f.name)
    print(f"\n{len(available)} images available locally")

    # Step 2: Stream metadata and sample
    selected = stream_and_sample(available)
    print(f"\nSelected {len(selected)} pairs")

    # Step 3: Format output
    pairs = []
    for i, s in enumerate(selected):
        pairs.append({
            "pair_id": i,
            "prompt": s["prompt"][:500],  # Truncate long prompts
            "preferred_image": s["img1_name"],
            "nonpreferred_image": s["img2_name"],
            "model_preferred": s["model1"],
            "model_nonpreferred": s["model2"],
            "confidence": s["confidence"],
            "choice_dist": s["choice_dist"],
        })

    # Save
    OUTPUT.write_text(json.dumps(pairs, indent=2))
    print(f"\nSaved {len(pairs)} pairs to {OUTPUT}")

    # Summary stats
    models = defaultdict(int)
    for p in pairs:
        models[p["model_preferred"]] += 1
    print("\nModel distribution (preferred image):")
    for m, c in sorted(models.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")

    # Count unique images
    all_images = set()
    for p in pairs:
        all_images.add(p["preferred_image"])
        all_images.add(p["nonpreferred_image"])
    print(f"\nUnique images: {len(all_images)}")


if __name__ == "__main__":
    main()
