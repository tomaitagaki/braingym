"""
Recover all cached per-video results from the Modal volume.
Run this if the main pipeline dies mid-batch.

Usage:
    modal run recover_results.py
"""

import json
import modal

app = modal.App("braingym-recover")

vol = modal.Volume.from_name("braingym-cache")

RESULTS_DIR = "/cache/results"
TSINGHUA_RESULTS_DIR = "/cache/tsinghua_results"


@app.function(volumes={"/cache": vol})
def collect_results() -> dict:
    """Read all per-video result JSONs from both result dirs on the Modal volume."""
    from pathlib import Path

    output = {"tiktok": [], "tsinghua": []}

    for _label, _dir in [("tiktok", RESULTS_DIR), ("tsinghua", TSINGHUA_RESULTS_DIR)]:
        _path = Path(_dir)
        if not _path.exists():
            print(f"No {_label} results directory")
            continue
        for _f in sorted(_path.glob("*.json")):
            try:
                _data = json.loads(_f.read_text())
                output[_label].append(_data)
            except Exception as e:
                print(f"  SKIP {_f.name}: {e}")
        print(f"{_label}: {len(output[_label])} results")

    return output


@app.local_entrypoint()
def main():
    from pathlib import Path

    output = collect_results.remote()

    cache = Path("./cache")

    # Save TikTok results
    _tiktok = [r for r in output["tiktok"] if "error" not in r]
    if _tiktok:
        _path = cache / "tribe_tiktok_train_results.json"
        with open(_path, "w") as f:
            json.dump(_tiktok, f, indent=2)
        print(f"TikTok: {len(_tiktok)} results -> {_path}")

    # Save Tsinghua results
    _tsinghua = [r for r in output["tsinghua"] if "error" not in r]
    if _tsinghua:
        _path = cache / "tsinghua" / "selected" / "tribe_results.json"
        _path.parent.mkdir(parents=True, exist_ok=True)
        with open(_path, "w") as f:
            json.dump(_tsinghua, f, indent=2)
        print(f"Tsinghua: {len(_tsinghua)} results -> {_path}")

    print(f"\nTotal recovered: {len(_tiktok)} TikTok + {len(_tsinghua)} Tsinghua")
