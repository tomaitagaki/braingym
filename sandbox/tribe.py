"""
TribeScorer: wraps TRIBEv2 model for the sandbox loop.

Handles model loading, video -> prediction, and network signal extraction.
Supports mock mode for testing without GPU/Tribe installed.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from pathlib import Path


class TribeScorer:
    """Score videos with TRIBEv2 brain predictions.

    Args:
        mode: "local" (load model in-process), "mock" (random predictions for testing)
        device: torch device for local mode
        cache_dir: directory to cache predictions
    """

    N_VERTICES = 20484  # fsaverage5: 10242 per hemisphere

    def __init__(
        self,
        mode: str = "mock",
        device: str = "mps",
        cache_dir: Path | str = "cache/sandbox_tribe",
    ):
        self.mode = mode
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._network_masks = None

    def predict(self, media_path: Path | str) -> np.ndarray:
        """Run Tribe on a video or audio file, return (n_timesteps, 20484) predictions."""
        media_path = Path(media_path)
        cache_key = self._cache_key(media_path)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            return np.load(cache_path)

        if self.mode == "mock":
            preds = self._mock_predict(media_path)
        elif self.mode == "local":
            preds = self._local_predict(media_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        np.save(cache_path, preds)
        return preds

    def extract_signals(self, preds: np.ndarray) -> dict[str, np.ndarray]:
        """Extract per-network mean signals from vertex predictions.

        Uses quantify.py's Schaefer 400-parcel 7-network parcellation.
        """
        if self._network_masks is None:
            self._load_parcellation()

        n_hemi = self.N_VERTICES // 2
        preds_lh = preds[:, :n_hemi]
        preds_rh = preds[:, n_hemi : 2 * n_hemi]

        signals = {}
        for net_name, masks in self._network_masks.items():
            lh_idx, rh_idx = masks["lh"], masks["rh"]
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else 0
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else 0
            signals[net_name] = (sig_lh + sig_rh) / 2

        return signals

    # -- private --

    def _cache_key(self, video_path: Path) -> str:
        stat = video_path.stat()
        raw = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _mock_predict(self, video_path: Path) -> np.ndarray:
        """Generate deterministic mock predictions based on filename."""
        # Use filename as seed for reproducibility
        seed = int(hashlib.md5(video_path.name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # Estimate duration: ~1 TR per second, assume 15-30s videos
        n_trs = rng.integers(10, 40)
        preds = rng.standard_normal((n_trs, self.N_VERTICES)) * 0.1
        # Add some temporal structure (smoothing)
        for t in range(1, n_trs):
            preds[t] = 0.7 * preds[t - 1] + 0.3 * preds[t]
        return preds

    def _local_predict(self, media_path: Path) -> np.ndarray:
        """Run TRIBEv2 locally. Accepts video (.mp4) or audio (.wav/.mp3)."""
        if self._model is None:
            self._load_model()

        suffix = media_path.suffix.lower()
        if suffix in (".wav", ".mp3", ".flac", ".ogg"):
            df = self._model.get_events_dataframe(audio_path=str(media_path))
        else:
            df = self._model.get_events_dataframe(video_path=str(media_path))
        preds, _segments = self._model.predict(events=df)
        return preds

    def _load_model(self) -> None:
        import torch
        from tribev2.demo_utils import TribeModel

        device = self.device
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=str(self.cache_dir / "model"),
            device=device,
            config_update={
                "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
                "data.text_feature.device": "cpu",
                "data.audio_feature.device": "cpu",
                "data.video_feature.image.device": "cpu",
            },
        )

    def _load_parcellation(self) -> None:
        """Load Schaefer 7-network parcellation for fsaverage5."""
        import nibabel as nib
        from urllib.request import urlretrieve

        schaefer_base = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/FreeSurfer5.3/fsaverage5/label"
        )
        annot_name = "Schaefer2018_400Parcels_7Networks_order.annot"
        aliases = {
            "DorsAttn": "DAN",
            "Default": "DMN",
            "Cont": "FPN",
            "SalVentAttn": "VAN",
            "Limbic": "Limbic",
            "SomMot": "SomMot",
            "Vis": "Vis",
        }

        parc_dir = self.cache_dir / "schaefer"
        parc_dir.mkdir(exist_ok=True)

        self._network_masks = {}
        for hemi in ("lh", "rh"):
            local = parc_dir / f"{hemi}.{annot_name}"
            if not local.exists():
                urlretrieve(f"{schaefer_base}/{hemi}.{annot_name}", local)

            labels, _ctab, names = nib.freesurfer.read_annot(local)
            names = [n.decode() if isinstance(n, bytes) else n for n in names]

            for idx, name in enumerate(names):
                parts = name.split("_")
                if len(parts) >= 3 and parts[0] == "7Networks":
                    net = aliases.get(parts[2], parts[2])
                    if net not in self._network_masks:
                        self._network_masks[net] = {"lh": [], "rh": []}
                    self._network_masks[net][hemi].extend(
                        np.where(labels == idx)[0].tolist()
                    )

        # Convert to numpy arrays
        for net in self._network_masks:
            for hemi in ("lh", "rh"):
                self._network_masks[net][hemi] = np.array(
                    self._network_masks[net][hemi]
                )
