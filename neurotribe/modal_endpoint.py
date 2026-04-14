"""
Modal web endpoint for TRIBEv2 inference.

Accepts a file upload (video, audio, or text), runs TRIBEv2,
returns predicted fMRI + network timeseries + raw vertex data.

Deploy: modal deploy neurotribe/modal_endpoint.py
Test:   modal serve neurotribe/modal_endpoint.py
"""

import modal
import json

app = modal.App("neurotribe-api")

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
        "fastapi",
    )
    .pip_install("neuralset==0.0.2", "neuraltrain==0.0.2")
    .pip_install("git+https://github.com/facebookresearch/tribev2.git")
)

vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)
CACHE_DIR = "/cache"


@app.cls(
    image=tribe_image,
    gpu="A10G",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    memory=32768,
    scaledown_window=300,
)
class TribeEndpoint:
    @modal.enter()
    def load_model(self):
        import os
        import time
        import torch
        import numpy as np
        import nibabel as nib
        from pathlib import Path
        from urllib.request import urlretrieve
        from tribev2.demo_utils import TribeModel

        os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

        print("Loading TRIBEv2...")
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=CACHE_DIR,
            device=device,
            config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
        )
        print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

        # Load Schaefer parcellation
        SCHAEFER_BASE = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/FreeSurfer5.3/fsaverage5/label"
        )
        ANNOT_NAME = "Schaefer2018_400Parcels_7Networks_order.annot"
        self.NETWORK_ALIASES = {
            "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
            "SalVentAttn": "VAN", "Limbic": "Limbic", "SomMot": "SomMot", "Vis": "Vis",
        }

        parcellation_dir = Path(CACHE_DIR) / "schaefer"
        parcellation_dir.mkdir(parents=True, exist_ok=True)
        self.networks = {}
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
                    net = self.NETWORK_ALIASES.get(parts[2], parts[2])
                    if net not in nets:
                        nets[net] = []
                    nets[net].extend(np.where(labels == idx)[0].tolist())
            self.networks[hemi] = {k: np.array(v) for k, v in nets.items()}

        self.N_VERTICES_HEMI = 10242
        print("Ready.")

    @modal.method()
    def predict(self, file_bytes: bytes, filename: str, modality: str) -> dict:
        """Run TRIBE on uploaded file. Returns network timeseries + downsampled vertex data."""
        import numpy as np
        import tempfile
        import time
        from pathlib import Path

        t0 = time.time()

        # Write uploaded file to temp
        suffix = Path(filename).suffix or {
            "video": ".mp4", "audio": ".wav", "text": ".txt"
        }.get(modality, ".mp4")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name

        # Build events dataframe
        try:
            if modality == "video":
                df = self.model.get_events_dataframe(video_path=tmp_path)
            elif modality == "audio":
                df = self.model.get_events_dataframe(audio_path=tmp_path)
            elif modality == "text":
                df = self.model.get_events_dataframe(text_path=tmp_path)
            else:
                return {"error": f"Unknown modality: {modality}"}

            # Run prediction
            preds, segments = self.model.predict(events=df)
        except Exception as e:
            return {"error": str(e)[:1000]}

        n_tp = preds.shape[0]
        preds_lh = preds[:, :self.N_VERTICES_HEMI]
        preds_rh = preds[:, self.N_VERTICES_HEMI:2 * self.N_VERTICES_HEMI]

        # Network-level timeseries
        net_signals = {}
        for net_name in self.NETWORK_ALIASES.values():
            lh_idx = self.networks["lh"].get(net_name, np.array([], dtype=int))
            rh_idx = self.networks["rh"].get(net_name, np.array([], dtype=int))
            sig_lh = preds_lh[:, lh_idx].mean(axis=1) if len(lh_idx) > 0 else np.zeros(n_tp)
            sig_rh = preds_rh[:, rh_idx].mean(axis=1) if len(rh_idx) > 0 else np.zeros(n_tp)
            net_signals[net_name] = ((sig_lh + sig_rh) / 2).tolist()

        net_means = {k: float(np.mean(v)) for k, v in net_signals.items()}

        # Downsample vertex data for visualization (every vertex, sampled timepoints)
        # Send full vertex data for a subset of timepoints to keep response size manageable
        viz_step = max(1, n_tp // 20)  # at most ~20 timepoints for viz
        viz_indices = list(range(0, n_tp, viz_step))[:20]
        viz_preds = preds[viz_indices].tolist()

        # Segment info (text/events at each timestep)
        segment_info = []
        for seg in segments:
            info = {"events": []}
            if hasattr(seg, "events"):
                for ev in seg.events:
                    ev_dict = {}
                    if hasattr(ev, "text") and ev.text:
                        ev_dict["text"] = str(ev.text)
                    if hasattr(ev, "type"):
                        ev_dict["type"] = str(ev.type)
                    info["events"].append(ev_dict)
            segment_info.append(info)

        elapsed = time.time() - t0

        return {
            "n_timepoints": n_tp,
            "n_vertices": int(preds.shape[1]),
            "elapsed_seconds": round(elapsed, 1),
            "filename": filename,
            "modality": modality,
            # Scalar summaries
            "net_means": net_means,
            "attention": net_means["DAN"] - net_means["DMN"],
            "engagement": -net_means["DMN"],
            "cognitive_load": net_means["FPN"] + net_means["DAN"] - net_means["DMN"],
            # Timeseries
            "timeseries": net_signals,
            "attention_ts": [net_signals["DAN"][t] - net_signals["DMN"][t] for t in range(n_tp)],
            # Visualization data (downsampled timepoints, full vertices)
            "viz_timepoint_indices": viz_indices,
            "viz_preds": viz_preds,
            # Segment info
            "segments": segment_info[:n_tp],
        }


@app.local_entrypoint()
def test():
    """Quick test with a sample video."""
    from pathlib import Path

    endpoint = TribeEndpoint()

    # Test with text
    text = "Stop scrolling. This is the most important thing you'll hear today."
    result = endpoint.predict.remote(
        file_bytes=text.encode("utf-8"),
        filename="test.txt",
        modality="text",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "viz_preds"}, indent=2))
