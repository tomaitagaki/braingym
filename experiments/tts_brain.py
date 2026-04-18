"""
Experiment: Does TTS delivery affect predicted brain response?

Same text, different TTS voices → Tribe → compare brain predictions.
Tests whether prosody/pacing/voice quality modulates brain encoding
beyond the text content itself.

Voices:
  - macOS Samantha (US, neural)
  - macOS Daniel (UK, neural)
  - gTTS (Google Translate, flat/robotic)

Usage:
    source tribev2/.venv/bin/activate
    python experiments/tts_brain.py
"""

import subprocess
import time
import json
import numpy as np
from pathlib import Path

CACHE = Path("cache/tts_experiment")
CACHE.mkdir(parents=True, exist_ok=True)

# --- Test passages: varying complexity ---
PASSAGES = {
    "simple": (
        "The sun is shining today. Birds are singing in the trees. "
        "It is a beautiful morning. The air feels warm and fresh."
    ),
    "narrative": (
        "She opened the door slowly, unsure of what she would find. "
        "The room was dark, but a faint light flickered in the corner. "
        "Her heart raced as she stepped inside, the floorboards creaking beneath her feet."
    ),
    "technical": (
        "The transformer architecture uses multi-head self-attention "
        "to compute representations of input sequences in parallel. "
        "Each attention head learns different aspects of the relationships "
        "between tokens, enabling the model to capture both local and global dependencies."
    ),
}

# --- Generate TTS audio files ---

def generate_macos_tts(text, voice, output_path):
    """Generate audio using macOS say command."""
    aiff_path = output_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
    # Convert AIFF to WAV using ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", str(aiff_path), "-ar", "16000", "-ac", "1",
        str(output_path)
    ], capture_output=True, check=True)
    aiff_path.unlink()
    return output_path


def generate_gtts(text, output_path):
    """Generate audio using Google TTS."""
    from gtts import gTTS
    mp3_path = output_path.with_suffix(".mp3")
    tts = gTTS(text, lang="en")
    tts.save(str(mp3_path))
    # Convert MP3 to WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1",
        str(output_path)
    ], capture_output=True, check=True)
    mp3_path.unlink()
    return output_path


def main():
    voices = {
        "macos_samantha": lambda text, path: generate_macos_tts(text, "Samantha", path),
        "macos_daniel": lambda text, path: generate_macos_tts(text, "Daniel", path),
        "gtts_google": lambda text, path: generate_gtts(text, path),
    }

    # Step 1: Generate all audio files
    print("=== Generating TTS audio ===")
    audio_files = {}
    for passage_name, text in PASSAGES.items():
        for voice_name, gen_fn in voices.items():
            key = f"{passage_name}__{voice_name}"
            wav_path = CACHE / f"{key}.wav"
            if not wav_path.exists():
                print(f"  Generating: {key}")
                gen_fn(text, wav_path)
            else:
                print(f"  Cached: {key}")
            audio_files[key] = wav_path

    # Step 2: Run Tribe on each audio file
    print("\n=== Running Tribe v2 ===")
    import torch
    from tribev2.demo_utils import TribeModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(CACHE),
        device=device,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )

    results = {}
    for key, wav_path in audio_files.items():
        print(f"\n  Processing: {key}")
        t0 = time.time()
        df = model.get_events_dataframe(audio_path=str(wav_path))
        preds, segments = model.predict(events=df)
        elapsed = time.time() - t0
        print(f"    {preds.shape} in {elapsed:.1f}s")

        # Extract network means
        N = 10242
        preds_lh = preds[:, :N]
        preds_rh = preds[:, N:2*N]

        # Quick parcellation (use quantify if available)
        result = {
            "key": key,
            "passage": key.split("__")[0],
            "voice": key.split("__")[1],
            "n_timepoints": int(preds.shape[0]),
            "mean_activation": float(preds.mean()),
            "std_activation": float(preds.std()),
            "preds_mean_per_vertex": preds.mean(axis=0).tolist(),
        }
        results[key] = result

    # Step 3: Compare voices for same passage
    print("\n" + "=" * 70)
    print("RESULTS: Same text, different voices")
    print("=" * 70)

    for passage_name in PASSAGES:
        print(f"\n--- {passage_name} ---")
        passage_results = {k: v for k, v in results.items() if v["passage"] == passage_name}

        # Compare mean activation
        for key, r in passage_results.items():
            print(f"  {r['voice']:20s}: mean={r['mean_activation']:+.4f}  std={r['std_activation']:.4f}  TRs={r['n_timepoints']}")

        # Compute pairwise correlation between voice predictions
        keys = list(passage_results.keys())
        if len(keys) >= 2:
            print(f"\n  Pairwise spatial correlation (do voices produce similar brain patterns?):")
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    p1 = np.array(passage_results[keys[i]]["preds_mean_per_vertex"])
                    p2 = np.array(passage_results[keys[j]]["preds_mean_per_vertex"])
                    r = np.corrcoef(p1, p2)[0, 1]
                    v1 = passage_results[keys[i]]["voice"]
                    v2 = passage_results[keys[j]]["voice"]
                    print(f"    {v1} vs {v2}: r={r:.4f}")

    # Step 4: Compare passages for same voice
    print("\n" + "=" * 70)
    print("RESULTS: Same voice, different text")
    print("=" * 70)

    for voice_name in voices:
        print(f"\n--- {voice_name} ---")
        voice_results = {k: v for k, v in results.items() if v["voice"] == voice_name}
        for key, r in voice_results.items():
            print(f"  {r['passage']:15s}: mean={r['mean_activation']:+.4f}  TRs={r['n_timepoints']}")

        keys = list(voice_results.keys())
        if len(keys) >= 2:
            print(f"\n  Pairwise spatial correlation:")
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    p1 = np.array(voice_results[keys[i]]["preds_mean_per_vertex"])
                    p2 = np.array(voice_results[keys[j]]["preds_mean_per_vertex"])
                    r = np.corrcoef(p1, p2)[0, 1]
                    t1 = voice_results[keys[i]]["passage"]
                    t2 = voice_results[keys[j]]["passage"]
                    print(f"    {t1} vs {t2}: r={r:.4f}")

    # Save results (without huge vertex arrays for readability)
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "preds_mean_per_vertex"}
                    for k, v in results.items()}
    with open(CACHE / "tts_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nSaved to {CACHE / 'tts_results.json'}")


if __name__ == "__main__":
    main()
