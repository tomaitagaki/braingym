import time
from pathlib import Path
import numpy as np
import torch
from tribev2.demo_utils import TribeModel

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

STIMULI = {
    "simple": """
        The sun is warm. The sky is blue. Birds fly in the air.
        Dogs like to run. Cats like to sleep. Fish swim in water.
        Trees grow tall. Flowers smell nice. Rain falls from clouds.
    """,
    "narrative": """
        To be or not to be, that is the question.
        Whether 'tis nobler in the mind to suffer
        the slings and arrows of outrageous fortune,
        or to take arms against a sea of troubles
        and by opposing end them. To die, to sleep,
        no more; and by a sleep to say we end
        the heartache and the thousand natural shocks
        that flesh is heir to.
    """,
    "technical": """
        The mitochondrial electron transport chain consists of four
        protein complexes embedded in the inner mitochondrial membrane.
        Complex I, NADH dehydrogenase, catalyzes the transfer of electrons
        from NADH to ubiquinone, coupled with the translocation of four
        protons across the membrane, establishing the electrochemical
        gradient essential for oxidative phosphorylation via ATP synthase.
    """,
    "dense": """
        The renormalization group analysis of quantum chromodynamics
        reveals asymptotic freedom at high energies, where the running
        coupling constant alpha_s decreases logarithmically. The beta
        function coefficient b_0 equals 33 minus 2n_f over 12 pi,
        governing one-loop evolution and implying confinement at low
        momentum transfer through infrared slavery of the gluon
        propagator in the Landau gauge.
    """,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"PyTorch {torch.__version__}, Device: {DEVICE}")

    log("Loading TRIBEv2...")
    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_FOLDER,
        device=DEVICE,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
            "data.audio_feature.device": "cpu",
            "data.video_feature.image.device": "cpu",
        },
    )
    log(f"Model loaded in {time.time() - t0:.1f}s")

    # --- Run TRIBEv2 on each stimulus (text → TTS → predicted fMRI) ---
    predictions = {}
    for level, text in STIMULI.items():
        cache_path = CACHE_FOLDER / f"preds_{level}.npy"
        if cache_path.exists():
            log(f"Loading cached '{level}' from {cache_path}")
            predictions[level] = np.load(cache_path)
            continue

        text_path = CACHE_FOLDER / f"{level}.txt"
        text_path.write_text(text.strip())

        log(f"Processing '{level}'...")
        t0 = time.time()
        df = model.get_events_dataframe(text_path=text_path)
        preds, segments = model.predict(events=df)
        np.save(cache_path, preds)
        predictions[level] = preds
        log(f"  -> {preds.shape} in {time.time() - t0:.1f}s, saved to {cache_path}")

    # --- Quantify: full table with cognitive metrics + spatial regions ---
    from quantify import compute_full_table, print_full_table

    rows = compute_full_table(predictions)
    print_full_table(rows)

    log("Done!")


if __name__ == "__main__":
    main()
