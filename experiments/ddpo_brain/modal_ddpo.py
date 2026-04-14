"""
DDPO + Tribe: Brain-reward RL fine-tuning of Stable Diffusion 1.5.

For each content category, DDPO fine-tunes SD 1.5 LoRA weights to generate
images that maximize predicted brain activation (scored by TRIBEv2).

The algorithm:
    1. SAMPLE: Generate N images with stochastic DDIM, tracking denoising log probs
    2. SCORE:  Convert each image → 5s video → Tribe → fMRI prediction → reward
    3. ADVANTAGE: Group-relative normalization (reward - mean) / std
    4. UPDATE: PPO-clipped policy gradient on LoRA weights, per denoising step

Usage:
    # Single category
    modal run experiments/ddpo_brain/modal_ddpo.py --category faces --epochs 20

    # All 5 categories (sequential)
    modal run experiments/ddpo_brain/modal_ddpo.py

    # Download results
    modal volume get braingym-cache ddpo_results/ ./cache/ddpo_results/
"""

import json
import modal
from pathlib import Path

# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("braingym-ddpo")
vol = modal.Volume.from_name("braingym-cache", create_if_missing=True)

CACHE_DIR = "/cache"

ddpo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        # PyTorch — same constraint as working Tribe pipeline (CUDA-enabled)
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
        # Stable Diffusion + LoRA
        # diffusers >=0.33 works with modern transformers (no FLAX_WEIGHTS_NAME)
        # peft <0.17 avoids float8_e8m0fnu requirement
        # transformers >=4.48 for AutoVideoProcessor (Tribe), <5 to avoid float8
        "diffusers>=0.33,<0.35",
        "peft>=0.14,<0.17",
        "accelerate>=0.34,<0.36",
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
# CATEGORIES
# ============================================================

CATEGORIES = {
    "faces": {
        "prompt": "a close-up portrait photograph of a person, natural lighting, detailed skin texture",
        "negative": "cartoon, illustration, blurry, deformed, extra limbs",
    },
    "landscapes": {
        "prompt": "a photograph of natural landscape scenery, high quality, detailed",
        "negative": "people, text, watermark, blurry, low quality",
    },
    "diagrams": {
        "prompt": "a clean educational diagram with labels, infographic, clear layout, organized",
        "negative": "photograph, blurry, messy, cluttered",
    },
    "action": {
        "prompt": "a photograph of a person in dynamic motion, sports, athletic movement, sharp",
        "negative": "static, sitting, portrait, blurry, cartoon",
    },
    "abstract": {
        "prompt": "an abstract painting with bold geometric shapes and vivid colors, modern art",
        "negative": "photograph, realistic, text, blurry, dull",
    },
}

# ============================================================
# DDPO CORE
# ============================================================


def ddim_step_with_logprob(scheduler, noise_pred, timestep, sample, eta=1.0):
    """Stochastic DDIM step that returns (prev_sample, log_prob).

    Uses eta=1.0 for stochasticity — necessary for RL since pure DDIM
    (eta=0) is deterministic and has no policy gradient signal.

    The log prob is computed up to a constant that cancels in the
    importance ratio (new_log_prob - old_log_prob).
    """
    import torch

    prev_timestep = (
        timestep
        - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    alpha_t = scheduler.alphas_cumprod[timestep]
    alpha_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    # Predicted x_0
    pred_x0 = (sample - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    pred_x0 = pred_x0.clamp(-1, 1)

    # Variance and std for stochastic step
    variance = (
        (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
    )
    std = eta * variance.clamp(min=1e-20).sqrt()

    # Mean of x_{t-1}
    coeff = (1 - alpha_prev - std**2).clamp(min=0).sqrt()
    mean = alpha_prev.sqrt() * pred_x0 + coeff * noise_pred

    # Sample: x_{t-1} = mean + std * noise
    noise = torch.randn_like(sample)
    prev_sample = mean + std * noise

    # Log prob (per image, sum over spatial dims; constants cancel in ratio)
    log_prob = -0.5 * noise.flatten(1).pow(2).sum(1)

    return prev_sample, log_prob


def compute_log_prob_for_stored(
    scheduler, noise_pred, timestep, sample, stored_prev, eta=1.0
):
    """Compute log prob of a stored x_{t-1} under the current policy.

    During the update phase, the UNet has new LoRA weights, so it predicts
    a different mean. We compute how likely the OLD sample is under the
    NEW policy to get the importance ratio.
    """
    import torch

    prev_timestep = (
        timestep
        - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    alpha_t = scheduler.alphas_cumprod[timestep]
    alpha_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    pred_x0 = (sample - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    pred_x0 = pred_x0.clamp(-1, 1)

    variance = (
        (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
    )
    std = eta * variance.clamp(min=1e-20).sqrt()

    coeff = (1 - alpha_prev - std**2).clamp(min=0).sqrt()
    mean = alpha_prev.sqrt() * pred_x0 + coeff * noise_pred

    # Implied noise: what noise would have produced stored_prev from this mean?
    implied_noise = (stored_prev - mean) / std
    log_prob = -0.5 * implied_noise.flatten(1).pow(2).sum(1)

    return log_prob


# ============================================================
# TRIBE SCORING
# ============================================================


def image_to_video(image_path, video_path, duration=5):
    """Convert static image to a 5-second video for Tribe's V-JEPA2 encoder."""
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1",
            "-i", str(image_path),
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-vf", "scale=512:512",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )


def load_parcellation(cache_dir):
    """Load Schaefer 7-network parcellation for fsaverage5."""
    import numpy as np
    import nibabel as nib
    from urllib.request import urlretrieve

    schaefer_base = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage5/label"
    )
    annot_name = "Schaefer2018_400Parcels_7Networks_order.annot"
    aliases = {
        "DorsAttn": "DAN", "Default": "DMN", "Cont": "FPN",
        "SalVentAttn": "VAN", "Limbic": "Limbic",
        "SomMot": "SomMot", "Vis": "Vis",
    }

    parc_dir = Path(cache_dir) / "schaefer"
    parc_dir.mkdir(parents=True, exist_ok=True)

    networks = {}
    for hemi in ("lh", "rh"):
        local = parc_dir / f"{hemi}.{annot_name}"
        if not local.exists():
            urlretrieve(f"{schaefer_base}/{hemi}.{annot_name}", local)
        labels, _, names = nib.freesurfer.read_annot(str(local))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        for idx, name in enumerate(names):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "7Networks":
                net = aliases.get(parts[2], parts[2])
                if net not in networks:
                    networks[net] = {"lh": [], "rh": []}
                networks[net][hemi].extend(
                    np.where(labels == idx)[0].tolist()
                )

    for net in networks:
        for hemi in ("lh", "rh"):
            networks[net][hemi] = np.array(networks[net][hemi])

    return networks


def score_image_with_tribe(tribe_model, image_path, networks):
    """Image → 5s video → Tribe → network signals → reward."""
    import numpy as np

    video_path = Path(str(image_path).replace(".png", "_tmp.mp4"))
    image_to_video(image_path, video_path)

    df = tribe_model.get_events_dataframe(video_path=str(video_path))
    preds, _ = tribe_model.predict(events=df)
    video_path.unlink(missing_ok=True)

    # Extract network signals
    N = 10242
    preds_lh = preds[:, :N]
    preds_rh = preds[:, N : 2 * N]

    signals = {}
    for net_name, masks in networks.items():
        lh_idx, rh_idx = masks["lh"], masks["rh"]
        sig_lh = (
            preds_lh[:, lh_idx].mean(axis=1)
            if len(lh_idx) > 0
            else np.zeros(preds.shape[0])
        )
        sig_rh = (
            preds_rh[:, rh_idx].mean(axis=1)
            if len(rh_idx) > 0
            else np.zeros(preds.shape[0])
        )
        signals[net_name] = (sig_lh + sig_rh) / 2

    # Composite reward: same weights as BrainReward default
    dan, dmn = signals["DAN"], signals["DMN"]
    van, limbic = signals["VAN"], signals["Limbic"]

    reward = float(
        0.4 * (-np.mean(dmn))
        + 0.3 * (np.mean(dan) - np.mean(dmn))
        + 0.2 * np.mean(van)
        + 0.1 * np.mean(limbic)
    )

    metrics = {
        "engagement": float(-np.mean(dmn)),
        "attention": float(np.mean(dan) - np.mean(dmn)),
        "salience": float(np.mean(van)),
        "emotional": float(np.mean(limbic)),
        "cognitive_load": float(
            np.mean(signals["FPN"]) + np.mean(dan) - np.mean(dmn)
        ),
        "net_DAN": float(np.mean(dan)),
        "net_DMN": float(np.mean(dmn)),
        "net_VAN": float(np.mean(van)),
        "net_FPN": float(np.mean(signals["FPN"])),
        "net_Vis": float(np.mean(signals["Vis"])),
        "net_Limbic": float(np.mean(limbic)),
        "net_SomMot": float(np.mean(signals["SomMot"])),
    }

    return reward, metrics


# ============================================================
# TRAINING
# ============================================================


@app.function(
    image=ddpo_image,
    gpu="A100",
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=65536,
)
def train_category(
    category: str,
    num_epochs: int = 20,
    samples_per_epoch: int = 8,
    num_inference_steps: int = 20,
    lr: float = 3e-5,
    clip_eps: float = 0.2,
    guidance_scale: float = 7.5,
    eta: float = 1.0,
):
    """DDPO training loop for one content category.

    Loads SD 1.5 + LoRA and Tribe on the same A100, runs the full
    sample → score → update loop.
    """
    import os
    import time
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from peft import LoraConfig, get_peft_model

    device = "cuda"
    cat_info = CATEGORIES[category]
    prompt = cat_info["prompt"]
    negative_prompt = cat_info["negative"]

    results_dir = Path(CACHE_DIR) / "ddpo_results" / category
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # ---- Load Stable Diffusion 1.5 ----
    print("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ---- Add LoRA to UNet ----
    print("Adding LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        lr=lr,
    )

    # ---- Load Tribe ----
    print("Loading Tribe v2...")
    from tribev2.demo_utils import TribeModel

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
    networks = load_parcellation(CACHE_DIR)

    # ---- Encode text once (reused every epoch) ----
    tok = pipe.tokenizer
    max_len = tok.model_max_length

    text_ids = tok(
        [prompt] * samples_per_epoch,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        text_embeds = pipe.text_encoder(text_ids)[0].detach().to(torch.float16)

    neg_ids = tok(
        [negative_prompt] * samples_per_epoch,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        neg_embeds = pipe.text_encoder(neg_ids)[0].detach().to(torch.float16)

    # ---- Training loop ----
    print(f"\n{'=' * 60}")
    print(f"DDPO Training: {category}")
    print(f"  prompt:  {prompt}")
    print(f"  epochs:  {num_epochs}")
    print(f"  samples: {samples_per_epoch}/epoch")
    print(f"  steps:   {num_inference_steps}")
    print(f"{'=' * 60}\n")

    all_results = []

    for epoch in range(num_epochs):
        t0 = time.time()

        # ========== SAMPLE PHASE (no gradients) ==========
        pipe.unet.eval()
        pipe.scheduler.set_timesteps(num_inference_steps)

        latents = torch.randn(
            samples_per_epoch, 4, 64, 64, device=device, dtype=torch.float16,
        )

        # Trajectory storage (moved to CPU to save GPU memory)
        traj_latents = []
        traj_next = []
        traj_log_probs = []
        traj_timesteps = []

        for t in pipe.scheduler.timesteps:
            traj_latents.append(latents.detach().cpu())
            traj_timesteps.append(int(t.item()))

            # Classifier-free guidance
            lat_in = torch.cat([latents, latents])
            emb_in = torch.cat([neg_embeds, text_embeds])

            with torch.no_grad():
                noise_pred = pipe.unet(lat_in, t, emb_in).sample

            noise_neg, noise_pos = noise_pred.chunk(2)
            noise_pred = noise_neg + guidance_scale * (noise_pos - noise_neg)

            latents, log_prob = ddim_step_with_logprob(
                pipe.scheduler, noise_pred, t, latents, eta,
            )

            traj_next.append(latents.detach().cpu())
            traj_log_probs.append(log_prob.detach().cpu())

        # Decode to images
        with torch.no_grad():
            imgs = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor
            ).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        # ========== SCORE PHASE (Tribe) ==========
        rewards = []
        metrics_list = []

        for i in range(samples_per_epoch):
            # Save PNG
            arr = imgs[i].cpu().permute(1, 2, 0).float().numpy()
            arr = (arr * 255).astype(np.uint8)
            img_path = images_dir / f"e{epoch:03d}_s{i:02d}.png"
            Image.fromarray(arr).save(str(img_path))

            # Score
            reward, metrics = score_image_with_tribe(
                tribe, img_path, networks,
            )
            rewards.append(reward)
            metrics_list.append(metrics)

        rewards_np = np.array(rewards)

        # ========== COMPUTE ADVANTAGES (group-relative) ==========
        if rewards_np.std() > 1e-8:
            advantages = (rewards_np - rewards_np.mean()) / rewards_np.std()
        else:
            advantages = np.zeros_like(rewards_np)
        adv_tensor = torch.tensor(
            advantages, dtype=torch.float16, device=device,
        )

        # ========== UPDATE PHASE (per-step gradient accumulation) ==========
        pipe.unet.train()
        optim.zero_grad()

        total_loss = 0.0
        n_steps = len(traj_timesteps)

        for step_idx in range(n_steps):
            x_t = traj_latents[step_idx].to(device)
            x_prev = traj_next[step_idx].to(device)
            t_val = traj_timesteps[step_idx]
            old_lp = traj_log_probs[step_idx].to(device)

            # Forward with CURRENT LoRA weights (gradients enabled)
            lat_in = torch.cat([x_t, x_t])
            emb_in = torch.cat([neg_embeds, text_embeds])

            noise_pred = pipe.unet(lat_in, t_val, emb_in).sample

            noise_neg, noise_pos = noise_pred.chunk(2)
            noise_pred = noise_neg + guidance_scale * (noise_pos - noise_neg)

            # Log prob of stored x_prev under new policy
            new_lp = compute_log_prob_for_stored(
                pipe.scheduler, noise_pred, t_val, x_t, x_prev, eta,
            )

            # PPO-clipped loss
            ratio = (new_lp - old_lp).exp()
            clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
            step_loss = -torch.min(
                ratio * adv_tensor, clipped * adv_tensor,
            ).mean()

            # Accumulate gradients (scale by 1/n_steps)
            (step_loss / n_steps).backward()
            total_loss += step_loss.item()

        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        # Free trajectory memory
        del traj_latents, traj_next, traj_log_probs
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        avg_loss = total_loss / n_steps

        # ========== LOG ==========
        best_idx = int(rewards_np.argmax())
        epoch_result = {
            "epoch": epoch,
            "category": category,
            "rewards": rewards_np.tolist(),
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "loss": avg_loss,
            "elapsed_s": elapsed,
            "best_metrics": metrics_list[best_idx],
            "all_metrics": metrics_list,
        }
        all_results.append(epoch_result)

        print(
            f"  Epoch {epoch:3d}  "
            f"reward={rewards_np.mean():+.4f}±{rewards_np.std():.4f}  "
            f"best={rewards_np.max():+.4f}  "
            f"loss={avg_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )
        m = metrics_list[best_idx]
        print(
            f"           "
            f"attn={m['attention']:.3f}  "
            f"engage={m['engagement']:.3f}  "
            f"sal={m['salience']:.3f}  "
            f"vis={m['net_Vis']:.3f}"
        )

        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = results_dir / f"lora_e{epoch:03d}"
            pipe.unet.save_pretrained(str(ckpt))
            print(f"           → saved LoRA: {ckpt.name}")

        # Persist results
        (results_dir / "training_log.json").write_text(
            json.dumps(all_results, indent=2)
        )
        vol.commit()

    # ---- Final ----
    print(f"\n{'=' * 60}")
    print(f"DONE: {category}")
    r0 = all_results[0]["reward_mean"]
    rN = all_results[-1]["reward_mean"]
    print(f"  Epoch  0: reward={r0:+.4f}")
    print(f"  Epoch {num_epochs - 1:2d}: reward={rN:+.4f}")
    print(f"  Δ reward: {rN - r0:+.4f}")
    print(f"{'=' * 60}")

    pipe.unet.save_pretrained(str(results_dir / "lora_final"))
    vol.commit()

    return all_results


# ============================================================
# ENTRYPOINT
# ============================================================


@app.local_entrypoint()
def main(
    category: str = "all",
    epochs: int = 20,
    samples: int = 8,
):
    """Run DDPO training.

    Args:
        category: One of faces/landscapes/diagrams/action/abstract, or "all"
        epochs: Training epochs per category
        samples: Images per epoch
    """
    cats = list(CATEGORIES.keys()) if category == "all" else [category]

    for cat in cats:
        print(f"\n{'#' * 60}")
        print(f"# Starting: {cat}")
        print(f"{'#' * 60}\n")

        results = train_category.remote(
            cat,
            num_epochs=epochs,
            samples_per_epoch=samples,
        )

        # Save locally
        local_dir = Path(f"cache/ddpo_results/{cat}")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "training_log.json").write_text(
            json.dumps(results, indent=2)
        )
        print(f"Saved local: {local_dir / 'training_log.json'}")
