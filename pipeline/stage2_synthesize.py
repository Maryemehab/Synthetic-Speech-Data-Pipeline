

import time
from pathlib import Path

from .utils import (
    Checkpoint, append_jsonl, load_jsonl,
    get_logger, load_config, ensure_dirs
)

log = get_logger("stage2")


def _load_model(cfg: dict):
    """
    Load the TTS model according to config.
    
    This function handles all three variants.  The import of chatterbox
    is inside this function so the rest of the pipeline can still be
    imported (for tests etc.) even if chatterbox is not installed.
    
    Returns (model, sample_rate) or raises ImportError with instructions.
    """
    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors
        from chatterbox import mtl_tts   # chatterbox-tts package
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install with: pip install chatterbox-tts safetensors huggingface-hub torch\n"
            "Then download a model variant (see config.yaml tts_synthesis.model_variant)"
        ) from e

    tts = cfg["tts_synthesis"]
    variant  = tts["model_variant"]
    model_ids = tts["model_ids"]
    device   = tts.get("device", "cpu")

    log.info(f"Loading TTS model variant='{variant}' on device='{device}'")

    if variant in ("namaa", "oddadmix"):
        # Load the base Chatterbox multilingual model
        model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)

        # Download and apply the Egyptian fine-tune checkpoint
        repo_id   = model_ids[variant]
        tfile_key = f"{variant}_tfile"
        tfile     = model_ids.get(tfile_key, "model.safetensors")

        log.info(f"Downloading fine-tune checkpoint from {repo_id}")
        ckpt_dir  = snapshot_download(repo_id=repo_id, repo_type="model")
        ckpt_path = Path(ckpt_dir) / tfile

        if not ckpt_path.exists():
            # Some repos have the file at root level
            safetensor_files = list(Path(ckpt_dir).glob("*.safetensors"))
            if safetensor_files:
                ckpt_path = safetensor_files[0]
                log.info(f"Using checkpoint file: {ckpt_path.name}")
            else:
                raise FileNotFoundError(
                    f"No .safetensors file found in {ckpt_dir}\n"
                    f"Check the repo {repo_id} on HuggingFace."
                )

        log.info(f"Loading checkpoint: {ckpt_path}")
        state = load_safetensors(str(ckpt_path), device=device)
        model.t3.load_state_dict(state, strict=False)
        model.t3.to(device).eval()
        log.info("Fine-tune checkpoint loaded successfully")

    elif variant == "base_arabic":
        # Use base multilingual model with Arabic language tag (no fine-tune)
        model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
        log.info("Base multilingual model loaded (no Egyptian fine-tune)")

    else:
        raise ValueError(f"Unknown model_variant: '{variant}'. "
                         f"Options: namaa, oddadmix, base_arabic")

    return model



# Audio utilities

def _get_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using soundfile (no heavy deps)."""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        return 0.0


def _resample_to_16k(src: Path, dst: Path):
    
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(src))
        if waveform.shape[0] > 1:               # stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform  = resampler(waveform)
        torchaudio.save(str(dst), waveform, 16000)
    except Exception as e:
        log.warning(f"Resample failed ({e}), copying original")
        import shutil
        shutil.copy2(src, dst)


# Main synthesis loop


def synthesize_audio(cfg: dict, prompts: list[dict]) -> list[dict]:
   
    tts = cfg["tts_synthesis"]
    out_dir  = Path(tts["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(tts["manifest_file"])
    ckpt          = Checkpoint(tts["checkpoint_file"])

    existing_ids: set[str] = {r["id"] for r in load_jsonl(manifest_path)}
    log.info(f"Stage 2 start — {len(existing_ids)} already synthesized, "
             f"{len(prompts)} total prompts")

    if len(existing_ids) >= len(prompts):
        log.info("All prompts already synthesized — nothing to do")
        return load_jsonl(manifest_path)

    # Load model (only once, expensive)
    model = _load_model(cfg)
    log.info("TTS model ready")

    exaggeration = float(tts.get("exaggeration", 0.5))
    cfg_weight   = float(tts.get("cfg_weight",   0.3))
    temperature  = float(tts.get("temperature",  0.8))
    retries      = int(tts.get("retry_attempts", 3))
    retry_delay  = float(tts.get("retry_delay_seconds", 3))
    batch_size   = int(tts.get("batch_size", 10))
    variant      = tts.get("model_variant", "namaa")

    batch_count  = 0
    ok_count     = 0
    fail_count   = 0

    for prompt in prompts:
        pid  = prompt["id"]
        text = prompt["text"]

        # Skip if already done
        if pid in existing_ids or ckpt.done(pid):
            continue

        wav_path = out_dir / f"{pid}.wav"
        success  = False

        for attempt in range(retries):
            try:
                log.info(f"  Synthesizing {pid}: '{text[:50]}...' (attempt {attempt+1})")

                # ── Core synthesis call ───────────────────────
                wav = model.generate(
                    text,
                    language_id="ar",          # Arabic — applies to all variants
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )

                
                import torchaudio
                raw_path = out_dir / f"{pid}_raw.wav"
                torchaudio.save(str(raw_path), wav, model.sr)

        
                _resample_to_16k(raw_path, wav_path)
                raw_path.unlink(missing_ok=True)   # remove intermediate file

                duration = _get_duration(wav_path)

                record = {
                    "id":            pid,
                    "text":          text,
                    "domain":        prompt.get("domain", ""),
                    "audio_path":    str(wav_path),
                    "duration_sec":  round(duration, 3),
                    "sample_rate":   16000,
                    "model_variant": variant,
                    "status":        "ok",
                    "word_count":    prompt.get("word_count", len(text.split())),
                }
                append_jsonl(manifest_path, record)
                ckpt.mark_done(pid, {"duration": duration})
                existing_ids.add(pid)
                ok_count  += 1
                success    = True
                log.info(f"  ✓  {pid}  {duration:.2f}s")
                break   # exit retry loop

            except Exception as e:
                log.warning(f"  ✗  {pid} attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        if not success:
            fail_count += 1
            record = {
                "id":            pid,
                "text":          text,
                "domain":        prompt.get("domain", ""),
                "audio_path":    "",
                "duration_sec":  0.0,
                "sample_rate":   0,
                "model_variant": variant,
                "status":        "failed",
                "word_count":    prompt.get("word_count", 0),
            }
            append_jsonl(manifest_path, record)
            ckpt.mark_failed(pid, "max_retries_exceeded")
            log.error(f"  ✗✗  {pid} FAILED after {retries} attempts")

        # Progress checkpoint every batch_size items
        batch_count += 1
        if batch_count % batch_size == 0:
            total_done = ok_count + fail_count
            log.info(f"  ── Batch checkpoint: {total_done}/{len(prompts)} "
                     f"({ok_count} ok, {fail_count} failed)")

    manifest = load_jsonl(manifest_path)
    log.info(f"Stage 2 complete — {ok_count} ok, {fail_count} failed, "
             f"{len(manifest)} total in manifest")
    return manifest


if __name__ == "__main__":
    cfg = load_config()
    from .utils import setup_logging
    setup_logging(cfg)
    ensure_dirs(cfg)
    prompts = load_jsonl(cfg["text_generation"]["output_file"])
    if not prompts:
        print("No prompts found — run stage 1 first")
    else:
        manifest = synthesize_audio(cfg, prompts)
        print(f"\n✓ {len(manifest)} items in manifest")