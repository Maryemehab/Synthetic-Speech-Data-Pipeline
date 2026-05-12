

import math
from pathlib import Path

from .utils import get_logger, load_jsonl, append_jsonl, load_config

log = get_logger("stage3_quality")



# Signal computation

def compute_signals(audio_path: str, text: str, cfg: dict) -> dict:
   
    signals_cfg = cfg["review"]["auto_signals"]

    # Defaults for when audio can't be read
    out = {
        "duration_sec":    None,
        "silence_ratio":   None,
        "rms_energy":      None,
        "snr_estimate_db": None,
        "word_rate_wps":   None,
        "has_numbers":     _has_numbers(text),
        "has_latin":       _has_latin(text),
        "auto_pass":       False,
        "auto_reject_reason": "not_computed",
    }

    path = Path(audio_path)
    if not path.exists() or not audio_path:
        out["auto_reject_reason"] = "file_missing"
        return out

    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)   # stereo → mono

        n_samples = len(data)
        duration  = n_samples / sr
        out["duration_sec"] = round(duration, 3)

        #  Silence ratio 
        # Frame energy in 25ms windows
        frame_len = max(1, int(0.025 * sr))
        frames    = [data[i: i + frame_len] for i in range(0, n_samples, frame_len)]
        energies  = [float(np.sqrt(np.mean(f ** 2))) for f in frames if len(f) > 0]
        if energies:
            noise_floor   = 0.005   # below this = silence
            silent_frames = sum(1 for e in energies if e < noise_floor)
            silence_ratio = silent_frames / len(energies)
        else:
            silence_ratio = 1.0
        out["silence_ratio"] = round(silence_ratio, 3)

        #  RMS energy 
        rms = float(np.sqrt(np.mean(data ** 2))) if n_samples > 0 else 0.0
        out["rms_energy"] = round(rms, 6)

        #  SNR estimate 
        # Simple approach: take top 10% energy frames as signal,
        # bottom 10% as noise estimate
        if energies:
            sorted_e = sorted(energies)
            n        = len(sorted_e)
            noise_e  = float(np.mean(sorted_e[: max(1, n // 10)])) + 1e-10
            signal_e = float(np.mean(sorted_e[max(1, 9 * n // 10):])) + 1e-10
            snr_db   = 20 * math.log10(signal_e / noise_e)
            out["snr_estimate_db"] = round(snr_db, 1)

        #  Word rate 
        word_count = len(text.split())
        if duration > 0:
            out["word_rate_wps"] = round(word_count / duration, 2)

        #  Auto-pass decision 
        reason = _check_thresholds(out, signals_cfg)
        out["auto_pass"]          = (reason is None)
        out["auto_reject_reason"] = reason or "passed"

    except Exception as e:
        log.warning(f"Signal computation failed for {audio_path}: {e}")
        out["auto_reject_reason"] = f"computation_error: {e}"

    return out


def _check_thresholds(signals: dict, cfg: dict) -> str | None:
    
    dur  = signals.get("duration_sec")
    slr  = signals.get("silence_ratio")
    rms  = signals.get("rms_energy")
    wps  = signals.get("word_rate_wps")

    if dur is None:
        return "audio_unreadable"

    if dur < cfg["min_duration_seconds"]:
        return f"too_short ({dur:.2f}s)"

    if dur > cfg["max_duration_seconds"]:
        return f"too_long ({dur:.2f}s — likely repetition loop)"

    if slr is not None and slr > cfg["max_silence_ratio"]:
        return f"too_silent (silence_ratio={slr:.2f})"

    if rms is not None and rms < cfg["min_rms_energy"]:
        return f"too_quiet (rms={rms:.5f})"

    # Sanity check on word rate (flagged but not hard-rejected)
    if wps is not None and (wps < 0.8 or wps > 8.0):
        return f"abnormal_word_rate ({wps:.2f} wps)"

    return None   # all checks passed


def _has_numbers(text: str) -> bool:
    return any(c.isdigit() or "\u0660" <= c <= "\u0669" for c in text)


def _has_latin(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)



# Batch scoring

def score_manifest(cfg: dict, manifest: list[dict]) -> list[dict]:
    
    scored_path = Path(cfg["tts_synthesis"]["manifest_file"]).parent / "scored_manifest.jsonl"

    # Don't re-score already-scored records
    existing_ids: set[str] = {r["id"] for r in load_jsonl(scored_path)}

    scored: list[dict] = list(load_jsonl(scored_path))   # pre-existing

    auto_pass_count = 0
    auto_fail_count = 0

    for rec in manifest:
        if rec["id"] in existing_ids:
            continue
        if rec.get("status") == "failed":
            signals = {
                "auto_pass": False,
                "auto_reject_reason": "synthesis_failed",
            }
        else:
            signals = compute_signals(rec.get("audio_path", ""), rec["text"], cfg)

        enriched = {**rec, **signals}
        append_jsonl(scored_path, enriched)
        scored.append(enriched)
        existing_ids.add(rec["id"])

        if signals.get("auto_pass"):
            auto_pass_count += 1
        else:
            auto_fail_count += 1

    log.info(f"Quality scoring complete — "
             f"{auto_pass_count} auto-pass, {auto_fail_count} auto-fail")
    return scored


if __name__ == "__main__":
    cfg = load_config()
    from .utils import setup_logging, ensure_dirs
    setup_logging(cfg)
    ensure_dirs(cfg)
    manifest = load_jsonl(cfg["tts_synthesis"]["manifest_file"])
    scored = score_manifest(cfg, manifest)
    passed = sum(1 for r in scored if r.get("auto_pass"))
    print(f"\nScored {len(scored)} items — {passed} auto-pass, {len(scored)-passed} auto-fail")
