

import json
import random
import shutil
from pathlib import Path

from .utils import get_logger, load_jsonl, load_config

log = get_logger("stage4_export")


def export_dataset(cfg: dict) -> None:
    
    export_cfg = cfg["export"]
    out_dir    = Path(export_cfg["output_dir"])
    audio_dir  = out_dir / export_cfg["audio_subdir"]
    meta_file  = out_dir / export_cfg["metadata_file"]

    # Load decisions and manifest
    decisions = {d["id"]: d for d in load_jsonl(cfg["review"]["review_db"])}
    manifest  = load_jsonl(cfg["tts_synthesis"]["manifest_file"])

    # Filter approved items
    approved = []
    for rec in manifest:
        pid = rec["id"]
        if decisions.get(pid, {}).get("status") == export_cfg["accepted_status"]:
            approved.append(rec)

    log.info(f"Exporting {len(approved)} approved items out of {len(manifest)} total")

    if not approved:
        log.warning("No approved items found — run the review UI first")
        return

    # Create output directories
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio files and build metadata
    metadata = []
    for rec in approved:
        pid = rec["id"]
        src_path = Path(rec["audio_path"])
        dst_path = audio_dir / f"{pid}.wav"

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            log.warning(f"Audio file missing: {src_path}")
            continue

        # Build metadata record
        meta_rec = {
            "id":            pid,
            "text":          rec["text"],
            "audio":         str(dst_path.relative_to(out_dir)),
            "duration_sec":  rec.get("duration_sec", 0),
            "domain":        rec.get("domain", ""),
            "word_count":    rec.get("word_count", 0),
            "source":        rec.get("source", ""),
        }
        metadata.append(meta_rec)

    # Write metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Optional train/test split
    if export_cfg["split"]["enabled"]:
        split_cfg = export_cfg["split"]
        random.seed(split_cfg["seed"])
        random.shuffle(metadata)

        n_train = int(len(metadata) * split_cfg["train_ratio"])
        train_data = metadata[:n_train]
        test_data  = metadata[n_train:]

        # Write split files
        for split_name, data in [("train", train_data), ("test", test_data)]:
            split_file = out_dir / f"{split_name}.jsonl"
            with open(split_file, "w", encoding="utf-8") as f:
                for rec in data:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        log.info(f"Split: {len(train_data)} train, {len(test_data)} test")

    log.info(f"Dataset exported to: {out_dir}")
    log.info(f"Metadata: {meta_file}")
    log.info(f"Audio files: {audio_dir}")


if __name__ == "__main__":
    cfg = load_config()
    from .utils import setup_logging
    setup_logging(cfg)
    export_dataset(cfg)