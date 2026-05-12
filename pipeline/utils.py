
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml



def load_config(config_path: str = "config/config.yaml") -> dict:
    
    root_dir = Path(__file__).resolve().parent.parent
    path = (root_dir / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            f"Make sure you run from the project root directory."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        debug_path = Path("logs/config_debug.txt")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"PATH: {path}\n")
            f.write(f"SIZE: {path.stat().st_size}\n")
            f.write(f"RAW_BYTES: {raw!r}\n")
            f.write("FALLBACK: using default review UI config values\n")

        default_cfg = {
            "pipeline": {
                "name": "SSDP-Egyptian-Arabic",
                "version": "1.0.0",
                "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                          + "_" + str(uuid.uuid4()).replace("-", "")[:8],
            },
            "text_generation": {
                "target_count": 120,
                "output_file": "data/raw_text/prompts.jsonl",
                "checkpoint_file": "data/raw_text/.gen_checkpoint.json",
                "retry_attempts": 3,
                "use_seed_fallback": True,
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "qwen2.5:7b",
                    "timeout_seconds": 120,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "num_predict": 1024,
                    },
                },
                "domains": {
                    "daily_life": {
                        "count": 20,
                        "hint": "مواقف الحياة اليومية: التسوق والطبخ والمواصلات والكلام مع الجيران",
                    },
                    "family_home": {
                        "count": 15,
                        "hint": "محادثات عيلية في البيت، طلبات، شجارات ودية بين إخوات",
                    },
                    "food_cafe": {
                        "count": 15,
                        "hint": "طلب أكل في مطعم أو كافيه، الكلام عن الأكل المصري، باعة الشارع",
                    },
                    "transport": {
                        "count": 10,
                        "hint": "توجيهات تاكسي ومترو ومكروباص، المفاوضة على الأجرة",
                    },
                    "work_school": {
                        "count": 10,
                        "hint": "كلام شغل أو مدرسة بالعامية، شكاوى وخطط ومواعيد",
                    },
                    "emotions_opinions": {
                        "count": 10,
                        "hint": "التعبير عن المشاعر، الموافقة والرفض، التعليق على الأخبار",
                    },
                    "phone_calls": {
                        "count": 10,
                        "hint": "بداية مكالمة تليفون مصرية، سؤال عن أحوال، طلب معروف",
                    },
                    "religion_daily": {
                        "count": 10,
                        "hint": "عبارات دينية شائعة في الكلام اليومي: إن شاء الله، الحمد لله، ربنا يخليك",
                    },
                    "numbers_mixed": {
                        "count": 10,
                        "hint": "جمل فيها أرقام وتواريخ وأسعار بالعامية المصرية",
                    },
                    "slang_expressions": {
                        "count": 10,
                        "hint": "تعبيرات وعبارات مصرية خالصة: يسطا، يابني، مش معقول، عال العال",
                    },
                },
                "length_distribution": {
                    "short":   {"min": 3,  "max": 8,  "weight": 0.25},
                    "medium":  {"min": 9,  "max": 18, "weight": 0.50},
                    "long":    {"min": 19, "max": 35, "weight": 0.25},
                },
            },
            "tts_synthesis": {
                "model_variant": "namaa",
                "model_ids": {
                    "namaa": "NAMAA-Space/NAMAA-Egyptian-TTS",
                    "namaa_tfile": "t3_mtl23ls_v2.safetensors",
                    "oddadmix": "oddadmix/chatterbox-egyptian-v0",
                    "oddadmix_tfile": "model.safetensors",
                    "base_arabic": "ResembleAI/chatterbox",
                },
                "exaggeration": 0.5,
                "cfg_weight": 0.3,
                "temperature": 0.8,
                "device": "cpu",
                "output_format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "batch_size": 10,
                "retry_attempts": 3,
                "retry_delay_seconds": 3,
                "output_dir": "data/audio",
                "manifest_file": "data/audio/synthesis_manifest.jsonl",
                "checkpoint_file": "data/audio/.synth_checkpoint.json",
            },
            "review": {
                "auto_signals": {
                    "min_duration_seconds": 0.5,
                    "max_duration_seconds": 30.0,
                    "max_silence_ratio": 0.80,
                    "min_rms_energy": 0.001,
                },
                "ui_host": "0.0.0.0",
                "ui_port": 8003,
                "statuses": ["approved", "rejected", "uncertain"],
                "review_db": "data/reviewed/review_decisions.jsonl",
            },
            "export": {
                "format": "huggingface",
                "accepted_status": "approved",
                "output_dir": "data/exports/final_dataset",
                "metadata_file": "metadata.jsonl",
                "audio_subdir": "audio",
                "split": {"enabled": True, "train_ratio": 0.9, "seed": 42},
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/pipeline.log",
                "json_log_file": "logs/pipeline_events.jsonl",
            },
        }
        path.write_text(yaml.safe_dump(default_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return default_cfg

    cfg = yaml.safe_load(text)
    if cfg is None:
        snippet = repr(text[:300])
        raise ValueError(
            f"Config file {path} loaded empty after YAML parsing. "
            f"Text length={len(text)}. Head={snippet}"
        )
    if not isinstance(cfg, dict):
        raise TypeError(
            f"Config file {path} must contain a YAML mapping at the top level. "
            f"Parsed type: {type(cfg).__name__}"
        )

    # Generate a unique run ID if not set (format: 20250511_143022_a3f9b1c2)
    if not cfg["pipeline"].get("run_id"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4()).replace("-", "")[:8]
        cfg["pipeline"]["run_id"] = f"{ts}_{uid}"

    return cfg



# Logging


_json_fh = None   # module-level handle so we don't open the file repeatedly


def setup_logging(cfg: dict) -> logging.Logger:
    """
    Configure the root 'ssdp' logger with three outputs:
      stdout, plain log file, JSONL log file.
    Call once at pipeline startup; all child loggers inherit.
    """
    global _json_fh

    log_cfg = cfg.get("logging", {})
    level   = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file      = Path(log_cfg.get("log_file",      "logs/pipeline.log"))
    json_log_file = Path(log_cfg.get("json_log_file", "logs/pipeline_events.jsonl"))

    # Ensure directories exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    json_log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("ssdp")
    logger.setLevel(level)
    logger.handlers.clear()

     
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _json_fh = open(json_log_file, "a", encoding="utf-8")
    run_id   = cfg["pipeline"].get("run_id", "unknown")

    class JsonlHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            obj = {
                "ts":     datetime.now(timezone.utc).isoformat(),
                "level":  record.levelname,
                "logger": record.name,
                "msg":    record.getMessage(),
                "run_id": run_id,
            }
            if record.exc_info:
                obj["exc"] = self.formatException(record.exc_info)
            _json_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            _json_fh.flush()

    logger.addHandler(JsonlHandler())
    logger.info(f"Pipeline initialised — run_id={run_id}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger that inherits all handlers from the 'ssdp' root."""
    return logging.getLogger(f"ssdp.{name}")


# Checkpointing


class Checkpoint:
    

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ── internal ──────────────────────────────────────────────
    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)   # atomic on POSIX

    # ── public API ────────────────────────────────────────────
    def done(self, key: str) -> bool:
        entry = self._data.get(key, {})
        return bool(entry) and entry.get("status") != "failed"

    def mark_done(self, key: str, meta: dict | None = None):
        self._data[key] = {
            "status":       "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        self._save()

    def mark_failed(self, key: str, reason: str):
        self._data[key] = {
            "status":    "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error":     reason,
        }
        self._save()

    def count_done(self)   -> int: return sum(1 for v in self._data.values() if v.get("status") == "done")
    def count_failed(self) -> int: return sum(1 for v in self._data.values() if v.get("status") == "failed")

    @property
    def completed_keys(self) -> set:
        return {k for k, v in self._data.items() if v.get("status") == "done"}


# JSONL helpers


def append_jsonl(path: str | Path, obj: dict):
    """Append one JSON object as a new line to a .jsonl file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a .jsonl file. Returns [] if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def ensure_dirs(cfg: dict):
    """Pre-create every output directory the pipeline will write to."""
    paths = [
        cfg["text_generation"]["output_file"],
        cfg["tts_synthesis"]["manifest_file"],
        cfg["review"]["review_db"],
        cfg["logging"]["log_file"],
        cfg["logging"]["json_log_file"],
    ]
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["tts_synthesis"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["export"]["output_dir"]).mkdir(parents=True, exist_ok=True)
