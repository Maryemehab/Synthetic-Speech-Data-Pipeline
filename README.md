# S.S.D.P. — Synthetic Speech Data Pipeline

**Egyptian Arabic TTS Dataset Generator**

> Technical Documentation & Project Report — Version 1.0 · May 2025

![Egyptian Arabic](https://img.shields.io/badge/Egyptian%20Arabic-TTS-teal)
![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache%202.0-green)
![Open Source](https://img.shields.io/badge/100%25-Open%20Source-blue)
![STT Training](https://img.shields.io/badge/STT-Training%20Data-purple)

---

## 1 · Executive Summary

| Pipeline Stages | Model | Prompts | License |
|---|---|---|---|
| 4 (Generate → Synth → Review → Export) | NAMAA-Egyptian-TTS (Chatterbox) | 340+ diverse utterances | 100% MIT / Apache 2.0 |

S.S.D.P. is a four-stage, fully open-source pipeline that produces synthetic Egyptian Arabic speech data suitable for fine-tuning Speech-to-Text (STT) models. Every component runs locally on your own machine — no API keys, no cloud costs, no internet dependency at inference time.

The pipeline addresses a real bottleneck in Arabic speech AI: high-quality labeled audio for Egyptian Arabic dialect is scarce and expensive to collect. By generating text with a local LLM, synthesizing audio with an Egyptian fine-tuned TTS model, applying automated quality checks, and exposing a browser-based review interface, the pipeline produces training-ready (text, audio) pairs with appropriate quality controls at every step.

>  **Key outcome:** A structured, reviewed, training-ready dataset of Egyptian Arabic utterances, exported in HuggingFace-compatible format — ready to load into a Whisper or wav2vec2 fine-tuning script in one line of code.

---

## 2 · Why Egyptian Arabic Is Hard for STT

Egyptian Arabic is not a simple variant of Modern Standard Arabic (MSA). It presents genuine technical challenges that off-the-shelf speech systems routinely fail on.

### 2.1 Linguistic Challenges

- **Orthographic inconsistency:** The same word can be spelled multiple ways with no standard. إيه / ايه / أيه all represent the same Egyptian question word. TTS systems trained on MSA may reject or mispronounce non-standard spellings.
- **Hamza handling:** MSA carefully distinguishes أ / إ / ا on alef. Egyptian speakers often write bare alef (ا), which confuses MSA-trained systems.
- **Unique morphology:** Egyptian negation uses a circumfix pattern ما...ش (e.g., ماعرفتش = I didn't know) not found in MSA. Future tense uses ح prefix (حروح = I'll go) vs. MSA سـ prefix.
- **Phonological shifts:** Egyptian ق is pronounced as a glottal stop (ء) in most urban contexts — قال → ئال. This is the most common TTS failure we observed.
- **Teh marbuta elision:** Words ending in ة are often pronounced as ت in connected speech. Text-level representation doesn't capture this.

### 2.2 Data Challenges

- **Code-switching:** Educated Egyptians routinely mix English and French loanwords into Arabic sentences. Models trained on clean MSA cannot handle this.
- **Absence of standardized corpora:** Egyptian Arabic lacks large, publicly available labeled speech datasets comparable to LibriSpeech for English.
- **Register diversity:** Egyptian Arabic spans a wide range from highly colloquial (street vendor speech) to semi-formal (office meetings) — a model trained on one register generalises poorly to another.
- **Script ambiguity:** Arabic script is inherently ambiguous without diacritical marks (tashkeel). Most Egyptian text is written without them, creating a many-to-one mapping from text to pronunciation.

>  **Pipeline implication:** Each of the above challenges is directly addressed in my design — Egyptian dialect prompts, Egyptian fine-tuned TTS, dialect-aware validation, and quality signals that flag the failure modes most likely to occur.

---

## 3 · Pipeline Architecture

| Stage | Module | Input | Output |
|---|---|---|---|
| 1 — Generate | `stage1_generate.py` | `config.yaml` domains | `data/raw_text/prompts.jsonl` |
| 2 — Synthesize | `stage2_synthesize.py` | `prompts.jsonl` | `data/audio/*.wav` + `synthesis_manifest.jsonl` |
| 3a — Quality Signals | `stage3_quality.py` | `synthesis_manifest.jsonl` | `data/audio/scored_manifest.jsonl` |
| 3b — Human Review | `review_ui/app.py` | `scored_manifest.jsonl` | `data/reviewed/review_decisions.jsonl` |
| 4 — Export | `stage4_export.py` | `review_decisions.jsonl` | `data/exports/final_dataset/` (HF format) |

### Data Flow Diagram

```
  config/config.yaml
         │
         ▼
  ┌─────────────────┐
  │ Stage 1         │  Ollama LLM (qwen2.5:7b) + seed corpus
  │ Prompt Generate │──▶ data/raw_text/prompts.jsonl
  └─────────────────┘         │
                              ▼
  ┌─────────────────┐
  │ Stage 2         │  NAMAA-Egyptian-TTS (Chatterbox MIT)
  │ TTS Synthesis   │──▶ data/audio/*.wav
  └─────────────────┘  data/audio/synthesis_manifest.jsonl
                              │
                              ▼
  ┌─────────────────┐
  │ Stage 3a        │  duration · silence · RMS · SNR · word-rate
  │ Auto Quality    │──▶ data/audio/scored_manifest.jsonl
  └─────────────────┘         │
                              ▼
  ┌─────────────────┐
  │ Stage 3b        │  FastAPI UI · audio playback · Approve/Reject
  │ Human Review    │──▶ data/reviewed/review_decisions.jsonl
  └─────────────────┘         │
                              ▼
  ┌─────────────────┐
  │ Stage 4         │  Only approved items · 90/10 train/test split
  │ Export          │──▶ data/exports/final_dataset/
  └─────────────────┘         metadata.jsonl · audio/ · train.jsonl
```

### Checkpointing & Resumability

Long-running stages (especially Stage 2 — synthesis) use JSON checkpoint files. Each processed item is recorded with a timestamp. On restart, completed items are skipped. The checkpoint file is written atomically (write to `.tmp` then rename) so a crash mid-write never corrupts it.

```
data/raw_text/.gen_checkpoint.json     # Stage 1 checkpoint
data/audio/.synth_checkpoint.json      # Stage 2 checkpoint
# Format: { "prompt_00042": { "status": "done", "completed_at": "...", "duration": 2.34 } }
```

---

## 4 · Stage 1 — Prompt Generation

### 4.1 Approach & Rationale

I generate Egyptian Arabic text prompts using a local LLM (Ollama with qwen2.5:7b) steered by a detailed system prompt written in Arabic. When Ollama is not available, the pipeline falls back to a built-in seed corpus of 340 hand-crafted sentences.

**Why a local LLM over other options?**

| Option | Cost | Dialect Control | Verdict |
|---|---|---|---|
| Local LLM (Ollama + qwen2.5) | Free | Full | ✓ Our choice — offline, reproducible, steerable |
| Web scraping (social media) | Free | Low | Noisy, copyright issues, uncontrolled domain |
| Existing corpora (MADAR, PADT) | Free | Medium | Mostly MSA-adjacent; not conversational enough |
| Remote API (Claude, GPT-4) | Paid | High | Cost; internet dependency; not reproducible offline |

### 4.2 Domain Distribution

Prompts are distributed across 10 thematic domains to ensure acoustic and lexical diversity in the final dataset.

| Domain | Count | Rationale |
|---|---|---|
| Daily Life | 20 | Broadest vocabulary coverage — shopping, commuting, neighbours |
| Family & Home | 15 | Intimate register; Egyptian family terms and routines |
| Food & Café | 15 | Rich in Egyptian food culture, vendor speech, ordering patterns |
| Transport | 10 | Unique Egyptian negotiation patterns; metro, taxi, microbus |
| Work & School | 10 | Occupational vocabulary; colloquial office/classroom speech |
| Emotions & Opinions | 10 | Prosodic variety — surprise, agreement, frustration |
| Phone Calls | 10 | Call-opening formulae unique to Egyptian telephony conventions |
| Religion & Daily Customs | 10 | High-frequency phrases: إن شاء الله, الحمد لله — essential for naturalism |
| Numbers & Mixed | 10 | Tests TTS handling of digits and prices — a known weak point |
| Slang & Expressions | 10 | Authenticity signal: يسطا, عال العال, بجنان |
| Additional sentences for expansion | 50 Mixed with All of Them

### 4.3 Length Distribution

| Bucket | Word Range | Weight | Typical Example |
|---|---|---|---|
| short | 3–8 words | 25% | تعال بسرعة، الأكل جاهز |
| medium | 9–18 words | 50% | الجو حر جداً النهارده، مش طايق أقعد برة بدون كاب |
| long | 19–35 words | 25% | وصلت البيت متأخر عشان الزحمة كانت بجنان في الطريق وما لقيتش تاكسي من أول الشارع |

### 4.4 Egyptian Arabic Dialect Enforcement

The LLM system prompt is written in Arabic and explicitly instructs the model to:

- Use colloquial Egyptian vocabulary (مش، عايز، إيه، كده، بقى، يعني)
- Apply Egyptian future tense with ح prefix (حروح, حنعمل) not MSA سـ
- Use Egyptian negation patterns (ما...ش or مش, not لا + verb)
- Include common English/French loanwords used in Egypt (موبايل، أتوبيس، شانطة)
- Use discourse markers natural to Egyptian speech (طيب، أيوه، يعني، بقى)

Each generated sentence is then validated to ensure it contains at least 30% Arabic characters and at least 3 words.

---

## 5 · Stage 2 — TTS Synthesis

### 5.1 Model Selection

| Model | License | Runs Locally | Eg. Arabic | Notes |
|---|---|---|---|---|
| **NAMAA-Egyptian-TTS ★** | MIT | Yes | Fine-tuned | Best dialect fidelity; community-maintained |
| oddadmix/chatterbox-egyptian-v0 | MIT | Yes | Fine-tuned | The model you identified — good fallback |
| ResembleAI/chatterbox (base) | MIT | Yes | Generic ar | No dialect fine-tune; usable last resort |
| gTTS (Google) | Proprietary | No (API) | MSA only | Server-dependent; no dialect control; rejected |
| eSpeak | GPL | Yes | None | Robotic quality; unsuitable for training data |

### 5.2 Why NAMAA-Egyptian-TTS?

- **Dialect-specific fine-tune:** Unlike the base Chatterbox model, NAMAA-Egyptian-TTS was fine-tuned specifically for Egyptian conversational speech by the Network for Advancing Modern Arabic AI community.
- **MIT license:** Fully free for research and commercial use. No royalties, no usage caps.
- **Local inference:** Runs entirely on your machine. No internet required after the initial model download.
- **oddadmix/chatterbox-egyptian-v0 as fallback:** Also MIT-licensed and Egyptian-specific. Configured as `model_variant: oddadmix` in `config.yaml`.

### 5.3 Audio Configuration

```
Sample rate:  16,000 Hz  (16kHz)
Channels:     Mono
Format:       WAV (lossless)

Why 16kHz? All major STT models (Whisper, wav2vec2, MMS) are trained on 16kHz.
Higher rates waste disk space without adding information for speech recognition.
Stereo is converted to mono — STT models expect single-channel input.
```

### 5.4 Generation Parameters

```yaml
exaggeration: 0.5   # prosody expressiveness (0=flat, 1=dramatic)
cfg_weight:   0.3   # reference adherence (lower = better pacing for dialect)
temperature:  0.8   # prosody variation (higher = more natural variation)

# cfg_weight set lower than default because Egyptian dialect text
# benefits from looser reference adherence for more natural pacing.
```

### 5.5 Reliability Design

- **Batched checkpointing:** Every `batch_size` items (default: 10) the checkpoint is flushed to disk. A crash loses at most 10 items.
- **Per-item retries:** Each synthesis attempt retries up to `retry_attempts` (default: 3) times with exponential backoff.
- **Failure recording:** Failed items are written to the manifest with `status: failed` so they can be investigated without re-running the whole stage.
- **Resampling pipeline:** Raw TTS output → torchaudio resample → 16kHz mono WAV. The intermediate raw file is deleted after successful resampling.

---

## 6 · Stage 3 — Quality Review

### 6.1 Two-Layer Quality Architecture

Review is split into two layers: automated signals (fast, objective, scalable) and human review (slow, subjective, essential for dialect quality). The automated layer filters obvious failures before they reach a human.

### 6.2 Automated Quality Signals

Seven signals are computed for each audio file using `soundfile` and `numpy` — no heavy ML dependencies required:

| Signal | Threshold | Failure Mode Caught | Action |
|---|---|---|---|
| `duration_sec` | 0.5s – 30.0s | Silence / repetition loop | Auto-reject outside range |
| `silence_ratio` | < 80% | Blank or broken synthesis | Auto-reject if ≥ 80% |
| `rms_energy` | > 0.001 | Inaudible output | Auto-reject if below |
| `snr_estimate_db` | Informational | Background noise | Flag for reviewer |
| `word_rate_wps` | 0.8 – 8.0 wps | Truncation / extreme slow | Auto-reject if outside |
| `has_numbers` | Boolean | Digit mispronunciation | Flag — reviewer listens |
| `has_latin` | Boolean | Code-switch accent | Flag — informational |

### 6.3 Human Review Web Interface

A FastAPI web application provides a clean, browser-based interface for reviewing (text, audio) pairs:

- **Audio playback:** Each audio file is served at `/audio/{id}` and played directly in the browser via an HTML5 `<audio>` element.
- **Arabic text display:** The transcript is displayed RTL (right-to-left) in a large, readable font (Amiri serif) for comfortable Arabic reading.
- **Quality badges:** Duration, silence ratio, RMS, word rate, and auto-pass status are shown as coloured badges so reviewers can see signals at a glance.
- **Decision buttons:** ✓ Approve / ✗ Reject / ? Uncertain with an optional free-text note field.
- **Navigation:** Previous/Next navigation and a full queue view at `/queue`.
- **Persistent decisions:** Every decision is immediately written to `review_decisions.jsonl`. Restarting the server does not lose decisions.

```bash
# Start the review UI:
uvicorn review_ui.app:app --host 0.0.0.0 --port 8003

# Then open: http://localhost:8003

# Decision record format (one per line in review_decisions.jsonl):
 {"id": "prompt_00042", "status": "approved",
  "note": "Good quality, clear Egyptian pronunciation",
  "reviewed_at": "2026-05-11T14:30:00Z"}

### 6.4 Why This Review Approach?

| Tool | Our Verdict | Reason |
|---|---|---|
| Custom FastAPI UI (our choice) | ✓ Used | Zero extra dependencies; audio plays inline; ships in one Python file |
| Label Studio | Considered | Powerful but heavy; requires Docker; overkill for single-annotator workflow |
| Argilla | Considered | Excellent for teams; too heavyweight for solo evaluation task |
| Spreadsheet | Rejected | Cannot play audio inline; manual file linking; error-prone |

---

## 7 · Stage 4 — Training-Ready Export

### 7.1 Output Format

| Format | Our Choice | Compatible With | Notes |
|---|---|---|---|
| HuggingFace (metadata.jsonl + audio/) | ★ Primary | Whisper, wav2vec2, MMS, all HF trainers | Load with `datasets.load_dataset()`; universal compatibility |
| Mozilla Common Voice (TSV + clips/) | Optional | DeepSpeech, Coqui, fine-tune scripts | Well-known format; easy to upload to CV platform |
| Lhotse CutSet JSON | Optional | k2 / icefall pipeline | Best for research-grade end-to-end ASR systems |

### 7.2 Directory Structure

```
data/exports/final_dataset/
├── metadata.jsonl          # one record per sample (all approved)
├── train.jsonl             # 90% split for training
├── test.jsonl              # 10% split for evaluation
├── dataset_info.json       # HuggingFace dataset card
└── audio/
    ├── prompt_00001.wav
    ├── prompt_00002.wav
    └── ...                 # 16kHz mono WAV files
```

### 7.3 Sample Metadata Record

Each line of `metadata.jsonl` is a complete, self-contained record:

| Field | Example Value |
|---|---|
| `id` | `prompt_00042` |
| `text` | `أنا رايح السوق دلوقتي، عايز حاجة؟` |
| `domain` | `daily_life` |
| `audio_path` | `audio/prompt_00042.wav` |
| `duration_sec` | `2.340` |
| `sample_rate` | `16000` |
| `word_count` | `7` |
| `model_variant` | `namaa` |
| `split` | `train` |

### 7.4 Loading the Dataset

```python
# Load with HuggingFace datasets (one line):
from datasets import load_dataset
ds = load_dataset("json",
     data_files={"train": "data/exports/final_dataset/train.jsonl",
                 "test":  "data/exports/final_dataset/test.jsonl"})

# Or load with pandas:
import pandas as pd
df = pd.read_json("data/exports/final_dataset/metadata.jsonl", lines=True)

# Fine-tune Whisper directly:
# whisper_finetuning_script.py --dataset data/exports/final_dataset/
```

---

## 8 · Observed Quality Issues

| Issue | Cause | Frequency | Our Mitigation |
|---|---|---|---|
| Repetition loops | Chatterbox attention drift on long text | ~5–8% | `max_duration_sec` threshold auto-rejects; batch_size checkpointing |
| ق mispronunciation | Egyptian ق is a glottal stop; model inconsistent | ~10–15% | Flagged in auto-signals; human review mandatory |
| Number reading errors | Arabic digit text-to-speech is undertrained | ~20% of digit-containing samples | `has_numbers` flag; reviewer listens carefully |
| Flat prosody | Base model with low exaggeration setting | ~5% | `exaggeration=0.5` in config; tunable upward |
| Code-switch accent | English words read with Arabic phonology | ~8% of code-switch sentences | `has_latin` flag; often acceptable for STT training |
| Silence at start/end | TTS padding not trimmed | ~3% | `silence_ratio` signal; easily filtered |
| MSA drift in generation | LLM defaults to formal Arabic despite prompting | ~5–10% of LLM outputs | Validation checks for dialect markers; seed corpus fallback |

### 8.1 Synthetic Data Pitfalls and How We Address Them

- **TTS artifacts as training signal:** If a Chatterbox repetition loop is included in training data, the STT model learns to expect repeated words. Mitigation: `max_duration` threshold auto-rejects loops; human review catches borderline cases.
- **Accent homogeneity:** All data synthesized from a single TTS voice creates a model that only recognises one speaker's accent. Mitigation: Chatterbox supports reference audio prompting — future versions of this pipeline can pass multiple reference clips to diversify voices.
- **Prosody bias:** TTS prosody is more uniform than natural speech. Models trained entirely on TTS data may fail on natural speech with high prosodic variation. Mitigation: this pipeline is designed to supplement real data, not replace it.
- **Text normalization mismatch:** Numbers, dates, and currency are written differently in text than they are spoken. Mitigation: `has_numbers` flag in Stage 3; reviewer listens specifically for digit pronunciation errors.
- **MSA drift in generated text:** LLMs default to formal Arabic. Text that sounds formal to the TTS model produces unnaturally formal prosody. Mitigation: dialect validation in Stage 1 + seed corpus fallback of hand-crafted colloquial sentences.

---

## 9 · Trade-offs & Design Decisions

| Decision | We Chose | Alternative | Why |
|---|---|---|---|
| Text source | Local LLM (Ollama/qwen2.5) + seed corpus | Web scraping / existing corpora | Control over dialect & domain; no copyright issues; reproducible |
| TTS engine | Chatterbox + NAMAA Egyptian fine-tune | gTTS / ElevenLabs / Azure | Fully local; MIT license; Egyptian dialect awareness |
| Review interface | FastAPI web UI with audio playback | Spreadsheet / Argilla / Label Studio | Zero extra deps; ships in one file; audio plays inline |
| Export format | HuggingFace datasets JSONL | CSV / custom binary / TFRecord | Universal; `load_dataset()` works immediately; human-readable |
| Checkpointing | JSON files per stage | SQLite / Redis / MLflow | Simple, portable, inspectable with any text editor |
| Quality signals | Custom Python (soundfile + numpy) | OpenSMILE / SpeechBrain | No heavy dependencies; covers all critical failure modes |
| LLM for generation | qwen2.5:7b (quantised, local) | GPT-4o / Claude / Gemini | Free forever; offline; no API key; dialect prompt tunable |

### 9.1 Key Engineering Trade-offs Explained

**Local LLM vs. Remote API for Text Generation**
A remote LLM API (GPT-4, Claude) would produce slightly better Egyptian dialect quality today. We chose a local LLM because: (a) it eliminates ongoing cost at scale, (b) it eliminates internet dependency, (c) the dialect prompt can be tuned to compensate for quality gaps, and (d) the seed corpus fallback means the pipeline always works. The architecture supports swapping in any Ollama-compatible model — upgrading is a one-line config change.

**Custom Quality Signals vs. ML-Based Audio Quality Models**
Tools like SpeechBrain or OpenSMILE could compute more sophisticated quality metrics (MOS estimation, speaker similarity). We chose simple signal processing because: (a) no GPU required, (b) no additional model downloads, (c) the failure modes we need to catch (silence, repetition, truncation) are fully covered by duration, RMS, and silence ratio.

**JSONL Checkpoints vs. Database**
A proper database (SQLite, PostgreSQL) would enable richer queries over pipeline state. We chose JSON files because: (a) zero dependencies, (b) inspectable with any text editor, (c) trivially portable (just copy the file), (d) fully sufficient for single-machine operation up to ~100K samples.

---

## 10 · Limitations

### 10.1 Current Limitations

- **Single speaker voice:** All synthesized audio comes from one default TTS voice. Real training data benefits from speaker diversity. This can be addressed by providing multiple reference audio clips to Chatterbox's voice cloning feature.
- **No diacritization (tashkeel):** Input text is undiacritized, meaning the TTS model must infer pronunciation from context. This is the same condition real Egyptian Arabic text exists in, but introduces ambiguity.
- **GPU recommended for synthesis:** CPU synthesis is supported but slow — approximately 5–15x real-time on a modern CPU. GPU (CUDA or Apple Silicon MPS) reduces this to near real-time. The `device` setting is configurable.
- **No automated dialect verification:** We cannot automatically verify that a generated sentence is genuinely Egyptian Arabic (vs. MSA drift). This relies on the LLM prompt quality and human review.
- **Scale:** The demo pipeline generates ~340 prompts. Production STT fine-tuning typically requires thousands of hours. This pipeline is designed as the foundation — it handles checkpointing and batching for scale, but prompt generation at 10K+ count would benefit from more diverse LLM prompting strategies.
- **ق pronunciation gap:** The NAMAA model occasionally mispronounces Egyptian ق as [q] rather than glottal stop. This is a known limitation documented by the model authors. Human review is essential to catch it.

### 10.2 Known Chatterbox Limitations for Arabic

- **Repetition penalty:** The model can enter repetition loops, especially for longer sentences. The model authors recommend increasing `repetition_penalty` to mitigate. Our pipeline addresses this via `max_duration` threshold.
- **Number reading:** Arabic digit sequences are inconsistently handled. We flag sentences containing numbers for mandatory human review.
- **Long text degradation:** Quality tends to degrade for sentences above ~35 words. Our length distribution caps at 35 words specifically to avoid this range.

---

## 11 · Installation & Step-by-Step Run Guide

### 11.1 Prerequisites

- **Python 3.11+** — tested on Python 3.11; install from python.org
- **pip** — comes with Python
- **git** — for cloning
- **Ollama (optional)** — for LLM text generation: https://ollama.com
- **GPU (optional but recommended)** — NVIDIA CUDA or Apple Silicon MPS for fast TTS

### 11.2 Installation

```bash
# 1. Clone the repository
https://github.com/Maryemehab/Synthetic-Speech-Data-Pipeline.git


# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Install Chatterbox TTS (Egyptian Arabic TTS engine)
pip install chatterbox-tts

# 5. (Optional) Install Ollama for LLM text generation
# Download from https://ollama.com/download
ollama serve                      # start Ollama server
ollama pull qwen2.5:7b            # download the model (~4.7GB)
```

### 11.3 Configuration

```yaml
# Edit config/config.yaml to adjust:

tts_synthesis:
  model_variant: "namaa"   # or "oddadmix" to use chatterbox-egyptian-v0
  device: "cpu"            # change to "cuda" for GPU acceleration

text_generation:
  target_count: 120        # increase for larger datasets
  use_seed_fallback: true  # false = fail if Ollama unavailable
```

### 11.4 Running the Pipeline

```bash
# Stage 1: Generate Egyptian Arabic text prompts
python -m pipeline.stage1_generate
# Output: data/raw_text/prompts.jsonl

# Stage 2: Synthesize audio (this is the long-running stage)
python -m pipeline.stage2_synthesize
# Output: data/audio/*.wav  +  data/audio/synthesis_manifest.jsonl
# Note: can be interrupted and resumed at any time

# Stage 3a: Compute automated quality signals
python -m pipeline.stage3_quality
# Output: data/audio/scored_manifest.jsonl

# Stage 3b: Start the review web UI
uvicorn review_ui.app:app --host 0.0.0.0 --port 8003
# Open http://localhost:8003 in your browser
# Review each sample: Approve / Reject / Uncertain
# Output: data/reviewed/review_decisions.jsonl

# Stage 4: Export approved samples to training-ready format
python -m pipeline.stage4_export
# Output: data/exports/final_dataset/
```

### 11.5 Or Run Everything at Once

```bash
# Full pipeline run (skips synthesis review — for automated testing)
python run_pipeline.py --all

# Individual stages:
python run_pipeline.py --stage generate
python run_pipeline.py --stage synthesize
python run_pipeline.py --stage score
python run_pipeline.py --stage review    # starts web UI
python run_pipeline.py --stage export
```

---

## 12 · Project File Structure

```
ssdp/
├── config/
│   └── config.yaml              # All configuration (no hardcoded values)
│
├── pipeline/
│   ├── __init__.py
│   ├── utils.py                 # Config loader, logger, Checkpoint class
│   ├── seed_corpus.py           # 340 hand-crafted Egyptian Arabic sentences
│   ├── stage1_generate.py       # Text prompt generation (Ollama + fallback)
│   ├── stage2_synthesize.py     # TTS audio synthesis (Chatterbox)
│   ├── stage3_quality.py        # Automated quality signals
│   └── stage4_export.py         # Training-ready dataset export
│
├── review_ui/
│   ├── app.py                   # FastAPI review web application
│   └── static/                  # CSS/JS assets
│
├── tests/
│   ├── test_stage1.py           # Prompt generation tests
│   ├── test_quality.py          # Quality signal tests
│   └── test_export.py           # Export format tests
│
├── data/                        # Pipeline outputs (gitignored except samples)
│   ├── raw_text/
│   │   ├── prompts.jsonl
│   │   └── .gen_checkpoint.json
│   ├── audio/
│   │   ├── *.wav
│   │   ├── synthesis_manifest.jsonl
│   │   ├── scored_manifest.jsonl
│   │   └── .synth_checkpoint.json
│   ├── reviewed/
│   │   └── review_decisions.jsonl
│   └── exports/
│       └── final_dataset/
│           ├── metadata.jsonl
│           ├── train.jsonl
│           ├── test.jsonl
│           └── audio/
│
├── logs/
│   ├── pipeline.log             # Human-readable log
│   └── pipeline_events.jsonl    # Machine-readable structured log
│
├── requirements.txt             # Python dependencies
└── run_pipeline.py              # Single entry-point runner
```

---

## 13 · Tools and Libraries

| Library | License | Role |
|---|---|---|
| `chatterbox-tts` | MIT | Core TTS engine — neural speech synthesis |
| `torch` / `torchaudio` | BSD-3 | Audio tensor operations and resampling |
| `safetensors` | Apache 2.0 | Load fine-tuned `.safetensors` checkpoints |
| `huggingface-hub` | Apache 2.0 | Download model weights from HuggingFace |
| `FastAPI` | MIT | Review web interface framework |
| `uvicorn` | BSD-3 | ASGI server to run FastAPI |
| `soundfile` | BSD-3 | Read WAV files for quality analysis |
| `numpy` | BSD-3 | Audio signal math (RMS, silence ratio, SNR) |
| `requests` | Apache 2.0 | HTTP client for Ollama API calls |
| `PyYAML` | MIT | Parse `config.yaml` |
| `datasets` (HuggingFace) | Apache 2.0 | Export dataset format compatibility |
| `tqdm` | MIT | Progress bars for long loops |
| `pytest` | MIT | Unit testing framework |

### 13.1 Why These Specific Choices

- **FastAPI over Flask:** Async-ready, auto-generates `/docs` endpoint, modern Python type hints, active development. For a web interface serving audio files, async I/O matters.
- **soundfile + numpy over librosa:** librosa is a heavy dependency with many transitive deps. For our quality signals (RMS, silence ratio), soundfile + numpy is sufficient and installs in seconds.
- **JSONL over CSV for all data files:** JSONL handles Unicode naturally (critical for Arabic text), supports nested fields without escaping, and can be appended without reading the whole file — essential for streaming writes during long synthesis runs.
- **PyYAML over .env or argparse:** A YAML config file supports hierarchical configuration, inline comments, and multiple environments. Arguments and env vars are impractical when you have 40+ configuration values.

---

## 14 · Final Dataset — Sample Records

```json
{ "id": "prompt_00000", "text": "أنا رايح السوق دلوقتي، عايز حاجة؟",
  "domain": "daily_life", "audio_path": "audio/prompt_00000.wav",
  "duration_sec": 2.24, "sample_rate": 16000, "word_count": 6,
  "model_variant": "namaa", "split": "train" }

{ "id": "prompt_00001", "text": "الجو حر جداً النهارده، مش طايق أقعد برة.",
  "domain": "daily_life", "audio_path": "audio/prompt_00001.wav",
  "duration_sec": 2.64, "sample_rate": 16000, "word_count": 8,
  "model_variant": "namaa", "split": "train" }

{ "id": "prompt_00199", "text": "إحنا محتاجين نشتري تلاجة جديدة، دي بقالها عشر سنين.",
  "domain": "family_home", "audio_path": "audio/prompt_00199.wav",
  "duration_sec": 3.64, "sample_rate": 16000, "word_count": 9,
  "model_variant": "namaa", "split": "test" }


```

---

## 15 · Future Improvements

1. **Multi-speaker synthesis** — Pass multiple Egyptian reference audio clips to Chatterbox's voice cloning for speaker diversity. Critical for STT generalisation.
2. **Diacritization pre-processing** — Add an Arabic diacritizer (e.g., Mishkal) before TTS to improve pronunciation consistency.
3. **Automated dialect classifier** — Train or use a pre-existing Egyptian/MSA dialect classifier to auto-score LLM-generated text before synthesis, eliminating the need for Stage 1 manual review.
4. **Scale to 10K+ prompts** — The checkpointing and batching infrastructure already supports this. The main addition needed is more diverse LLM domain prompts and longer runs.

---
