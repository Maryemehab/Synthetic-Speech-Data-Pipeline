
import json
import re
import time
from pathlib import Path

import requests

from .utils import (
    Checkpoint, append_jsonl, load_jsonl,
    get_logger, load_config, ensure_dirs
)
from .seed_corpus import get_seed_sentences

log = get_logger("stage1")



SYSTEM_PROMPT = """أنت خبير في اللهجة المصرية العامية البحتة.
مهمتك كتابة جمل طبيعية بالعامية المصرية كما يتكلمها المصريون في حياتهم اليومية.

القواعد الإلزامية:
- اكتب بالعامية المصرية فقط، مش بالفصحى.
- استخدم: مش، عايز، إيه، كده، بقى، يعني، طيب، أيوه، لأ، والله، ده، دي، دول
- استخدم ح للمستقبل: حروح، حنعمل، هيجي
- استخدم النفي المصري: ما...ش أو مش
- ممكن تحط كلمات أجنبية شائعة: موبايل، كمبيوتر، أتوبيس، شانطة، باص
- الجمل لازم تبان زي كلام ناس حقيقيين مش كتب
- رد بـ JSON array فقط، بدون أي كلام تاني"""


def _build_generation_prompt(domain: str, hint: str, count: int,
                              min_w: int, max_w: int) -> str:
    """Build the per-batch user prompt."""
    return (
        f"اكتب {count} جملة بالعامية المصرية.\n"
        f"الموضوع: {hint}\n"
        f"طول كل جملة: بين {min_w} و {max_w} كلمة تقريباً.\n"
        f"الجمل لازم تكون متنوعة ومختلفة عن بعض.\n\n"
        f"اكتب فقط JSON array هكذا (بدون أي نص إضافي):\n"
        f'["الجملة الأولى", "الجملة التانية", ...]'
    )



# Ollama client


def _call_ollama(base_url: str, model: str, system: str, user: str,
                 options: dict, timeout: int) -> str | None:
    """
    Call the Ollama /api/chat endpoint and return the assistant's message.

    Ollama runs locally — install from https://ollama.com then:
      ollama serve
      ollama pull qwen2.5:7b

    Returns None on any error so the caller can fall back gracefully.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model":    model,
        "stream":   False,
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ],
        "options": options,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.ConnectionError:
        log.warning("Ollama not reachable — is `ollama serve` running?")
        return None
    except Exception as e:
        log.warning(f"Ollama error: {e}")
        return None



# Parsing


def _parse_json_array(raw: str) -> list[str]:
    """
    Extract a JSON array from an LLM response.

    LLMs sometimes wrap JSON in markdown fences (```json ... ```) or add
    commentary before/after.  We strip the fences and find the first [].
    """
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.replace("```", "").strip()

    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        log.debug(f"No JSON array found in: {raw[:120]}")
        return []

    try:
        items = json.loads(raw[start: end + 1])
        return [s.strip() for s in items if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError as e:
        log.debug(f"JSON parse error: {e}")
        return []



# Validation


def _validate(sentence: str) -> tuple[bool, str]:
    """
    Quick sanity checks on a generated sentence.

    We reject:
    • Sentences shorter than 3 words (fragments)
    • Sentences with <30% Arabic characters (LLM went off-rails to Latin)
    • Sentences that are just punctuation or numbers
    """
    if len(sentence.split()) < 3:
        return False, "too_few_words"

    arabic_chars = sum(1 for c in sentence if "\u0600" <= c <= "\u06FF")
    alpha_chars  = sum(1 for c in sentence if c.isalpha())

    if alpha_chars == 0:
        return False, "no_alpha"

    if arabic_chars / alpha_chars < 0.30:
        return False, "too_little_arabic"

    return True, ""




def generate_prompts(cfg: dict) -> list[dict]:
    """
    Run Stage 1: generate Egyptian Arabic text prompts.

    Strategy
    ─────────
    1. Try Ollama (local LLM).  Generate per (domain × length_bucket).
    2. If Ollama unavailable AND use_seed_fallback=true, merge in the
       120 hand-crafted seed sentences.
    3. Deduplicate by text.
    4. Write each record to prompts.jsonl immediately (streaming writes).
    5. Checkpoint each (domain, bucket) batch so restarts skip done work.

    Returns the full list of prompt records.
    """
    tg = cfg["text_generation"]
    ollama_cfg = tg["ollama"]

    out_path = Path(tg["output_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = Checkpoint(tg["checkpoint_file"])

    # Load already-generated texts for deduplication
    existing_texts: set[str] = {r["text"] for r in load_jsonl(out_path)}
    log.info(f"Stage 1 start — {len(existing_texts)} prompts already on disk")

    id_counter = len(existing_texts)
    domains    = tg["domains"]
    length_dist = tg["length_distribution"]
    buckets    = list(length_dist.items())

    ollama_ok  = False   # will be set True on first successful call

    #  Try Ollama 
    for domain_name, domain_spec in domains.items():
        for bucket_name, bucket_spec in buckets:
            ckpt_key = f"{domain_name}__{bucket_name}"
            if ckpt.done(ckpt_key):
                log.info(f"  ↩  checkpoint skip: {ckpt_key}")
                continue

            count = max(1, round(domain_spec["count"] * bucket_spec["weight"]))
            min_w = bucket_spec["min"]
            max_w = bucket_spec["max"]
            hint  = domain_spec["hint"]

            log.info(f"  ▶  {domain_name}/{bucket_name}  ({count} sentences, {min_w}–{max_w} words)")

            accepted = []
            for attempt in range(tg.get("retry_attempts", 3)):
                raw = _call_ollama(
                    base_url=ollama_cfg["base_url"],
                    model=ollama_cfg["model"],
                    system=SYSTEM_PROMPT,
                    user=_build_generation_prompt(domain_name, hint, count, min_w, max_w),
                    options=ollama_cfg.get("options", {}),
                    timeout=ollama_cfg.get("timeout_seconds", 120),
                )

                if raw is None:
                    log.warning(f"  ⚠  Ollama unavailable (attempt {attempt+1})")
                    break   # no point retrying if connection refused

                ollama_ok = True
                sentences = _parse_json_array(raw)

                if not sentences:
                    log.warning(f"  ⚠  Empty parse for {ckpt_key}, attempt {attempt+1}")
                    time.sleep(2)
                    continue

                for sent in sentences:
                    if sent in existing_texts:
                        continue
                    ok, reason = _validate(sent)
                    if not ok:
                        log.debug(f"  ✗  Rejected '{sent[:40]}' ({reason})")
                        continue
                    existing_texts.add(sent)
                    words = sent.split()
                    wc = len(words)
                    record = {
                        "id":            f"prompt_{id_counter:05d}",
                        "text":          sent,
                        "domain":        domain_name,
                        "length_bucket": bucket_name,
                        "word_count":    wc,
                        "char_count":    len(sent),
                        "source":        "ollama",
                    }
                    append_jsonl(out_path, record)
                    accepted.append(record)
                    id_counter += 1

                log.info(f"  ✓  Accepted {len(accepted)}/{len(sentences)}")
                ckpt.mark_done(ckpt_key, {"count": len(accepted)})
                break   # success

            else:
                # Exhausted retries without success — checkpoint as failed
                if ollama_ok:
                    ckpt.mark_failed(ckpt_key, "max_retries_exceeded")

    # Seed fallback 
    if not ollama_ok and tg.get("use_seed_fallback", True):
        # Ollama failed — use only the unique seed corpus for fallback.
        # Do not append text like "(variation)" because that changes the prompt.
        seeds = get_seed_sentences()
        target = tg.get("target_count", 120)
        added = 0
        for rec in seeds:
            if rec["text"] in existing_texts:
                continue
            existing_texts.add(rec["text"])
            rec["id"] = f"prompt_{id_counter:05d}"
            id_counter += 1
            append_jsonl(out_path, rec)
            added += 1
            if added >= target:
                break
        if added < target:
            log.warning(
                f"Only {added} unique seed sentences available; target {target} cannot be reached without Ollama."
            )
        log.info(f"  ✓  Added {added} seed sentences")

    elif tg.get("use_seed_fallback", True) and ollama_ok:
        # Ollama worked, but let's also top-up with seeds if we're under target
        current_count = len(load_jsonl(out_path))
        target = tg.get("target_count", 120)
        if current_count < target:
            gap = target - current_count
            log.info(f"Topping up with {gap} seed sentences (have {current_count}/{target})")
            seeds = get_seed_sentences()
            added = 0
            for rec in seeds:
                if added >= gap:
                    break
                if rec["text"] in existing_texts:
                    continue
                existing_texts.add(rec["text"])
                rec["id"] = f"prompt_{id_counter:05d}"
                id_counter += 1
                append_jsonl(out_path, rec)
                added += 1

    all_prompts = load_jsonl(out_path)
    log.info(f"Stage 1 complete — {len(all_prompts)} total prompts on disk")

    # Print domain distribution summary
    domain_counts: dict[str, int] = {}
    for r in all_prompts:
        domain_counts[r.get("domain", "unknown")] = domain_counts.get(r.get("domain", "unknown"), 0) + 1
    log.info("Domain distribution:")
    for d, c in sorted(domain_counts.items()):
        log.info(f"  {d:25s}: {c}")

    return all_prompts


if __name__ == "__main__":
    cfg = load_config()
    from .utils import setup_logging
    setup_logging(cfg)
    ensure_dirs(cfg)
    prompts = generate_prompts(cfg)
    print(f"\n✓ {len(prompts)} prompts ready in data/raw_text/prompts.jsonl")
    for p in prompts[:5]:
        print(f"  [{p['domain']:20s}] {p['text']}")