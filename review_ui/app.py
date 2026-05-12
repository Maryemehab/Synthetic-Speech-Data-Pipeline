
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
importlib.invalidate_caches()

from pipeline.utils import load_config, load_jsonl, append_jsonl, get_logger

log = get_logger("review_ui")


app = FastAPI(title="SSDP Review UI", version="1.0.0")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_cfg: dict = {}


@app.on_event("startup")
async def startup():
    global _cfg
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    _cfg = load_config(str(config_path))
    log.info("Review UI started")



def _load_scored_manifest() -> list[dict]:
    scored_path = (
        Path(_cfg["tts_synthesis"]["manifest_file"]).parent
        / "scored_manifest.jsonl"
    )
    # Fall back to plain manifest if scored not available
    if not scored_path.exists():
        return load_jsonl(_cfg["tts_synthesis"]["manifest_file"])
    return load_jsonl(scored_path)


def _load_decisions() -> dict[str, dict]:
    """Load existing review decisions as id → record dict."""
    records = load_jsonl(_cfg["review"]["review_db"])
    return {r["id"]: r for r in records}


def _save_decision(item_id: str, status: str, note: str = ""):
    rec = {
        "id":          item_id,
        "status":      status,
        "note":        note,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    db_path = Path(_cfg["review"]["review_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_jsonl(db_path)
    updated  = {r["id"]: r for r in existing}
    updated[item_id] = rec

    with open(db_path, "w", encoding="utf-8") as f:
        for r in updated.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_stats(manifest: list[dict], decisions: dict[str, dict]) -> dict:
    total     = len(manifest)
    reviewed  = len(decisions)
    approved  = sum(1 for d in decisions.values() if d["status"] == "approved")
    rejected  = sum(1 for d in decisions.values() if d["status"] == "rejected")
    uncertain = sum(1 for d in decisions.values() if d["status"] == "uncertain")
    pending   = total - reviewed
    return {
        "total":     total,
        "reviewed":  reviewed,
        "approved":  approved,
        "rejected":  rejected,
        "uncertain": uncertain,
        "pending":   pending,
        "pct_done":  round(100 * reviewed / max(total, 1), 1),
    }



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main review page — shows the next unreviewed item."""
    manifest  = _load_scored_manifest()
    decisions = _load_decisions()
    stats     = _get_stats(manifest, decisions)

    # Find the first unreviewed item
    item = None
    for rec in manifest:
        if rec["id"] not in decisions and rec.get("status") != "failed":
            if rec.get("auto_pass", True):   # skip auto-rejected
                item = rec
                break

    # If all auto-pass items reviewed, show uncertain ones
    if not item:
        for rec in manifest:
            if rec["id"] not in decisions and rec.get("status") != "failed":
                item = rec
                break

    return HTMLResponse(_render_main_page(item, stats, decisions, manifest))


@app.get("/item/{item_id}", response_class=HTMLResponse)
async def show_item(item_id: str):
    """Show a specific item by ID."""
    manifest  = _load_scored_manifest()
    decisions = _load_decisions()
    stats     = _get_stats(manifest, decisions)

    item = next((r for r in manifest if r["id"] == item_id), None)
    return HTMLResponse(_render_main_page(item, stats, decisions, manifest))


@app.post("/decide")
async def decide(request: Request):
    """Record a review decision and redirect to next item."""
    form   = await request.form()
    pid    = form.get("id")
    status = form.get("status")
    note   = form.get("note", "")

    if pid and status in ("approved", "rejected", "uncertain"):
        _save_decision(pid, status, note)
        log.info(f"Decision: {pid} → {status}")

    # Redirect back to main page (next item)
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/", status_code=303)


@app.get("/audio/{item_id}")
async def serve_audio(item_id: str):
    """Serve the WAV file for a given item ID directly in browser."""
    manifest = _load_scored_manifest()
    rec = next((r for r in manifest if r["id"] == item_id), None)
    if not rec or not rec.get("audio_path"):
        return JSONResponse({"error": "not found"}, status_code=404)
    path = Path(rec["audio_path"])
    if not path.exists():
        return JSONResponse({"error": "file missing"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/api/stats")
async def api_stats():
    """JSON stats endpoint for the dashboard."""
    manifest  = _load_scored_manifest()
    decisions = _load_decisions()
    return _get_stats(manifest, decisions)


@app.get("/queue", response_class=HTMLResponse)
async def queue():
    """Show the full review queue with status for every item."""
    manifest  = _load_scored_manifest()
    decisions = _load_decisions()
    stats     = _get_stats(manifest, decisions)
    return HTMLResponse(_render_queue_page(manifest, decisions, stats))




def _badge(label: str, value, good: bool | None = None) -> str:
    colour = "#555"
    if good is True:  colour = "#1a7a3a"
    if good is False: colour = "#b91c1c"
    return (f'<span style="background:{colour};color:#fff;padding:3px 8px;'
            f'border-radius:4px;font-size:12px;margin:2px;display:inline-block">'
            f'{label}: {value}</span>')


def _signal_badges(rec: dict) -> str:
    """Render quality signal badges for one record."""
    badges = []

    dur = rec.get("duration_sec")
    if dur is not None:
        ok = 0.5 <= dur <= 30
        badges.append(_badge("duration", f"{dur:.1f}s", ok))

    slr = rec.get("silence_ratio")
    if slr is not None:
        ok = slr < 0.8
        badges.append(_badge("silence", f"{slr:.0%}", ok))

    rms = rec.get("rms_energy")
    if rms is not None:
        ok = rms > 0.001
        badges.append(_badge("RMS", f"{rms:.4f}", ok))

    wps = rec.get("word_rate_wps")
    if wps is not None:
        ok = 0.8 <= wps <= 8.0
        badges.append(_badge("wps", f"{wps:.1f}", ok))

    if rec.get("has_numbers"):
        badges.append(_badge("⚠ numbers", "may misread", False))
    if rec.get("has_latin"):
        badges.append(_badge("code-switch", "has Latin", None))

    ap = rec.get("auto_pass")
    if ap is True:
        badges.append(_badge("AUTO", "PASS", True))
    elif ap is False:
        reason = rec.get("auto_reject_reason", "")
        badges.append(_badge("AUTO", f"FAIL: {reason}", False))

    return " ".join(badges)


def _render_main_page(item, stats: dict, decisions: dict, manifest: list) -> str:
    """Render the main review page HTML."""

    if item:
        decision = decisions.get(item["id"], {})
        current_status = decision.get("status", "")
        current_note   = decision.get("note", "")

        # Previous / next navigation
        ids = [r["id"] for r in manifest if r.get("status") != "failed"]
        idx = ids.index(item["id"]) if item["id"] in ids else 0
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx < len(ids) - 1 else None

        nav_html = ""
        if prev_id:
            nav_html += f'<a href="/item/{prev_id}" style="margin-right:16px">◀ Previous</a>'
        nav_html += f'<span style="color:#888">Item {idx+1} of {len(ids)}</span>'
        if next_id:
            nav_html += f'<a href="/item/{next_id}" style="margin-left:16px">Next ▶</a>'

        item_html = f"""
        <div style="background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <code style="color:#94a3b8;font-size:13px">{item['id']}</code>
            <span style="color:#64748b;font-size:13px">domain: {item.get('domain','')}</span>
          </div>

          <!-- Arabic text — RTL direction -->
          <div style="font-size:26px;line-height:1.8;direction:rtl;text-align:right;
                      color:#f1f5f9;background:#0f172a;padding:16px;border-radius:8px;
                      margin-bottom:16px;font-family:'Amiri',serif">
            {item['text']}
          </div>

          <!-- Audio player -->
          <audio controls style="width:100%;margin-bottom:16px" preload="auto">
            <source src="/audio/{item['id']}" type="audio/wav">
            Your browser does not support audio.
          </audio>

          <!-- Quality signal badges -->
          <div style="margin-bottom:16px">{_signal_badges(item)}</div>

          <!-- Decision form -->
          <form method="POST" action="/decide">
            <input type="hidden" name="id" value="{item['id']}">
            <div style="display:flex;gap:12px;margin-bottom:12px">
              <button name="status" value="approved"
                style="flex:1;padding:12px;border:none;border-radius:8px;cursor:pointer;
                       font-size:15px;font-weight:600;
                       background:{'#14532d' if current_status=='approved' else '#166534'};
                       color:#dcfce7">
                ✓ Approve
              </button>
              <button name="status" value="rejected"
                style="flex:1;padding:12px;border:none;border-radius:8px;cursor:pointer;
                       font-size:15px;font-weight:600;
                       background:{'#450a0a' if current_status=='rejected' else '#7f1d1d'};
                       color:#fee2e2">
                ✗ Reject
              </button>
              <button name="status" value="uncertain"
                style="flex:1;padding:12px;border:none;border-radius:8px;cursor:pointer;
                       font-size:15px;font-weight:600;
                       background:{'#422006' if current_status=='uncertain' else '#78350f'};
                       color:#fef3c7">
                ? Uncertain
              </button>
            </div>
            <input type="text" name="note" value="{current_note}"
              placeholder="Optional note (mispronunciation, accent issue, etc.)"
              style="width:100%;padding:10px;background:#0f172a;border:1px solid #334155;
                     border-radius:6px;color:#f1f5f9;font-size:14px;box-sizing:border-box">
          </form>

          <!-- Navigation -->
          <div style="margin-top:16px;text-align:center;color:#94a3b8">{nav_html}</div>
        </div>
        """
        current_id_for_queue = item["id"]
    else:
        item_html = """
        <div style="text-align:center;padding:60px;color:#64748b">
          <div style="font-size:48px;margin-bottom:16px">✓</div>
          <h2>All items reviewed!</h2>
          <p>Run Stage 4 to export approved samples.</p>
        </div>
        """
        current_id_for_queue = ""

    progress_pct = stats["pct_done"]
    progress_bar = f"""
    <div style="background:#1e293b;border-radius:8px;overflow:hidden;margin-bottom:8px">
      <div style="background:linear-gradient(90deg,#166534,#15803d);height:8px;
                  width:{progress_pct}%;transition:width 0.3s"></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;text-align:center">
      <div><div style="font-size:24px;font-weight:700;color:#f1f5f9">{stats['total']}</div><div style="color:#64748b;font-size:12px">Total</div></div>
      <div><div style="font-size:24px;font-weight:700;color:#4ade80">{stats['approved']}</div><div style="color:#64748b;font-size:12px">Approved</div></div>
      <div><div style="font-size:24px;font-weight:700;color:#f87171">{stats['rejected']}</div><div style="color:#64748b;font-size:12px">Rejected</div></div>
      <div><div style="font-size:24px;font-weight:700;color:#fbbf24">{stats['uncertain']}</div><div style="color:#64748b;font-size:12px">Uncertain</div></div>
      <div><div style="font-size:24px;font-weight:700;color:#94a3b8">{stats['pending']}</div><div style="color:#64748b;font-size:12px">Pending</div></div>
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="ar" dir="ltr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSDP Review — Egyptian Arabic TTS</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f172a; color: #f1f5f9; font-family: 'IBM Plex Mono', monospace;
           min-height: 100vh; padding: 24px; }}
    a {{ color: #60a5fa; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1 {{ font-size: 18px; font-weight: 600; letter-spacing: 0.05em; }}
  </style>
</head>
<body>
  <div style="max-width:800px;margin:0 auto">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
      <h1>🎙 SSDP · Egyptian Arabic Review</h1>
      <a href="/queue" style="font-size:13px;color:#94a3b8">View Queue</a>
    </div>

    <!-- Progress -->
    <div style="background:#1e293b;border-radius:12px;padding:20px;margin-bottom:24px">
      <div style="font-size:13px;color:#64748b;margin-bottom:12px">
        Progress: {progress_pct}% complete
      </div>
      {progress_bar}
    </div>

    <!-- Main item -->
    {item_html}
  </div>
</body>
</html>"""


def _render_queue_page(manifest: list, decisions: dict, stats: dict) -> str:
    """Render the full queue listing."""
    rows = []
    for rec in manifest:
        pid     = rec["id"]
        decision = decisions.get(pid, {})
        status  = decision.get("status", "pending")
        colour  = {"approved": "#4ade80", "rejected": "#f87171",
                   "uncertain": "#fbbf24", "pending": "#94a3b8"}.get(status, "#94a3b8")
        auto    = "✓" if rec.get("auto_pass") else "✗"
        auto_c  = "#4ade80" if rec.get("auto_pass") else "#f87171"
        dur     = f"{rec.get('duration_sec',0):.1f}s" if rec.get("duration_sec") else "—"
        rows.append(
            f'<tr style="border-bottom:1px solid #1e293b">'
            f'<td style="padding:8px"><a href="/item/{pid}">{pid}</a></td>'
            f'<td style="padding:8px;color:#64748b">{rec.get("domain","")}</td>'
            f'<td style="padding:8px;direction:rtl;text-align:right;max-width:300px;'
            f'overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{rec["text"]}</td>'
            f'<td style="padding:8px;color:{auto_c}">{auto}</td>'
            f'<td style="padding:8px">{dur}</td>'
            f'<td style="padding:8px;color:{colour}">{status}</td>'
            f'</tr>'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>SSDP Review Queue</title>
<style>
  body {{ background:#0f172a;color:#f1f5f9;font-family:monospace;padding:24px }}
  a {{ color:#60a5fa;text-decoration:none }}
  table {{ width:100%;border-collapse:collapse }}
  th {{ text-align:left;padding:8px;color:#64748b;border-bottom:2px solid #334155 }}
</style></head>
<body>
  <div style="max-width:1000px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
      <h2>Review Queue ({stats['total']} items)</h2>
      <a href="/">← Back to Review</a>
    </div>
    <table>
      <thead><tr>
        <th>ID</th><th>Domain</th><th>Text</th><th>Auto</th><th>Duration</th><th>Status</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body></html>"""