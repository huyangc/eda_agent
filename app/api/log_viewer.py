"""Log viewer — serves a browser-based UI at /logs.

Routes:
    GET /logs                      → HTML single-page app
    GET /logs/api/dates            → list of date folders
    GET /logs/api/sessions/{date}  → sessions for a date
    GET /logs/api/events/{date}/{filename}  → events for a session
"""

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings

router = APIRouter()


def _log_root() -> Path:
    return Path(settings.log_dir)


# ---------------------------------------------------------------------------
# Data API
# ---------------------------------------------------------------------------

@router.get("/logs/api/dates")
def list_dates():
    root = _log_root()
    if not root.exists():
        return JSONResponse([])
    dates = sorted(
        [d.name for d in root.iterdir() if d.is_dir()],
        reverse=True,
    )
    return JSONResponse(dates)


@router.get("/logs/api/sessions/{date}")
def list_sessions(date: str):
    day_dir = _log_root() / date
    if not day_dir.exists():
        return JSONResponse([])
    files = sorted(
        [f.name for f in day_dir.glob("*.jsonl")],
        reverse=True,
    )
    sessions = []
    for fname in files:
        parts = fname.replace(".jsonl", "").split("_", 1)
        sessions.append({
            "filename": fname,
            "time": parts[0] if len(parts) == 2 else "",
            "session_id": parts[1] if len(parts) == 2 else parts[0],
        })
    return JSONResponse(sessions)


@router.get("/logs/api/events/{date}/{filename}")
def get_events(date: str, filename: str):
    path = _log_root() / date / filename
    if not path.exists():
        return JSONResponse([])
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return JSONResponse(events)


# ---------------------------------------------------------------------------
# Single-page app HTML
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>EDA Agent — Log Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Menlo', 'Consolas', monospace; font-size: 13px;
         background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  #sidebar { width: 260px; min-width: 260px; background: #161b22; border-right: 1px solid #30363d;
             display: flex; flex-direction: column; overflow: hidden; }
  #sidebar h2 { padding: 14px 16px; font-size: 12px; color: #8b949e; text-transform: uppercase;
                letter-spacing: .08em; border-bottom: 1px solid #30363d; }
  #dates { overflow-y: auto; flex: 1; }
  .date-group { border-bottom: 1px solid #21262d; }
  .date-header { padding: 9px 16px; cursor: pointer; color: #58a6ff; font-size: 12px;
                 display: flex; justify-content: space-between; align-items: center;
                 user-select: none; }
  .date-header:hover { background: #1c2128; }
  .date-header .arrow { transition: transform .2s; }
  .date-header.open .arrow { transform: rotate(90deg); }
  .session-list { display: none; padding: 4px 0; }
  .session-list.open { display: block; }
  .session-item { padding: 6px 16px 6px 28px; cursor: pointer; color: #8b949e;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .session-item:hover { background: #1c2128; color: #c9d1d9; }
  .session-item.active { background: #1f2d3d; color: #58a6ff; }
  .session-time { font-size: 11px; color: #3fb950; }
  .session-id  { font-size: 11px; }

  /* Main */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #header { padding: 12px 20px; border-bottom: 1px solid #30363d; background: #161b22;
            display: flex; align-items: center; gap: 12px; }
  #header h1 { font-size: 13px; color: #8b949e; }
  #header span { color: #c9d1d9; font-size: 13px; }
  #events { flex: 1; overflow-y: auto; padding: 16px 20px; }
  #placeholder { color: #484f58; text-align: center; margin-top: 80px; font-size: 14px; }

  /* Event cards */
  .event { margin-bottom: 10px; border-radius: 6px; border: 1px solid #30363d; overflow: hidden; }
  .event-header { padding: 6px 12px; display: flex; gap: 12px; align-items: center;
                  background: #161b22; cursor: pointer; user-select: none; }
  .event-header:hover { background: #1c2128; }
  .ev-ts   { color: #484f58; font-size: 11px; min-width: 90px; }
  .ev-type { font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 4px; }
  .ev-summary { color: #8b949e; font-size: 12px; flex: 1; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
  .ev-toggle { color: #484f58; font-size: 11px; }
  .event-body { padding: 10px 12px; background: #0d1117; border-top: 1px solid #21262d;
                font-size: 12px; line-height: 1.6; display: none; }
  .event-body.open { display: block; }
  .event-body pre { white-space: pre-wrap; word-break: break-word; color: #c9d1d9; }

  /* Type colors */
  .t-request      { background: #1f3a5f; color: #58a6ff; }
  .t-intent       { background: #2d2a1f; color: #e3b341; }
  .t-cmd_extract  { background: #1f2d3d; color: #79c0ff; }
  .t-rag_query    { background: #1f3a2f; color: #3fb950; }
  .t-rag_response { background: #1a3a2a; color: #56d364; }
  .t-llm_response { background: #2d1f3d; color: #bc8cff; }
  .t-tool_calls   { background: #3a1f1f; color: #ff7b72; }
  .t-tool_results { background: #3a2a1f; color: #ffa657; }
  .t-default      { background: #21262d; color: #8b949e; }
</style>
</head>
<body>

<div id="sidebar">
  <h2>Sessions</h2>
  <div id="dates"><div style="padding:16px;color:#484f58">Loading…</div></div>
</div>

<div id="main">
  <div id="header">
    <h1>EDA Agent Log Viewer</h1>
    <span id="current-session"></span>
  </div>
  <div id="events">
    <div id="placeholder">← Select a session from the sidebar</div>
  </div>
</div>

<script>
const TYPE_COLORS = {
  request: 't-request', intent: 't-intent', cmd_extract: 't-cmd_extract',
  rag_query: 't-rag_query', rag_response: 't-rag_response',
  llm_response: 't-llm_response', tool_calls: 't-tool_calls',
  tool_results: 't-tool_results',
};

function typeClass(t) { return TYPE_COLORS[t] || 't-default'; }

function summarize(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case 'request': {
      const msgs = d.messages || [];
      const last = msgs[msgs.length - 1] || {};
      const text = (last.content || '').replace(/\\n/g, ' ').slice(0, 80);
      return `[${last.role || '?'}] ${text}`;
    }
    case 'intent':
      return `${d.intent}  conf=${(d.confidence||0).toFixed(2)}  ${(d.reason||'').slice(0,60)}`;
    case 'cmd_extract':
      return `ns=${d.namespace}  cmds=[${(d.commands||[]).join(', ')}]  mode=${d.output_mode}`;
    case 'rag_query':
      return `ns=${d.namespace}  q=${(d.query||'').slice(0,60)}`;
    case 'rag_response': {
      const docs = d.docs || [];
      return `${docs.length} docs  q=${(d.query||'').slice(0,50)}`;
    }
    case 'llm_response':
      return `mode=${d.output_mode}  len=${(d.content||'').length}  ${(d.content||'').slice(0,60)}`;
    case 'tool_calls': {
      const calls = d.calls || [];
      return calls.map(c => c.name).join(', ');
    }
    case 'tool_results': {
      const results = d.results || [];
      return `${results.length} result(s)`;
    }
    default:
      return JSON.stringify(d).slice(0, 80);
  }
}

function renderEvent(ev) {
  const div = document.createElement('div');
  div.className = 'event';

  const header = document.createElement('div');
  header.className = 'event-header';
  header.innerHTML = `
    <span class="ev-ts">${ev.ts || ''}</span>
    <span class="ev-type ${typeClass(ev.type)}">${ev.type}</span>
    <span class="ev-summary">${summarize(ev)}</span>
    <span class="ev-toggle">▶</span>`;

  const body = document.createElement('div');
  body.className = 'event-body';
  body.innerHTML = `<pre>${JSON.stringify(ev.data, null, 2)}</pre>`;

  header.addEventListener('click', () => {
    body.classList.toggle('open');
    header.querySelector('.ev-toggle').textContent =
      body.classList.contains('open') ? '▼' : '▶';
  });

  div.appendChild(header);
  div.appendChild(body);
  return div;
}

async function loadEvents(date, filename, sessionId) {
  document.getElementById('current-session').textContent = `${date} / ${sessionId}`;
  document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`[data-file="${filename}"]`)?.classList.add('active');

  const eventsDiv = document.getElementById('events');
  eventsDiv.innerHTML = '<div style="color:#484f58;padding:20px">Loading…</div>';

  const resp = await fetch(`/logs/api/events/${date}/${encodeURIComponent(filename)}`);
  const events = await resp.json();

  eventsDiv.innerHTML = '';
  if (!events.length) {
    eventsDiv.innerHTML = '<div style="color:#484f58;padding:20px">No events found.</div>';
    return;
  }
  events.forEach(ev => eventsDiv.appendChild(renderEvent(ev)));
}

async function loadSessions(date, groupEl) {
  const listEl = groupEl.querySelector('.session-list');
  if (listEl.innerHTML.trim()) { return; }  // already loaded

  const resp = await fetch(`/logs/api/sessions/${date}`);
  const sessions = await resp.json();

  listEl.innerHTML = '';
  sessions.forEach(s => {
    const item = document.createElement('div');
    item.className = 'session-item';
    item.dataset.file = s.filename;
    item.innerHTML = `<div class="session-time">${s.time.replace(/(\\d{2})(\\d{2})(\\d{2})/, '$1:$2:$3')}</div>
                      <div class="session-id">${s.session_id}</div>`;
    item.addEventListener('click', () => loadEvents(date, s.filename, s.session_id));
    listEl.appendChild(item);
  });
}

async function init() {
  const resp = await fetch('/logs/api/dates');
  const dates = await resp.json();
  const datesDiv = document.getElementById('dates');
  datesDiv.innerHTML = '';

  dates.forEach((date, i) => {
    const group = document.createElement('div');
    group.className = 'date-group';
    group.innerHTML = `
      <div class="date-header ${i===0?'open':''}">
        <span>${date}</span><span class="arrow">▶</span>
      </div>
      <div class="session-list ${i===0?'open':''}"></div>`;

    const header = group.querySelector('.date-header');
    const list   = group.querySelector('.session-list');

    header.addEventListener('click', () => {
      const isOpen = list.classList.toggle('open');
      header.classList.toggle('open', isOpen);
      if (isOpen) loadSessions(date, group);
    });

    datesDiv.appendChild(group);
    if (i === 0) loadSessions(date, group);
  });
}

init();
</script>
</body>
</html>
"""


@router.get("/logs", response_class=HTMLResponse)
def log_viewer():
    return HTMLResponse(_HTML)
