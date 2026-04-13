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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Agent — Log Insights</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-dark: #0f172a;
    --bg-gradient: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    --glass-bg: rgba(30, 41, 59, 0.4);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-hover: rgba(51, 65, 85, 0.6);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    
    /* Event Colors */
    --c-req: #3b82f6;     /* blue */
    --c-req-bg: rgba(59, 130, 246, 0.15);
    --c-int: #eab308;     /* yellow */
    --c-int-bg: rgba(234, 179, 8, 0.15);
    --c-cmd: #0ea5e9;     /* sky */
    --c-cmd-bg: rgba(14, 165, 233, 0.15);
    --c-rag: #10b981;     /* emerald */
    --c-rag-bg: rgba(16, 185, 129, 0.15);
    --c-llm: #a855f7;     /* purple */
    --c-llm-bg: rgba(168, 85, 247, 0.15);
    --c-tool: #f43f5e;    /* rose */
    --c-tool-bg: rgba(244, 63, 94, 0.15);
    --c-def: #64748b;     /* slate */
    --c-def-bg: rgba(100, 116, 139, 0.15);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.4); }

  body { 
    font-family: 'Inter', sans-serif; 
    font-size: 14px;
    background: var(--bg-gradient); 
    color: var(--text-primary); 
    display: flex; 
    height: 100vh; 
    overflow: hidden; 
    -webkit-font-smoothing: antialiased;
  }

  /* Sidebar */
  #sidebar { 
    width: 280px; 
    min-width: 280px; 
    background: var(--glass-bg); 
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid var(--glass-border);
    display: flex; 
    flex-direction: column; 
    overflow: hidden; 
    box-shadow: 4px 0 24px rgba(0,0,0,0.2);
    z-index: 10;
  }
  #sidebar h2 { 
    padding: 20px; 
    font-size: 11px; 
    font-weight: 600;
    color: var(--text-muted); 
    text-transform: uppercase;
    letter-spacing: 0.1em; 
    border-bottom: 1px solid var(--glass-border); 
  }
  #dates { overflow-y: auto; flex: 1; padding: 12px; }
  
  .date-group { margin-bottom: 8px; }
  .date-header { 
    padding: 10px 12px; 
    cursor: pointer; 
    color: var(--text-primary); 
    font-weight: 500;
    font-size: 13px;
    display: flex; 
    justify-content: space-between; 
    align-items: center;
    user-select: none; 
    border-radius: 6px;
    transition: background 0.2s ease;
  }
  .date-header:hover { background: var(--glass-hover); }
  .date-header .arrow { 
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
    color: var(--text-muted);
    font-size: 10px;
  }
  .date-header.open .arrow { transform: rotate(90deg); }
  
  .session-list { 
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows 0.3s ease;
  }
  .session-list.open { grid-template-rows: 1fr; }
  .session-list-inner { overflow: hidden; }
  
  .session-item { 
    padding: 8px 12px 8px 24px; 
    margin: 4px 0;
    cursor: pointer; 
    color: var(--text-secondary);
    border-radius: 6px;
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .session-item:hover { 
    background: var(--glass-hover); 
    color: var(--text-primary); 
    transform: translateX(4px);
  }
  .session-item.active { 
    background: rgba(59, 130, 246, 0.1); 
    border-left: 3px solid var(--accent);
    color: var(--text-primary); 
    padding-left: 21px;
  }
  .session-time { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--c-rag); }
  .session-id  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); }

  /* Main */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }
  
  /* Decorative background blur objects */
  #main::before {
    content: ''; position: absolute; top: -100px; right: -100px; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
    z-index: 0; pointer-events: none;
  }

  #header { 
    padding: 20px 32px; 
    background: linear-gradient(to bottom, rgba(15, 23, 42, 0.8), transparent);
    backdrop-filter: blur(8px);
    display: flex; 
    align-items: baseline; 
    gap: 16px; 
    z-index: 10;
  }
  #header h1 { font-size: 18px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.02em; }
  #header span { color: var(--text-muted); font-size: 13px; font-family: 'JetBrains Mono', monospace;}
  
  #events { flex: 1; overflow-y: auto; padding: 20px 32px 60px; z-index: 1; scroll-behavior: smooth; }
  #placeholder { 
    color: var(--text-muted); 
    text-align: center; 
    margin-top: 120px; 
    font-size: 14px; 
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }

  /* Event cards */
  .event { 
    margin-bottom: 16px; 
    border-radius: 12px; 
    background: var(--glass-bg);
    border: 1px solid var(--glass-border); 
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
  }
  .event:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border-color: rgba(255, 255, 255, 0.15);
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .event-header { 
    padding: 14px 20px; 
    display: flex; 
    gap: 16px; 
    align-items: center;
    cursor: pointer; 
    user-select: none; 
  }
  .ev-ts { 
    color: var(--text-muted); 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 12px; 
    min-width: 95px; 
  }
  .ev-type { 
    font-size: 11px; 
    font-weight: 600; 
    padding: 4px 10px; 
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .ev-summary { 
    color: var(--text-secondary); 
    font-size: 13px; 
    flex: 1; 
    overflow: hidden;
    text-overflow: ellipsis; 
    white-space: nowrap; 
    font-family: 'JetBrains Mono', monospace;
  }
  .ev-toggle { 
    color: var(--text-muted); 
    font-size: 10px; 
    transition: transform 0.3s ease;
  }
  .event.open .ev-toggle { transform: rotate(180deg); }
  
  .event-body { 
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(0,0,0,0.2); 
  }
  .event.open .event-body { grid-template-rows: 1fr; border-top: 1px solid var(--glass-border); }
  .event-body-inner { 
    overflow: hidden; 
    padding: 0 20px; /* Paddings applied correctly with grid animation */
  }
  .event.open .event-body-inner {
    padding: 16px 20px;
  }

  /* JSON Syntax Highlighting */
  .json-view { 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 12px; 
    line-height: 1.6; 
    white-space: pre-wrap; 
    word-break: break-word; 
  }
  .json-key { color: #818cf8; } /* Indigo */
  .json-string { color: #a3e635; } /* Lime */
  .json-number { color: #f472b6; } /* Pink */
  .json-boolean { color: #fbbf24; } /* Amber */
  .json-null { color: #94a3b8; font-style: italic; }

  /* Type Colors */
  .t-request      { background: var(--c-req-bg); color: var(--c-req); border: 1px solid rgba(59, 130, 246, 0.2); }
  .t-intent       { background: var(--c-int-bg); color: var(--c-int); border: 1px solid rgba(234, 179, 8, 0.2); }
  .t-cmd_extract  { background: var(--c-cmd-bg); color: var(--c-cmd); border: 1px solid rgba(14, 165, 233, 0.2); }
  .t-rag_query    { background: var(--c-rag-bg); color: var(--c-rag); border: 1px solid rgba(16, 185, 129, 0.2); }
  .t-rag_response { background: var(--c-rag-bg); color: var(--c-rag); border: 1px solid rgba(16, 185, 129, 0.2); }
  .t-llm_response { background: var(--c-llm-bg); color: var(--c-llm); border: 1px solid rgba(168, 85, 247, 0.2); }
  .t-tool_calls   { background: var(--c-tool-bg); color: var(--c-tool); border: 1px solid rgba(244, 63, 94, 0.2); }
  .t-tool_results { background: var(--c-tool-bg); color: var(--c-tool); border: 1px solid rgba(244, 63, 94, 0.2); }
  .t-default      { background: var(--c-def-bg); color: var(--c-def); border: 1px solid rgba(100, 116, 139, 0.2); }
</style>
</head>
<body>

<div id="sidebar">
  <h2>Sessions</h2>
  <div id="dates">
    <div style="padding:16px;color:var(--text-muted);font-size:12px;text-align:center;">Loading…</div>
  </div>
</div>

<div id="main">
  <div id="header">
    <h1>EDA Insights</h1>
    <span id="current-session"></span>
  </div>
  <div id="events">
    <div id="placeholder">Select a session from the sidebar to view logs</div>
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

function syntaxHighlightJSON(json) {
    if (typeof json !== 'string') {
         json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
                return '<span class="' + cls + '">' + match.replace(/":$/, '"') + '</span>:';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function summarize(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case 'request': {
      const msgs = d.messages || [];
      const last = msgs[msgs.length - 1] || {};
      let text = (last.content || '');
      if (Array.isArray(text)) text = JSON.stringify(text);
      text = text.replace(/\\n/g, ' ').slice(0, 80);
      return `[${last.role || '?'}] ${text}`;
    }
    case 'intent':
      return `Intent: ${d.intent} | Conf: ${(d.confidence||0).toFixed(2)} | ${(d.reason||'').slice(0,60)}`;
    case 'cmd_extract':
      return `${d.namespace} » [${(d.commands||[]).join(', ')}] » ${d.output_mode}`;
    case 'rag_query':
      return `${d.namespace} » ${(d.query||'').slice(0,60)}`;
    case 'rag_response': {
      const docs = d.docs || [];
      return `Found ${docs.length} docs for: ${(d.query||'').slice(0,50)}`;
    }
    case 'llm_response':
      return `${d.output_mode.toUpperCase()} | Length: ${(d.content||'').length} | ${(d.content||'').slice(0,60).replace(/\\n/g, ' ')}`;
    case 'tool_calls': {
      const calls = d.calls || [];
      return `Called: ${calls.map(c => c.name).join(', ')}`;
    }
    case 'tool_results': {
      const results = d.results || [];
      return `Returned ${results.length} results`;
    }
    default:
      return JSON.stringify(d).slice(0, 80);
  }
}

function renderEvent(ev, index) {
  const div = document.createElement('div');
  div.className = 'event';
  div.style.animationDelay = `${Math.min(index * 0.05, 0.5)}s`;

  const header = document.createElement('div');
  header.className = 'event-header';
  header.innerHTML = `
    <span class="ev-ts">${ev.ts || ''}</span>
    <span class="ev-type ${typeClass(ev.type)}">${ev.type}</span>
    <span class="ev-summary">${summarize(ev)}</span>
    <span class="ev-toggle">▼</span>`;

  const bodyWrapper = document.createElement('div');
  bodyWrapper.className = 'event-body';
  
  const bodyInner = document.createElement('div');
  bodyInner.className = 'event-body-inner json-view';
  bodyInner.innerHTML = syntaxHighlightJSON(ev.data);
  bodyWrapper.appendChild(bodyInner);

  header.addEventListener('click', () => {
    div.classList.toggle('open');
  });

  div.appendChild(header);
  div.appendChild(bodyWrapper);
  return div;
}

async function loadEvents(date, filename, sessionId) {
  document.getElementById('current-session').textContent = `${date} / ${sessionId}`;
  document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
  const activeItem = document.querySelector(`[data-file="${filename}"]`);
  if(activeItem) activeItem.classList.add('active');

  const eventsDiv = document.getElementById('events');
  eventsDiv.innerHTML = '<div style="color:var(--text-muted);padding:40px;text-align:center;">Loading logs...</div>';

  try {
    const resp = await fetch(`/logs/api/events/${date}/${encodeURIComponent(filename)}`);
    const events = await resp.json();

    eventsDiv.innerHTML = '';
    if (!events.length) {
      eventsDiv.innerHTML = '<div style="color:var(--text-muted);padding:40px;text-align:center;">No events found in this session.</div>';
      return;
    }
    events.forEach((ev, i) => eventsDiv.appendChild(renderEvent(ev, i)));
  } catch (err) {
    eventsDiv.innerHTML = '<div style="color:var(--c-tool);padding:40px;text-align:center;">Failed to load events.</div>';
  }
}

async function loadSessions(date, groupEl) {
  const listEl = groupEl.querySelector('.session-list');
  const innerEl = listEl.querySelector('.session-list-inner');
  if (innerEl.innerHTML.trim()) { return; }  // already loaded

  try {
    const resp = await fetch(`/logs/api/sessions/${date}`);
    const sessions = await resp.json();

    innerEl.innerHTML = '';
    if(!sessions.length) {
        innerEl.innerHTML = '<div style="padding:10px 24px; color:var(--text-muted); font-size: 12px;">No sessions</div>';
        return;
    }

    sessions.forEach(s => {
      const item = document.createElement('div');
      item.className = 'session-item';
      item.dataset.file = s.filename;
      item.innerHTML = `<span class="session-time">${s.time.replace(/(\\d{2})(\\d{2})(\\d{2})/, '$1:$2:$3')}</span>
                        <span class="session-id">${s.session_id}</span>`;
      item.addEventListener('click', () => loadEvents(date, s.filename, s.session_id));
      innerEl.appendChild(item);
    });
  } catch(err) {
    innerEl.innerHTML = '<div style="padding:10px 24px; color:var(--c-tool); font-size: 12px;">Error</div>';
  }
}

async function init() {
  const datesDiv = document.getElementById('dates');
  try {
    const resp = await fetch('/logs/api/dates');
    const dates = await resp.json();
    datesDiv.innerHTML = '';

    if (!dates.length) {
        datesDiv.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;text-align:center;">No logs available</div>';
        return;
    }

    dates.forEach((date, i) => {
      const group = document.createElement('div');
      group.className = 'date-group';
      group.innerHTML = `
        <div class="date-header ${i===0?'open':''}">
          <span>${date}</span><span class="arrow">▶</span>
        </div>
        <div class="session-list ${i===0?'open':''}">
          <div class="session-list-inner"></div>
        </div>`;

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
  } catch (err) {
    datesDiv.innerHTML = '<div style="padding:16px;color:var(--c-tool);font-size:12px;text-align:center;">Failed to load dates</div>';
  }
}

init();
</script>
</body>
</html>
"""


@router.get("/logs", response_class=HTMLResponse)
def log_viewer():
    return HTMLResponse(_HTML)
