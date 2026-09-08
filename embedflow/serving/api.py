import html
import math
from typing import Any


def dashboard_html(title: str = "EmbedFlow") -> str:
    # Keep this as an ordinary template instead of a Python f-string.  The
    # dashboard contains JavaScript template literals, and escaping every
    # JavaScript brace in a f-string made browser failures very easy to hide.
    escaped_title = html.escape(title, quote=True)
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
body{font:15px system-ui,sans-serif;background:#0b1020;color:#e8eefc;max-width:1100px;margin:0 auto;padding:28px}
.card{background:#141d35;border:1px solid #2d3b62;border-radius:14px;padding:18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.label{color:#9fb0d8;font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.value{font-size:25px;font-weight:650;margin-top:5px}
input{width:min(75%,650px);padding:12px;border-radius:8px;border:1px solid #45567f;background:#0b1020;color:white}
button{padding:12px 18px;border:0;border-radius:8px;background:#6ea8fe;color:#071021;font-weight:700;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
table{width:100%;border-collapse:collapse;margin-top:12px}td,th{padding:8px;border-bottom:1px solid #2d3b62;text-align:left}code{color:#a8d5ff}
.error{color:#ffb4b4;background:#3a1820;border-color:#7c3343}.muted{color:#9fb0d8}
</style></head>
<body><h1>EMBEDFLOW</h1><p>Progressive Embedding Migration</p><div id="status" class="card">Connecting to EmbedFlow…</div>
<div class="card"><h2>Search</h2><form id="form"><input id="query" placeholder="Ask a question…" autocomplete="off" required><button id="submit" type="submit">Search</button></form><div id="search" class="muted">Enter a query to test source retrieval, target reranking, and progressive caching.</div></div>
<script>
const fmt=(x)=>x===undefined||x===null?'—':(typeof x==='number'?x.toFixed(2):x);
const esc=(x)=>String(x??'').replace(/[&<>"']/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function jsonFetch(url, options){
  const response=await fetch(url, options);
  const text=await response.text();
  let data={};
  try{data=text?JSON.parse(text):{};}catch(_){data={detail:text||response.statusText};}
  if(!response.ok){
    const detail=data&&data.detail;
    const message=Array.isArray(detail)?detail.map((item)=>typeof item==='string'?item:(item.msg||JSON.stringify(item))).join('; '):(detail&&typeof detail==='object'?JSON.stringify(detail):(detail||('HTTP '+response.status)));
    throw new Error(message);
  }
  return data;
}
async function refresh(){
  try{
    const s=await jsonFetch('/status');
    const m=s.migration||{},c=s.cache||{},w=s.worker||{},l=s.latency||{};
    document.querySelector('#status').className='card';
    document.querySelector('#status').innerHTML=`<div class="grid"><div><div class="label">Source</div><div class="value"><code>${esc(m.source_model||'configured')}</code></div></div><div><div class="label">Target</div><div class="value"><code>${esc(m.target_model||'configured')}</code></div></div><div><div class="label">Diagnostic</div><div class="value">${esc(m.diagnostic)}</div></div><div><div class="label">Candidate depth</div><div class="value">K=${fmt(m.candidate_depth)}</div></div><div><div class="label">Cache progress</div><div class="value">${fmt(c.cached_target_vectors)} / ${fmt(m.corpus_size)} (${fmt(100*(m.cache_fraction||0))}%)</div></div><div><div class="label">Background throughput</div><div class="value">${fmt(w.last_throughput_docs_sec)} docs/s</div></div><div><div class="label">Query p50/p95</div><div class="value">${fmt(l.p50_ms)} / ${fmt(l.p95_ms)} ms</div></div></div><p>Status: <b>${esc(m.status)}</b> · ANN: <b>${esc(m.ann_status)}</b></p>`;
  }catch(error){
    document.querySelector('#status').className='card error';
    document.querySelector('#status').textContent='Unable to load migration status: '+error.message;
  }
}
refresh();
setInterval(refresh,3000);
document.querySelector('#form').addEventListener('submit',async(event)=>{
  event.preventDefault();
  const input=document.querySelector('#query'),button=document.querySelector('#submit'),out=document.querySelector('#search');
  const query=input.value.trim(); if(!query)return;
  button.disabled=true; button.textContent='Searching…'; out.className='muted'; out.textContent='Running source retrieval and target reranking…';
  try{
    const x=await jsonFetch('/search',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({query:query,top_k:10})});
    const migration=x.migration||{}, timing=x.timing_ms||{};
    out.className='';
    out.innerHTML=`<p><b>${esc(migration.status)}</b> · ${fmt(timing.total_ms)} ms · cache ${fmt(migration.target_cache_hits)}/${fmt(migration.source_candidates)} · queued ${fmt(migration.async_misses_queued)}</p><table><tr><th>Target rank</th><th>Source rank</th><th>Cached</th><th>Document</th></tr>${(x.results||[]).map((a)=>`<tr><td>${fmt(a.target_rank)}</td><td>${fmt(a.source_rank)}</td><td>${a.target_vector_cached?'yes':'no'}</td><td>${esc(a.text)}</td></tr>`).join('')}</table><p class="muted">Total ${fmt(timing.total_ms)} ms · source encode ${fmt(timing.source_query_encode_ms)} ms · ANN ${fmt(timing.source_ann_ms)} ms · target encode ${fmt(timing.target_query_encode_ms)} ms · scoring ${fmt(timing.target_score_ms)} ms</p>`;
  }catch(error){
    out.className='card error'; out.textContent='Search failed: '+error.message+' (check the server terminal for details)';
  }finally{button.disabled=false;button.textContent='Search';}
});
</script></body></html>'''.replace('__TITLE__', escaped_title)


def create_app(engine: Any):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse

        from .schemas import PrewarmRequest, SearchRequest, SearchResponse
    except ImportError as exc:
        raise RuntimeError("API requires fastapi, uvicorn, and pydantic") from exc

    app = FastAPI(title="EmbedFlow", version="0.1.1")

    @app.get("/", response_class=HTMLResponse)
    def root(): return dashboard_html(engine.cfg.dashboard_title)

    @app.get("/health")
    def health(): return {"status": "ok"}

    @app.get("/status")
    def status(): return engine.status()

    @app.get("/plan")
    def plan(): return engine.plan.to_dict()

    @app.post("/analyze")
    def analyze():
        """Return the persisted finite-tail decision and deployment plan."""
        return {
            "diagnostic": engine.plan.diagnostic,
            "recommended_initial_k": engine.plan.candidate_depth,
            "ann_status": engine.plan.ann_status,
            "plan": engine.plan.to_dict(),
            "probe": engine.probe,
            "warning": "T2-v1 is an empirical finite-tail diagnostic, not a compatibility guarantee.",
        }

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest):
        try:
            result = engine.search(body.query, body.top_k, body.candidate_depth, body.max_sync_misses)
            migration = result.get("migration", {})
            # Keep the detailed nested object and expose the compact fields
            # shown in the public API example at the top level as well.
            result.update({
                "state": migration.get("status"),
                "candidate_depth": migration.get("candidate_depth"),
                "cache_hits": migration.get("cache_hits", migration.get("target_cache_hits")),
                "cache_misses": migration.get("cache_misses", migration.get("target_cache_misses")),
                "sync_encoded": migration.get("sync_encoded", migration.get("sync_misses_encoded")),
                "async_queued": migration.get("async_queued", migration.get("async_misses_queued")),
            })
            return result
        except (ValueError, KeyError, RuntimeError, TypeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # keep backend/model failures as structured API errors
            raise HTTPException(status_code=500, detail=f"search backend failure: {exc}") from exc

    @app.post("/prewarm")
    def prewarm(body: PrewarmRequest):
        try:
            return engine.prewarm(body.document_ids, body.asynchronous)
        except (ValueError, KeyError, RuntimeError, TypeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prewarm backend failure: {exc}") from exc

    @app.get("/metrics")
    def metrics():
        from ..metrics import aggregate_records, summarize

        values: list[float] = []
        for row in engine.records:
            try:
                value = float(row.get("total_ms"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                values.append(value)
        return {
            "latency": summarize(values) if values else {"count": 0},
            "stages": aggregate_records(engine.records),
            "cache": engine.cache.stats(),
            "queue": engine.worker.stats(),
        }

    @app.get("/economics")
    def economics():
        from ..cli import economics_for
        return economics_for(engine.documents.size(), engine.cfg.economics.target_docs_per_second, engine.cfg.economics.gpu_price_per_hour,
                             engine.cache.stats()["cached_target_vectors"])

    return app
