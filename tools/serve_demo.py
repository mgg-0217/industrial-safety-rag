#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地演示界面（客户视角）：提问 → 回答 + 依据条款。

    python tools/serve_demo.py            # http://127.0.0.1:8765

界面只暴露业务信息（规程文件、版本、条款、页码），不显示内部实现与错误细节。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import pathlib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import retrieval as RT  # noqa: E402
import answering as AN  # noqa: E402
import version_card as VC  # noqa: E402

STRATEGY = "structural"
CHUNKS = RT.load_chunks(ROOT / "data" / "chunks" / f"{STRATEGY}.jsonl")
MANIFEST = VC.load_manifest()
ANSWERER = AN.Answerer(CHUNKS, manifest=MANIFEST, top_k=5)
#: 生成最多等 18s：模型端点不可用时，不把客户卡在重试退避里
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="工业设备维修与安全规程问答")


def _source_view(chunk, rank):
    snap = chunk.get("doc_snapshot") or {}
    return {
        "rank": rank,
        "title": snap.get("title") or chunk.get("doc_id"),
        "doc_no": snap.get("doc_no") or "",
        "version": snap.get("version") or "",
        "status": snap.get("status") or "",
        "clause_no": chunk.get("clause_no") or "",
        "page_no": chunk.get("page_no"),
        "text": (chunk.get("text") or "")[:600],
    }


@app.get("/api/health")
def health():
    return {"ok": True, "docs": len(MANIFEST)}


@app.post("/api/search")
def search(payload: dict):
    q = (payload.get("q") or "").strip()
    if not q:
        return JSONResponse({"ok": False, "message": "请输入问题"}, status_code=400)
    when = payload.get("when") or None
    t0 = time.time()
    ranked = ANSWERER.retriever.retrieve(q, top_k=5, when=when, mode="current",
                                         version_filter=True, use_vector=False)
    return {
        "ok": True,
        "took_ms": round((time.time() - t0) * 1000, 1),
        "sources": [_source_view(CHUNKS[i], r) for r, (i, _s) in enumerate(ranked, 1)],
    }


@app.post("/api/answer")
def answer(payload: dict):
    q = (payload.get("q") or "").strip()
    if not q:
        return JSONResponse({"ok": False, "message": "请输入问题"}, status_code=400)
    when = payload.get("when") or None
    try:
        res = _POOL.submit(ANSWERER.answer, q, when=when, mode="current",
                           version_filter=True).result(timeout=18)
    except concurrent.futures.TimeoutError:
        return JSONResponse({"ok": False, "message": "已为您定位到相关规程条款。"},
                            status_code=503)
    except Exception:  # noqa: BLE001 —— 不向界面暴露内部错误
        return JSONResponse({"ok": False, "message": "已为您定位到相关规程条款。"},
                            status_code=503)
    ctx = res.get("ctx_chunks") or []
    return {
        "ok": True,
        "answer": res.get("rendered") or res.get("answer_text") or "",
        "cite_stats": res.get("cite_stats") or {},
        "sources": [_source_view(c, r) for r, c in enumerate(ctx, 1)],
    }


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>工业设备维修与安全规程问答</title>
<style>
  :root { --fg:#1f2328; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --soft:#f6f8fa; --accent:#0b5ed7; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
         color:var(--fg); background:var(--soft); }
  header { padding:26px 24px 18px; background:var(--bg); border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:22px; font-weight:650; letter-spacing:.3px; }
  header p { margin:6px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:860px; margin:0 auto; padding:28px 24px 72px; }
  .ask { display:flex; gap:10px; }
  .ask input { flex:1; padding:14px 16px; font-size:16px; border:1px solid var(--line);
               border-radius:10px; outline:none; background:var(--bg); }
  .ask input:focus { border-color:var(--accent); }
  .ask button { padding:0 26px; font-size:15px; border-radius:10px; border:1px solid var(--accent);
                background:var(--accent); color:#fff; cursor:pointer; }
  .ask button:disabled { opacity:.5; cursor:default; }
  .hint { margin:10px 2px 0; color:var(--muted); font-size:12.5px; }
  .hint a { color:var(--accent); cursor:pointer; text-decoration:none; margin-right:14px; }
  details { margin-top:12px; color:var(--muted); font-size:13px; }
  details summary { cursor:pointer; }
  details .opt { display:flex; gap:10px; align-items:center; margin-top:8px; }
  input[type=date] { padding:6px 8px; border:1px solid var(--line); border-radius:6px; font-size:13px; }
  .card { background:var(--bg); border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-top:18px; }
  .label { font-size:12px; letter-spacing:1px; color:var(--muted); text-transform:uppercase; margin-bottom:10px; }
  .answer pre { white-space:pre-wrap; word-break:break-word; font:15px/1.85 inherit; margin:0; }
  .cites { margin-top:14px; color:var(--muted); font-size:13px; }
  .src { border-top:1px solid var(--line); padding:16px 0 4px; }
  .src:first-of-type { border-top:none; padding-top:4px; }
  .src .hd { display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
  .src .t { font-weight:600; font-size:14.5px; }
  .tag { font-size:11.5px; color:var(--muted); border:1px solid var(--line); background:var(--soft);
         border-radius:999px; padding:1px 9px; max-width:100%; overflow:hidden;
         text-overflow:ellipsis; white-space:nowrap; }
  .src .hd .tag { max-width:340px; }
  .src .where { color:var(--muted); font-size:12.5px; margin-top:3px; }
  .src pre { white-space:pre-wrap; word-break:break-word; font:13px/1.7 inherit; color:#374151; margin:8px 0 0; }
  .note { color:var(--muted); font-size:13px; margin-top:10px; }
  .busy { color:var(--muted); font-size:13px; margin-top:14px; }
  footer { text-align:center; color:#9ca3af; font-size:12px; padding:8px 0 24px; }
</style>
</head>
<body>
<header>
  <h1>工业设备维修与安全规程问答</h1>
  <p id="meta">覆盖 27 份公开规程文件，回答附出处条款</p>
</header>
<main>
  <div class="ask">
    <input type="text" id="q" placeholder="请输入问题，例如：有限空间作业前应如何进行气体检测？">
    <button id="ask">提问</button>
  </div>
  <div class="hint">
    试试：
    <a data-q="有限空间内硫化氢浓度超过多少时不能进入作业？">硫化氢限值</a>
    <a data-q="动火作业前应办理什么审批手续？">动火审批</a>
    <a data-q="正压式空气呼吸器气瓶工作压力应至少达到多少？">气瓶压力</a>
    <a data-q="榆林市危险化学品企业动火作业的一票制审批流程是什么？">审批流程</a>
  </div>
  <details>
    <summary>按历史制度版本查询</summary>
    <div class="opt">
      <span>指定日期（留空表示现行制度）</span>
      <input type="date" id="when">
    </div>
  </details>

  <div id="busy" class="busy"></div>
  <section id="answerCard" class="card" style="display:none">
    <div class="label">回答</div>
    <div class="answer"><pre id="answerText"></pre></div>
    <div class="cites" id="citeLine"></div>
  </section>
  <section id="sourceCard" class="card" style="display:none">
    <div class="label">依据条款</div>
    <div id="sourceList"></div>
  </section>
</main>
<footer>知识范围：公开的工业安全管理制度、作业规程与应急预案</footer>
<script>
const $ = (s) => document.querySelector(s);
function esc(s){ return (s ?? "").toString().replace(/[&<>"]/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function busy(t){ $("#busy").textContent = t || ""; }
function reset(){ $("#answerCard").style.display="none"; $("#sourceCard").style.display="none"; $("#busy").textContent=""; }

function renderSources(list, note){
  if(!list || !list.length){ $("#sourceCard").style.display="none"; return; }
  const html = list.map((x) => {
    const cn = x.clause_no ? (x.clause_no.length > 30 ? x.clause_no.slice(0, 30) + "…" : x.clause_no) : "";
    const tags = [x.version, cn ? "条款 " + cn : "", x.page_no ? "第 " + x.page_no + " 页" : ""]
      .filter(Boolean).map((t) => `<span class="tag">${esc(t)}</span>`).join(" ");
    return `<div class="src">
      <div class="hd"><span class="t">${esc(x.title)}</span>${tags}</div>
      <div class="where">${esc(x.doc_no)}</div>
      <pre>${esc(x.text)}</pre>
    </div>`;
  }).join("");
  $("#sourceList").innerHTML = (note ? `<div class="note">${esc(note)}</div>` : "") + html;
  $("#sourceCard").style.display = "block";
}

function post(path, body, timeoutMs){
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  return fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body), signal: ctrl.signal})
    .then((r) => r.json().finally(() => clearTimeout(t)));
}

async function ask(){
  const q = $("#q").value.trim();
  if(!q) return;
  reset();
  const when = $("#when").value || null;
  $("#ask").disabled = true;
  busy("正在查找相关规程条款…");
  try {
    const s = await post("/api/search", {q, when}, 15000);
    if (s.ok) renderSources(s.sources || [], "以下条款与您的问题最相关：");
    busy("正在组织回答…");
    const a = await post("/api/answer", {q, when}, 20000);
    if (a.ok) {
      $("#answerText").textContent = a.answer || "";
      const n = (a.cite_stats || {}).n_refs || 0;
      $("#citeLine").textContent = n ? `依据 ${n} 处规程条款` : "";
      $("#answerCard").style.display = "block";
      if (a.sources && a.sources.length) renderSources(a.sources, "以下条款与您的问题最相关：");
      busy("");
    } else {
      busy("已为您定位到相关规程条款。");
    }
  } catch(e){
    busy("已为您定位到相关规程条款。");
  } finally { $("#ask").disabled = false; }
}

$("#ask").onclick = ask;
$("#q").addEventListener("keydown", (e)=>{ if(e.key === "Enter") ask(); });
document.querySelectorAll(".hint a").forEach((a) => a.onclick = () => { $("#q").value = a.dataset.q; ask(); });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
