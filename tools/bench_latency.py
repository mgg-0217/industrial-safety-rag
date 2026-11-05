# -*- coding: utf-8 -*-
"""检索延迟基准（PRD 7.6 的硬约束）。

> "**延迟也要记**：检索延迟（向量路 / BM25 路 / 过滤重算 / 融合分别计时）与端到端延迟。
> 过滤后重算 BM25 的代价（第 4.4 节的实现细节）必须被实测验证，
> 而不是靠'规模小应该没事'。"

## 为什么要单独做一个基准

M5 只记了**总查询时长**（`query_seconds`），没有拆阶段。
本项目最贵的一个设计决定是 **"先过滤、再排序"**（PRD 4.4）——
过滤之后 BM25 的 IDF 与 `avgdl` 必须在**存活子集**上重算，
这意味着**每条查询都要多扫一遍 postings**。
这条设计值不值，必须用实测回答，不能靠推理。

## 计时口径

| 阶段 | 计时范围 |
|---|---|
| `filter` | `version_alive()`：算存活下标 |
| `bm25_nofilter` | BM25 **不过滤**（alive=None）：全库 IDF/avgdl |
| `bm25_filtered` | BM25 **过滤后重算**（alive=存活集）：子集 IDF/avgdl |
| `recompute_delta` | `bm25_filtered` − `bm25_nofilter` = **重算的净代价** |
| `vector` | 向量路（本机默认 L4 纯 BM25 → 没有这一路） |
| `rrf` | RRF 融合 |
| `end2end` | `Retriever.retrieve()` 整体 |

🔴 **计时用 `time.perf_counter()`，并且每条查询**先跑一次预热**再计时** ——
否则第一次查询会把解释器的懒加载（分词器、numpy 首次调用）算进去，
单查询耗时会被高估一个量级。

🔴 **查询串用评测集里的真实问题**（306 条），不是造出来的字符串 ——
造出来的串分词后词项数不代表真实分布。

跑法：
    python tools/bench_latency.py --strategy fixed
    python tools/bench_latency.py --strategy fixed --use-vector
"""
import argparse
import json
import pathlib
import statistics
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bm25 as BM          # noqa: E402
import retrieval as RT     # noqa: E402

CHUNK_DIR = ROOT / "data" / "chunks"
EVAL_DIR = ROOT / "data" / "eval"
RUN_DIR = ROOT / "eval_runs"


def _pct(xs, p):
    """分位数（样本量小时用最近秩法）。"""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 6)


def summarize(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean_ms": round(statistics.mean(xs) * 1000, 3),
        "median_ms": round(statistics.median(xs) * 1000, 3),
        "p95_ms": round(_pct(xs, 95) * 1000, 3),
        "max_ms": round(max(xs) * 1000, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fixed")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--use-vector", action="store_true",
                    help="顺带测向量路（本机默认 L4 纯 BM25 → 无向量路）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条查询")
    args = ap.parse_args()

    RUN_DIR.mkdir(exist_ok=True)
    chunks = RT.load_chunks(CHUNK_DIR / f"{args.strategy}.jsonl")

    queries = []
    qpath = EVAL_DIR / "test_set.jsonl"
    if qpath.exists():
        queries = [json.loads(l)["question"] for l in
                   open(qpath, encoding="utf-8") if l.strip()]
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        queries = ["设备检修周期", "动火作业审批流程", "受限空间作业气体检测限值"]

    embedder = None
    if args.use_vector:
        import embed as EMB
        corpus = [c.get("norm_text") or c.get("text") or "" for c in chunks]
        embedder, backend, msg = EMB.build_embedder(levels=("L1", "L2"),
                                                    corpus_texts=corpus)
        print(f"[embed] {backend}（{msg}）")
    else:
        backend, msg = "L4_pure_bm25", "无向量路（PRD 4.6 的 L4，本项目实际配置）"

    r = RT.Retriever(chunks, embedder=embedder)
    bm = r.bm25

    # 预热：把分词器 / numpy 的懒加载排除在计时之外
    for q in queries[:3]:
        bm.search(q, alive=None, top_k=args.top_k)
        r.retrieve(q, top_k=args.top_k, when="2026-11-02", mode="current",
                   use_vector=args.use_vector)

    t_filter, t_bm_no, t_bm_fi, t_vec, t_rrf, t_e2e = ([] for _ in range(6))
    t_filter_cold = []
    n_alive, n_all = [], []

    for q in queries:
        # 🔴 三个口径分清楚（2026-11-04 优化版本过滤后）：
        #   cold —— **每次都重算**（不共享缓存）：最坏情况
        #   warm —— 命中 Retriever 的跨查询缓存（同一 when/mode 时存活集固定）
        #   filter —— 本轮优化后的**按文档缓存**（单次调用内只判 26 个文档）
        t0 = time.perf_counter()
        alive_cold, _u = RT.version_alive(chunks, "2026-11-02", "current",
                                          cache={})
        t1 = time.perf_counter()
        t_filter_cold.append(t1 - t0)

        alive, _unknown = RT.version_alive(chunks, "2026-11-02", "current",
                                            cache=r._ver_cache)
        t2 = time.perf_counter()
        t_filter.append(t2 - t1)
        n_alive.append(len(alive))
        n_all.append(len(chunks))

        bm.search(q, alive=None, top_k=args.top_k)
        t3 = time.perf_counter()
        t_bm_no.append(t3 - t2)

        bm.search(q, alive=alive, top_k=args.top_k)
        t4 = time.perf_counter()
        t_bm_fi.append(t4 - t3)

        if args.use_vector and embedder is not None:
            r.vector_search(q, alive=alive, top_k=args.top_k)
            t5 = time.perf_counter()
            t_vec.append(t5 - t4)
        else:
            t5 = t4
            t_vec.append(0.0)

        # RRF 只吃排名，单独计时
        routes = [[(0, 1.0), (1, 0.9)], [(2, 1.0), (3, 0.8)]]
        BM.rrf_fuse(routes, k=r.rrf_k, top_k=args.top_k)
        t6 = time.perf_counter()
        t_rrf.append(t6 - t5)

        r.retrieve(q, top_k=args.top_k, when="2026-11-02", mode="current",
                   use_vector=args.use_vector)
        t_e2e.append(time.perf_counter() - t6)

    delta = [f - n for f, n in zip(t_bm_fi, t_bm_no)]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_strategy": args.strategy,
        "embed_backend": backend,
        "embed_note": msg,
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "alive_chunks_mean": round(sum(n_alive) / len(n_alive), 1),
        "stages": {
            "filter": summarize(t_filter),
            "filter_cold": summarize(t_filter_cold),
            "bm25_nofilter": summarize(t_bm_no),
            "bm25_filtered": summarize(t_bm_fi),
            "recompute_delta": summarize(delta),
            "vector": summarize(t_vec) if args.use_vector else None,
            "rrf": summarize(t_rrf),
            "end2end": summarize(t_e2e),
        },
        "note": (
            "recompute_delta = 过滤后重算 BM25 的净代价（存活子集上的 IDF/avgdl "
            "必须重算，PRD 4.4）。负值表示过滤反而更快（子集更小、posting 更少）。"
            "｜filter = 命中 Retriever 跨查询缓存（同一 when/mode 的存活集固定，"
            "库里的版本关系不随查询变）；"
            "filter_cold = 每次都重算（按文档缓存：26 份文档判 26 次，不是逐块判）。"
        ),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    p = RUN_DIR / f"{ts}_latency_{args.strategy}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"-> {p.relative_to(ROOT)}")
    print(json.dumps(payload, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
