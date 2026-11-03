# -*- coding: utf-8 -*-
"""检索层评测 —— PRD M5（基线 B2 / B3 / B2′）。

## 产出

    eval_runs/<YYYYMMDD_HHMM>_chunk_<strategy>.json     每个切分策略一份（M4 的产出物）
    eval_runs/<YYYYMMDD_HHMM>_baseline_<name>.json        每组基线一份

**都不覆盖**（PRD 3.3 / 7.5 的约定）：重跑会生成新的时间戳目录。

## 🔴 gold chunk 的定义（本脚本最要紧的部分）

Recall@k 的分母是"有效题数"，而"有效题"取决于 **gold 怎么定**。
本脚本按题型分三档，**每档的粒度都写进报告**：

| 题型 | gold 粒度 | 定义 | n |
|---|---|---|---|
| 事实题 / 数值题 | **条款级** | 候选池记录的条款**原始 md 偏移**与 chunk 的 `[char_start, char_end)` 求重叠，取重叠最大者 | 198 |
| 版本陷阱题 | **文档级** | 期望文档 `expect_doc_id` 的任一块 —— 版本陷阱问的正是"该用哪一版"，文档级正是它的正确粒度 | 41 |
| 跨文档推理题 | **版本链级** | 该文档**整条版本链**的块（从 manifest 的版本链分组推导，**不硬编码**）；题目问"两版有何差异"，两版都要召回才算命中 | 25 |
| 拒答题 | 无 gold | **不参与召回指标**（生成层另有"拒答误答率"，PRD 7.3） | 42 |

⚠️ **诚实说明**：文档级/版本链级 gold 比条款级"松"，
所以它们的召回率**天然偏高**，不能与条款级直接比大小。
报告里**分档给出**，不合并成一个数。

## 指标（PRD 7.3，全部写明 n / 分子 / 分母）

- `recall_at_5` / `recall_at_10`：top-k 含 gold 的题数 / 有效题数
- `mrr_at_10`：gold 首次出现排名的倒数均值（无命中记 0）
- `stale_version_false_recall_rate`：结果里出现"在 query_time 不生效"的 chunk 的题数占比。
  🔴 **B3 下恒为 0**（过滤在检索层），它的对照价值在 **B2 vs B3**（PRD 3.3 第 3 条）。
- 95% 置信区间用 **Wilson 区间**（小样本比正态近似稳）。
"""
import argparse
import csv
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bm25 as BM          # noqa: E402
import embed as EMB        # noqa: E402
import retrieval as RT     # noqa: E402
import versioning as V     # noqa: E402

CHUNK_DIR = ROOT / "data" / "chunks"
EVAL_DIR = ROOT / "data" / "eval"
RUN_DIR = ROOT / "eval_runs"

STRATEGIES = ["fixed", "structural", "clause", "table", "semantic", "qa"]
#: 策略 6（QA-pair）的 gold 口径特殊：PRD 3.1 策略 6 第 3 条已写死
#: 「策略 6 的 Recall@k 按定义恒为 0 —— 它的 gold 与不能出现在索引里的东西是同一个东西」。
#: 所以策略 6 **不进 M4 的策略排序**，单独报告（等价答案命中率另算）。
SORTABLE_STRATEGIES = ["fixed", "structural", "clause", "table", "semantic"]


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def wilson(successes, n, z=1.96):
    """Wilson 置信区间（小样本比正态近似稳）。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (c - m) / d
    hi = (c + m) / d
    return (round(p, 4), round(max(0.0, lo), 4), round(min(1.0, hi), 4))


# ---------------------------------------------------------------------------
# gold 定义
# ---------------------------------------------------------------------------

class Gold:
    """一道题的 gold。"""

    def __init__(self, kind, chunk_ids, required_docs=None):
        self.kind = kind                    # clause | doc | chain | none
        self.chunk_ids = set(chunk_ids)
        self.required_docs = set(required_docs or [])

    def hit_and_rank(self, ranked_idx):
        """在有序的 chunk 下标列表里，返回 ``(是否命中, 命中时的排名(1-based)``。

        - clause：命中的 chunk 在列表里即算命中
        - doc：期望文档的任一块在列表里即算命中
        - chain：**每个**期望文档都要至少有一块在列表里才算命中，
          排名取"最后一个文档首次被覆盖"的位置
        """
        if self.kind == "none" or not ranked_idx:
            return False, None
        if self.kind in ("clause", "doc"):
            for rank, idx in enumerate(ranked_idx, start=1):
                if idx in self.chunk_ids:
                    return True, rank
            return False, None
        # chain
        seen = set()
        for rank, idx in enumerate(ranked_idx, start=1):
            doc = self.doc_of.get(idx)
            if doc in self.required_docs:
                seen.add(doc)
                if len(seen) == len(self.required_docs):
                    return True, rank
        return False, None


def build_gold(q, cand_by_qid, chunks, chunks_by_doc, chain_by_doc):
    """给一道题构造 gold。"""
    qt = q.get("question_type")
    if qt in ("事实题", "数值题"):
        cand = cand_by_qid.get(q.get("src_qid"))
        if not cand:
            return Gold("none", [])
        cs, ce = cand.get("clause_char_start"), cand.get("clause_char_end")
        if cs is None or ce is None:
            return Gold("none", [])
        best = []
        best_ov = 0
        for idx in chunks_by_doc.get(q["doc_id"], []):
            c = chunks[idx]
            a = c.get("char_start")
            b = c.get("char_end")
            if a is None or b is None:
                continue
            ov = min(b, ce) - max(a, cs)
            if ov > best_ov:
                best_ov = ov
                best = [idx]
            elif ov > 0 and ov == best_ov:
                best.append(idx)
        if not best:
            return Gold("none", [])
        g = Gold("clause", best)
        g.doc_of = {i: chunks[i]["doc_id"] for i in best}
        return g

    expect = (q.get("expect_doc_id") or "").strip()
    if not expect:
        return Gold("none", [])
    if qt == "跨文档推理题":
        docs = chain_by_doc.get(expect) or [expect]
        ids = []
        for d in docs:
            ids.extend(chunks_by_doc.get(d, []))
        g = Gold("chain", ids, required_docs=set(docs))
        g.doc_of = {i: chunks[i]["doc_id"] for i in ids}
        return g
    # 版本陷阱题：文档级
    ids = chunks_by_doc.get(expect, [])
    g = Gold("doc", ids, required_docs={expect})
    g.doc_of = {i: chunks[i]["doc_id"] for i in ids}
    return g


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def gold_evaluable(g, alive_set, chunks):
    """这道题在当前配置下**可评**吗？返回 ``(可评, 原因)``。

    原因有三种，落盘时要分开计数（混在一起就看不出问题出在哪）：

    ============== ==========================================================
    ``none``      这题**没有 gold**（拒答题）→ 不参与召回指标
    ``absent``    gold 集合**为空** —— 该切分策略根本没索引这份文档
                  （如策略 4 只覆盖 5/26 份文档）→ **不可评**
    ``filtered``  gold 在候选集里，但被**版本过滤**掉了
                  （如 as_of 口径下旧版被滤掉）→ **不可评**
    ============== ==========================================================

    🔴 **IR 惯例**：gold 不在候选集里就**不可评** ——
    算成"漏召"是冤枉检索器。

    🔴🔴 **2026-11-03 修的 bug**：`absent` 这一支原来只在
    `alive_set is None`（B2 不过滤）时被跳过 ——
    结果**过滤模式下被排除、不过滤模式下被算成漏召**，
    同一道题在两组基线里的待遇不一样，B2 被人为压低。
    实测后果：策略 4（表格）B2 的 R@5 = 0.077，
    其实是因为 66 道题的文档根本没被策略 4 索引。
    """
    if g.kind == "none":
        return False, "none"
    if not g.chunk_ids:
        # 该策略没有这个文档的任何块 → 答案不在索引里 → 不可评
        return False, "absent"
    if alive_set is None:
        return True, ""
    if g.kind in ("clause", "doc"):
        if any(i in alive_set for i in g.chunk_ids):
            return True, ""
        return False, "filtered"
    # chain：每个必需文档都要有存活的块
    for d in g.required_docs:
        if not any(i in alive_set and chunks[i]["doc_id"] == d
                   for i in g.chunk_ids):
            return False, "filtered"
    return True, ""


def chunk_size_stats(chunks):
    """PRD 3.3 要求的**规模代价**字段。

    ⚠️ 为什么要单列："一个召回高 2% 但 chunk 数翻 5 倍的策略不一定是更好的工程选择"
    （PRD 3.3 第 4 条）。所以规模不是附注，是**并列的选型判据**。
    """
    lens = sorted(len((c.get("text") or "").strip()) for c in chunks)
    n = len(lens)
    if n == 0:
        return {"n_chunks": 0, "avg_chunk_chars": 0, "p95_chunk_chars": 0,
                "table_chunks": 0, "pseudo_table_chunks": 0,
                "cross_page_chunks": 0, "empty_chunks": 0}
    p95 = lens[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n_chunks": n,
        "avg_chunk_chars": round(sum(lens) / n, 1),
        "p95_chunk_chars": p95,
        "table_chunks": sum(1 for c in chunks if c.get("table") is not None),
        "pseudo_table_chunks": sum(
            1 for c in chunks
            if (c.get("strategy_meta") or {}).get("table_is_pseudo")),
        "cross_page_chunks": sum(
            1 for c in chunks
            if (c.get("strategy_meta") or {}).get("cross_page")),
        "empty_chunks": sum(1 for c in chunks
                            if not (c.get("text") or "").strip()),
    }


def eval_strategy(strategy, questions, cand_by_qid, chain_by_doc, embedder,
                  baselines, alpha_grid=(0.2, 0.5, 0.8, 1.0), top_k=10):
    """跑一个切分策略的全部基线，返回 ``{baseline_name: 指标}``。"""
    chunks = RT.load_chunks(CHUNK_DIR / f"{strategy}.jsonl")
    chunks_by_doc = defaultdict(list)
    for i, c in enumerate(chunks):
        chunks_by_doc[c["doc_id"]].append(i)

    size_stats = chunk_size_stats(chunks)
    # 🔴 策略 6 的评测污染断言（R7）：不通过就抛异常，不报数
    qa_guard = assert_qa_no_leakage(chunks) if strategy == "qa" else None
    ret = RT.Retriever(chunks, embedder=embedder)
    out = {}
    per_q_all = {}

    for bname, bcfg in baselines.items():
        use_filter = bcfg.get("version_filter", True)
        use_vector = bcfg.get("use_vector", True)
        use_bm25 = bcfg.get("bm25", True)
        alpha = bcfg.get("alpha", 0.5)
        t0 = time.time()
        hits5 = hits10 = 0
        rr_sum = 0.0
        stale = 0
        n_eff = 0
        n_filtered = 0
        n_absent = 0
        per_type = defaultdict(lambda: Counter())
        per_q = {}
        for q in questions:
            g = build_gold(q, cand_by_qid, chunks, chunks_by_doc, chain_by_doc)
            if g.kind == "none":
                continue
            when = q.get("query_time")
            mode = q.get("time_mode", "current")
            alive = None
            if use_filter:
                alive, _u = RT.version_alive(chunks, when, mode)
                alive_set = set(alive)
            else:
                alive_set = None
            ok, why = gold_evaluable(g, alive_set, chunks)
            if not ok:
                if why == "filtered":
                    n_filtered += 1
                    per_type[q["question_type"]]["gold_filtered"] += 1
                elif why == "absent":
                    n_absent += 1
                    per_type[q["question_type"]]["gold_absent"] += 1
                continue
            n_eff += 1
            ranked = ret.retrieve(
                q["question"], top_k=top_k, when=when, mode=mode,
                version_filter=use_filter, use_vector=use_vector, alpha=alpha,
                use_bm25=use_bm25)
            idxs = [i for i, _s in ranked]
            hit, rank = g.hit_and_rank(idxs)
            per_q[q["qid"]] = (bool(hit and rank and rank <= 5), bool(hit),
                               rank, q["question_type"])
            if hit:
                hits10 += 1
                per_type[q["question_type"]]["hits10"] += 1
                if rank <= 5:
                    hits5 += 1
                    per_type[q["question_type"]]["hits5"] += 1
                rr_sum += 1.0 / rank
            else:
                per_type[q["question_type"]]["miss"] += 1
            per_type[q["question_type"]]["n"] += 1
            # 过期版本误召回
            # 🔴 cross_version 口径**不查**：那个口径本来就要历史版本在场
            if mode != "cross_version":
                for i in idxs:
                    doc = chunks[i].get("doc_snapshot") or {}
                    if V.is_effective(doc, when) is False:
                        stale += 1
                        break
        dt = time.time() - t0
        m = {
            "strategy": strategy,
            "baseline": bname,
            "n_questions": len(questions),
            "n_effective": n_eff,
            "n_gold_filtered": n_filtered,
            "n_gold_absent": n_absent,
            "recall_at_5": wilson(hits5, n_eff),
            "recall_at_10": wilson(hits10, n_eff),
            "mrr_at_10": round(rr_sum / n_eff, 4) if n_eff else 0.0,
            "numeric_recall_at_5": wilson(per_type["数值题"]["hits5"],
                                          per_type["数值题"]["n"]),
            "stale_version_false_recall_rate": wilson(stale, n_eff),
            "embed_backend": ret.stats()["embed_backend"],
            "alpha": alpha,
            "version_filter": use_filter,
            "use_vector": use_vector,
            "query_seconds": round(dt, 2),
            # PRD 3.3 的规模代价字段（并列的选型判据，不是附注）
            **size_stats,
            "qa_leakage_guard": qa_guard,
            "in_ranking": strategy in SORTABLE_STRATEGIES,
            "note": (
                "策略 6 的 chunk 没有 char_start/char_end（QA 对不是原文切片），"
                "条款级 gold 无法定位 → 事实题/数值题不参与本组；"
                "按 PRD 3.1 策略 6 第 3 条，本组数字**不与其他策略横比**，单独报告"
                if strategy == "qa" else ""),
            "per_type": {
                k: {
                    "n": v["n"],
                    "miss": v["miss"],
                    "gold_filtered": v["gold_filtered"],
                    "gold_absent": v["gold_absent"],
                    "recall_at_5": wilson(v["hits5"], v["n"]),
                    "recall_at_10": wilson(v["hits10"], v["n"]),
                }
                for k, v in sorted(per_type.items())
            },
        }
        out[bname] = m
        per_q_all[bname] = per_q
    return out, per_q_all


def assert_qa_no_leakage(chunks):
    """PRD 3.1 策略 6 第 3 条 / 风险 R7 的**运行时断言**。

    🔴 策略 6（QA-pair）把"问-答对"直接放进索引 ——
    如果索引侧 QA 里混进了评测侧的 QA，评测就变成"拿答案去查答案"，
    指标会虚高到 100%，整个评测作废。

    → 断言失败即**判定该次运行无效，不报数**（R7 的处置条款）。
    """
    eval_ids = set()
    with open(EVAL_DIR / "qa_eval_side.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            eval_ids.add(row["qa_id"])
    idx_ids = {(c.get("strategy_meta") or {}).get("qa_id") for c in chunks}
    idx_ids.discard(None)
    overlap = idx_ids & eval_ids
    if overlap:
        raise AssertionError(
            f"策略 6 评测污染：索引侧 QA 与评测侧 QA 有 {len(overlap)} 条交集 "
            f"（例：{sorted(overlap)[:3]}）—— 本次运行无效，不报数")
    return {"n_index_qa": len(idx_ids), "n_eval_qa": len(eval_ids),
            "overlap": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--baselines", default="B2,B3,B2prime")
    ap.add_argument("--levels", default="L1,L2",
                    help="向量后端的尝试顺序（L4 = 纯 BM25，不进这个列表）")
    ap.add_argument("--primary-baseline", default="B2prime_filter_a0.5",
                    help="算共同子集用的主基线（要存在且可评集合稳定）")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    RUN_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    questions = [json.loads(l) for l in
                 open(EVAL_DIR / "test_set.jsonl", encoding="utf-8") if l.strip()]
    cands = [json.loads(l) for l in
             open(EVAL_DIR / "candidates.jsonl", encoding="utf-8") if l.strip()]
    cand_by_qid = {c["qid"]: c for c in cands}
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    docs = manifest if isinstance(manifest, list) else manifest.get("documents", manifest)
    chains = V.group_version_chains(docs)
    chain_by_doc = {}
    for _key, members in chains.items():
        if len(members) > 1:
            for d in members:
                chain_by_doc[d["doc_id"]] = [x["doc_id"] for x in members]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    base_names = [b.strip() for b in args.baselines.split(",") if b.strip()]

    # 向量后端：按可用性选一个
    corpus = []
    for line in open(CHUNK_DIR / "structural.jsonl", encoding="utf-8"):
        c = json.loads(line)
        corpus.append(c.get("norm_text") or c.get("text") or "")
    embedder, backend, msg = EMB.build_embedder(
        levels=tuple(args.levels.split(",")), corpus_texts=corpus)

    baselines = {}
    for b in base_names:
        if b == "B2":
            baselines["B2_bm25_nofilter"] = {"version_filter": False,
                                             "use_vector": False, "alpha": 0.5}
        elif b == "B3":
            baselines["B3_hybrid_filter"] = {"version_filter": True,
                                            "use_vector": embedder is not None,
                                            "alpha": 0.5}
        elif b == "B2prime":
            for a in (0.2, 0.5, 0.8, 1.0):
                baselines[f"B2prime_filter_a{a}"] = {"version_filter": True,
                                                     "use_vector": False,
                                                     "alpha": a}
        elif b == "V":
            # 仅向量路（+版本过滤）：量化"向量路自己"能召回多少。
            # B3 的混合分如果低于 B2′（纯 BM25），必须能说清是**哪一路**拖的。
            baselines["V_vector_filter"] = {"version_filter": True,
                                            "use_vector": True, "alpha": 0.5,
                                            "bm25": False}

    print(f"\n题集 {len(questions)} 条 | 向量后端 {backend}（{msg}）")
    print(f"策略 {strategies}\n基线 {list(baselines)}\n")

    all_out = {}
    per_q_all = {}
    for st in strategies:
        res, pq = eval_strategy(st, questions, cand_by_qid, chain_by_doc,
                                embedder, baselines, top_k=args.top_k)
        all_out[st] = res
        per_q_all[st] = pq
        for bname, m in res.items():
            mark = "" if m["in_ranking"] else "  [不进排序]"
            print(f"[{st:11s}|{bname:22s}] R@5 {m['recall_at_5'][0]:.3f} "
                  f"[{m['recall_at_5'][1]:.3f},{m['recall_at_5'][2]:.3f}]  "
                  f"R@10 {m['recall_at_10'][0]:.3f}  MRR@10 {m['mrr_at_10']:.3f}  "
                  f"数值R@5 {m['numeric_recall_at_5'][0]:.3f}  "
                  f"stale {m['stale_version_false_recall_rate'][0]:.3f}  "
                  f"n={m['n_effective']}(abs {m['n_gold_absent']}/filt "
                  f"{m['n_gold_filtered']})  chunks={m['n_chunks']}  "
                  f"{m['query_seconds']}s{mark}")
        print()

    # 落盘（PRD 3.3：每策略一份；PRD 7.5：带时间戳、不覆盖）
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embed_backend": backend,
        "embed_note": msg,
        "n_questions": len(questions),
        "strategies": strategies,
        "baselines": list(baselines),
        "ranking_eligible": {st: (st in SORTABLE_STRATEGIES) for st in strategies},
        "ranking_rule": (
            "PRD 3.3：主判据 Recall@5，子判据 numeric_recall_at_5；"
            "stale_version_false_recall_rate 只记录不参与排序；"
            "策略 6（QA-pair）结果单独标注、不进排序"),
        "gold_definitions": {
            "事实题/数值题": "条款级：条款原始 md 偏移与 chunk 区间重叠最大者",
            "版本陷阱题": "文档级：expect_doc_id 的任一块",
            "跨文档推理题": "版本链级：整条版本链的块都要覆盖（由 manifest 版本链推导）",
            "拒答题": "无 gold，不参与召回指标",
        },
    }
    for st, res in all_out.items():
        payload = dict(meta)
        payload["strategy"] = st
        payload["results"] = res
        p = RUN_DIR / f"{ts}_chunk_{st}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"-> {p.relative_to(ROOT)}")
    for bname in baselines:
        payload = dict(meta)
        payload["baseline"] = bname
        payload["results"] = {st: all_out[st][bname] for st in all_out
                              if bname in all_out[st]}
        p = RUN_DIR / f"{ts}_baseline_{bname}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"-> {p.relative_to(ROOT)}")

    # ---- 共同子集对比（不同策略的可答题集合不同时，唯一的公平口径）--------
    # 🔴 策略 4（表格）只索引 5/26 份文档、策略 6（QA）只索引 10 份 ——
    # 它们的分母与其它策略**不是同一批题**，R@5 不能直接横比。
    # 实测：策略 6 的可评题（版本陷阱题）与策略 4 的可评题（事实/数值题）
    # **完全不相交** → 六策略的交集是空集，共同子集这条路走不通。
    # → 改为：以"可评题最多的那一组策略"（覆盖全库的那几个）取交集，
    #    专用策略单列，标注"分母不同，不可横比"。
    primary = args.primary_baseline
    if primary not in baselines:
        primary = list(baselines)[0]
    sets = {st: set(per_q_all[st].get(primary, {}).keys()) for st in strategies}
    sizes = {st: len(s) for st, s in sets.items()}
    max_n = max(sizes.values()) if sizes else 0
    full = [st for st in strategies if sizes[st] == max_n]
    common = set.intersection(*[sets[st] for st in full]) if full else set()
    common_rows = {}
    for st in strategies:
        pq = per_q_all[st].get(primary, {})
        sub = {q: pq[q] for q in common if q in pq}
        h5 = sum(1 for v in sub.values() if v[0])
        h10 = sum(1 for v in sub.values() if v[1])
        rr = sum(1.0 / v[2] for v in sub.values() if v[2])
        n_here = len(sub)
        common_rows[st] = {
            "n_own": sizes[st],
            "n_common": len(common),
            "n_comparable": n_here,
            "comparable": st in full,
            "recall_at_5": wilson(h5, n_here),
            "recall_at_10": wilson(h10, n_here),
            "mrr_at_10": round(rr / n_here, 4) if n_here else 0.0,
        }
    payload = dict(meta)
    payload["primary_baseline"] = primary
    payload["note"] = (
        "共同子集 = 覆盖全库的策略（可评题最多的那一组）的交集。"
        "策略 4 只索引 5/26 份文档、策略 6 只索引 10 份，"
        "它们的可评题与其它策略**不相交**，分母不同 → 标 comparable=false，"
        "R@5 不与其它策略横比。")
    payload["per_strategy_own_set"] = sizes
    payload["full_coverage_strategies"] = full
    payload["common_subset"] = common_rows
    payload["common_qids"] = sorted(common)
    p = RUN_DIR / f"{ts}_common_subset.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"-> {p.relative_to(ROOT)}")
    print(f"\n共同子集（{primary}，覆盖全库的策略 {full}）n={len(common)}：")
    for st in strategies:
        r = common_rows[st]
        flag = "" if r["comparable"] else "  [分母不同，不可横比]"
        print(f"  {st:11s} 自己可评 {r['n_own']:4d} 题 | "
              f"共同子集 R@5 {r['recall_at_5'][0]:.3f} "
              f"[{r['recall_at_5'][1]:.3f},{r['recall_at_5'][2]:.3f}]{flag}")


if __name__ == "__main__":
    main()
