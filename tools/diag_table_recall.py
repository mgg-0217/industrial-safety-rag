# -*- coding: utf-8 -*-
"""诊断：**附录表召不回**到底是不是真问题，以及多路索引能不能解决。

## 要回答的三个问题

1. **规模**：有多少题的 gold 落在**表格**里？（"该召回的是表"的题有多少）
2. **现状**：这些题在 fixed 单路下的 R@5 是多少？
3. **解法**：加一路表格索引（RRF 融合）后，R@5 与**内容完整度**各是多少？

## 🔴 为什么还要看"内容完整度"

R@5 是**二值命中**：top-5 里有一块与 gold 区间重叠就算命中。
但对表格题来说，**命中一块碎片 ≠ 能答对** ——
模型要的是那张表的**限值**，碎片里可能只有半张表。

→ 本脚本额外算一个连续指标：

    coverage = |top-5 命中块的字符区间并集 ∩ gold 区间| / |gold 区间|

对表格题，coverage 才是"上下文里到底有多少答案"。

## 口径与 M5 完全一致

复用 `tools/run_retrieval_eval.py` 的 `build_gold` / `gold_evaluable`，
所以数字可以和 M5 的批次**直接对比**。

## 🔴 多路融合的实现约束

`bm25.rrf_fuse` 用 **chunk_idx** 做 key，而各路的 idx **不在同一坐标系**
（主索引第 3 块 ≠ 表格索引第 3 块）→ 融合前必须统一成 **chunk_id**。
本脚本先按这个约束做，验证有效后再进 `src/`。

跑法：
    python tools/diag_table_recall.py
    python tools/diag_table_recall.py --main structural
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import bm25 as BM              # noqa: E402
import retrieval as RT         # noqa: E402
from run_retrieval_eval import (  # noqa: E402
    build_gold, gold_evaluable, wilson,
)

CHUNK_DIR = ROOT / "data" / "chunks"
EVAL_DIR = ROOT / "data" / "eval"
RUN_DIR = ROOT / "eval_runs"


def route_alive(chunks, when, mode):
    """某一路的存活下标。"""
    alive, _unknown = RT.version_alive(chunks, when, mode)
    return alive


def multi_route_search(routes, query, when, mode, top_k=10, alpha=0.5,
                       candidate_k=None, version_filter=True, weights=None):
    """多路 BM25 + RRF 融合。

    :param routes: ``[(name, chunks, bm25_index), ...]``
    :param weights: 各路权重（默认全 1）
    :returns: ``[(chunk_id, route_name, rrf_score), ...]``

    🔴 **各路的 chunk_idx 不在同一坐标系** —— 主索引第 3 块 ≠ 表格索引第 3 块。
    `bm25.rrf_fuse` 直接拿 idx 当 key，两路的 idx 会**互相撞车**：
    主索引的 idx=3 和表格索引的 idx=3 被当成同一个 chunk 融合，
    反查时又只认第一路 → **第二路的块被整路吞掉**。
    （本脚本第一版就踩了这个坑，表现是"表格路占 0 个名额"。）

    → 解法：融合**之前**给每路的块编**全局连续号**，
    融合之后按号反查回 (路名, 局部下标)。
    """
    ck = candidate_k or max(top_k * 4, 50)
    # 全局编号：第 ri 路的局部下标 i → 全局号 offsets[ri] + i
    offsets = []
    total = 0
    for _name, chunks, _index in routes:
        offsets.append(total)
        total += len(chunks)

    rankings = []
    for (_name, chunks, index), off in zip(routes, offsets):
        alive = None
        if version_filter and when:
            alive = route_alive(chunks, when, mode)
            if not alive:
                continue
        partial = {i for i, c in enumerate(chunks) if c.get("quality") == "partial"}
        res = index.search(query, alive=alive, top_k=ck,
                           alpha=alpha, partial=partial)
        rankings.append([(off + i, s) for i, s in res])
    if not rankings:
        return []
    fused = BM.rrf_fuse(rankings, k=BM.RRF_K, top_k=top_k * 3, weights=weights)
    out = []
    for gid, score in fused:
        for ri, (name, chunks, _index) in enumerate(routes):
            off = offsets[ri]
            if off <= gid < off + len(chunks):
                out.append((chunks[gid - off].get("chunk_id"), name, score))
                break
    return out


def interval_union(intervals):
    """区间并集的总长度。**已移到 src/retrieval.py**（单一来源）。"""
    import retrieval as _RT
    return _RT.interval_union_len(intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="fixed", help="主索引策略")
    ap.add_argument("--aux", default="table", help="辅助（表格）索引策略")
    ap.add_argument("--aux-mode", default="table",
                    choices=["table", "samedocs", "merged"],
                    help="table = 第二路放表格块（RRF 融合）；"
                         "samedocs = 对照组：第二路放**同一批文档的正文块**"
                         "（规模相近、内容是碎片）→ 判别提升来自'表格整块'"
                         "还是'双路加权'；"
                         "merged = 表格块**并进主索引**，单路检索（不做 RRF）")
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="第二路的 RRF 权重")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--char-budget", type=int, default=0,
                    help="按**字符预算**定 top-k（各策略块长差 3 倍，固定 top-k "
                         "对它们不公平）：top_k = budget / 该索引的平均块长。"
                         "例：--char-budget 2500 → fixed 取 5、structural 取 15")
    ap.add_argument("--detail", default="", help="打印某题的 top-5 明细")
    args = ap.parse_args()

    main_chunks = RT.load_chunks(CHUNK_DIR / f"{args.main}.jsonl")
    aux_chunks = RT.load_chunks(CHUNK_DIR / f"{args.aux}.jsonl")

    if args.aux_mode == "samedocs":
        # 🔴 对照组：把 aux 索引覆盖的那些文档，从**主索引**里再抽一路出来。
        #    这一路的内容是正文碎片（不是整表），规模与表格路相当。
        #    → 若它也能拿到同样的提升，说明提升来自"双路加权"，
        #      而不是"表格整块比碎片强"。
        docs = {c["doc_id"] for c in aux_chunks}
        aux_chunks = [c for c in main_chunks if c["doc_id"] in docs]

    if args.aux_mode == "merged":
        # 🔴 合并索引：表格块**直接并进**主索引，单路 BM25，不做 RRF。
        #    为什么值得测：RRF 只用排名不用分数 → 新路的权重是**门槛式**的
        #    （实测 w<1 时辅助路一个都进不了，w=1 时占掉 40% 的名额），
        #    所以"多路融合"这条路对"补充一路新内容"天然不友好。
        #    合并索引没有这个问题：表格块与正文块在同一个倒排表里竞争。
        main_chunks = main_chunks + aux_chunks
        aux_chunks = []
    # 判"gold 是不是一张表"始终用**原始的表格索引**，与 aux-mode 无关
    aux_probe_chunks = RT.load_chunks(CHUNK_DIR / f"{args.aux}.jsonl")

    main_idx = BM.BM25Index([c.get("norm_text") or c.get("text") or ""
                              for c in main_chunks])
    aux_idx = BM.BM25Index([c.get("norm_text") or c.get("text") or ""
                            for c in aux_chunks])

    questions = [json.loads(l) for l in
                 open(EVAL_DIR / "test_set.jsonl", encoding="utf-8") if l.strip()]
    cands = [json.loads(l) for l in
             open(EVAL_DIR / "candidates.jsonl", encoding="utf-8") if l.strip()]
    cand_by_qid = {c["qid"]: c for c in cands}

    main_by_doc = defaultdict(list)
    for i, c in enumerate(main_chunks):
        main_by_doc[c["doc_id"]].append(i)
    # 🔴 表格路的块**单独**记一份：判"gold 是不是一张表"时只看这一路的块。
    #    （主索引里也含表格内容 —— 定长切分会把表切碎混进正文 ——
    #      用主索引判会得到"所有题都是表格题"这种废话结论。）
    aux_by_doc = defaultdict(list)
    for i, c in enumerate(aux_probe_chunks):
        aux_by_doc[c["doc_id"]].append(i)

    routes = [(args.main, main_chunks, main_idx),
              (args.aux, aux_chunks, aux_idx)]
    # 合并视图（两路的块放在一起，便于算覆盖度）
    merged = main_chunks + aux_chunks
    merged_by_doc = defaultdict(list)
    for i, c in enumerate(merged):
        merged_by_doc[c["doc_id"]].append(i)

    # 🔴 **等字符预算**：固定 top-k 对不同块长的策略不公平 ——
    #    fixed 均长 501、structural 均长 169，top-5 时前者装 2505 字符、
    #    后者只装 845 字符，差 3 倍。→ 预算相同才谈得上"谁更好"。
    def avg_len(cs):
        lens = [len(c.get("text") or "") for c in cs]
        return (sum(lens) / len(lens)) if lens else 0.0

    if args.char_budget:
        main_k = max(1, int(round(args.char_budget / max(1.0, avg_len(main_chunks)))))
        aux_k = max(1, int(round(args.char_budget / max(1.0, avg_len(aux_chunks))))) \
            if aux_chunks else main_k
    else:
        main_k = aux_k = args.top_k
    print(f"主索引均长 {avg_len(main_chunks):.1f} 字符 → top-{main_k}"
          + (f"；辅助索引均长 {avg_len(aux_chunks):.1f} 字符 → top-{aux_k}"
             if aux_chunks else ""))

    rows = []
    for q in questions:
        if q.get("question_type") not in ("事实题", "数值题"):
            continue
        cand = cand_by_qid.get(q.get("src_qid"))
        if not cand:
            continue
        cs, ce = cand.get("clause_char_start"), cand.get("clause_char_end")
        if cs is None or ce is None:
            continue
        when = q.get("query_time")
        mode = q.get("time_mode", "current")

        # --- gold：主索引里与条款区间重叠最大的块（与 M5 同口径）---
        best, best_ov = [], 0
        for idx in main_by_doc.get(q["doc_id"], []):
            c = main_chunks[idx]
            a, b = c.get("char_start"), c.get("char_end")
            if a is None or b is None:
                continue
            ov = min(b, ce) - max(a, cs)
            if ov > best_ov:
                best_ov, best = ov, [idx]
            elif ov > 0 and ov == best_ov:
                best.append(idx)
        if not best:
            continue
        gold_ids = {main_chunks[i]["chunk_id"] for i in best}

        # --- 这题的 gold 区间在**表格索引**里有没有对应的表？---
        aux_ov = 0
        for idx in aux_by_doc.get(q["doc_id"], []):
            c = aux_probe_chunks[idx]
            a, b = c.get("char_start"), c.get("char_end")
            if a is None or b is None:
                continue
            ov = min(b, ce) - max(a, cs)
            if ov > aux_ov:
                aux_ov = ov
        is_table_q = aux_ov > 0
        gold_len = max(1, ce - cs)

        # --- A：主索引单路 ---
        alive = route_alive(main_chunks, when, mode)
        ret = RT.Retriever(main_chunks, embedder=None)
        ranked_a = ret.retrieve(q["question"], top_k=main_k, when=when,
                                mode=mode, version_filter=True,
                                use_vector=False, alpha=0.5)
        ids_a = [main_chunks[i]["chunk_id"] for i, _ in ranked_a]
        # 🔴 **命中一律按前 5 名判**（"R@5"就是 R@5），
        #    否则扫 top-k 时指标名会跟着变 → 扫出来的数没法比。
        hit_a = bool(gold_ids & set(ids_a[:5]))
        cov_a = coverage(main_chunks, ids_a, cs, ce)

        # --- B：主 + 辅助 双路 RRF ---
        ranked_b = multi_route_search(routes, q["question"], when, mode,
                                      top_k=aux_k, alpha=0.5,
                                      weights=[1.0, args.aux_weight])
        ids_b = [cid for cid, _n, _s in ranked_b][:aux_k]
        hit_b = bool(gold_ids & set(ids_b[:5]))
        cov_b = coverage(merged, ids_b, cs, ce)

        rows.append({
            "qid": q["qid"], "type": q["question_type"], "doc_id": q["doc_id"],
            "is_table_q": is_table_q, "gold_len": gold_len,
            "hit_a": hit_a, "hit_b": hit_b,
            "cov_a": round(cov_a, 3), "cov_b": round(cov_b, 3),
            # 🔴 只统计 **top-k 之内**的辅助路块数（上一版统计了 top_k*3，
            #    看起来像"表格路占 7 个名额"，其实是统计口径错了）
            "n_aux_in_top": sum(1 for _c, n, _s in ranked_b[:aux_k]
                                if n == args.aux),
        })

    def report(title, subset):
        if not subset:
            print(f"\n{title}：无题")
            return
        n = len(subset)
        h_a = sum(1 for r in subset if r["hit_a"])
        h_b = sum(1 for r in subset if r["hit_b"])
        ca = sum(r["cov_a"] for r in subset) / n
        cb = sum(r["cov_b"] for r in subset) / n
        slot = sum(r["n_aux_in_top"] for r in subset) / n
        print(f"\n{title}（n={n}）")
        print(f"  R@5  主索引单路        : {h_a}/{n} = {h_a/n:.3f} "
              f"{wilson(h_a, n)}")
        print(f"  R@5  主+辅助双路 RRF   : {h_b}/{n} = {h_b/n:.3f} "
              f"{wilson(h_b, n)}")
        print(f"  内容覆盖度 主索引单路  : {ca:.3f}")
        print(f"  内容覆盖度 主+辅助双路 : {cb:.3f}")
        print(f"  辅助路平均占用名额 : {slot:.2f} / {aux_k}")

    print(f"主索引 = {args.main}（{len(main_chunks)} 块），"
          f"辅助索引 = {args.aux}（{len(aux_chunks)} 块，模式 {args.aux_mode}，"
          f"权重 {args.aux_weight}），top-k = {args.top_k}")
    report("全部条款级可评题", rows)
    report("🎯 gold 落在表格里的题", [r for r in rows if r["is_table_q"]])
    report("gold 不含表格的题", [r for r in rows if not r["is_table_q"]])

    # 变化明细
    changed = [r for r in rows if r["hit_a"] != r["hit_b"]]
    print(f"\n命中状态发生变化的题：{len(changed)}")
    for r in changed[:15]:
        d = "变好" if (r["hit_b"] and not r["hit_a"]) else "变差"
        print(f"  [{d}] {r['qid']} {r['type']} {r['doc_id']} "
              f"cov {r['cov_a']}→{r['cov_b']} "
              f"(表格路占 {r['n_aux_in_top']} 个名额)")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "main_strategy": args.main, "aux_strategy": args.aux,
        "aux_mode": args.aux_mode, "aux_weight": args.aux_weight,
        "top_k": args.top_k,
        "char_budget": args.char_budget,
        "main_top_k": main_k, "aux_top_k": aux_k,
        "main_avg_chars": round(avg_len(main_chunks), 1),
        "aux_avg_chars": round(avg_len(aux_chunks), 1) if aux_chunks else None,
        "n_rows": len(rows),
        "n_table_questions": sum(1 for r in rows if r["is_table_q"]),
        "rows": rows,
    }
    RUN_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    p = RUN_DIR / f"{ts}_diag_table_recall.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"\n-> {p.relative_to(ROOT)}")


def coverage(chunks, chunk_ids, cs, ce):
    """转调 ``src/retrieval.py`` 的实现 —— **单一来源**。

    🔴 指标实现只能有一份：诊断脚本和常规评测脚本各写一份，
    迟早会出现"两边算出不同的数"，然后没人知道该信哪个。
    """
    return RT.content_coverage(chunks, chunk_ids, cs, ce)


if __name__ == "__main__":
    main()
