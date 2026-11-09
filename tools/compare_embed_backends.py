#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**向量后端对照**（L1 真实句向量 vs L4 纯 BM25）。

🔴 **要回答的问题**（PRD 4.6 的 L1/L2/L3/L4 分级）：
本机实测 **L2（TF-IDF+SVD）这条向量路是负贡献** ——
「仅向量」基线 V 的 R@5 只有 0.633–0.724，而纯 BM25 是 0.856–0.883。
但 **L2 不是预训练语义模型** → **不能**据此断言"语义检索无效"。
→ 装了 `sentence-transformers` 之后，**L1（`bge-small-zh-v1.5`）才是能下结论的那一级**。

🔴 **本脚本对比两个批次的同一组基线**，重点看三件事：
1. **V（仅向量）的 R@5** —— 语义路自己能召回多少；
2. **B3（混合）vs B2′（纯 BM25）** —— 混合到底是加分还是拖后腿；
3. **B2′ 不受向量影响**（它 `use_vector=False`）→ 它是**对照基线**，
   两批次之间 B2′ 的差异就是**题目/切分变化**造成的，不是后端造成的。

⚠️ **横比的前提**：两批次必须**同策略、同题集、同 top-k**。
B2′ 是这三条之外的第四个自查项 —— 它变了就说明有别的东西变了。

用法：
    python tools/compare_embed_backends.py \
        --a eval_runs/<L4批次前缀> --b eval_runs/<L1批次前缀>

🔴 **批次前缀**：`run_retrieval_eval.py` 会把每个切分策略写成独立文件
`<前缀>_chunk_<策略>.json`，本工具按 `<前缀>_chunk_<策略>.json` 逐个读。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "eval_runs"


def load_batch(prefix):
    """读一个批次的所有 `<prefix>_chunk_<策略>.json`。"""
    p = pathlib.Path(prefix)
    if p.suffix == ".json" and p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("strategies") or [], {st: d["results"] for st in (d.get("strategies") or [])}
    files = sorted(RUN_DIR.glob(f"{p.name}_chunk_*.json"))
    if not files:
        raise SystemExit(f"找不到 {prefix}_chunk_*.json")
    strat, res = [], {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        st = d.get("strategy") or f.stem.split("_chunk_")[-1]
        strat.append(st)
        res[st] = d["results"]
    return strat, res


def backend_of(prefix):
    files = sorted(RUN_DIR.glob(f"{pathlib.Path(prefix).name}_chunk_*.json"))
    if not files:
        return "?"
    return json.loads(files[0].read_text(encoding="utf-8")).get("embed_backend")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="对照批次（通常是 L4 纯 BM25）")
    ap.add_argument("--b", required=True, help="实验批次（L1）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    strat_a, A = load_batch(args.a)
    strat_b, B = load_batch(args.b)
    print(f"A = {args.a}  后端 {backend_of(args.a)}")
    print(f"B = {args.b}  后端 {backend_of(args.b)}\n")

    rows = []
    for st in strat_a:
        ra = A.get(st)
        rb = B.get(st)
        if not ra or not rb:
            continue
        print(f"=== 切分策略 `{st}` ===")
        print(f"{'基线':<24}{'A: R@5':>10}{'B: R@5':>10}{'差':>9}"
              f"{'A: 数值R@5':>12}{'B: 数值R@5':>12}{'A: 秒':>9}{'B: 秒':>9}")
        for bname in ra:
            if bname not in rb:
                continue
            x, y = ra[bname], rb[bname]
            r5a = x["recall_at_5"][0]
            r5b = y["recall_at_5"][0]
            na = x.get("numeric_recall_at_5") or [None]
            nb = y.get("numeric_recall_at_5") or [None]
            print(f"{bname:<24}{r5a:>10.3f}{r5b:>10.3f}{r5b-r5a:>+9.3f}"
                  f"{(na[0] if na else 0):>12.3f}{(nb[0] if nb else 0):>12.3f}"
                  f"{x.get('query_seconds', 0):>9}{y.get('query_seconds', 0):>9}")
            rows.append({"strategy": st, "baseline": bname,
                         "r5_a": r5a, "r5_b": r5b, "diff": round(r5b - r5a, 4),
                         "numeric_a": na[0] if na else None,
                         "numeric_b": nb[0] if nb else None,
                         "sec_a": x.get("query_seconds"), "sec_b": y.get("query_seconds")})
        # 三个关键读数
        v_a, v_b = ra.get("V_vector_filter"), rb.get("V_vector_filter")
        p_a, p_b = ra.get("B2prime_filter_a0.5"), rb.get("B2prime_filter_a0.5")
        h_a, h_b = ra.get("B3_hybrid_filter"), rb.get("B3_hybrid_filter")
        if v_a and v_b and p_a and p_b and h_a and h_b:
            print(f"  🔴 仅向量 V：R@5 {v_a['recall_at_5'][0]:.3f} → "
                  f"{v_b['recall_at_5'][0]:.3f}")
            print(f"  🔴 纯 BM25 B2′(α=0.5)：R@5 {p_a['recall_at_5'][0]:.3f} → "
                  f"{p_b['recall_at_5'][0]:.3f}"
                  f"   （**不受向量影响** → 差异说明题集/切分变了，不是后端）")
            print(f"  🔴 混合 B3：R@5 {h_a['recall_at_5'][0]:.3f} → "
                  f"{h_b['recall_at_5'][0]:.3f}"
                  f"   （B3 vs B2′：{h_a['recall_at_5'][0]-p_a['recall_at_5'][0]:+.3f}"
                  f" → {h_b['recall_at_5'][0]-p_b['recall_at_5'][0]:+.3f}）")
            print(f"  🔴 延迟：向量路 {v_a.get('query_seconds')}s → "
                  f"{v_b.get('query_seconds')}s；纯 BM25 {p_a.get('query_seconds')}s → "
                  f"{p_b.get('query_seconds')}s\n")

    out = {"a": args.a, "a_backend": backend_of(args.a),
           "b": args.b, "b_backend": backend_of(args.b), "rows": rows}
    outp = ROOT / (args.out or "eval_runs/embed_backend_compare.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
