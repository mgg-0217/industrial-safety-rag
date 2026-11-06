#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**拒答批次的配对比较**（同一批题、同一判据，比较两个配置）。

🔴 **为什么必须配对**：
拒答误答率的**分母大、事件数小**（150 条里只有 3–6 条误答），
两个独立批次的 CI **必然重叠** → "0.040 → 0.020"这种说法**没法证明什么**。
配对设计（同一批 qid、同一个检索结果）能把**个体层面的翻转**拎出来：
到底有几条从"误答"翻成"通过"、又有几条反向翻。

🔴 **判定口径**：用**当前** `src/scoring.py::refusal_judgement`（strict / lenient）
重判两个批次 —— 两边用同一份代码，杜绝"跨判据版本不能直接比"（K-61）。

🔴 **主体从 `equipment_id` 字段取**（K-51：靠正则从题干抽会静默失效，
中文主体抽不到 → 该判误答的被降级成"人工复核" → 误答率被系统性低估）。

用法：
    python tools/compare_refusal_paired.py \
        --a eval_runs/20261105_1123_generation_retrieval_refuse_v2sys2_fixed.answers.jsonl \
        --b eval_runs/20261106_0925_generation_retrieval_拒答题_pending_v3refuse.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def mcnemar_exact(b, c):
    """双侧精确 McNemar（二项检验）。b = A误答B通过, c = A通过B误答。"""
    n = b + c
    if n == 0:
        return 1.0, "无不一致样本"
    # P(X <= min(b,c)) * 2, X ~ Binom(n, 0.5)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4), f"不一致 {n} 对（A误B对 {b} / A对B误 {c}）"


def load(path):
    out = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        out[r.get("qid")] = r
    return out


def judge(rec):
    """用当前判据重判一条记录 → (严格判定, 宽松判定)。"""
    ans = rec.get("answer") or ""
    subj = rec.get("equipment_id") or rec.get("subject")
    q = rec.get("question") or ""
    strict, _ = SC.refusal_judgement(ans, subject=subj, question=q, mode="strict")
    lenient, _ = SC.refusal_judgement(ans, subject=subj, question=q, mode="lenient")
    return strict, lenient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    print(f"A = {args.label_a}：{len(A)} 条")
    print(f"B = {args.label_b}：{len(B)} 条")
    print(f"配对交集：{len(common)} 条\n")

    if not common:
        raise SystemExit("没有共同 qid")

    va = Counter()
    vb = Counter()
    la = Counter()
    lb = Counter()
    both_false = []
    flipped = []
    table = {"A误B对": 0, "A对B误": 0, "都误": 0, "都对": 0}

    for qid in common:
        sa, laa = judge(A[qid])
        sb, lbb = judge(B[qid])
        va[sa] += 1
        vb[sb] += 1
        la[laa] += 1
        lb[lbb] += 1
        fa, fb = sa == "false_answer", sb == "false_answer"
        if fa and fb:
            table["都误"] += 1
            both_false.append(qid)
        elif fa and not fb:
            table["A误B对"] += 1
            flipped.append((qid, sa, sb))
        elif fb and not fa:
            table["A对B误"] += 1
            flipped.append((qid, sa, sb))
        else:
            table["都对"] += 1

    n = len(common)
    print(f"=== 严格口径判定分布（配对 {n} 条）===")
    for k in ("pass", "bounded", "manual", "false_answer"):
        print(f"  {k:14s} A {va.get(k,0):4d}   B {vb.get(k,0):4d}")
    print("\n=== 宽松口径判定分布 ===")
    for k in ("pass", "false_answer"):
        print(f"  {k:14s} A {la.get(k,0):4d}   B {lb.get(k,0):4d}")

    fa_n, fb_n = va.get("false_answer", 0), vb.get("false_answer", 0)
    print("\n=== 误答率 ===")
    print(f"  A {args.label_a}: {fa_n}/{n} = {fa_n/n:.4f}  CI {wilson(fa_n, n)}")
    print(f"  B {args.label_b}: {fb_n}/{n} = {fb_n/n:.4f}  CI {wilson(fb_n, n)}")
    print(f"  宽松 A {la.get('false_answer',0)}/{n}   B {lb.get('false_answer',0)}/{n}")

    print("\n=== 配对 2×2（严格口径）===")
    for k, v in table.items():
        print(f"  {k:10s} {v:4d}")
    p, note = mcnemar_exact(table["A误B对"], table["A对B误"])
    print(f"\n  McNemar 精确检验：p = {p}   （{note}）")
    if table["A误B对"] or table["A对B误"]:
        print("  → 逐条翻转（qid / A判定 / B判定）：")
        for qid, sa, sb in flipped:
            print(f"     {qid}  A={sa}  B={sb}")

    out = {"a": args.a, "b": args.b, "n_paired": n,
           "verdicts_a": dict(va), "verdicts_b": dict(vb),
           "lenient_a": dict(la), "lenient_b": dict(lb),
           "false_rate_a": fa_n / n, "false_rate_b": fb_n / n,
           "ci_a": wilson(fa_n, n), "ci_b": wilson(fb_n, n),
           "table": table, "mcnemar_p": p,
           "flipped": [{"qid": q, "a": s, "b": t} for q, s, t in flipped],
           "both_false": both_false}
    outp = ROOT / (args.out or "eval_runs/refusal_paired_compare.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
