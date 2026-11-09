#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**人工核对队列的决策辅助**：把每条疑点的证据摊开，并给出建议动作。

🔴 **为什么需要它**：
`data/eval/review_queue.md` 只说"这里有问题"，但**没说问题出在哪一层**，
人工核对时每条都要自己回原文翻一遍。
本工具对每条给出：
* **根因分类**（机器能判的那几种）
* **证据**（锚点实际出现在哪些条款 / 原文里根本没有）
* **建议动作**（改题干 / 改出处 / 改锚点 / 弃题 / 可机器修）

机器判不了的（"题干问的量 ↔ 锚点答的量"是否对应）**明确标成"必须人判"**，
不装成结论 —— 这是第五轮 §4 的教训（「新发现」必须问：有第二个实现吗）。

用法：
    python tools/diag_review_decisions.py \
        --questions data/eval/test_set.relinked.jsonl \
        --report eval_runs/review_report_relinked.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import review_eval_set as RV  # noqa: E402
from fix_review_findings import load_clause_index_with_pos  # noqa: E402


def classify(rec, problems, text_of, pos_of):
    """给一条疑点判定根因 + 建议动作。返回 (根因, 动作, 证据行列表)。"""
    qtype = rec.get("question_type")
    doc = rec.get("doc_id")
    kps = list(rec.get("key_points") or [])
    nums = list(rec.get("must_include_numbers") or [])
    ev = []

    # 每个锚点实际出现在哪些条款
    per_anchor = []
    for a in kps + nums:
        hits = []
        for (d, cn), txt in text_of.items():
            if d != doc:
                continue
            if RV.contains(txt, a):
                hits.append(cn)
        per_anchor.append((a, hits))

    unreachable = [a for a, h in per_anchor if not h]
    if unreachable:
        return ("锚点在原文里根本找不到",
                "改出处文档 或 改 key_point 为原文逐字片段（或弃题）",
                [f"锚点「{a[:34]}」在该文档全文里没有任何匹配" for a in unreachable])

    # 各锚点的条款集合：取交集（同一条款能同时容纳全部锚点？）
    sets = [set(h) for _, h in per_anchor if h]
    common = set.intersection(*sets) if sets else set()
    if common and qtype != "数值题":
        return ("锚点分散，但存在**同时容纳全部锚点**的条款",
                "把 clause_no 改到该条款（机器可修）",
                [f"共同条款：{sorted(common)[:3]}"])

    # 落在不同条款
    covered = set()
    for _, h in per_anchor:
        covered |= set(h)
    ev.append(f"锚点分布在 {len(covered)} 个条款：{sorted(covered)[:4]}")
    if common:
        ev.append(f"存在同时容纳全部锚点的条款：{sorted(common)[:3]}")
    if qtype == "数值题":
        # 🔴 **数值题优先判"必须人判"**：溯源定位（clause_no）是**表层**，
        #    「题干问的量 ↔ 锚点答的量」是否对应才是**根子** ——
        #    实测 t_87dac2691148 问"氮气吹扫压力限值"，
        #    锚点却含 "71 ℃" / "85 ℃" 等与压力无关的量。
        #    把 clause_no 指对了，题照样是错的。
        q = rec.get("question") or ""
        ev.append(f"题干：{q[:60]}")
        ev.append(f"锚点：{[a for a, _ in per_anchor][:6]}")
        return ("数值题锚点分散（短数字串在文档多处命中，定位不可靠）",
                "🔴 **必须人判**：题干问的量 ↔ 锚点答的量是否对应？"
                "不对应就改题干或弃题（溯源定位是次要问题）",
                ev)
    return ("锚点分散在多个条款（题目天然跨条款）",
            "把溯源升级成条款集合 `clause_nos`（本次已做）或拆题",
            ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/test_set.jsonl")
    ap.add_argument("--report", default="data/eval/review_report.json")
    ap.add_argument("--out", default="docs/人工核对决策表.md")
    args = ap.parse_args()

    rows = {json.loads(l)["qid"]: json.loads(l)
            for l in (ROOT / args.questions).open(encoding="utf-8") if l.strip()}
    rep = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    idx = load_clause_index_with_pos()
    text_of = {k: "\n".join(t for t, _ in v) for k, v in idx.items()}
    pos_of = {k: min(s for _, s in v) for k, v in idx.items()}

    bad = [r for r in rep["results"] if r.get("verdict") == "likely_bad"]
    L = ["# 人工核对决策表", "",
         f"题集：`{args.questions}`　队列来源：`{args.report}`",
         f"疑点 {len(bad)} 条", "",
         "> 🔴 本表只把**证据**摊开，**不代替判断**：",
         "> 「题干问的量 ↔ 锚点答的量」是否对应，机器判不了，明确标成「必须人判」。", ""]

    counts = {}
    for r in bad:
        rec = rows.get(r["qid"]) or {}
        cause, action, ev = classify(rec, r.get("problems") or [], text_of, pos_of)
        counts[action] = counts.get(action, 0) + 1
        L += [f"## `{r['qid']}` · {rec.get('question_type')}", "",
              f"- **题干**：{rec.get('question')}",
              f"- **出处**：`{rec.get('doc_id')}`",
              f"- **机器判出的问题**：{'; '.join(r.get('problems') or [])[:200]}",
              f"- **根因**：{cause}",
              f"- **建议动作**：{action}"]
        for e in ev:
            L.append(f"  - {e}")
        L.append("")

    L += ["## 汇总", "", "| 建议动作 | 条数 |", "|---|---|"]
    for a, n in sorted(counts.items(), key=lambda x: -x[1]):
        L.append(f"| {a} | {n} |")
    L.append("")

    (ROOT / args.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-8:]))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
