#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**数值锚点被抽取正则 bug 污染**的题：给出修复建议。

🔴 **背景**（2026-11-09）：
`tools/build_eval_candidates.py::NUM_RE` 的单位分支把单字母 `m` 排在 `mg/m³` **前面**
→ 对 "10mg/m3"，正则先在 `m` 分支命中 → 锚点变成 **"10 m"**（浓度被当成长度）。
已修正正则；本脚本把**受影响的题**逐条列出来，并给出**修正后的抽取结果**，
供人工逐条确认（"题干问的量 ↔ 锚点答的量"仍然需要人判）。

用法：
    python tools/diag_numeric_anchor_fix.py [--questions data/eval/test_set.jsonl]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import normalize as N  # noqa: E402
import review_eval_set as RV  # noqa: E402
from build_eval_candidates import extract_numbers  # noqa: E402

MD = ROOT / "data" / "md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/test_set.jsonl")
    ap.add_argument("--out", default="docs/数值锚点修复建议.md")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.questions).open(encoding="utf-8") if l.strip()]
    mds = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in MD.glob("*.md")}

    L = ["# 数值锚点修复建议（抽取正则 bug 的受害题）", "",
         f"题集：`{args.questions}`", "",
         "> 🔴 **根因**：`NUM_RE` 的单位分支顺序 —— 单字母 `m` 排在 `mg/m³` 前面 →",
         "> `10mg/m3` 被切成 `10 m`（质量浓度当成长度）。**正则已修**。",
         "> 下表给出每道受害题的**修正后候选锚点**；",
         "> 「题干问的量 ↔ 锚点答的量」**仍需人判**。", ""]

    n_bad = 0
    for r in rows:
        if r.get("question_type") != "数值题":
            continue
        doc = r.get("doc_id") or ""
        txt = mds.get(doc, "")
        nums = r.get("must_include_numbers") or []
        if isinstance(nums, str):
            nums = [x for x in nums.split(";") if x.strip()]
        # 锚点在原文里找不到 → 受污染（或本来就是错题）
        missing = [n for n in nums if not RV.contains_num(txt, n)]
        if not missing:
            continue
        n_bad += 1
        # 修正后的抽取：全文重抽
        fixed = extract_numbers(txt)
        # 只保留与缺失锚点"数值相同"的（同一个量，只是单位被切错）
        def core(x):
            m = re.match(r"(-?\d+(?:\.\d+)?)", str(x))
            return m.group(1) if m else str(x)
        want = {core(x) for x in missing}
        cand = [f for f in fixed if core(f) in want]
        L += [f"## `{r['qid']}`", "",
              f"- **题干**：{r.get('question')}",
              f"- **出处**：`{doc}`",
              f"- **当前锚点（错）**：{missing}",
              f"- **修正后候选**（数值相同、单位才对）：{cand or '（全文重抽里没有同数值的 → 可能整条锚点都不成立）'}",
              f"- **该文档修正后抽到的全部工程量**（供人工挑）：{fixed[:12]}",
              ""]

    L += ["## 汇总", "", f"受害题：**{n_bad} / 39 条数值题**", ""]
    (ROOT / args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"受害题 {n_bad} 条 -> {args.out}")
    print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
