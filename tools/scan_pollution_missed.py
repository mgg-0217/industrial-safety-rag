#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**漏判风险扫描** —— 找"真污染可能被放过"的句子。

🔴 为什么需要：判据越宽，越容易把真污染放过（假阴性）。
而"误答率下降"有两种原因：
   ① 修掉了假阳性（好）
   ② 把真污染也放过了（坏，且**数字会好看**）
→ 必须主动去查第 ② 种。

🔴 **高风险判据**：
   一句话同时满足 ① 含主体 ② 含断言谓语 ③ 含**具体数值/条款号**
   —— 却**没有被判为污染句** → 值得人工看。
   （带数值/条款号的断言几乎不可能是"模型在澄清归属"）

用法：
    python tools/scan_pollution_missed.py --answers <批次>.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--all", action="store_true",
                    help="连不含数值/条款号的候选也列出来")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.answers, encoding="utf-8")
               if l.strip()]
    high, mid = [], []
    for r in records:
        subject = r.get("equipment_id")
        if not subject:
            continue
        ans = r.get("answer") or ""
        poll = {p[0] for p in SC.find_pollution_sentences(ans, subject)}
        for s, _st, _en in SC._sentences_with_span(ans):
            if subject not in s or s in poll:
                continue
            if not SC._ASSERT_RE.search(s):
                continue
            s_wo = SC._strip_subject(s, subject)
            specific = bool(SC._CLAUSE_RE.search(s_wo) or SC._NUMBER_RE.search(s_wo))
            why = []
            if SC._NEG_PATTERN.search(s):
                why.append(f"NEG:{SC._NEG_PATTERN.search(s).group(0)}")
            if SC._UNCERTAIN_RE.search(s):
                why.append("UNC")
            if SC._ABSENCE_RE.search(s):
                why.append("ABSENCE")
            if SC._CONDITIONAL_RE.search(s):
                why.append(f"COND:{SC._CONDITIONAL_RE.search(s).group(0)}")
            if any(m in s for m in SC._BOUNDARY_STRONG):
                why.append("BOUNDARY")
            if not why:
                continue
            item = (r.get("qid"), subject, s, why)
            (high if specific else mid).append(item)

    print(f"高风险（含数值/条款号却被放过）：{len(high)} 条")
    for qid, subj, s, why in high[:args.limit]:
        print(f"\n  [{subj}] {s[:120]}")
        print(f"    被排除的原因：{why}")
    print(f"\n中风险（含断言谓语但无数值）：{len(mid)} 条")
    if args.all:
        for qid, subj, s, why in mid[:args.limit]:
            print(f"\n  [{subj}] {s[:120]}")
            print(f"    被排除的原因：{why}")


if __name__ == "__main__":
    main()
