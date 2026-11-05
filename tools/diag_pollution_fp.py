#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：每条被判"污染句"的句子，**具体命中了哪一条判据**。

🔴 为什么需要：假阳性（把"模型在承认没有规定"当成"模型在给主体下规定"）
会直接污染所有拒答指标。必须逐条说清**是哪条判据把它抓进来的**，
才能知道该修判据还是该修预期。

用法：
    python tools/diag_pollution_fp.py --answers <批次>.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402


def why(sentence, subject):
    """逐条判据命中情况。"""
    hits = []
    if subject not in sentence:
        hits.append("①未提主体")
    if SC._NEG_PATTERN.search(sentence):
        hits.append(f"②_NEG:{SC._NEG_PATTERN.search(sentence).group(0)}")
    if SC._UNCERTAIN_RE.search(sentence):
        hits.append(f"②UNC:{SC._UNCERTAIN_RE.search(sentence).group(0)}")
    if not SC._ASSERT_RE.search(sentence):
        hits.append("③无断言谓语")
    else:
        hits.append(f"③断言谓语:{SC._ASSERT_RE.search(sentence).group(0)}")
    if any(m in sentence for m in SC._BOUNDARY_STRONG):
        hits.append(f"④强边界:{[m for m in SC._BOUNDARY_STRONG if m in sentence]}")
    if SC._ABSENCE_RE.search(sentence):
        hits.append(f"⑤缺失声明:{SC._ABSENCE_RE.search(sentence).group(0)}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.answers, encoding="utf-8")
               if l.strip()]
    n = 0
    for r in records:
        subject = r.get("equipment_id")
        if not subject:
            continue
        for s, _st, _en in SC.find_pollution_sentences(r.get("answer") or "", subject):
            n += 1
            if n > args.limit:
                return
            print(f"[{subject}] {s[:100]}")
            print(f"    命中：{why(s, subject)}")


if __name__ == "__main__":
    main()
