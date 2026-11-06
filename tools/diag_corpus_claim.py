#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**扫「假称语料里有规定」这种失败模式**。

🔴 **背景**：`SYSTEM_V3` 全量 150 条里的 3 条误答，有 1 条形态看着像新的：
「资料中**有**合成氨装置**检维修作业许可**的明确规定」。

🔴 **本工具 2026-11-06 的重要修正**：
初版在本文件里**自带一套正则**，测出「8/150 条含这类断言」。
🔴 **那个数字是假阳性** —— 宽松的模式把**边界声明句**
（「以下为通用要求，其适用性不等同于对 X 的专门规定」）也判成了假称，
而那恰恰是**我们想要的正确行为**。
初版还踩了 K-58 同款的**子串陷阱**：「**现有**资料仅能说明…」里的"有"。

→ **现在本工具直接复用 `src/scoring.py::find_corpus_claim_sentences`**，
判据只有一份，改一处两边同时生效（不再有"两套正则各说各话"）。

⚠️ **实测结论（修正后）**：
* 真正的"假称语料里有规定"：**1/150**
* 且这 1 条**同时也是污染句**（`find_pollution_sentences` 命中）
  → **它不是一类新的失败模式，就是原有的"归因污染"**
* → "代码级兜底要额外处理假称句"这个设计**仍然保留**（它是安全网：
  假称句不一定同时满足污染句的全部条件），但在本批次上**新增划除 0 条**。

用法：
    python tools/diag_corpus_claim.py --answers eval_runs/<batch>.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = [json.loads(l) for l in pathlib.Path(args.answers).open(encoding="utf-8")
            if l.strip()]
    n_claim = 0
    n_also_pollution = 0
    n_pollution_only = 0
    by_mech = Counter()
    ex = []
    for r in recs:
        subj = r.get("equipment_id") or r.get("subject")
        ans = r.get("answer") or ""
        claims = SC.find_corpus_claim_sentences(ans, subj)
        poll = SC.find_pollution_sentences(ans, subj)
        if poll and not claims:
            n_pollution_only += 1
        if not claims:
            continue
        n_claim += 1
        by_mech[r.get("refusal_mechanism") or "?"] += 1
        if poll:
            n_also_pollution += 1
        if len(ex) < 12:
            ex.append((r.get("qid"), r.get("refusal_mechanism"),
                       [x[0] for x in claims], bool(poll)))

    print(f"批次 {args.answers}")
    print(f"  总条数                      {len(recs)}")
    print(f"  判为「假称语料里有规定」      {n_claim}")
    print(f"    其中**同时也是污染句**      {n_also_pollution}"
          f"  → 不是独立的失败模式")
    print(f"  判为污染句但不是假称          {n_pollution_only}")
    if by_mech:
        print(f"  按机制：{by_mech.most_common()}")
    for qid, mech, ss, also in ex:
        print(f"    {qid} [{mech}]{'（也是污染句）' if also else ''}")
        for s in ss:
            print(f"      {s[:130]}")

    out = {"batch": args.answers, "n": len(recs), "n_claim": n_claim,
           "n_also_pollution": n_also_pollution,
           "n_pollution_only": n_pollution_only,
           "by_mechanism": dict(by_mech),
           "note": "判据：src/scoring.py::find_corpus_claim_sentences（与代码级兜底同源）"}
    outp = ROOT / (args.out or "eval_runs/corpus_claim_diag.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
