#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**扫「假称资料里有规定」这种失败模式**（SYSTEM_V3 残留误答的新靶子）。

🔴 **现象**：SYSTEM_V3 全量 150 条里的 3 条误答，有 1 条形态与前两轮都不同 ——
模型**断言语料里有该主体的专门规定**：

    「资料中有合成氨装置**检维修作业许可**的明确规定」

→ 这不是"把通用条款套到主体上"（归因污染），
而是**对语料内容本身的错误断言**（假称有规定）。

🔴 **为什么值得单独测**：
* 它**可以完全用代码治** —— 代码知道语料里到底有没有该主体的专门规定，
  不需要靠模型自觉（这是"结构化事实"→代码强制拼接的典型场景）；
* 前两轮的提示词约束（规则 5/7）**管不到它** ——
  规则管的是"怎么表述"，这里错的是"语料里有什么"这个事实判断。

用法：
    python tools/diag_corpus_claim.py --answers eval_runs/<batch>.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: 「假称语料里有规定」的模式。
#: 🔴 故意**不用**引号包裹，避免和"未找到"类表述混淆。
CLAIM_PATTERNS = [
    re.compile(r"资料(?:中|里)?(?:有|存在|载有|对)[^。；\n]{0,24}明确规定"),
    re.compile(r"资料(?:中|里)?(?:有|存在|载有)[^。；\n]{0,24}(?:规定|要求|条款)"),
    re.compile(r"(?:根据|依据)资料(?:中|里)?[^。；\n]{0,24}的?(?:规定|要求)"),
    re.compile(r"资料(?:已|已)?(?:明确)?(?:规定|要求|写明)"),
    re.compile(r"检索到[^。；\n]{0,20}关于[^。；\n]{0,16}的(?:规定|要求|条款)"),
]

#: 反向：明确说"没有找到"的表述（不算假称）。
ABSENT_PATTERNS = [
    re.compile(r"未找到|没有找到|未检索到|没有检索到|未提及|没有提及"),
    re.compile(r"未提供|没有提供|资料中未|资料里未"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = [json.loads(l) for l in pathlib.Path(args.answers).open(encoding="utf-8")
            if l.strip()]
    n_claim = 0
    n_claim_without_absent = 0
    ex = []
    per_mech = Counter()
    for r in recs:
        ans = r.get("answer") or ""
        hits = [m.group(0) for p in CLAIM_PATTERNS for m in p.finditer(ans)]
        if not hits:
            continue
        n_claim += 1
        per_mech[r.get("refusal_mechanism") or "?"] += 1
        # 🔴 只有在**整篇答案里没有"未找到"表态**时，才是"真·假称"
        has_absent = any(p.search(ans) for p in ABSENT_PATTERNS)
        if not has_absent:
            n_claim_without_absent += 1
            if len(ex) < 12:
                i = ans.find(hits[0])
                ex.append((r.get("qid"), r.get("refusal_mechanism"),
                           ans[max(0, i - 40):i + 90].replace("\n", " ")))

    print(f"批次 {args.answers}")
    print(f"  总条数                {len(recs)}")
    print(f"  含「假称资料里有规定」  {n_claim}")
    print(f"  其中**整篇无未找到表态** {n_claim_without_absent}  ← 更硬的假称")
    if per_mech:
        print(f"  按机制：{per_mech.most_common()}")
    for qid, mech, snip in ex:
        print(f"    {qid} [{mech}] …{snip}…")

    out = {"batch": args.answers, "n": len(recs), "n_claim": n_claim,
           "n_claim_without_absent": n_claim_without_absent,
           "by_mechanism": dict(per_mech)}
    outp = ROOT / (args.out or "eval_runs/corpus_claim_diag.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
