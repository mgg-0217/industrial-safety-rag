#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**排除「答案变长 → 更容易命中 key_point」的混淆**。

🔴 **为什么要查**：
`system_v3` 在事实题上三个指标都显著提升（软 +6.4 点 / 严 +4.9 点 / 全命中 +7.6 点），
但**答案也长了 51 字**（250 → 301）。
→ 更长的答案本来就更可能**逐字包含** key_point 片段 →
提升可能只是"字数效应"，不是"内容更准"。

🔴 **判别方法**：按**答案长度分层**再比。
若提升在**短答案层**也成立 → 是内容提升；
若提升只出现在长答案层 → 是字数效应。

用法：
    python tools/diag_length_confound.py --a <system批次> --b <system_v3批次>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(p):
    return {json.loads(l)["qid"]: json.loads(l)
            for l in pathlib.Path(p).open(encoding="utf-8") if l.strip()}


def boot_ci(a, b, n_boot=2000, seed=20261106):
    n = len(a)
    if n < 3:
        return None, None
    rng = random.Random(seed)
    d = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            s += a[j] - b[j]
        d.append(s / n)
    d.sort()
    return d[int(0.025 * n_boot)], d[min(n_boot - 1, int(0.975 * n_boot))]


def metrics(rec):
    d = rec.get("kp_detail") or []
    tot = len(d)
    if not tot:
        return None
    soft = sum(1 for x in d if x.get("hit")) / tot
    strict = sum(1 for x in d if x.get("strict")) / tot
    return soft, strict, len(rec.get("answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    pairs = []
    for qid in common:
        ma, mb = metrics(A[qid]), metrics(B[qid])
        if ma and mb:
            pairs.append((ma, mb))
    print(f"配对 {len(pairs)} 条\n")

    lens_a = [p[0][2] for p in pairs]
    lens_b = [p[1][2] for p in pairs]
    med_a = statistics.median(lens_a)
    med_b = statistics.median(lens_b)
    print(f"答案长度中位数：{args.label_a} {med_a:.0f} / {args.label_b} {med_b:.0f}")
    cut = (med_a + med_b) / 2
    print(f"分层切点（两批中位数的均值）：{cut:.0f}\n")

    strata = {
        "短答案层（两批都短）": lambda a, b: a[2] <= cut and b[2] <= cut,
        "长答案层（两批都长）": lambda a, b: a[2] > cut and b[2] > cut,
        "A短B长": lambda a, b: a[2] <= cut < b[2],
        "A长B短": lambda a, b: b[2] <= cut < a[2],
    }
    print(f"{'层':<26}{'n':>5}{'软A':>9}{'软B':>9}{'差':>10}{'CI':>22}{'显著':>7}")
    for name, pred in strata.items():
        sel = [(p[0], p[1]) for p in pairs if pred(p[0], p[1])]
        if len(sel) < 5:
            print(f"{name:<26}{len(sel):>5}   （样本太少）")
            continue
        sa = [x[0][0] for x in sel]
        sb = [x[1][0] for x in sel]
        lo, hi = boot_ci(sa, sb)
        sig = "**是**" if lo is not None and (lo > 0 or hi < 0) else ""
        ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "-"
        d = sum(sa) / len(sa) - sum(sb) / len(sb)
        print(f"{name:<26}{len(sel):>5}{sum(sa)/len(sa):>9.4f}{sum(sb)/len(sb):>9.4f}"
              f"{d:>+10.4f}{ci:>22}{sig:>7}")

    # 严格口径同理
    print(f"\n{'层':<26}{'n':>5}{'严A':>9}{'严B':>9}{'差':>10}{'CI':>22}{'显著':>7}")
    for name, pred in strata.items():
        sel = [(p[0], p[1]) for p in pairs if pred(p[0], p[1])]
        if len(sel) < 5:
            continue
        sa = [x[0][1] for x in sel]
        sb = [x[1][1] for x in sel]
        lo, hi = boot_ci(sa, sb)
        sig = "**是**" if lo is not None and (lo > 0 or hi < 0) else ""
        ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "-"
        d = sum(sa) / len(sa) - sum(sb) / len(sb)
        print(f"{name:<26}{len(sel):>5}{sum(sa)/len(sa):>9.4f}{sum(sb)/len(sb):>9.4f}"
              f"{d:>+10.4f}{ci:>22}{sig:>7}")


if __name__ == "__main__":
    main()
