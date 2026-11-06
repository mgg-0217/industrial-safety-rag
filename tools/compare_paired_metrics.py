#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**两个批次的配对指标比较**（按题为单位做 bootstrap CI）。

🔴 **为什么不用两个独立批次的 CI 重叠来判断**：
两批答案的 CI 重叠与否，对**同一个模型在同一批题上的两次运行**几乎没有信息量
（题目难度是主导项）→ 只能靠**配对**把题目难度消掉。

🔴 **单位 = 题**：题目难度是最大的方差来源，
所以重采样必须在**题**这个单位上做，不能在 key_point 上做
（在 kp 上重采样会把同一题的多个 kp 当成独立样本 → CI 假窄）。

🔴 **哪些指标能配对**：
`key_point_coverage` 这类**聚合口径**（命中数/总数）**不能**直接配对 ——
两批的 kp 总数可能不同（题集变了）→ 分子分母都变。
本工具用**题目级均值**做配对：每题算一个比例，再对题做重采样。
`refusal_false_answer`（0/1）本身就是题目级 → 可以直接配对。

用法：
    python tools/compare_paired_metrics.py \
        --a eval_runs/20261106_09XX_..._pending_main.answers.jsonl \
        --b eval_runs/20261106_0930_..._pending_v3fact.answers.jsonl \
        --label-a system --label-b system_v3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402

BOOT = 2000
SEED = 20261106


def load(path):
    out = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r.get("qid")] = r
    return out


def per_question_metrics(rec, rows):
    """返回该题的各项**题目级**指标。"""
    q = rows.get(rec.get("qid")) or {}
    m = {}
    detail = rec.get("kp_detail") or []
    tot = len(detail)
    if tot:
        soft = sum(1 for d in detail if d.get("hit"))
        strict = sum(1 for d in detail if d.get("strict"))
        m["kp_cov_soft"] = soft / tot
        m["kp_cov_strict"] = strict / tot
        m["kp_full_hit"] = 1.0 if soft == tot else 0.0
    else:
        m["kp_cov_soft"] = m["kp_cov_strict"] = None
        m["kp_full_hit"] = None
    m["answer_len"] = len(rec.get("answer") or "")
    m["cited_n"] = rec.get("cited_n") or 0
    v = rec.get("refusal_verdict")
    if v is not None:
        m["refusal_false"] = 1.0 if v == "false_answer" else 0.0
    m["type"] = q.get("question_type") or rec.get("question_type") or "?"
    return m


def boot_diff(a_vals, b_vals, n_boot=BOOT, seed=SEED):
    """按题重采样的配对差值 95% CI（题目是重采样单位）。"""
    n = len(a_vals)
    if n < 2:
        return None, None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            s += a_vals[j] - b_vals[j]
        diffs.append(s / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    mean = sum(a - b for a, b in zip(a_vals, b_vals)) / n
    return round(mean, 4), round(lo, 4), round(hi, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    rows = {json.loads(l)["qid"]: json.loads(l)
            for l in (ROOT / "data/eval/test_set.jsonl").open(encoding="utf-8")
            if l.strip()}
    common = sorted(set(A) & set(B))
    print(f"A = {args.label_a}：{len(A)} 条")
    print(f"B = {args.label_b}：{len(B)} 条")
    print(f"配对交集：{len(common)} 条\n")

    per_type = defaultdict(lambda: defaultdict(list))
    for qid in common:
        ma = per_question_metrics(A[qid], rows)
        mb = per_question_metrics(B[qid], rows)
        for key in ("kp_cov_soft", "kp_cov_strict", "kp_full_hit",
                    "answer_len", "cited_n", "refusal_false"):
            if ma.get(key) is None or mb.get(key) is None:
                continue
            t = per_type[ma["type"]]
            t[key].append((ma[key], mb[key]))

    print(f"{'题型':<12}{'指标':<18}{'n':>5}{'A':>10}{'B':>10}"
          f"{'差(A-B)':>11}{'95%CI(配对)':>24}{'显著':>8}")
    result = {}
    for qt, metrics in sorted(per_type.items()):
        for key, pairs in sorted(metrics.items()):
            av = [p[0] for p in pairs]
            bv = [p[1] for p in pairs]
            mean_a = sum(av) / len(av)
            mean_b = sum(bv) / len(bv)
            d, lo, hi = boot_diff(av, bv)
            sig = ""
            if lo is not None and (lo > 0 or hi < 0):
                sig = "**是**"
            ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "-"
            print(f"{qt:<12}{key:<18}{len(pairs):>5}{mean_a:>10.4f}{mean_b:>10.4f}"
                  f"{d:>+11.4f}{ci:>24}{sig:>8}")
            result.setdefault(qt, {})[key] = {
                "n": len(pairs), "a": round(mean_a, 4), "b": round(mean_b, 4),
                "diff": d, "ci": [lo, hi], "significant": bool(sig)}

    print("\n🔴 口径：题目级均值 + 按题重采样（2000 次，seed=20261106）。")
    print("   「显著」= 配对差值的 95% CI 不含 0。")

    out = {"a": args.a, "b": args.b, "label_a": args.label_a,
           "label_b": args.label_b, "n_paired": len(common), "per_type": result}
    outp = ROOT / (args.out or "eval_runs/paired_metrics_compare.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
