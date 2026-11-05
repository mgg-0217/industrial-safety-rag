#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比**两套拒答判据**在同一批答案上的判定差异。

🔴 **为什么需要这个工具**（K-56 的延伸）：

改判分必然让历史数字漂移。只报"新数字"会让人误以为
"模型变好了"；只报"旧数字"则是掩盖。
→ 必须能**逐条**说出：哪条从 A 挪到 B、为什么。

用法：
    python tools/compare_judge_versions.py \
        --old eval_runs/xxx_rescored_t0.7.judge_v1.json \
        --new eval_runs/xxx_rescored_t0.7.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_records(path: pathlib.Path):
    """批次汇总 JSON → {qid: record}。

    逐题明细在同名的 ``*.answers.jsonl`` 里。
    """
    meta = json.loads(path.read_text(encoding="utf-8"))
    stem = path.name
    for suffix in (".judge_v1.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    answers = path.parent / f"{stem}.answers.jsonl"
    if not answers.exists():
        # 汇总 JSON 本身没有逐题明细时，退回只用汇总
        return meta, {}
    recs = {}
    for line in open(answers, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r.get("qid") or r.get("qid") or r.get("question_id")] = r
    return meta, recs


def refusal_type(meta, recs):
    """取出拒答题子集。"""
    out = {}
    for qid, r in recs.items():
        if r.get("type") == "拒答题" or (r.get("question_type") == "拒答题"):
            out[qid] = r
    if out:
        return out
    # 退而求其次：汇总里只有拒答题，就用全量
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--show", type=int, default=8,
                    help="打印多少条具体变化样本")
    ap.add_argument("--show-text", action="store_true",
                    help="打印变化样本的答案正文")
    args = ap.parse_args()

    old_meta, old_recs = load_records(pathlib.Path(args.old))
    new_meta, new_recs = load_records(pathlib.Path(args.new))
    old_recs = refusal_type(old_meta, old_recs)
    new_recs = refusal_type(new_meta, new_recs)

    common = sorted(set(old_recs) & set(new_recs))
    print(f"共同题数：{len(common)}（旧 {len(old_recs)} / 新 {len(new_recs)}）\n")

    # ---- 汇总对比 ----------------------------------------------------------
    def stats(recs):
        c = collections.Counter(r.get("refusal_verdict") for r in recs.values())
        n = len(recs) or 1
        return c, {k: round(v / n, 4) for k, v in c.items()}

    oc, orate = stats(old_recs)
    nc, nrate = stats(new_recs)
    keys = sorted(set(oc) | set(nc))
    print("verdict 分布：")
    print(f"{'verdict':<14}{'旧':>6}{'新':>6}{'Δ':>8}")
    for k in keys:
        print(f"{k:<14}{oc.get(k,0):>6}{nc.get(k,0):>6}{nc.get(k,0)-oc.get(k,0):>+8}")
    n = len(common) or 1
    fo = sum(1 for q in common if old_recs[q].get("refusal_verdict") == "false_answer")
    fn = sum(1 for q in common if new_recs[q].get("refusal_verdict") == "false_answer")
    print(f"\n严格误答率：旧 {fo}/{n} = {fo/n:.4f}   "
          f"新 {fn}/{n} = {fn/n:.4f}   Δ {(fn-fo)/n:+.4f}")

    # ---- 逐条变化 ----------------------------------------------------------
    moves = collections.Counter()
    changed = []
    for q in common:
        o = old_recs[q].get("refusal_verdict")
        nw = new_recs[q].get("refusal_verdict")
        if o != nw:
            moves[(o, nw)] += 1
            changed.append((q, o, nw))
    print("\n判定迁移：")
    for (o, nw), c in moves.most_common():
        print(f"  {o} → {nw}: {c} 条")

    if changed:
        print(f"\n变化样本（前 {args.show} 条）：")
        for q, o, nw in changed[: args.show]:
            r = new_recs[q]
            print(f"\n  [{q}] {o} → {nw}")
            print(f"    理由：{r.get('refusal_reason', '')[:150]}")
            if args.show_text:
                print(f"    答案：{(r.get('answer') or '')[:500]}")
    else:
        print("\n逐条判定完全一致 —— 汇总差异来自样本口径而非判据。")


if __name__ == "__main__":
    main()
