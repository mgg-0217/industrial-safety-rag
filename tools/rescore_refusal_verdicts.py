#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用**指定的判据版本**重算一批答案的拒答判定。

🔴 为什么需要这个工具（K-56 的延伸）：
判据改了之后，"新数字更低"里分不清是**模型变好**还是**口径漂移**。
唯一的办法是**把每一版判据都跑一遍同一批答案**，把差异摊开。

用法：
    # 用当前判据
    python tools/rescore_refusal_verdicts.py --answers <批次>.answers.jsonl

    # 用历史判据（从 git 里取出来的那一版）
    git show 1023783:src/scoring.py > _tmp/scoring_v1.py
    python tools/rescore_refusal_verdicts.py --answers <批次>.answers.jsonl \
        --scoring _tmp/scoring_v1.py --tag v1_1023783
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_scoring(path: pathlib.Path):
    """动态加载一个 scoring 模块（可以是历史版本）。

    🔴 历史版本的 scoring.py 依赖同目录的 `normalize` 模块
    → 必须先把 `src/` 放进 sys.path，否则 import 失败。
    """
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    if path.resolve() == (ROOT / "src" / "scoring.py").resolve():
        import scoring
        return scoring
    spec = importlib.util.spec_from_file_location("scoring_variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--scoring", default="src/scoring.py")
    ap.add_argument("--git-scoring", default="",
                    help="直接从 git 取某个提交的 src/scoring.py 当判据用"
                         "（如 `--git-scoring 1023783`），不用手工导出快照")
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    scoring_path = ROOT / args.scoring
    if args.git_scoring:
        import subprocess
        import tempfile
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             f"{args.git_scoring}:src/scoring.py"],
            capture_output=True, check=True).stdout
        tmp = pathlib.Path(tempfile.mkdtemp()) / "scoring_snapshot.py"
        tmp.write_bytes(blob)
        scoring_path = tmp
        print(f"[判据版本] git {args.git_scoring}:src/scoring.py")

    sc = load_scoring(scoring_path)
    records = [json.loads(l) for l in open(args.answers, encoding="utf-8")
               if l.strip()]

    verdicts = {}
    for r in records:
        if r.get("question_type") != "拒答题":
            continue
        subject = r.get("equipment_id")
        v, why = sc.refusal_judgement(r.get("answer") or "", subject=subject)
        verdicts.setdefault(subject or "?", []).append(
            (r.get("qid"), v, why))

    flat = [x for vs in verdicts.values() for x in vs]
    n = len(flat)
    counts = {}
    for _q, v, _w in flat:
        counts[v] = counts.get(v, 0) + 1
    n_false = counts.get("false_answer", 0)

    # 按主体聚类的 bootstrap CI（拒答题同主体多条时不能当独立样本）
    payload = {
        "tag": args.tag,
        "scoring_module": args.scoring,
        "answers": args.answers,
        "n": n,
        "n_subjects": len(verdicts),
        "verdicts": counts,
        "false_answer_rate": round(n_false / n, 4) if n else None,
        "ci_wilson": wilson(n_false, n),
        "by_subject": {s: {v: sum(1 for _q, vv, _w in vs if vv == v)
                            for v in set(x[1] for x in vs)}
                       for s, vs in verdicts.items()},
    }
    brief = {k: payload[k] for k in
             ("tag", "n", "n_subjects", "verdicts", "false_answer_rate",
              "ci_wilson")}
    print(json.dumps(brief, ensure_ascii=False))

    if args.out:
        out = ROOT / args.out
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
