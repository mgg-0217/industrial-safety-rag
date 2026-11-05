#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**待跑流水线**：额度恢复后按顺序跑完剩下的全部实验，并打印前后对比。

🔴 **为什么要有这个脚本**：
上一轮 Token Plan 额度在批次中途耗尽（错误码 2056），
改进的证据链断在半路（`system_v3` 只跑了 45/150）。
→ 把"额度一恢复就跑什么"固化成脚本，避免下次又是零散地跑。

🔴 **每一阶段都能单独中断**（额度可能又耗尽）：
已完成的阶段不会重跑（产物存在就跳过），中断后重跑本脚本即可续上。

用法：
    python tools/run_pending.py              # 跑所有待办阶段
    python tools/run_pending.py --list       # 只看清单
    python tools/run_pending.py --stage kp   # 只跑某一阶段
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

#: 阶段定义：(名称, 说明, 产物判据, 命令)
STAGES = [
    ("kp", "LLM 重提炼事实题 key_point（159 条）",
     "data/eval/test_set.jsonl",
     [PY, "tools/refine_key_points_llm.py", "--workers", "3",
      "--out", "eval_runs/pending_kp_refine.json"]),
    ("review", "自动核对 + 溯源重定位",
     "data/eval/review_queue.md",
     [PY, "tools/fix_review_findings.py"]),
    ("check", "重跑质检",
     "data/eval/check_report.md",
     [PY, "tools/check_test_set.py"]),
    ("v3refuse", "system_v3 全量拒答评测（150 条）",
     "eval_runs/pending_v3refuse.done",
     [PY, "tools/run_generation_eval.py", "--configs", "retrieval",
      "--strategy", "fixed", "--questions-file", "refusal_extended_v2.jsonl",
      "--only-type", "拒答题", "--system", "system_v3",
      "--workers", "3", "--tag", "pending_v3refuse"]),
    ("v3fact", "system_v3 副作用检查：主集事实题（159 条）",
     "eval_runs/pending_v3fact.done",
     [PY, "tools/run_generation_eval.py", "--configs", "retrieval",
      "--strategy", "fixed", "--questions-file", "test_set.jsonl",
      "--only-type", "事实题", "--system", "system_v3",
      "--workers", "3", "--tag", "pending_v3fact"]),
    ("main", "主集重跑（306 条，题集变了必须重跑）",
     "eval_runs/pending_main.done",
     [PY, "tools/run_generation_eval.py", "--configs", "retrieval",
      "--strategy", "fixed", "--questions-file", "test_set.jsonl",
      "--system", "system", "--workers", "3", "--tag", "pending_main"]),
]


def newest(pattern):
    """目录下最新匹配的文件。"""
    files = sorted((ROOT / "eval_runs").glob(pattern),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def done_marker(stage):
    return ROOT / f"eval_runs/pending_{stage}.done"


def run(cmd, timeout=7200):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print("  !! 退出码", p.returncode)
        print(p.stderr[-2000:])
    print(f"  用时 {time.time()-t0:.0f}s")
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("待跑阶段：")
        for name, desc, _prod, cmd in STAGES:
            print(f"  [{name}] {desc}")
            print(f"        {' '.join(str(c) for c in cmd)}")
        return

    print("=== 待跑流水线 ===\n")
    for name, desc, prod, cmd in STAGES:
        if args.stage and args.stage != name:
            continue
        print(f"--- [{name}] {desc}")
        marker = done_marker(name)
        if marker.exists() and not args.force:
            print("  已完成（标记存在），跳过\n")
            continue
        if name == "v3refuse":
            out = newest("*pending_v3refuse*.json")
            if out and "pending_v3refuse" in out.name and not args.force:
                print(f"  已存在 {out.name}\n")
                continue
        ok = run(cmd)
        if ok:
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            print(f"  -> 标记 {marker.name}\n")
        else:
            print(f"  -> 失败，停在 [{name}]；额度恢复后重跑本脚本续上\n")
            break

    print("=== 完成；对比见 docs/题集自动核对.md §4 ===")


if __name__ == "__main__":
    main()
