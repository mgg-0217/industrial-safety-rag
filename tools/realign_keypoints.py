#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**重对齐 `aligned_to_source` 的 key_point**（修 `best_span` 坐标 bug 的后果）。

🔴 **为什么需要**：
`src/eval_align.py::best_span` 有一个坐标 bug（用 kp 侧坐标 `block.a` 去切原文段，
应为 `block.b`）—— 已于 2026-11-09 修复。
题集里 `key_point_source = "aligned_to_source"` 的那些 kp **全部出自 bug 修复前的代码**，
于是它们是**从中间切断的碎片**：

| 旧（碎片） | 重对齐后 |
|---|---|
| 「6.清除置换有」 | 「A. 安全隔离 B. 清除置换」 |
| 「处理建议:由泉港区纪委监委进一」 | 「处理建议：由泉港区纪委监委进一步调查处理」 |
| 「处理落实情况:202」 | 「处理落实情况：2024年6月25日，泉州市应急管理局…」 |
| 「一,指导思想以深入贯彻落实」 | 「以深入贯彻落实科学发展观，规范和强化特种设备事故紧急处置工作…」 |

🔴 **kp 不进提示词**，只用于评分 →
所以这是**纯标尺修正**：答案一个字不用改，离线重算即可看到指标变化
（`tools/compare_generation.py`）。

🔴 **只改 `aligned_to_source`**：其它来源的 kp（`llm_from_evidence` 是 LLM 从证据提炼的整句，
`must_include_numbers` 是数值锚点）都不该被对齐算法碰。

用法：
    python tools/realign_keypoints.py [--dry-run] [--out <新文件>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import eval_align as A  # noqa: E402
import review_eval_set as RV  # noqa: E402
from fix_fact_keypoints import align_by_sentence, align_with_ratio  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="写到新文件（默认原地覆盖）")
    ap.add_argument("--questions", default="", help="题集路径（默认主题集）")
    ap.add_argument("--source", default="aligned_to_source",
                    help="只重对齐这个 key_point_source 的 kp")
    args = ap.parse_args()

    path = pathlib.Path(args.questions) if args.questions else TEST_SET
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    n_q = n_k = n_changed = 0
    samples = []
    for r in rows:
        if r.get("key_point_source") != args.source:
            continue
        n_q += 1
        doc = r.get("doc_id") or ""
        full = A.read_md(doc)
        if not full:
            continue
        new_kps = []
        for kp in (r.get("key_points") or []):
            n_k += 1
            new = align_by_sentence(kp, full) or align_with_ratio(kp, full)
            # 🔴 新片段必须**逐字可达于出处文档正文**，且不能是空壳
            if new and new != kp and RV.contains(full, new) and len(A.flat(new)) >= 8:
                new_kps.append(new)
                n_changed += 1
                if len(samples) < 8:
                    samples.append((r["qid"], kp, new))
            else:
                new_kps.append(kp)
        if new_kps != (r.get("key_points") or []):
            r["key_points"] = new_kps
            r["kp_realign_note"] = "aligned_to_source 的 kp 用修好的 best_span 重新对齐"

    print(f"处理 {args.source} 的题 {n_q} 条 / kp {n_k} 个；"
          f"重对齐 {n_changed} 个\n")
    for qid, old, new in samples:
        print(f"  {qid}\n    旧「{old[:44]}」\n    新「{new[:44]}」")

    if args.dry_run:
        print("\n[dry-run] 未写盘")
        return
    if args.out:
        out = ROOT / args.out
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n-> {args.out}（**新文件**，题集未改动）")
        return
    bak = path.with_suffix(".jsonl.bak_realign")
    shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {path.name}（备份 {bak.name}）")


if __name__ == "__main__":
    main()
