#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**把"改写型"的 key_point 对齐回原文逐字**（事实题）。

🔴 **要修的现象**：机器判出「锚点在原文里根本找不到」的事���题，
其中一部分**不是数据坏，而是 kp 本身是改写**：

    原文：特种设备**安全**事故应急救援工作坚持"以人为本"的理念…
    kp ：特种设备事故应急救援工作坚持以人为本的理念…        ← 少"安全"二字

→ kp 不是原文片段 → `key_point_coverage` 的分母是假的（永远命中不了）。

🔴 **为什么用整篇文档正文当对齐源**（而不是 `gold_answer`）：
`gold_answer` 是 M2 时代按 `clause_no` **机械切片**的产物，
正是它跟 kp 对不上（第五轮 §2.2 的根因）→ 拿它当对齐源会把 kp 对到错误的切片上。

🔴 **只修"能逐字对齐"的**：对齐后必须**逐字出现在出处文档正文里**，
否则不动（宁可留一条人工，也不要塞一个假片段进去）。

用法：
    python tools/fix_fact_keypoints.py [--dry-run] [--out <新文件>]
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import eval_align as A  # noqa: E402
import review_eval_set as RV  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"

#: 对齐的**比例门槛**：公共片段至少要占 kp 的这个比例才算对齐成功。
#: 🔴 **为什么不能直接用 `align_key_point`**（它的默认门槛是 4 字）：
#: 它是为**切片**设计的（"LLM 做了概括，只要公共片段 ≥4 字就算对齐"）。
#: 但本工具的对齐源是**整篇文档正文**（几万字）→ 4 字门槛会**乱抓**：
#: 实测 "检查装备。结合有限空间辨识情况…" 被对成 **"局文件高"**（完全无关）。
#: → 这里按**比例**判定（≥40% 且 ≥8 字），且对齐结果必须逐字可达。
MIN_RATIO = 0.4
MIN_LEN = 8

#: 句级切分（对齐到**整句**时用）
_SENT_SPLIT = re.compile(r"[。；;！!？?\n]")


def _cover_ratio(kp_f: str, seg_f: str) -> float:
    """kp 被 seg **覆盖**的比例（所有公共块之和 / kp 长度）。

    🔴 **为什么不用"最长公共块"**：模型改写往往**多点开花** ——
    原文「特种设备**安全**事故应急救援工作坚持…；贯彻"统一领导、分级管理，条块结合、
    以块为主"以及单位自救和社会救援相结合的原则；」
    kp 写成「特种设备事故应急救援工作坚持…；贯彻政府统一领导，分级管理、条块结合、
    条块为主、单位自救和社会救援相结合的原则。」
    → 最长公共块只有「统一领导,分级管理,条块结合,」（0.38），
    但**公共块总和**覆盖了大半个 kp → 按总和判才是"这题考的是这段"的正确语义。
    """
    if not kp_f or not seg_f:
        return 0.0
    sm = difflib.SequenceMatcher(None, kp_f, seg_f, autojunk=False)
    total = sum(b.size for b in sm.get_matching_blocks())
    return total / len(kp_f)


def align_by_sentence(kp, text, min_ratio=0.5):
    """把 kp 对齐到原文里**最相似的一整句**（覆盖率达标才返回）。"""
    kp_f = A.flat(kp)
    if not kp_f:
        return None
    best, best_ratio = None, 0.0
    for seg in _SENT_SPLIT.split(text):
        seg_f = A.flat(seg)
        if len(seg_f) < MIN_LEN:
            continue
        r = _cover_ratio(kp_f, seg_f)
        if r > best_ratio:
            # 🔴 保留**原文形态**（不是归一化形态）——
            #    归一化后的片段虽然"归一化口径下可达"，
            #    但它**不是原文**（标点被换成半角）→ 溯源时不好核对。
            best_ratio, best = r, seg.strip()
    if best and best_ratio >= min_ratio:
        return best
    return None


def align_with_ratio(kp, text):
    """把 kp 对齐回原文：公共片段 ≥40% 且 ≥8 字 → 返回该片段，否则 None。"""
    span = A.best_span(kp, text)
    if not span:
        return None
    kp_f = A.flat(kp)
    if kp_f and len(span) / len(kp_f) < MIN_RATIO:
        return None
    if len(span) < MIN_LEN:
        return None
    return span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="写到新文件（默认原地覆盖）")
    ap.add_argument("--questions", default="", help="题集路径（默认主题集）")
    args = ap.parse_args()

    path = pathlib.Path(args.questions) if args.questions else TEST_SET
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    fixed, still = [], []
    for r in rows:
        if r.get("question_type") != "事实题":
            continue
        doc = r.get("doc_id") or ""
        full = A.read_md(doc)
        if not full:
            continue
        kps = r.get("key_points") or []
        if not kps:
            continue
        # 只有"现在就不在原文里"的 kp 才需要对齐
        need = [k for k in kps if not RV.contains(full, k)]
        if not need:
            continue
        new_kps, changed = [], False
        for k in kps:
            if RV.contains(full, k):
                new_kps.append(k)
                continue
            # 先试**句级**（覆盖率 ≥50%），再退回**片段级**（≥40%）
            aligned = align_by_sentence(k, full) or align_with_ratio(k, full)
            # 🔴 对齐结果必须**逐字可达于出处文档正文**，否则不动
            if aligned and RV.contains(full, aligned) and aligned != k:
                new_kps.append(aligned)
                changed = True
                fixed.append((r["qid"], k, aligned))
            else:
                new_kps.append(k)
                still.append((r["qid"], k))
        if changed:
            r["key_points"] = new_kps
            r["kp_fix_note"] = "key_point 由改写形态对齐回原文逐字"

    print(f"对齐回原文 {len(fixed)} 个；仍不可达 {len(still)} 个\n")
    for qid, k, a in fixed:
        print(f"  [fix] {qid}\n        旧「{k[:40]}」\n        新「{a[:40]}」")
    for qid, k in still:
        print(f"  ⚠️  {qid}: 「{k[:40]}」")

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
    bak = path.with_suffix(".jsonl.bak_kpfix")
    shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {path.name}（备份 {bak.name}）")


if __name__ == "__main__":
    main()
