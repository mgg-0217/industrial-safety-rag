#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**key_point 不可达到底是谁的问题** —— 质检判据 vs 数据。

🔴 **背景**：
LLM 重提炼 key_point 之后，`check_test_set.py` 的「key_point 不可达」从 4 条涨到 46 条。
两种解释，**修复动作完全相反**：

| 解释 | 含义 | 该修什么 |
|---|---|---|
| (A) 数据坏 | kp 是幻觉 / 串到了别的文档 | 改数据（重提炼 / 改题） |
| (B) 判据错 | kp 是**真原文**，只是不在**遗留的 `gold_answer`** 里 | 改判据（`gold_answer` 该刷新） |

🔴 **本脚本用「逐层可达性」把 46 条分箱**：
`gold_answer` → **来源文档全文** → 检索证据块 → 其它文档 → 哪里都没有。

⚠️ **不能只把判据放宽**（K-59 的反面）：判据收紧让数字好看要扫漏判，
判据放宽让数字好看同样要**先证明放宽后剩下的仍然是真问题**。
所以本脚本同时给出**收紧版判据**下的命中数：
「kp 不在来源文档正文里」→ 那是**真·幻觉**，无论判据怎么改都必须修。

用法：
    python tools/diag_kp_reachability.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval_align as A  # noqa: E402

MD_DIR = ROOT / "data" / "md"
CHUNKS_DIR = ROOT / "data" / "chunks"
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
FLAGS = ROOT / "data" / "eval" / "check_flags.json"


def nz(s) -> str:
    """归一化：去空白 + 去零宽字符（PDF 抽取常见）。"""
    s = re.sub(r"[\s  ​﻿]+", "", s or "")
    return s


def load_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def load_flags():
    raw = json.loads(FLAGS.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    out = {}
    for item in raw:
        qid = item.get("qid")
        if qid:
            # 🔴 嵌套结构：{"qid":..., "flags":{...}} —— 别把外层当 flag 用
            out[qid] = item.get("flags") or {}
    return out


def main():
    rows = load_jsonl(TEST_SET)
    flags = load_flags()
    mds = {p.stem: nz(p.read_text(encoding="utf-8", errors="replace"))
           for p in MD_DIR.glob("*.md")}
    chunks = load_jsonl(CHUNKS_DIR / "fixed.jsonl")
    chunk_by_doc = {}
    for c in chunks:
        chunk_by_doc.setdefault(c.get("doc_id") or "", []).append(nz(c.get("text") or ""))

    boxes = Counter()
    detail = []
    n_kp_total = 0
    n_unreach = 0

    for r in rows:
        f = flags.get(r.get("qid")) or {}
        unreach = f.get("unreachable_key_points") or []
        kps = r.get("key_points") or []
        n_kp_total += len(kps)
        if not unreach:
            continue
        doc = r.get("doc_id") or ""
        src, mode = A.alignment_source(r)
        src_text = src if isinstance(src, str) else ""
        for kp in unreach:
            n_unreach += 1
            k = nz(kp)
            if not k:
                boxes["空 kp"] += 1
                continue
            if k and k in nz(src_text):
                boxes["A1 在 gold_answer 里（与质检结果矛盾）"] += 1
                continue
            if k and k in mds.get(doc, ""):
                # 在来源文档里 —— 再看是不是检索证据里就有
                in_chunk = any(k in ct for ct in chunk_by_doc.get(doc, []))
                if in_chunk:
                    lab = "B1 在来源文档 + 检索证据里（真原文，判据字段错）"
                else:
                    lab = "B2 在来源文档但不在任何 fixed 块里（真原文，切分边界外）"
                boxes[lab] += 1
                detail.append((r.get("qid"), r.get("question_type"), lab, kp[:70]))
                continue
            # 不在来源文档 → 查是不是跑到别的文档去了
            other = [d for d, t in mds.items() if k in t]
            if other:
                lab = "C1 串到别的文档：" + ",".join(other[:2])
            else:
                lab = "C2 哪里都没有（真·幻觉/改写）"
            boxes[lab] += 1
            detail.append((r.get("qid"), r.get("question_type"), lab, kp[:70]))

    print("=== 不可达 key_point 的归因分箱 ===")
    for k, v in boxes.most_common():
        print(f"  {v:4d}  {k}")
    print(f"\n总 key_point {n_kp_total}，判不可达 {n_unreach}")

    # 收紧版判据：不在来源文档正文里 = 真问题
    strict = sum(v for k, v in boxes.items()
                 if k.startswith("C") or k in ("空 kp",))
    print(f"\n🔴 收紧版判据（kp 不在来源文档正文里 = 真问题）：{strict} 条")
    print(f"   → 判据若改成「逐字可达于来源文档」，命中从 {n_unreach} 降到 {strict}")

    if detail:
        print("\n=== 真问题（C 箱）明细 ===")
        for qid, qt, lab, kp in [d for d in detail if d[2].startswith("C")][:20]:
            print(f"  {qid} [{qt}] {lab}\n      {kp}")

    # 按题型拆分（数值题的 key_point 是数值锚点，判据不同）
    by_type = Counter()
    for qid, qt, lab, kp in detail:
        by_type[(qt, lab.split("（")[0])] += 1
    print("\n=== 按题型 × 分箱 ===")
    for (qt, lab), v in sorted(by_type.items()):
        print(f"  {v:4d}  [{qt}] {lab}")

    out = {"boxes": dict(boxes), "n_kp_total": n_kp_total,
           "n_unreachable": n_unreach, "strict_hits": strict,
           "detail": [{"qid": q, "type": t, "box": l, "kp": k}
                      for q, t, l, k in detail]}
    (ROOT / "eval_runs" / "kp_reachability.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n-> eval_runs/kp_reachability.json")


if __name__ == "__main__":
    main()
