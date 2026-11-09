#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**把溯源从「单条款」升级为「条款集合」**（`clause_nos`）。

🔴 **要解决的结构性问题**（2026-11-09 实测）：
题集的 `clause_no` 是**单值**字段，但一条题的多个 key_point / 数值锚点
**天然分散在多个相邻条款**里 —— 24 条"锚点挂错条款"的事实题里，
**没有一条是"选错了条款"**：

| 分档 | 条数 | 形态 |
|---|---|---|
| A 单条款（纯选错） | **0** | — |
| B 邻接多条款 | 18 | `1.3.1/1.3.2/1.3.3`、`3.3.1(1)/3.3.1(2)`、`1、作业事项/2、作业内容` |
| C 远隔多条款 | 2 | |
| D 原文里根本找不到 | 4 | 真问题（出处文档选错 / kp 是改写） |

→ **单值 `clause_no` 这个 schema 表达不了跨条款题**：
无论把 `clause_no` 改成哪个，总有 kp"挂错"。

🔴 **改法**：
1. 题集新增 `clause_nos`（**列表**）= 该题所有锚点实际所在的条款（按文档内位置排序）；
2. `clause_no` 保留（取归属集合里位置最靠前的那个，或原值若它本就在集合里），
   **向后兼容**（`src/citation.py` 等仍读单值）；
3. `tools/review_eval_set.py` 的溯源 scope 改为**条款集合的并集**。

⚠️ **不影响已发布的召回数字**：
`tools/run_retrieval_eval.py::build_gold` 算 gold 区间用的是
**候选池的 `clause_char_start/end`**（切片时的字符偏移），**不读题集的 `clause_no`**。
→ 本工具只改溯源元数据与核对判据。

用法：
    python tools/relink_clauses.py [--dry-run]
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

import review_eval_set as RV  # noqa: E402
from fix_review_findings import load_clause_index_with_pos  # noqa: E402

TEST_SET = ROOT / "data/eval/test_set.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="",
                    help="写到这个**新文件**（默认原地覆盖题集）。"
                         "原地覆盖会被权限分类器拦，且没有 diff 就没有审阅依据 —— "
                         "建议先写新文件 → 核对 diff → 再替换。")
    args = ap.parse_args()

    path = TEST_SET
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    idx = load_clause_index_with_pos()

    # (doc_id, clause_no) → 起始字符位置 / 文本
    pos_of = {k: min(s for _, s in v) for k, v in idx.items()}
    text_of = {k: "\n".join(t for t, _ in v) for k, v in idx.items()}

    changed, multi, none_found = [], [], []
    for r in rows:
        qtype = r.get("question_type")
        # 🔴 **只处理事实题**：
        #   - 事实题的 kp 是**长句**（30+ 字）→ 定位可靠；
        #   - 数值题的锚点是**短数字串**（"150 mm"），在文档里多处命中
        #     → "位置最靠前"那个未必是出处条款 → **定位不可靠，不自动改**；
        #   - 版本陷阱题 / 跨文档推理题的溯源是 `expect_doc_id`（文档级），
        #     不依赖 `clause_no`；
        #   - 拒答题没有原文断言。
        if qtype != "事实题":
            continue
        doc = r.get("doc_id")
        if not doc:
            continue
        anchors = [k for k in (r.get("key_points") or [])]
        if not anchors:
            continue

        # 每个锚点 → 它实际所在的条款（取**位置最靠前**的那个）
        own = []
        missing = []
        for a in anchors:
            hits = [(cn, pos_of.get((doc, cn), 0)) for (d, cn) in text_of
                    if d == doc and RV.contains(text_of[(d, cn)], a)]
            if hits:
                hits.sort(key=lambda x: x[1])
                own.append(hits[0][0])
            else:
                missing.append(a)
        if not own:
            none_found.append((r["qid"], missing[:2]))
            continue

        # 归属条款集合（按文档内位置排序）
        uniq = sorted(set(own), key=lambda cn: pos_of.get((doc, cn), 0))
        old_cn = r.get("clause_no")
        old_list = r.get("clause_nos")
        if old_list == uniq and (old_cn == uniq[0] or old_cn in uniq):
            continue

        r["clause_nos"] = uniq
        # 单值向后兼容：原值若仍在集合里就保留，否则取位置最靠前的
        if old_cn not in uniq:
            r["clause_no"] = uniq[0]
        r["clause_nos_source"] = "relink_2026-11-09"
        if len(uniq) > 1:
            multi.append((r["qid"], len(uniq), r.get("clause_no")))
        changed.append(r["qid"])

    print(f"重挂溯源 {len(changed)} 条（其中跨条款 {len(multi)} 条）；"
          f"锚点在原文里找不到 {len(none_found)} 条\n")
    for qid, n, cn in multi[:12]:
        print(f"  [跨条款] {qid}: {n} 个条款，主条款 {str(cn)[:40]}")
    if len(multi) > 12:
        print(f"  … 另有 {len(multi)-12} 条")
    for qid, m in none_found:
        print(f"  ⚠️  {qid}: 锚点原文找不到 {m}")

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
    bak = path.with_suffix(".jsonl.bak_relink")
    shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {path.name}（备份 {bak.name}）")


if __name__ == "__main__":
    main()
