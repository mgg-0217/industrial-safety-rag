#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**key_point 精修（`--restrict-to-doc`）的前后对照分析**。

🔴 **要回答的问题**：
第六轮用 `tools/refine_key_points_llm.py --restrict-to-doc` 把事实题的 key_point
重新从**出处文档内部**提炼了一遍。这一刀下去：

1. **跨文档 key_point 降了多少？**（第五轮 §2.3 实测事实题 48/280 是跨文档套话）
2. **评测指标怎么变？** —— 🔴 **必须是零 API 成本的配对**：
   key_point **不进提示词**，只用于评分 → 同一批答案换一套 key_point 重判，
   变化**全部来自标尺**，不来自模型行为。
   （用 `src/scoring.py` 的 `key_point_hits`，与 `tools/compare_generation.py` 同源。）
3. **剩下没修好的那些是什么？** —— 两种可能：
   a) 精修**跳过**的题（LLM 在出处文档内也找不到能支撑的原文 → **出处文档可能选错了**）
   b) 精修**成功但仍有跨文档 kp** 的题

用法：
    python tools/analyze_kp_refine_impact.py \
        --old data/eval/test_set.jsonl.pre_restrict.bak \
        --batch eval_runs/20261106_0933_generation_retrieval_pending_main.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import normalize as NM  # noqa: E402
import scoring as SC  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
MD_DIR = ROOT / "data" / "md"


def nz(s) -> str:
    """🔴 判定口径**必须与评分同源**（`src/normalize.py::match_norm`）。

    原先这里用"只去空白"的归一化 → 同一批 kp 判出 41 个跨文档，
    换同源口径后只有 **10** 个 —— 31 个是**标点差异造成的假阳性**
    （全角/半角括号、中英文引号、破折号）。
    🔴 同 K-66：**两个工具用两套口径 → 结论会自相矛盾**。
    """
    return NM.match_norm(s)


def load_jsonl(p):
    return [json.loads(l) for l in pathlib.Path(p).open(encoding="utf-8") if l.strip()]


def coverage(answers_by_qid, rows_by_qid, kp_field="key_points"):
    """按题型算 (软覆盖, 严覆盖, 题目级全命中, kp总数, 跨文档数)。

    口径与 `tools/compare_generation.py` 一致（软=命中数/总数，严=逐字命中数/总数）。
    """
    per = defaultdict(lambda: Counter())
    for qid, rec in rows_by_qid.items():
        ans = answers_by_qid.get(qid)
        if ans is None:
            continue
        qt = rec.get("question_type") or "?"
        if qt == "拒答题":
            continue                      # 拒答题的 kp 是判分标准，不做覆盖率
        kps = rec.get(kp_field) or []
        if not kps:
            continue
        hit, tot, detail = SC.key_point_hits(ans, kps)
        strict = sum(1 for d in detail if d.get("strict"))
        t = per[qt]
        t["n"] += 1
        t["kp"] += tot
        t["soft_hit"] += hit
        t["strict_hit"] += strict
        t["fullhit_soft"] += 1 if hit == tot else 0
    return per


def fmt(per, label):
    L = [f"\n### {label}", "",
         "| 题型 | n | kp 总数 | 软覆盖 | 严格覆盖 | 题目级全命中 |",
         "|---|---|---|---|---|---|"]
    for qt, t in sorted(per.items()):
        if not t["kp"]:
            continue
        L.append(f"| {qt} | {t['n']} | {t['kp']} | {t['soft_hit']/t['kp']:.4f} | "
                 f"{t['strict_hit']/t['kp']:.4f} | {t['fullhit_soft']/t['n']:.4f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/eval/test_set.jsonl.pre_restrict.bak")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", default="eval_runs/kp_refine_impact.json")
    args = ap.parse_args()

    old_rows = load_jsonl(ROOT / args.old)
    new_rows = load_jsonl(TEST_SET)
    old_by = {r["qid"]: r for r in old_rows}
    new_by = {r["qid"]: r for r in new_rows}

    mds = {p.stem: nz(p.read_text(encoding="utf-8", errors="replace"))
           for p in MD_DIR.glob("*.md")}

    ans_recs = load_jsonl(ROOT / args.batch)
    answers_by = {r.get("qid"): (r.get("answer") or "") for r in ans_recs}

    # ---------- 1. 跨文档 kp 前后对照 ----------
    cross = defaultdict(Counter)
    cross_qids = defaultdict(list)
    changed_kp = Counter()
    old_cross = defaultdict(Counter)
    for qid, nr in new_by.items():
        qt = nr.get("question_type")
        if qt == "拒答题":
            continue
        doc = nr.get("doc_id") or nr.get("expect_doc_id") or ""
        src = mds.get(doc, "")
        for kp in nr.get("key_points") or []:
            is_cross = not (nz(kp) and nz(kp) in src)
            cross[qt]["tot"] += 1
            cross[qt]["cross" if is_cross else "src"] += 1
            if is_cross:
                cross_qids[qt].append((qid, kp))
        ok = old_by.get(qid, {}).get("key_points") or []
        if ok != (nr.get("key_points") or []):
            changed_kp[qt] += 1
        # 旧题集侧的跨文档统计（**分母必须是 kp 数，不是题数**）
        orec = old_by.get(qid) or {}
        odoc = orec.get("doc_id") or orec.get("expect_doc_id") or ""
        osrc = mds.get(odoc, "")
        for kp in (orec.get("key_points") or []):
            old_cross[qt]["tot"] += 1
            if not (nz(kp) and nz(kp) in osrc):
                old_cross[qt]["cross"] += 1

    L = ["# key_point 精修（`--restrict-to-doc`）前后对照", "",
         f"旧题集：`{args.old}`　新题集：`data/eval/test_set.jsonl`",
         f"批次：`{args.batch}`（**同一批答案，零 API 成本配对**）", "",
         "> 🔴 key_point **不进提示词**，只用于评分 → 下面所有指标变化"
         "**全部来自标尺变化**，不代表模型行为变化。",
         "> 判据：`src/scoring.py::key_point_hits`，与 `tools/compare_generation.py` 同源。", "",
         "## 1. 跨文档 key_point（不在出处文档正文里的 kp）", "",
         "| 题型 | 旧：kp 数 | 旧：跨文档 | 旧：跨文档率 | 新：kp 数 | 新：跨文档 | 新：跨文档率 |",
         "|---|---|---|---|---|---|---|"]
    for qt in sorted(cross):
        o, n = old_cross[qt], cross[qt]
        o_rate = f"{o['cross']/o['tot']*100:.1f}%" if o["tot"] else "-"
        n_rate = f"{n['cross']/n['tot']*100:.1f}%" if n["tot"] else "-"
        L.append(f"| {qt} | {o['tot']} | {o['cross']} | {o_rate} | "
                 f"{n['tot']} | {n['cross']} | {n_rate} |")
    L += ["", f"改写了 kp 的题数：{dict(changed_kp)}", ""]

    # ---------- 2. 覆盖率前后对照 ----------
    per_old = coverage(answers_by, old_by)
    per_new = coverage(answers_by, new_by)
    L.append("## 2. 评测指标前后对照（同一批答案）")
    L.append(fmt(per_old, "旧 key_point（`clause_no` 切片 → 第五轮 LLM 提炼）"))
    L.append(fmt(per_new, "新 key_point（限定出处文档内提炼）"))
    L.append("")

    # ---------- 3. 剩余跨文档 kp 明细 ----------
    L += ["## 3. 精修后仍跨文档的 key_point（剩余）", ""]
    for qt, items in sorted(cross_qids.items()):
        if not items:
            continue
        L.append(f"### {qt}（{len(items)} 个）")
        L.append("")
        L.append("| qid | key_point |")
        L.append("|---|---|")
        for qid, kp in items:
            L.append(f"| {qid} | {kp[:60]} |")
        L.append("")

    # ---------- 4. 精修跳过的题 = 出处文档存疑 ----------
    skip_log = ROOT / "data/eval/kp_refine_restrict_log.json"
    if skip_log.exists():
        log = json.loads(skip_log.read_text(encoding="utf-8"))
        skipped = {x["qid"] for x in log if not x.get("ok")}
        remain = {q for items in cross_qids.values() for q, _ in items}
        both = sorted(skipped & remain)
        L += ["## 4. 精修跳过的题（= 出处文档存疑）", "",
              f"精修跳过 {len(skipped)} 条（LLM 在**限定出处文档内**也找不到能支撑"
              f"该题的原文 → **出处文档很可能选错了**，不是 key_point 的问题）。",
              "",
              f"其中仍带跨文档 kp 的有 **{len(both)}** 条：", ""]
        if both:
            L.append("| qid | 出处文档 | 题干 |")
            L.append("|---|---|---|")
            for qid in both:
                r = new_by.get(qid) or {}
                L.append(f"| {qid} | {r.get('doc_id')} | {(r.get('question') or '')[:40]} |")
        L.append("")

    out = {
        "old": str(args.old), "batch": str(args.batch),
        "cross_doc_by_type": {qt: dict(c) for qt, c in cross.items()},
        "changed_kp_questions": dict(changed_kp),
        "remaining_cross_doc": {qt: [{"qid": q, "kp": k} for q, k in v]
                                for qt, v in cross_qids.items()},
    }
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    (ROOT / "docs" / "key_point精修前后对照.md").write_text("\n".join(L),
                                                            encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> docs/key_point精修前后对照.md\n-> {args.out}")


if __name__ == "__main__":
    main()
