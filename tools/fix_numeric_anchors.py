#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**修正被抽取正则 bug 污染的数值锚点**（题集层定向修复）。

🔴 **背景**（2026-11-09）：
`NUM_RE` 的单位分支顺序 bug 把 "10mg/m3" 切成 "10 m"（浓度当成长度），
另有三条同源缺陷（白名单大小写、`\\w` 含中文、平方单位丢失），
题集 39 条数值题里 **6 条**的锚点是这些缺陷的产物。

🔴 **本工具只做"确定性"修复**，不做语义判断：
1. 锚点在原文里**不可达**（用与评分同源的 `contains_num` 判定）；
2. 用**修好的** `extract_numbers` 从出处文档全文重抽候选；
3. 候选里找**数值部分相同**的（"10 m" → 数值 "10"）；
4. **候选唯一** → 自动替换；**多个候选** → 记下来交给人判。

🔴 **不动题干**：题干问的量 ↔ 锚点答的量是否对应，是**语义问题**，
本工具**判不了也不该判**（K-79：表层问题 vs 根子问题）。

⚠️ **改锚点会改评分**：数值精确匹配的分母就是 `must_include_numbers`，
所以修完必须**重算**相关指标（`tools/compare_generation.py` 离线重算，零 API 成本）。

用法：
    python tools/fix_numeric_anchors.py [--dry-run] [--out <新文件>]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import normalize as N  # noqa: E402
import review_eval_set as RV  # noqa: E402
from build_eval_candidates import extract_numbers  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
MD_DIR = ROOT / "data" / "md"


def core_num(x) -> str:
    """锚点的**数值部分**（"10 m" / "10 mg/m3" → "10"）。"""
    m = re.match(r"(-?\d+(?:\.\d+)?)", str(x).strip())
    return m.group(1) if m else str(x).strip()


def core_unit(x) -> str:
    """锚点的**单位部分**（"10 m" → "m"、"10 mg/m3" → "mg/m3"）。"""
    s = re.sub(r"^-?\d+(?:\.\d+)?", "", str(x).strip()).strip()
    return s.lower()


def unit_compatible(old, new) -> bool:
    """候选单位是否是原单位的**延伸**（原单位是候选的前缀）。

    🔴 **必须有这条约束**：只按"数值相同"匹配会**跨单位乱配** ——
    实测 `"40 MPa"` 会被配到 `"40 %"`（压力配成百分比），
    而这两者在语义上毫无关系。

    → 允许的形态：**原单位被截断**（"10 m" ← "10 mg/m3"）、平方化（"25 mm" ← "25 mm2"）。
    """
    o, n = core_unit(old), core_unit(new)
    if not o:
        return False
    return n.startswith(o)


#: 单位 → 量纲族（用来判断"题干问的量 ↔ 锚点答的量"是否同族）。
_UNIT_FAMILY = {
    "压力": ["mpa", "kpa", "pa", "bar", "兆帕", "千帕"],
    "长度": ["m", "mm", "cm", "km", "米", "毫米", "厘米"],
    "面积": ["m2", "m²", "mm2", "mm²", "cm2", "cm²", "平方米", "平方毫米", "平方厘米"],
    "体积": ["m3", "m³", "立方米"],
    "时间": ["h", "min", "分钟", "小时", "秒", "天", "s"],
    "质量": ["kg", "t", "吨", "千克", "mg", "g"],
    "浓度": ["%", "‰", "mg/m3", "mg/m³", "ppm"],
    "温度": ["℃", "c", "°c"],
    "功率": ["kw", "w", "千瓦", "kw", "kva", "mva"],
}

#: 题干关键词 → 量纲族。
_QUESTION_FAMILY = [
    ("压力", "压力"), ("压强", "压力"),
    ("长度", "长度"), ("距离", "长度"), ("高度", "长度"), ("管径", "长度"),
    ("面积", "面积"), ("截面", "面积"),
    ("容积", "体积"), ("体积", "体积"), ("储罐容积", "体积"),
    ("时间", "时间"), ("多久", "时间"), ("小时", "时间"), ("分钟", "时间"),
    ("供气", "时间"), ("周期", "时间"),
    ("重量", "质量"), ("质量", "质量"), ("存量", "质量"),
    ("浓度", "浓度"), ("含量", "浓度"),
    ("温度", "温度"),
]


def question_family(question: str):
    """题干问的是哪个量纲族（判不出就返回 None）。"""
    q = question or ""
    for kw, fam in _QUESTION_FAMILY:
        if kw in q:
            return fam
    return None


def unit_family(unit: str):
    """单位属于哪个量纲族。"""
    u = core_unit(unit)
    for fam, us in _UNIT_FAMILY.items():
        if u in us:
            return fam
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="写到新文件（默认原地覆盖）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in TEST_SET.open(encoding="utf-8") if l.strip()]
    mds = {p.stem: p.read_text(encoding="utf-8", errors="replace")
           for p in MD_DIR.glob("*.md")}

    fixed, ambiguous = [], []
    for r in rows:
        if r.get("question_type") != "数值题":
            continue
        doc = r.get("doc_id") or ""
        txt = mds.get(doc, "")
        if not txt:
            continue
        nums = list(r.get("must_include_numbers") or [])
        if isinstance(nums, str):
            nums = [x for x in nums.split(";") if x.strip()]
        if not nums:
            continue
        cands = extract_numbers(txt)

        mapping = {}
        for n in nums:
            if RV.contains_num(txt, n):
                continue                      # 已经可达，不动
            same = [c for c in cands if core_num(c) == core_num(n)]
            # 候选里排除与原锚点归一后等价的（那是同一个东西）
            same = [c for c in same
                    if N.normalize_number(c) != N.normalize_number(n)]
            # 🔴 **单位必须兼容**（原单位是候选单位的前缀）——
            #    否则 "40 MPa" 会被配到 "40 %"（压力配成百分比）
            same = [c for c in same if unit_compatible(n, c)]
            if len(same) == 1:
                cand = same[0]
                # 🔴 **量纲族检查**：题干问"供气时间"、锚点修成 "15 MPa" →
                #    治不了根子，还可能**制造虚假高分**（模型答压力就算命中，
                #    但它根本没回答时间）→ 这类题该**弃题/改题干**，不是改锚点。
                qfam = question_family(r.get("question") or "")
                cfam = unit_family(cand)
                if qfam and cfam and qfam != cfam:
                    ambiguous.append((r["qid"], n,
                                      [f"⚠️ 量纲不符：题干问「{qfam}」，"
                                       f"候选是「{cfam}」→ 建议弃题/改题干"]))
                    continue
                mapping[n] = cand
            elif len(same) > 1:
                ambiguous.append((r["qid"], n, same[:4]))
            else:
                ambiguous.append((r["qid"], n, []))

        if not mapping:
            continue
        new_nums = [mapping.get(n, n) for n in nums]
        r["must_include_numbers"] = new_nums
        # key_points 里与锚点"归一后相等"的项**同步替换**
        kps = r.get("key_points") or []
        new_kps = []
        for kp in kps:
            hit = None
            for old, new in mapping.items():
                if N.normalize_number(kp) == N.normalize_number(old):
                    hit = new
                    break
            new_kps.append(hit if hit else kp)
        r["key_points"] = new_kps
        r["anchor_fix_note"] = "；".join(f"{k}→{v}" for k, v in mapping.items())
        fixed.append((r["qid"], mapping))

    print(f"确定性修复 {len(fixed)} 条；需人工判断 {len(ambiguous)} 个锚点\n")
    for qid, mp in fixed:
        print(f"  [fix] {qid}: " + "；".join(f"{k!r}→{v!r}" for k, v in mp.items()))
    for qid, n, c in ambiguous:
        print(f"  ⚠️  {qid}: {n!r} → 候选 {c or '（无）'}")

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
    bak = TEST_SET.with_suffix(".jsonl.bak_anchorfix")
    shutil.copy2(TEST_SET, bak)
    with TEST_SET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {TEST_SET.name}（备份 {bak.name}）")


if __name__ == "__main__":
    main()
