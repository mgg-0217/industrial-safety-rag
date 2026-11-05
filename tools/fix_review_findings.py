#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 `tools/review_eval_set.py` 的核对结论**自动修复题集的溯源元数据**。

🔴 **要修的是什么**：
`review_eval_set.py` 报"锚点挂错条款"的题，答案本身**大多是对的**（锚点在原文中），
错的是**溯源元数据** —— `clause_no` 指向了一个**空的章节标题**或**错误的兄弟章节**。

实测根因（预案型 / 表格型文档）：
```
clause_no = "五、应急救援的初期处理措施"   ← 这个 chunk 只有 14 个字（只有标题）
kp        = "(二)压力容器安全事故应急救"      ← 实际在"六、…分项应急救援措施"下面
```
→ **答案没错，溯源错了**。修法：把 `clause_no` 改成锚点**实际所在**的条款。

🔴 **为什么必须修**：
`clause_no` 决定评测的 **gold 区间**（R@5 的命中判定、`content_coverage` 的分母）。
溯源不准 → 召回评测的分母是错的 → **召回数字不可信**。

修不了的（交给人工）：
* 锚点在同文档**多个**条款里出现，且无法判定该挂哪个（多命中且距离远）
* 锚点在原文里根本找不到（字符形态不一致，如全角破折号）
* 题干问的量与锚点量纲不符（题干本身错了）

用法：
    python tools/fix_review_findings.py [--dry-run]
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

import review_eval_set as RV  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"


def load_clause_index_with_pos():
    """(doc_id, clause_no) → [(text, char_start), ...]"""
    idx = {}
    p = RV.CHUNKS
    if not p.exists():
        return idx
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("clause_no"):
            idx.setdefault((c["doc_id"], str(c["clause_no"])), []).append(
                (c.get("text") or "", c.get("char_start") or 0))
    return idx


def locate(idx, doc_id, needle):
    """锚点实际出现在哪些条款里 → [(clause_no, char_start, text)]"""
    hits = []
    for (d, cn), pieces in idx.items():
        if d != doc_id:
            continue
        text = "\n".join(t for t, _ in pieces)
        if RV.contains(text, needle):
            hits.append((cn, min(s for _, s in pieces), text))
    hits.sort(key=lambda x: x[1])
    return hits


def flat_lines(source):
    """把原文按行切开，逐行 flat，返回 [(原文行, flat 起, flat 止)]。

    🔴 `flat` 是 1:1 归一化（去空白 + 统一标点 + 小写），所以
    **flat 后的行边界与原文行边界一一对应** → 可以把 flat 位置映射回原文行。
    """
    out = []
    pos = 0
    for line in (source or "").split("\n"):
        f = RV.norm(line)
        out.append((line, pos, pos + len(f)))
        pos += len(f)
    return out


def expand_kp_to_line(kp, source):
    """把截断的 key_point 扩展成**原文里的完整一行**。

    🔴 **要修的现象**：候选池按 `clause_no` 切片时把 key_point 切成了半截
    （「七,事故调查7.1事故调查市市场监督管」—— 停在"管"字），
    这样的片段在原文里**字面不可达** → key_point_coverage 的分母是废的。

    解法：用归一化定位它在原文中的行，取整行原文作为新的 kp
    （仍是原文片段 → 字面可达，且人可读）。
    """
    f = RV.norm(kp)
    if not f:
        return None
    lines = flat_lines(source)
    for line, a, b in lines:
        if not line.strip():
            continue
        if f in RV.norm(line):
            s = line.strip()
            return s if len(s) >= 4 else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(TEST_SET))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.test_set)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    idx = load_clause_index_with_pos()
    doc_text = RV.load_doc_text()

    # 原始 clause_no 的字符位置（多命中时用来就近选择）
    orig_pos = {}
    for (d, cn), pieces in idx.items():
        orig_pos[(d, cn)] = min(s for _, s in pieces)

    fixed, unfixable, expanded = [], [], []
    for r in rows:
        qtype = r.get("question_type")
        if qtype in ("拒答题", "跨文档推理题"):
            continue
        doc_id = r.get("doc_id")
        old_cn = r.get("clause_no")
        if not doc_id or not old_cn:
            continue
        scope = "\n".join(t for t, _ in idx.get((doc_id, str(old_cn)), []))
        if not scope:
            continue
        anchors = list(r.get("must_include_numbers") or [])
        anchors += [k for k in (r.get("key_points") or [])
                    if k not in anchors]
        if not anchors:
            continue
        # 哪些锚点不在原条款里
        missing = [a for a in anchors if not RV.contains(scope, a)]
        if not missing:
            continue

        # ---- 第 1 步：锚点在**任何条款 chunk** 里都找不到 → 扩展成原文整行 --
        # 🔴 判据要基于「条款 chunk」而不是「全文」：
        #    截断的 kp 往往**跨了两个条款**（"七,事故调查7.1事故调查市市场监督管"
        #    跨了"七、事故调查"和"7.1 事故调查"两段）→ 全文里有、条款里没有。
        src = doc_text.get(doc_id, "")
        hits_all = [locate(idx, doc_id, a) for a in missing]
        if not any(hits_all):
            new_kps, ok = [], True
            for kp in (r.get("key_points") or []):
                if RV.contains(src, kp):
                    new_kps.append(kp)
                    continue
                e = expand_kp_to_line(kp, src)
                if e:
                    new_kps.append(e)
                    expanded.append((r["qid"], kp, e))
                else:
                    ok = False
                    new_kps.append(kp)
            if ok and new_kps:
                r["key_points"] = new_kps
                r["review_note"] = "key_point 由截断片段扩展为原文整行"
                continue
            unfixable.append((r, f"锚点在原文里找不到且无法扩展：{missing[:2]}"))
            continue

        # ---- 第 2 步：锚点在全文里有，但不在本条款 → 溯源修正 --------------
        if qtype == "数值题":
            # 数值题**不自动改 clause_no**：
            # "题干问的量 ↔ 锚点"是否对应，机器判不了 ——
            # 实测 150 mm 是"围堰"的数值，而题干问的是"专用仓库和人员管理"，
            # 改溯源解决不了**题干本身写错**。
            unfixable.append(
                (r, f"数值题溯源条款存疑（锚点在其它条款）：{missing[:2]}"))
            continue

        # 逐个找它们实际在哪个条款
        found_cns = []
        for a in missing:
            hits = locate(idx, doc_id, a)
            if hits:
                found_cns.append(hits)
            else:
                found_cns.append(None)
        cn_votes = {}
        for hits in found_cns:
            if not hits:
                continue
            if len(hits) == 1:
                cn_votes[hits[0][0]] = cn_votes.get(hits[0][0], 0) + 1
            else:
                # 多命中：取**离原 clause_no 最近**的那个
                base = orig_pos.get((doc_id, str(old_cn)), 0)
                best = min(hits, key=lambda h: abs(h[1] - base))
                cn_votes[best[0]] = cn_votes.get(best[0], 0) + 1
        if not cn_votes:
            unfixable.append((r, f"锚点在原文里找不到：{missing[:2]}"))
            continue
        # 多数票
        new_cn = max(cn_votes, key=cn_votes.get)
        if new_cn == old_cn:
            unfixable.append((r, "锚点挂到了同一个条款（改不动）"))
            continue
        r["clause_no"] = new_cn
        r["review_note"] = (f"溯源条款由 {old_cn} 自动更正为 {new_cn}"
                            f"（锚点实际所在条款）")
        fixed.append((r["qid"], old_cn, new_cn))

    print(f"自动修正溯源 {len(fixed)} 条；扩展截断 kp {len(expanded)} 条；"
          f"仍需人工 {len(unfixable)} 条\n")
    for qid, old, new in fixed:
        print(f"  [fix] {qid}: {str(old)[:38]} → {str(new)[:38]}")
    for qid, old, new in expanded:
        print(f"  [cut] {qid}: kp「{old[:26]}」→ 整行「{new[:34]}」")
    print()
    for r, why in unfixable:
        print(f"  ⚠️  {r['qid']}（{r['question_type']}）：{why}")
        print(f"      题干：{r['question'][:80]}")

    if args.dry_run:
        print("\n[dry-run] 未写盘")
        return
    bak = path.with_suffix(".jsonl.bak_review")
    shutil.copy2(path, bak)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {path.name}（备份 {bak.name}）")


if __name__ == "__main__":
    main()
