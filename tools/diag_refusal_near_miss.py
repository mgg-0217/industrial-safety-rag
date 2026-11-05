# -*- coding: utf-8 -*-
"""拒答主体的**语义近邻**检测 —— 精确匹配的补漏。

## 为什么需要

`build_refusal_pool_multitype.py` 用**精确子串匹配**验证主体不在语料里。
这是必要条件，但**不是充分条件**：

* 主体「液氮」在语料里查不到 → 通过验证
* 但语料里可能有「氮气」「液氨」→ 说的是**同类介质**，
  模型完全可能把氮气的规定套到液氮上 → 题目失去区分度

* 主体「加氢反应器」查不到 → 通过
* 但语料里有「加氢装置」「反应器」→ 模型可能说
  "参照加氢装置的检修要求" → 这算不算误答？取决于判分口径

这类**同义/近义存在**是精确匹配**看不见**的，必须单独查。

## 方法

对每个主体，在语料的归一化文本里找**长度相同**的所有 n-gram，
按**字符 bigram 的 Jaccard 相似度**排序，取最相近的几个 ——

* 相似度很高（≥0.6）→ 极可能是同一事物换了说法 → **主体选得不好**
* 相似度中等（0.3~0.6）→ 同领域，需人工判断是否构成"可套用"
* 相似度低（<0.3）→ 主体确实离语料远

输出到 `data/eval/refusal_near_miss.md`，供人工核对。
"""
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from normalize import build_chunk_text  # noqa: E402

POOL = ROOT / "data" / "eval" / "refusal_pool_multitype.json"
OUT = ROOT / "data" / "eval" / "refusal_near_miss.md"
MD_DIR = ROOT / "data" / "md"

HIGH = 0.6
MID = 0.3


def bigrams(s):
    s = s.strip()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    subjects = []
    for mech in ("fake_equipment", "fake_medium", "fake_process"):
        for it in pool[mech]:
            subjects.append((mech, it["subject"]))

    # 语料归一化文本
    texts = []
    for p in sorted(MD_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        _orig, norm = build_chunk_text(raw)
        texts.append(norm)
    total = sum(len(t) for t in texts)
    print(f"语料：{len(texts)} 份，{total} 字符")
    print(f"待检主体：{len(subjects)} 个")

    # 按长度建 n-gram 索引（每个长度只扫一遍语料）
    lengths = sorted({len(s) for _m, s in subjects})
    idx = defaultdict(set)
    for L in lengths:
        for t in texts:
            for i in range(len(t) - L + 1):
                idx[L].add(t[i:i + L])
        print(f"  长度 {L} 的 n-gram 种类：{len(idx[L])}")

    lines = []
    lines.append("# 拒答主体的语义近邻检测")
    lines.append("")
    lines.append(f"> 自动生成（`tools/diag_refusal_near_miss.py`）。"
                 f"语料 {total} 字符，主体 {len(subjects)} 个。")
    lines.append(">")
    lines.append(f"> 精确子串匹配已经确认这些主体**在语料里查不到**。")
    lines.append("> 本表查的是**精确匹配看不见**的东西：同义/近义存在"
                 "（「液氮」查不到，但语料里可能有「氮气」）。")
    lines.append(">")
    lines.append(f"> 判据：相似度 ≥ {HIGH} = 极可能是同一事物换了说法"
                 f"（主体选得不好）；{MID}~{HIGH} = 同领域，需人工判断；"
                 f"< {MID} = 主体确实离语料远。")
    lines.append("")

    n_high = n_mid = 0
    rows = []
    for mech, subj in subjects:
        L = len(subj)
        sb = bigrams(subj)
        scored = []
        for cand in idx.get(L, ()):
            s = jaccard(sb, bigrams(cand))
            if s >= MID:
                scored.append((s, cand))
        scored.sort(reverse=True)
        if not scored:
            rows.append((mech, subj, [], "—"))
            continue
        top = scored[:3]
        if top[0][0] >= HIGH:
            n_high += 1
            flag = "🔴 极近"
        elif top[0][0] >= MID:
            n_mid += 1
            flag = "🟡 同域"
        else:
            flag = "✅ 离得远"
        rows.append((mech, subj, top, flag))

    lines.append(f"## 汇总")
    lines.append("")
    lines.append(f"| 判定 | 主体数 |")
    lines.append(f"|---|---|")
    lines.append(f"| 🔴 极近（≥{HIGH}，疑似同一事物） | **{n_high}** |")
    lines.append(f"| 🟡 同域（{MID}~{HIGH}） | **{n_mid}** |")
    lines.append(f"| ✅ 离得远（<{MID}） | {len(subjects) - n_high - n_mid} |")
    lines.append("")

    for mech, subj, top, flag in rows:
        if flag == "✅ 离得远":
            continue
        lines.append(f"### {flag} `{subj}`（{mech}）")
        for s, cand in top:
            lines.append(f"- `{cand}` — 相似度 **{s:.2f}**")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n🔴 极近 {n_high} 个 / 🟡 同域 {n_mid} 个 / "
          f"✅ 离得远 {len(subjects) - n_high - n_mid} 个")
    print("=== 极近的主体（优先人工核对）===")
    for mech, subj, top, flag in rows:
        if flag.startswith("🔴"):
            print(f"  {subj:<16s} → {top[0][1]:<16s} {top[0][0]:.2f}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
