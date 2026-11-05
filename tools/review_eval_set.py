#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**题集自动核对** —— 把「人工核对 306 条」压成「人工核对 N 条可疑题」。

🔴 **为什么需要这个工具**：
README「下一步」第 1 项一直是"人工核对题集（唯一挡住结论可信的一步）"，
但 306 条逐条人读的成本很高。→ 折中做法：
**机器先把能机械判的判完，只把机器判不了 / 判出问题的挑出来给人看**。

🔴 **本工具比 `check_test_set.py` 多做了一件关键的事**：
**在原文里核对锚点，而且核对到"条款级"** ——
`check_test_set.py` 只在 `gold_answer` 里查 key_point 可达性，
而 `gold_answer` 本身可能就是错的（记忆里的 F-02：**锚点挂错条款**是最大一类问题）。
→ 这里用 `data/chunks/clause.jsonl` 的**条款级原文**核对：
   锚点必须出现在**该题指向的那个条款**里，否则就是挂错了。

核对项分三类：
* **溯源**（锚点是否真在原文条款里）
* **元数据自洽**（doc_id / clause_no / 题型字段）
* **题型专项**（版本陷阱题的唯一正答、拒答题的主体确实不存在）

产出：
* `data/eval/review_report.json` —— 逐题结论
* `data/eval/review_queue.md` —— **人工核对队列**（按严重度排序）
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import versioning as V  # noqa: E402

MD = ROOT / "data" / "md"
CHUNKS = ROOT / "data" / "chunks" / "clause.jsonl"
MANIFEST = ROOT / "data" / "manifest.json"

#: 严重度 → 排序权重（人工先看最严重的）
SEVERITY_ORDER = {"likely_bad": 0, "needs_human": 1, "auto_pass": 2}


def norm(text):
    """匹配用归一化：去空白 + 统一标点 + 全角转半角。"""
    if not text:
        return ""
    t = re.sub(r"\s+", "", text)
    table = str.maketrans({
        "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
        "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "－": "-", "～": "~", "％": "%", "－": "-",
        "–": "-", "—": "-", "−": "-", "﹣": "-",
    })
    return t.translate(table).lower()


def load_clause_index():
    """(doc_id, clause_no) → 该条款的原文。"""
    idx = collections.defaultdict(list)
    if not CHUNKS.exists():
        return idx
    for line in open(CHUNKS, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("clause_no"):
            idx[(c["doc_id"], str(c["clause_no"]))].append(c.get("text") or "")
    return idx


def load_doc_text():
    """doc_id → 全文（用于没有条款号的题）。"""
    out = {}
    if MD.exists():
        for p in MD.glob("*.md"):
            out[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def contains(haystack, needle):
    """归一化后的子串包含。"""
    return norm(needle) in norm(haystack)


def compress(text):
    """压缩掉所有分隔符（位号验证用：T-9001 → T9001）。"""
    return re.sub(r"[\s\-_/·、,，:：]", "", text or "")


def subject_in_corpus(subject, doc_text):
    """主体在语料里是否**真的不存在**（拒答题的前提）。

    🔴 **双口径**：位号在原文轨里可能写成 `T-9001`，在归一化轨里可能写成 `T9001`
    —— 只查一种形态等于只查了一半（记忆里的 K-05 类坑）。
    """
    if not subject:
        return None, []
    s_c = compress(subject)
    hits = []
    for did, text in doc_text.items():
        if subject in text or (s_c and s_c in compress(text)):
            hits.append(did)
    return (not hits), hits


def scope_docs(rec):
    """这道题的答案可能来自**哪些文档**。

    🔴 跨文档推理题有 `related_doc_ids`（另一份文档），
    只核对 `doc_id` 会把"锚点在另一份文档里"误判成不可达（实测 17/25 条）。
    """
    out = []
    if rec.get("doc_id"):
        out.append(rec.get("doc_id"))
    for d in rec.get("related_doc_ids") or []:
        if d not in out:
            out.append(d)
    exp = rec.get("expect_doc_id")
    if exp and exp not in out:
        out.append(exp)
    return out


def check(rec, clause_idx, doc_text, manifest_by_id):
    """返回 (verdict, [问题列表])。

    verdict ∈ {"auto_pass", "needs_human", "likely_bad"}
    """
    problems = []
    qtype = rec.get("question_type")
    doc_id = rec.get("doc_id")
    clause_no = rec.get("clause_no")
    kps = rec.get("key_points") or []
    nums = rec.get("must_include_numbers") or []

    # ---- 0. 元数据自洽 ----------------------------------------------------
    if doc_id and doc_id not in manifest_by_id:
        problems.append(f"doc_id 不在 manifest：{doc_id}")
    if qtype == "数值题" and not nums:
        problems.append("数值题缺 must_include_numbers（数值保真判据无分母）")
    if qtype == "事实题" and not kps:
        problems.append("事实题缺 key_points（key_point_coverage 分母为 0）")

    # ---- 1. 拒答题：核对的是"主体真的不存在"，不是锚点 -------------------
    if qtype == "拒答题":
        # 🔴 拒答题的 key_point 是**判分标准**（"有没有声明查不到"），
        #    本来就不指向原文 —— 要求它字面可达是逻辑错误（K-13）。
        if rec.get("key_point_source") != "refusal_criteria":
            problems.append(
                f"拒答题的 key_point_source 是 "
                f"{rec.get('key_point_source')}（应为 refusal_criteria）")
        subj = rec.get("equipment_id")
        if not subj:
            problems.append("拒答题缺 equipment_id（主体判定不成立）")
        else:
            absent, hits = subject_in_corpus(subj, doc_text)
            if absent is False:
                problems.append(
                    f"🔴 拒答题的主体 **{subj}** 在语料里其实存在"
                    f"（命中 {hits[:3]}）→ 这不是有效的拒答题")
        if rec.get("gold_answer") and subj and subj in (rec.get("gold_answer") or ""):
            problems.append("拒答题的主体出现在 gold_answer 里")
        return ("likely_bad" if problems else "auto_pass"), problems

    # ---- 2. 溯源：锚点必须真在**该条款**里 --------------------------------
    docs = scope_docs(rec)
    scope_parts = []
    scope_kind = ""
    clause_hit = False
    for d in docs:
        if clause_no and d == doc_id:
            pieces = clause_idx.get((d, str(clause_no)))
            if pieces:
                scope_parts.append("\n".join(pieces))
                clause_hit = True
                continue
        scope_parts.append(doc_text.get(d, ""))
    scope = "\n".join(scope_parts)
    scope_kind = (f"条款 {clause_no}" if clause_hit
                  else ("全文：" + "/".join(docs) if docs else "无"))

    if not scope:
        problems.append(f"找不到原文：doc_id={doc_id}")
        scope_kind = "无原文"

    def _miss(what, label):
        """锚点不在 scope 时，区分"挂错条款"与"根本不存在"。"""
        if clause_hit:
            # 只在**别的条款/全文**里有 → 挂错了
            for d in docs:
                if contains(doc_text.get(d, ""), what):
                    return (f"锚点挂错条款：{label}「{what[:24]}」不在 {scope_kind}，"
                            f"但在同一文档的其它位置")
            return f"锚点不可达：{label}「{what[:24]}」在该文档里找不到"
        return f"锚点不可达：{label}「{what[:24]}」在 {scope_kind} 里找不到"

    for kp in kps:
        if not contains(scope, kp):
            problems.append(_miss(kp, "key_point"))
    for nm in nums:
        if not contains(scope, nm):
            problems.append(_miss(nm, "数值"))

    # ---- 2. 题干 ↔ 锚点的量纲一致性 --------------------------------------
    if qtype == "数值题" and nums:
        q = rec.get("question") or ""
        units_in_q = set()
        for u in ("MPa", "kPa", "Pa", "kg", "t", "吨", "℃", "mm", "cm",
                  "小时", "分钟", "%", "m3", "m³"):
            if u in q:
                units_in_q.add(u)
        units_in_n = set()
        for nm in nums:
            for u in ("MPa", "kPa", "Pa", "kg", "t", "吨", "℃", "mm", "cm",
                      "小时", "分钟", "%", "m3", "m³"):
                if u in nm:
                    units_in_n.add(u)
        if units_in_q and units_in_n and not (units_in_q & units_in_n):
            problems.append(
                f"题干量纲与锚点对不上：题干问 {'/'.join(sorted(units_in_q))}，"
                f"锚点是 {'/'.join(sorted(units_in_n))}")

    # ---- 3. 题型专项 --------------------------------------------------------
    if qtype == "版本陷阱题":
        exp = rec.get("expect_doc_id")
        if not exp:
            problems.append("版本陷阱题缺 expect_doc_id")
        else:
            tm = rec.get("query_time")
            mode = rec.get("time_mode")
            m = manifest_by_id.get(exp)
            if m:
                # 🔴 三态判定：True / False / UNKNOWN（生效日未标注）
                state = V.is_effective(m, tm)
                if state is None:
                    problems.append(
                        f"版本陷阱题的正答文档生效日未标注，"
                        f"在 {mode}/{tm} 口径下**无法判定**（需人工）")
                elif state is not True:
                    problems.append(
                        f"版本陷阱题的正答文档在 {mode}/{tm} 口径下"
                        f"**不是现行有效**（is_effective={state}）")
            else:
                problems.append(f"expect_doc_id 不在 manifest：{exp}")

    if qtype == "拒答题":
        subj = rec.get("equipment_id") or ""
        gold = rec.get("gold_answer") or ""
        # 主体不应该在 gold_answer 里出现
        if subj and subj in gold:
            problems.append(f"拒答题的主体 {subj} 出现在 gold_answer 里")

    # ---- 4. 判定 -----------------------------------------------------------
    hard = [p for p in problems if "挂错条款" in p or "找不到" in p
            or "不在 manifest" in p or "不是现行有效" in p
            or "量纲" in p or "缺" in p]
    soft = [p for p in problems if p not in hard]
    if hard:
        return "likely_bad", hard + soft
    if soft:
        return "needs_human", soft
    return "auto_pass", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/test_set.jsonl")
    ap.add_argument("--out-json", default="data/eval/review_report.json")
    ap.add_argument("--out-md", default="data/eval/review_queue.md")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.questions, encoding="utf-8")
            if l.strip()]
    manifest_by_id = {}
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        docs = man if isinstance(man, list) else man.get("documents", [])
        for d in docs:
            manifest_by_id[d.get("doc_id")] = d
    clause_idx = load_clause_index()
    doc_text = load_doc_text()

    results = []
    counts = collections.Counter()
    by_type = collections.Counter()
    for r in rows:
        v, probs = check(r, clause_idx, doc_text, manifest_by_id)
        counts[v] += 1
        if v != "auto_pass":
            by_type[(r.get("question_type"), v)] += 1
        results.append({"qid": r.get("qid"), "question_type": r.get("question_type"),
                        "doc_id": r.get("doc_id"), "clause_no": r.get("clause_no"),
                        "verdict": v, "problems": probs,
                        "question": r.get("question")})

    n = len(rows)
    print(f"题数 {n}")
    for v in ("auto_pass", "needs_human", "likely_bad"):
        print(f"  {v:<12} {counts[v]:>4}  ({counts[v]/n:.1%})")
    print("\n可疑题按题型：")
    for (t, v), c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {v:<12} {c}")

    (ROOT / args.out_json).write_text(
        json.dumps({"n": n, "counts": dict(counts), "results": results},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 人工核对队列 ------------------------------------------------------
    queue = sorted([x for x in results if x["verdict"] != "auto_pass"],
                   key=lambda x: (SEVERITY_ORDER[x["verdict"]],
                                  x["question_type"] or "", x["qid"]))
    L = ["# 人工核对队列", "",
         f"生成脚本：`tools/review_eval_set.py`",
         f"题集：{args.questions}（{n} 条）", "",
         "| 结论 | 条数 | 说明 |", "|---|---|---|",
         f"| `auto_pass` | {counts['auto_pass']} | 机器核对项全过 |",
         f"| `needs_human` | {counts['needs_human']} | 有机器判不了/判软的疑点 |",
         f"| `likely_bad` | {counts['likely_bad']} | **机器判出明确问题**，建议直接改题 |",
         "", f"→ **人工只需要看下面 {len(queue)} 条**（而不是全部 {n} 条）。", "",
         "---", ""]
    for x in queue:
        L.append(f"### `{x['qid']}` · {x['question_type']} · `{x['verdict']}`")
        L.append("")
        L.append(f"- 题干：{x['question']}")
        L.append(f"- 出处：`{x['doc_id']}` 条款 `{x['clause_no']}`")
        for p in x["problems"]:
            L.append(f"- 🔴 {p}")
        L.append("")
    (ROOT / args.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n-> {args.out_json}")
    print(f"-> {args.out_md}")


if __name__ == "__main__":
    main()
