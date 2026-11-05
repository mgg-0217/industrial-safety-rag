#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**离线**模拟「代码级归因兜底」的效果（零 API 调用）。

🔴 **为什么可以离线模拟**：
代码级兜底**只改渲染层**，不改模型输出 ——
「模型说了什么」与「用户看到什么」是两个独立的事。
于是可以用**已落盘批次** + **chunk 库**离线复现兜底，
得到**完全配对**的对照（同一句话、同一个检索结果），
差异只来自兜底本身。→ 不烧额度、不受模型非确定性干扰。

🔴 **三个口径必须分开报**（本工具的核心价值）：

| 口径 | 问的问题 | 怎么算 |
|---|---|---|
| **A 模型自身** | 模型自己有没有把通用条款当成主体的规定？ | `verdict(模型原始答案)` |
| **B 最终输出** | **用户看到的**答案里有没有把通用条款当成主体的规定？ | `verdict(模型答案 + 兜底声明)` |
| **C 归因保障率** | 用户看到的答案里有没有**明确的适用性说明**？ | 边界声明 ∪ 缺失声明 ∪ 代码兜底 |

⚠️ **A 与 B 的差值 = 兜底的价值上限**，
但**它不等于"模型学会了"** —— 代码保证了输出安全，
不保证模型行为改变（K-55 的教训：判定分类的计数不能单独作为证据）。

用法：
    python tools/simulate_attribution_fallback.py \
        --answers eval_runs/20261105_1123_..._sys2_fixed.answers.jsonl \
        --chunks data/chunks/fixed.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import answering as AN     # noqa: E402
import scoring as SC      # noqa: E402


def load_chunks(path: pathlib.Path):
    """chunk_id → chunk（只要正文，用于"召回块里有没有提到主体"）。"""
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        out[c["chunk_id"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True,
                    help="批次的 *.answers.jsonl")
    ap.add_argument("--chunks", default="data/chunks/fixed.jsonl")
    ap.add_argument("--out", default="", help="把逐题明细写到哪（可选）")
    ap.add_argument("--show", type=int, default=6,
                    help="打印多少条'兜底改变了判定'的样本")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.answers, encoding="utf-8")
               if l.strip()]
    chunk_map = load_chunks(ROOT / args.chunks)
    if not chunk_map:
        raise SystemExit(f"chunk 库为空或不存在：{args.chunks}")

    n = len(records)
    va = collections.Counter()      # 口径 A：模型自身
    vb = collections.Counter()      # 口径 B：最终输出
    fb_count = 0                    # 代码兜底追加声明次数
    poll_count = 0                  # 划除的污染句总数
    poll_q = 0                      # 有污染句的题数
    residual = 0                    # 输出层（划线之外）残留的污染句
    mentioned = 0                   # 召回块里提到主体的题数
    rows = []

    for r in records:
        subject = r.get("equipment_id")
        ans = r.get("answer") or ""
        cids = r.get("ctx_chunk_ids") or []
        chunks = [chunk_map[c] for c in cids if c in chunk_map]
        if not chunks:
            # 🔴 召回块没对上 → 不能瞎判（宁可少算也不能瞎算）
            continue

        hit = AN.chunks_mention_subject(chunks, subject)
        if hit:
            mentioned += 1

        need, fb, pollution = AN.attribution_fallback(ans, chunks, subject)
        if need:
            fb_count += 1
        if pollution:
            poll_q += 1
            poll_count += len(pollution)

        # 用户看到的版本：污染句划掉 + 末尾声明
        marked = AN.mark_pollution(ans, pollution) if pollution else ans
        tail = ""
        if fb:
            tail += fb
        if pollution:
            tail += ("\n\n" if tail else "") + AN.POLLUTION_NOTICE.format(
                n=len(pollution), subject=subject)
        final_text = marked + (("\n\n" + tail) if tail else "")

        verdict_a, reason_a = SC.refusal_judgement(ans, subject=subject)
        # 🔴 口径 B = **用户实际读到的有效内容**：
        #    划掉的部分（系统标注"不予采信"）不参与判定，
        #    否则判分器还会看到被划掉的污染句 → B 与 A 恒等，指标失去意义。
        effective = SC.strip_marked_pollution(final_text)
        verdict_b, reason_b = SC.refusal_judgement(effective, subject=subject)

        va[verdict_a] += 1
        vb[verdict_b] += 1

        # 输出层残留：把划掉的部分剔掉之后，还有没有污染句
        residual_here = len(SC.find_pollution_sentences(
            SC.strip_marked_pollution(final_text), subject))
        residual += residual_here

        # 口径 C：最终答案里有没有适用性说明
        has_boundary, _ = SC.has_boundary_declaration(final_text, subject)
        has_absence, _ = SC.has_subject_absence_statement(final_text, subject)
        guaranteed = has_boundary or has_absence or bool(need)

        rows.append({
            "qid": r.get("qid"), "subject": subject,
            "mechanism": r.get("refusal_mechanism"),
            "ctx_mentions_subject": hit,
            "fallback_appended": bool(need),
            "n_pollution": len(pollution),
            "pollution": [p[0] for p in pollution],
            "verdict_model": verdict_a, "reason_model": reason_a,
            "verdict_final": verdict_b, "reason_final": reason_b,
            "residual_pollution": residual_here,
            "attribution_guaranteed": guaranteed,
        })

    # ---- 汇总 --------------------------------------------------------------
    print(f"题目数：{n}   召回块提到主体的：{mentioned}")
    print(f"代码兜底：追加声明 {fb_count} 次，划除污染句 {poll_count} 句"
          f"（涉及 {poll_q} 题）\n")

    def rate(c, k):
        return c.get(k, 0) / n if n else 0.0

    print(f"{'verdict':<14}{'A 模型自身':>12}{'B 最终输出':>12}{'Δ':>8}")
    for k in sorted(set(va) | set(vb)):
        print(f"{k:<14}{va.get(k,0):>8} ({rate(va,k):.3f})"
              f"{vb.get(k,0):>8} ({rate(vb,k):.3f})"
              f"{vb.get(k,0)-va.get(k,0):>+8}")

    fa_a, fa_b = va.get("false_answer", 0), vb.get("false_answer", 0)
    print(f"\n严格误答率  A 模型自身：{fa_a}/{n} = {fa_a/n:.4f}")
    print(f"严格误答率  B 最终输出：{fa_b}/{n} = {fa_b/n:.4f}"
          f"   （兜底把它压低 {fa_a-fa_b} 条）")

    # 口径 C
    n_guaranteed = sum(1 for x in rows if x["attribution_guaranteed"])
    n_bd_a = va.get("bounded", 0)
    print(f"\n归因保障率（C，输出里有适用性说明）：{n_guaranteed}/{n} = {n_guaranteed/n:.4f}")
    print(f"  其中模型自己做（bounded）：{n_bd_a}   代码兜底补上：{fb_count}")
    print(f"\n输出层残留污染（划线之外还有的污染句）：{residual} 句"
          f"  ← 用户真正可能被误导的部分")

    # ---- 变化样本 ----------------------------------------------------------
    changed = [x for x in rows if x["verdict_model"] != x["verdict_final"]]
    print(f"\n兜底改变判定的题：{len(changed)} 条（前 {args.show} 条）")
    for x in changed[: args.show]:
        print(f"\n  [{x['qid']}] 主体 {x['subject']}  机制 {x['mechanism']}")
        print(f"    A 模型自身：{x['verdict_model']}  ({x['reason_model'][:90]})")
        print(f"    B 最终输出：{x['verdict_final']}  ({x['reason_final'][:90]})")

    unchanged_fa = [x for x in rows
                    if x["verdict_model"] == "false_answer"
                    and x["verdict_final"] == "false_answer"]
    print(f"\n🔴 **兜底没能救回的误答**：{len(unchanged_fa)} 条"
          f"（模型给主体下了结论，追加声明不改变这一点）")

    if args.out:
        out = ROOT / args.out
        with out.open("w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
