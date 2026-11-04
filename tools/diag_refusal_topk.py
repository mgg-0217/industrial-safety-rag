# -*- coding: utf-8 -*-
"""诊断：**为什么加大 top-k 会让拒答误答率上升**。

## 要回答的问题

实测（四组完整 306 题）里有一个反直觉的现象：

| | 拒答误答率 |
|---|---|
| fixed + top-5 | **0.024**（1/42） |
| fixed + top-8 | **0.143**（6/42） |

🔴 **top-k 越大，模型越不愿意说"没找到"** ——
在安全规程问答里，这是最不该发生的失败模式。

## 假说

上下文里东西越多 → **别的设备的条款越多** → 模型更容易把
"某条款里写的限值"套到被问的主体上 → 判成误答。

## 验证方法

逐题对比两个批次的判定结果，把**判定发生变化**的题挑出来，
打印：判定变化、判分理由、答案里出现的条款号/数值。

如果变化题的答案里**确实出现了具体条款号或数值**（而主体本身没有规定），
假说成立。

跑法：
    python tools/diag_refusal_topk.py \
        --a "fixed+top5=eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl" \
        --b "fixed+top8=eval_runs/20261104_1514_generation_retrieval_k8.answers.jsonl"
"""
import argparse
import json
import pathlib
import re
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RUN_DIR = ROOT / "eval_runs"

#: 条款号（第 X 条 / X.X / 标准号）
CLAUSE_RE = re.compile(r"第\s*[一二三四五六七八九十百零〇\d]+\s*条"
                       r"|\b\d+\.\d+(?:\.\d+)?\b"
                       r"|GB\s*\d+|GB/T\s*\d+|AQ\s*\d+|TSG\s*\d+")
#: 具体数值（带单位或百分数）
NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|‰|kPa|KPa|MPa|℃|°C|mg/m3|mg/m³"
                     r"|m3|m³|L|kg|t|mm|cm|m|分钟|小时|天|年)")

#: 判定严重程度（用于排序）
SEVERITY = {"false_answer": 2, "manual": 1, "pass": 0}


def load(path):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / path
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return {r["qid"]: r for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--show", type=int, default=12, help="最多打印几道变化题")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    la, _, pa = args.a.partition("=")
    lb, _, pb = args.b.partition("=")
    ra, rb = load(pa.strip()), load(pb.strip())

    common = sorted(set(ra) & set(rb))
    refusals = [q for q in common
                if (ra[q].get("question_type") == "拒答题")]
    print(f"批次 A = {la.strip()}（{len(ra)} 题）")
    print(f"批次 B = {lb.strip()}（{len(rb)} 题）")
    print(f"拒答题交集 = {len(refusals)} 题\n")

    def counts(recs, qids):
        c = {"pass": 0, "false_answer": 0, "manual": 0}
        for q in qids:
            v = recs[q].get("refusal_verdict")
            if v in c:
                c[v] += 1
        return c

    ca, cb = counts(ra, refusals), counts(rb, refusals)
    print(f"A 判定分布：{ca}")
    print(f"B 判定分布：{cb}\n")

    changed = []
    for q in refusals:
        va = ra[q].get("refusal_verdict")
        vb = rb[q].get("refusal_verdict")
        if va != vb:
            changed.append((q, va, vb))
    changed.sort(key=lambda t: -(SEVERITY.get(t[2], 0) - SEVERITY.get(t[1], 0)))

    print(f"判定发生变化的题：{len(changed)} / {len(refusals)}")
    worse = [c for c in changed
             if SEVERITY.get(c[2], 0) > SEVERITY.get(c[1], 0)]
    print(f"  其中**变差**（pass → manual/false_answer）：{len(worse)}")
    print(f"  变好：{len(changed) - len(worse)}\n")

    # 假说验证：变差的题里，答案是否出现了**具体条款号/数值**
    print("=" * 78)
    for q, va, vb in changed[:args.show]:
        rec = rb[q]
        ans = rec.get("answer") or ""
        clauses = CLAUSE_RE.findall(ans)[:4]
        nums = NUM_RE.findall(ans)[:4]
        print(f"\n[{q}]  {va} → {vb}")
        print(f"  题：{(rec.get('question') or '')[:70]}")
        print(f"  B 的判分理由：{(rec.get('refusal_reason') or '')[:110]}")
        print(f"  答案里的条款号：{clauses or '无'}")
        print(f"  答案里的数值：{nums or '无'}")
        print(f"  答案片段：{ans[:160]}")

    # 统计：变差的题里有多少出现了"别的条款的编号/数值"
    n_with = 0
    for q, va, vb in worse:
        ans = rb[q].get("answer") or ""
        if CLAUSE_RE.search(ans) or NUM_RE.search(ans):
            n_with += 1
    print("\n" + "=" * 78)
    print(f"变差的 {len(worse)} 题里，答案中出现了具体条款号或数值的："
          f"{n_with} / {len(worse)}")
    if worse:
        print("→ 上下文里别的设备的条款越多，模型越容易把它们套到主体上"
              "（假说成立）" if n_with >= len(worse) * 0.6
              else "→ 多数变差的题**没有**引用具体条款/数值，"
                   "假说只解释了一部分，需另找原因")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "a": la.strip(), "b": lb.strip(),
        "n_refusal_common": len(refusals),
        "verdicts_a": ca, "verdicts_b": cb,
        "n_changed": len(changed), "n_worse": len(worse),
        "n_worse_with_clause_or_number": n_with,
        "changed": [{"qid": q, "from": va, "to": vb,
                     "reason_b": rb[q].get("refusal_reason"),
                     "answer_b": (rb[q].get("answer") or "")[:600]}
                    for q, va, vb in changed],
    }
    out = pathlib.Path(args.out) if args.out else \
        RUN_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M')}_diag_refusal_topk.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
