#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**用 LLM 从原文重新提炼事实题的 key_point**（替换"按 clause_no 机械切片"的碎片）。

🔴 **要解决的根因**：
M2 阶段事实题的 `key_points` 是**候选池按 `clause_no` 切片**的产物 ——
切点落在词中间，于是 key_point 长这样：

    「七,事故调查7.1事故调查市市场监督管」   ← 停在"管"字
    「九,应急保障1.」
    「3.2事故预防措施3.」

这些片段在原文里**字面不可达**（甚至跨了两个条款）→
`key_point_coverage` 的分母是废的，**"事实题严格口径 0.168"这个数字不可信**。

🔴 **本工具的做法**（用我们自己的检索器给 LLM 找证据）：
1. 用 BM25 按**题干**检索 top-k 原文块（`src/bm25.py` + `src/retrieval.py`）；
2. 连同题干交给 LLM，要求输出 1–3 个**关键信息点**，
   **每个必须是给定证据里的逐字片段**；
3. **机器校验**：每个 kp 必须字面出现在它所依据的证据块里，否则丢弃；
4. 回写题集。

🔴 **为什么这样更好**：
* kp 是从**相关原文**里提炼的，不是从切片边界切出来的 → 字面可达 ✓
* 检索用的是**我们自己的检索器** → 顺带检验了检索在"题库生成"上的可用性；
* 校验是机器做的 → 不会出现"LLM 说它引用了原文但其实没有"。

用法：
    python tools/refine_key_points_llm.py --limit 20 --workers 3   # 冒烟
    python tools/refine_key_points_llm.py --workers 3             # 全量
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bm25 as BM      # noqa: E402
import retrieval as RT  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
CHUNKS = ROOT / "data" / "chunks" / "fixed.jsonl"

SYSTEM = (
    "你是工业安全规程评测集的标注员。你的任务：从给定的【资料片段】里，"
    "提炼出回答该问题所必需的**关键信息点**。\n"
    "规则：\n"
    "1. 输出 1–3 个关键信息点，每个是**资料片段里的逐字片段**（不得改写、"
    "不得拼接不相邻的句子、不得补字）。\n"
    "2. 只写资料里**确实写明**的内容；资料没写的不要推测。\n"
    "3. 如果资料不足以回答该问题，输出空数组 []。\n"
    "4. 只输出 JSON 数组，每个元素是一个字符串，不要任何其它文字。"
)

PROMPT = (
    "【资料片段】\n{evidence}\n\n"
    "【问题】\n{question}\n\n"
    "请输出 JSON 数组（1–3 个逐字关键信息点）："
)


def load_chunks(path=CHUNKS):
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out


def build_retriever(chunks):
    """用项目自己的 BM25 做检索（不给答案，只给证据）。

    🔴 `use_vector=False`：本机没有预训练句向量，向量路实测是**负贡献**
    （PRD 4.6 的 L4 降级）→ 这里只用 BM25 路。
    """
    return RT.Retriever(chunks)


def extract_json_array(text):
    """从模型输出里抠出 JSON 数组（模型可能带解释文字）。"""
    if not text:
        return []
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        # 容错：逐条抠字符串
        return [s.strip() for s in re.findall(r'"([^"]{4,200})"', m.group(0))]
    return [str(x).strip() for x in arr if isinstance(x, (str, int, float))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--only-type", default="事实题")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    chunks = load_chunks()
    if not chunks:
        raise SystemExit("chunk 库为空")
    retr = build_retriever(chunks)

    rows = [json.loads(l) for l in TEST_SET.open(encoding="utf-8") if l.strip()]
    targets = [r for r in rows if r.get("question_type") == args.only_type]
    if args.limit:
        targets = targets[:args.limit]
    print(f"待精修：{len(targets)} 条（题型 {args.only_type}）")

    import llm_client

    def work(r):
        q = r.get("question") or ""
        try:
            hits = retr.retrieve(q, top_k=args.top_k, version_filter=False,
                                 use_vector=False)
        except Exception as e:      # 检索失败 → 不改这题
            return r, None, f"检索失败：{e}"
        evid = []
        for idx, _score in hits:
            if 0 <= idx < len(chunks):
                evid.append((chunks[idx].get("text") or "")[:1200])
        if not evid:
            return r, None, "检索无结果"
        evidence = "\n\n---\n\n".join(evid)
        prompt = PROMPT.format(evidence=evidence, question=q)
        try:
            raw = llm_client.call_llm(prompt, system=SYSTEM, temperature=0.0,
                                      max_tokens=900)
        except Exception as e:
            return r, None, f"LLM 失败：{e}"
        kps = extract_json_array(raw)
        # 🔴 **机器校验**：每个 kp 必须逐字出现在给它的证据里
        ok = []
        for k in kps:
            k = k.strip()
            if len(k) < 4:
                continue
            if k in evidence:
                ok.append(k)
            else:
                # 允许首尾空白差异
                if any(k in e for e in evid):
                    ok.append(k)
        if not ok:
            return r, None, "没有通过字面校验的 kp"
        return r, ok, None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in targets}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r, kps, err = fut.result()
            done += 1
            if done % 10 == 0 or err:
                print(f"  [{done}/{len(targets)}] {r.get('qid')} "
                      f"{'ok:' + str(len(kps)) if kps else 'skip:' + str(err)}")
            if kps:
                r["key_points"] = kps[:4]
                r["key_point_source"] = "llm_from_evidence"
                r.pop("review_note", None)
            results.append((r.get("qid"), bool(kps), err))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n成功精修 {n_ok}/{len(results)}")

    if args.dry_run:
        print("[dry-run] 未写盘")
        return
    with TEST_SET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {TEST_SET.name}")
    if args.out:
        (ROOT / args.out).write_text(
            json.dumps([{"qid": q, "ok": o, "err": e} for q, o, e in results],
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
