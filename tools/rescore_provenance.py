#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**溯源一致性重算**：把"答对了"和"答对了但引错文档"分开。

🔴 **要解决的混淆**：
`key_point_coverage` 只看"答案里有没有命中 key_point 的字面"，
**不看这份 key_point 出自哪份文档**。
而 27 条事实题的 key_point 来自**别的文档**（多为应急预案之间的通用套话，
例如"特种设备事故应急救援工作坚持'以人为本'的理念"在多份预案里一字不差）。

→ 于是模型**引错文档也能命中 key_point** → 事实题覆盖率被**虚高**，
而"归因污染"这一失败模式被**藏在覆盖率分子里**。

🔴 **本工具给出三个并列口径**（同一批答案、同一份 key_point 列表）：

| 口径 | 含义 |
|---|---|
| `kp_cov_all` | 现行口径：全部 key_point 的命中率（**与已发布数字可比**） |
| `kp_cov_src` | **溯源一致**：只算"逐字出现在出处文档正文里"的 key_point |
| `kp_cov_cross` | 只算"不在出处文档"的 key_point（跨文档套话） |

外加两个**与覆盖率正交**的可观测指标（K-55 的要求：不拿判定分类数当唯一证据）：
* `src_in_ctx`：出处文档是否出现在检索上下文里
* `src_cited`：出处文档的块**是否被引用**（这才是"答对了但引错文档"的直接证据）

⚠️ **口径不可混用**：`kp_cov_src` 的分母比 `kp_cov_all` 小，
**不能**说"溯源一致口径更高/更低"就等于"模型变好/变差"——
它只说明"有多少 key_point 本来就不属于那道题的出处文档"。

用法：
    python tools/rescore_provenance.py --batch eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402

MD_DIR = ROOT / "data" / "md"
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"


def nz(s) -> str:
    """归一化：去空白 + 零宽字符（与 refine 脚本的校验口径保持一致）。"""
    return re.sub(r"[\s　﻿]+", "", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="*.answers.jsonl 路径")
    ap.add_argument("--kp-source", default="testset",
                    choices=("batch", "testset"),
                    help="key_point 来源：batch=批次内嵌的（与 compare_generation 同源，"
                         "数字可直接对照已发布结论）；"
                         "testset=当前题集里的（用于衡量 kp 精修对同一批答案的影响）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = {json.loads(l)["qid"]: json.loads(l)
            for l in TEST_SET.open(encoding="utf-8") if l.strip()}
    mds = {p.stem: nz(p.read_text(encoding="utf-8", errors="replace"))
           for p in MD_DIR.glob("*.md")}

    batch = ROOT / args.batch if not os.path.isabs(args.batch) else pathlib.Path(args.batch)
    recs = [json.loads(l) for l in batch.open(encoding="utf-8") if l.strip()]
    print(f"批次 {batch.name}：{len(recs)} 条")

    per_type = defaultdict(lambda: Counter())
    rows_out = []
    n_no_src = 0

    for rec in recs:
        q = rows.get(rec.get("qid")) or {}
        qt = q.get("question_type") or rec.get("question_type") or "?"
        src_doc = q.get("expect_doc_id") or q.get("doc_id")
        if args.kp_source == "batch":
            # 🔴 用批次内嵌的 key_point（生成时的那一版）
            #    → 与 compare_generation.py 读的是同一份数据，数字可逐字对照
            kps = [d.get("key_point") for d in (rec.get("kp_detail") or [])
                   if d.get("key_point")]
        else:
            kps = q.get("key_points") or []
        answer = rec.get("answer") or ""

        # 出处文档 → 哪些 kp 属于它
        src_text = mds.get(src_doc or "", "")
        in_src, cross = [], []
        for kp in kps:
            (in_src if (nz(kp) and nz(kp) in src_text) else cross).append(kp)

        def cov(subset):
            """返回 ``(软命中数, 严格命中数, 总数)``。
            🔴 严格口径必须用 detail 里的 ``strict`` 字段（**逐字包含**），
            与已发布的"严格口径 0.168"同源；软命中是 `key_point_hits` 的默认口径。
            """
            if not subset:
                return None
            hit, tot, detail = SC.key_point_hits(answer, subset)
            strict = sum(1 for d in detail if d.get("strict"))
            return hit, strict, tot

        c_all = cov(kps)
        c_src = cov(in_src)
        c_cross = cov(cross)

        # 与覆盖率正交的可观测指标
        ctx_docs = rec.get("ctx_docs") or []
        ctx_ids = rec.get("ctx_chunk_ids") or []
        cited = (rec.get("cite_stats") or {}).get("cited") or []
        cited_docs = []
        for i in cited:
            if 1 <= i <= len(ctx_ids):
                cited_docs.append(ctx_ids[i - 1].split("__")[0])
        src_in_ctx = bool(src_doc) and src_doc in ctx_docs
        src_cited = bool(src_doc) and src_doc in cited_docs

        t = per_type[qt]
        t["n"] += 1
        if c_all:
            t["kp_cov_all_hits"] += c_all[0]
            t["kp_cov_all_strict_hits"] += c_all[1]
            t["kp_cov_all_tot"] += c_all[2]
            # 🔴 权威口径（run_generation_eval.py:604）：题目级全命中 = **软命中数 == 总数**
            t["kp_cov_all_fullhit_soft"] += 1 if c_all[0] == c_all[2] else 0
            t["kp_cov_all_fullhit_strict"] += 1 if c_all[1] == c_all[2] else 0
        if c_src:
            t["kp_cov_src_hits"] += c_src[0]
            t["kp_cov_src_strict_hits"] += c_src[1]
            t["kp_cov_src_tot"] += c_src[2]
            t["kp_cov_src_fullhit_soft"] += 1 if c_src[0] == c_src[2] else 0
            t["kp_cov_src_fullhit_strict"] += 1 if c_src[1] == c_src[2] else 0
        if c_cross:
            t["kp_cov_cross_hits"] += c_cross[0]
            t["kp_cov_cross_strict_hits"] += c_cross[1]
            t["kp_cov_cross_tot"] += c_cross[2]
            t["kp_cov_cross_fullhit_soft"] += 1 if c_cross[0] == c_cross[2] else 0
            t["kp_cov_cross_fullhit_strict"] += 1 if c_cross[1] == c_cross[2] else 0
        t["n_kp_total"] += len(kps)
        t["n_kp_in_src"] += len(in_src)
        t["n_kp_cross"] += len(cross)
        t["src_in_ctx"] += 1 if src_in_ctx else 0
        t["src_cited"] += 1 if src_cited else 0
        if not src_doc:
            n_no_src += 1

        rows_out.append({
            "qid": rec.get("qid"), "type": qt, "src_doc": src_doc,
            "kp_total": len(kps), "kp_in_src": len(in_src), "kp_cross": len(cross),
            "cov_all": c_all, "cov_src": c_src, "cov_cross": c_cross,
            "src_in_ctx": src_in_ctx, "src_cited": src_cited,
        })

    def ratio(h, t_):
        return round(h / t_, 4) if t_ else None

    print("\n=== 溯源一致性（按题型）===")
    print("🔴 口径与 `tools/compare_generation.py` 一致：")
    print("   软覆盖 = 命中数/kp总数；严格覆盖 = 逐字命中数/kp总数；全命中 = 题目级全中率")
    print(f"\n{'题型':<12}{'n':>5}{'kp总数':>8}{'源内':>7}{'跨文档':>8}"
          f"{'软all':>9}{'软src':>9}{'软cross':>9}"
          f"{'严all':>9}{'严src':>9}{'全中all':>9}{'全中src':>9}{'源被引':>9}")
    summary = {}
    for qt, t in sorted(per_type.items()):
        print(f"{qt:<12}{t['n']:>5}{t['n_kp_total']:>8}{t['n_kp_in_src']:>7}"
              f"{t['n_kp_cross']:>8}"
              f"{str(ratio(t['kp_cov_all_hits'], t['kp_cov_all_tot'])):>9}"
              f"{str(ratio(t['kp_cov_src_hits'], t['kp_cov_src_tot'])):>9}"
              f"{str(ratio(t['kp_cov_cross_hits'], t['kp_cov_cross_tot'])):>9}"
              f"{str(ratio(t['kp_cov_all_strict_hits'], t['kp_cov_all_tot'])):>9}"
              f"{str(ratio(t['kp_cov_src_strict_hits'], t['kp_cov_src_tot'])):>9}"
              f"{str(ratio(t['kp_cov_all_fullhit_soft'], t['n'])):>9}"
              f"{str(ratio(t['kp_cov_src_fullhit_soft'], t['n'])):>9}"
              f"{str(ratio(t['src_cited'], t['n'])):>9}")
        summary[qt] = dict(t)
        summary[qt]["cov_all_soft"] = ratio(t["kp_cov_all_hits"], t["kp_cov_all_tot"])
        summary[qt]["cov_all_strict"] = ratio(t["kp_cov_all_strict_hits"], t["kp_cov_all_tot"])
        summary[qt]["cov_all_fullhit"] = ratio(t["kp_cov_all_fullhit_soft"], t["n"])
        summary[qt]["cov_src_soft"] = ratio(t["kp_cov_src_hits"], t["kp_cov_src_tot"])
        summary[qt]["cov_src_strict"] = ratio(t["kp_cov_src_strict_hits"], t["kp_cov_src_tot"])
        summary[qt]["cov_src_fullhit"] = ratio(t["kp_cov_src_fullhit_soft"], t["n"])
        summary[qt]["cov_cross_soft"] = ratio(t["kp_cov_cross_hits"], t["kp_cov_cross_tot"])
        summary[qt]["cov_cross_strict"] = ratio(t["kp_cov_cross_strict_hits"], t["kp_cov_cross_tot"])
        summary[qt]["cov_cross_fullhit"] = ratio(t["kp_cov_cross_fullhit_soft"], t["n"])
        summary[qt]["src_cited_rate"] = ratio(t["src_cited"], t["n"])

    print(f"\n（无出处文档的题：{n_no_src} 条，拒答题本就没有出处）")

    out = {"batch": batch.name, "summary": summary, "rows": rows_out}
    outp = ROOT / (args.out or f"eval_runs/provenance_{batch.stem.split('.')[0]}.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
