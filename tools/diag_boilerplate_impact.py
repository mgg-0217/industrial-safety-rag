#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**站点导航噪声对检索的影响量化**（零 API 成本，纯离线）。

🔴 **为什么关心这个**：
`guide_shijiu_shoujiu`（有限空间作业事故安全施救指南）是从 gzip 里救回来的，
原始文件**没有走容器抽取**（`extract_mode: container`），所以整份文档里
**混进了站点导航和页脚**（左侧菜单 + 省份列表 + ICP 备案号）。
而 manifest 里它的 `extract_quality` 是 `ok` → **按 PRD 2.8 不会被降权或排除**。

🔴 **本脚本回答两个问题**：
1. **污染有多大**：每份 md 的导航/页脚字符占比（逐文档列出）；
2. **污染有没有真的伤害检索**：用项目自己的检索器跑评测题集，
   看 top-k 命中块里有多少字符是导航噪声、有多少块是"半个块都是导航"。

⚠️ **本脚本只读不写**，不改动语料 —— 改动会让已发布的检索/生成数字失效，
必须单开一轮"清洗语料 + 全量重跑"才可横比。

用法：
    python tools/diag_boilerplate_impact.py
    python tools/diag_boilerplate_impact.py --top-k 5 --questions-file data/eval/test_set.jsonl
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

MD_DIR = ROOT / "data" / "md"
CHUNKS = ROOT / "data" / "chunks" / "fixed.jsonl"

#: 明确的导航/页脚标记（命中即判为噪声行）。
BOILER_TOKENS = [
    "网站标识码", "备案号", "ICP备", "责任编辑", "扫一扫", "分享到", "版权所有",
    "官方微信", "打印本页", "关闭窗口", "字体：", "字号：", "上一篇", "下一篇",
    "相关链接", "国务院安全生产委员会办公室", "国家矿山安全监察局",
    "省应急厅（局）", "省（区、市）矿山安监局", "国家消防救援局",
    "中国森林草原防火网", "中国红十字会", "官方微博", "RSS订阅",
    "来源：", "访问量：", "发布时间：",
]

#: 纯行政区名单独成行（省份列表）→ 页脚噪声。
PROVINCES = set(
    "北京 天津 河北 山西 内蒙古 辽宁 吉林 黑龙江 上海 江苏 浙江 安徽 福建 江西 "
    "山东 河南 湖北 湖南 广东 广西 海南 重庆 四川 贵州 云南 西藏 陕西 甘肃 青海 "
    "宁夏 新疆 香港 澳门 台湾".split()
)


def norm_line(s: str) -> str:
    """行归一化（去空白、去全角标点），用于跨文档重复检测。"""
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[|｜·•\-—　]+", "", s)
    return s


def boiler_lines_for(text: str) -> set[int]:
    """返回该文档中判为噪声的行号集合（0-based）。"""
    lines = text.splitlines()
    hit = set()
    for i, l in enumerate(lines):
        st = l.strip()
        if not st:
            continue
        if any(t in st for t in BOILER_TOKENS):
            hit.add(i)
            continue
        if st in PROVINCES:
            hit.add(i)
    return hit


def cross_doc_repeats(docs: dict[str, list[str]]) -> set[str]:
    """跨文档重复出现的短行 → 站点导航的强信号。

    🔴 阈值 2 是刻意的：同一份 md 里重复出现两次的正文不算导航
    （比如"一、适用范围"），但**站点导航在同站的每一份文档里都会出现**。
    只在**长度 ≤ 20** 的行上启用，避免把长句正文误判。
    """
    cnt = Counter()
    for lines in docs.values():
        seen = {norm_line(l) for l in lines if 0 < len(norm_line(l)) <= 20}
        cnt.update(seen)
    return {k for k, v in cnt.items() if v >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--questions-file", default="data/eval/test_set.jsonl")
    ap.add_argument("--out", default="eval_runs/boilerplate_impact.json")
    args = ap.parse_args()

    mds = sorted(MD_DIR.glob("*.md"))
    if not mds:
        raise SystemExit("data/md 为空")

    docs: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for p in mds:
        t = p.read_text(encoding="utf-8", errors="replace")
        # 🔴 键用 **doc_id**（不带 .md 后缀），与 chunk 里的 `doc_id` 一致。
        #    早先用文件名做键 → 查表永远落空 → 噪声占比恒为 0.00%（假阴性）。
        key = p.stem
        texts[key] = t
        docs[key] = t.splitlines()

    repeats = cross_doc_repeats(docs)

    # 逐文档：噪声行号 + 噪声字符占比
    per_doc = {}
    for name, lines in docs.items():
        hit = boiler_lines_for(texts[name])
        # 跨文档重复的短行也算噪声
        for i, l in enumerate(lines):
            n = norm_line(l)
            if n and n in repeats and 0 < len(n) <= 20:
                hit.add(i)
        nchars = sum(len(lines[i]) for i in hit)
        total = sum(len(l) for l in lines)
        per_doc[name] = {
            "chars": total,
            "boiler_chars": nchars,
            "ratio": round(nchars / total, 4) if total else 0.0,
            "boiler_lines": len(hit),
            "total_lines": len(lines),
        }

    ranked = sorted(per_doc.items(), key=lambda kv: -kv[1]["ratio"])
    print("=== 逐文档导航噪声占比（top 12）===")
    for name, s in ranked[:12]:
        if s["boiler_chars"] == 0:
            continue
        print(f"  {name[:44]:46s} {s['chars']:7d} 字  噪声 {s['boiler_chars']:6d}"
              f"  {s['ratio']*100:5.1f}%  ({s['boiler_lines']}/{s['total_lines']} 行)")

    # ---- 检索影响 ----
    chunks = []
    if CHUNKS.exists():
        for line in CHUNKS.open(encoding="utf-8"):
            if line.strip():
                chunks.append(json.loads(line))
    impact = {}
    if chunks:
        import retrieval as RT

        retr = RT.Retriever(chunks)
        # 噪声行 → 文本片段集合（用片段而非行，容忍换行合并）
        boiler_frags: dict[str, set[str]] = {}
        for name, lines in docs.items():
            hit = boiler_lines_for(texts[name])
            for i, l in enumerate(lines):
                n = norm_line(l)
                if n and n in repeats and 0 < len(n) <= 20:
                    hit.add(i)
            frags = set()
            for i in hit:
                s = lines[i].strip()
                if len(s) >= 2:
                    frags.add(s)
            boiler_frags[name] = frags

        # 🔴 **覆盖率自检**：chunk 的 doc_id 必须能在噪声表里查到。
        #    查不到就说明键对不上（早先的 bug：噪声表键用文件名、chunk 用 doc_id，
        #    查表永远落空 → 噪声占比恒为 0%，**假阴性**）。
        chunk_docs = {c.get("doc_id") for c in chunks}
        hit_docs = chunk_docs & set(boiler_frags.keys())
        print(f"\n[自检] chunk 里的 doc_id {len(chunk_docs)} 个，"
              f"能在噪声表里查到 {len(hit_docs)} 个"
              f"（覆盖率 {len(hit_docs)/max(len(chunk_docs),1)*100:.1f}%）")
        if not hit_docs:
            raise SystemExit("🔴 键对不上：噪声表与 chunk 的 doc_id 无交集，"
                             "结果不可信，终止")

        def boiler_ratio(chunk_text: str, doc_id: str) -> float:
            frags = boiler_frags.get(doc_id) or set()
            if not frags:
                return 0.0
            hit = 0
            for f in frags:
                if f in chunk_text:
                    hit += len(f)
            return hit / max(len(chunk_text), 1)

        qfile = ROOT / args.questions_file
        qs = [json.loads(l) for l in qfile.open(encoding="utf-8") if l.strip()]
        n_hit = 0
        n_half = 0
        ratios = []
        by_doc = Counter()
        for q in qs:
            try:
                hits = retr.retrieve(q.get("question") or "", top_k=args.top_k,
                                     version_filter=False, use_vector=False)
            except Exception as e:
                print("检索失败：", e)
                break
            for idx, _s in hits:
                if not (0 <= idx < len(chunks)):
                    continue
                c = chunks[idx]
                r = boiler_ratio(c.get("text") or "", c.get("doc_id") or "")
                ratios.append(r)
                n_hit += 1
                if r >= 0.5:
                    n_half += 1
                    by_doc[c.get("doc_id")] += 1
        impact = {
            "chunks_retrieved": n_hit,
            "doc_id_coverage": round(len(hit_docs) / max(len(chunk_docs), 1), 4),
            "mean_boiler_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
            "chunks_over_50pct_boiler": n_half,
            "over_50pct_by_doc": dict(by_doc.most_common(10)),
        }
        print("\n=== 检索影响（top-%d，%d 题）===" % (args.top_k, len(qs)))
        print(f"  取回块数            {n_hit}")
        print(f"  块内噪声字符占比均值  {impact['mean_boiler_ratio']*100:.2f}%")
        print(f"  噪声占比 ≥50% 的块   {n_half}  ({n_half/max(n_hit,1)*100:.2f}%)")
        if by_doc:
            print("  主要来源文档：")
            for d, c in by_doc.most_common(8):
                print(f"    {d:44s} {c}")

    out = {"per_doc": per_doc, "impact": impact, "top_k": args.top_k}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
