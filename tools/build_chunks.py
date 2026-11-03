"""切分流水线驱动 —— PRD M3。

对 26 份 md 跑六种切分策略，产出带完整元数据的 chunk，落盘 `data/chunks/<strategy>.jsonl`，
并出一份各策略的规模统计（对应 PRD 3.3 的产出物字段的前半部分：
`n_chunks` / `avg_chunk_chars` / `p95_chunk_chars` / `table_chunks` /
`pseudo_table_chunks` / `cross_page_chunks` / `empty_chunks`）。

**召回类指标不在这里** —— 那要等 M5 的检索基线（M4 的六策略对比实验）。
本脚本只负责"切得出来、切得对"。

跑法：
    python tools/build_chunks.py                      # 全部策略
    python tools/build_chunks.py --strategies fixed,clause
    python tools/build_chunks.py --dry-run            # 只统计不落盘
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chunking as C  # noqa: E402

MD_DIR = ROOT / "data" / "md"
OUT_DIR = ROOT / "data" / "chunks"
MANIFEST = ROOT / "data" / "manifest.json"


def load_manifest():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = m if isinstance(m, list) else m.get("documents", m)
    return {d["doc_id"]: d for d in docs}


def p95(values):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def build_strategy(name, docs, doc_meta_by_id):
    """跑一个策略，返回 (chunks, stats)。"""
    chunks = []
    norm_cache = {}
    if name == "qa":
        chunks = C.strategy6_qa(None, doc_meta_by_id, ROOT)
    else:
        fn = C.STRATEGIES[name]
        for doc_id, dm in docs.items():
            path = MD_DIR / (doc_id + ".md")
            if not path.exists():
                continue
            fmt = dm.get("source_format") or "pdf_document"
            blocks = C.parse_md(path, source_format=fmt)
            raw = fn(blocks, {"doc_id": doc_id, "meta": dm})
            for i, ch in enumerate(raw):
                cid = f"{doc_id}__{name}_{i:05d}"
                chunks.append(C.assemble(ch, dm, name, cid, norm_cache))
    return chunks, stats_of(chunks)


def stats_of(chunks):
    lens = [len(c["text"]) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_chunk_chars": round(sum(lens) / max(len(lens), 1), 1),
        "p95_chunk_chars": p95(lens),
        "max_chunk_chars": max(lens) if lens else 0,
        "min_chunk_chars": min(lens) if lens else 0,
        "table_chunks": sum(1 for c in chunks if c.get("table")),
        "pseudo_table_chunks": sum(1 for c in chunks
                                   if (c.get("table") or {}).get("is_pseudo")),
        "cross_page_chunks": sum(1 for c in chunks
                                 if (c.get("strategy_meta") or {}).get("cross_page")),
        "empty_chunks": sum(1 for c in chunks if not c["text"].strip()),
        "docs_covered": len({c["doc_id"] for c in chunks}),
        # 🔴 M3 验收标准要求"实测统计**各文档**的伪表块数" ——
        # 汇总数看不出"哪个文档的表头不可信"，逐文档才有诊断价值。
        "per_doc": per_doc_stats(chunks),
    }


def per_doc_stats(chunks):
    """逐文档的块数 / 表块 / 伪表块。

    伪表 = 表头校验**没通过**的表格块（首行其实是正文散文，
    见 `src/chunking.py::header_looks_like_header`）。
    伪表率高 = 该文档的表格版式没有清晰列名，
    策略 4（表格切分）在它身上收益有限。
    """
    agg = {}
    for c in chunks:
        d = agg.setdefault(c["doc_id"], {
            "n_chunks": 0, "table_chunks": 0, "pseudo_table_chunks": 0,
            "cross_page_chunks": 0, "chars": 0,
        })
        d["n_chunks"] += 1
        d["chars"] += len(c["text"])
        if c.get("table"):
            d["table_chunks"] += 1
            if (c["table"] or {}).get("is_pseudo"):
                d["pseudo_table_chunks"] += 1
        if (c.get("strategy_meta") or {}).get("cross_page"):
            d["cross_page_chunks"] += 1
    for d in agg.values():
        d["pseudo_ratio"] = round(
            d["pseudo_table_chunks"] / max(d["table_chunks"], 1), 4)
        d["avg_chunk_chars"] = round(d["chars"] / max(d["n_chunks"], 1), 1)
    return dict(sorted(agg.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="fixed,structural,clause,table,semantic,qa")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    doc_meta_by_id = load_manifest()
    # 只切有 md 的文档（扫描件 yuncheng_paichabiao 没有 md）
    docs = {k: v for k, v in doc_meta_by_id.items()
            if (MD_DIR / (k + ".md")).exists()}
    print(f"语料 {len(doc_meta_by_id)} 份，其中有 md 的 {len(docs)} 份\n")

    out_dir = pathlib.Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for name in args.strategies.split(","):
        name = name.strip()
        if not name:
            continue
        chunks, st = build_strategy(name, docs, doc_meta_by_id)
        all_stats[name] = st
        print(f"[{name:11s}] chunk {st['n_chunks']:5d} | 均长 {st['avg_chunk_chars']:6.1f} "
              f"| p95 {st['p95_chunk_chars']:5d} | 表 {st['table_chunks']:4d} "
              f"(伪表 {st['pseudo_table_chunks']:3d}) | 跨页 {st['cross_page_chunks']:4d} "
              f"| 覆盖 {st['docs_covered']} 份")
        if not args.dry_run:
            p = out_dir / f"{name}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"             -> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")

    if not args.dry_run:
        (out_dir / "chunk_stats.json").write_text(
            json.dumps(all_stats, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n统计 -> {(out_dir / 'chunk_stats.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
