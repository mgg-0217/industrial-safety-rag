"""**基线重建 vs 现状**的逐页对比 —— 评估"重跑抽取层"的影响面。

对比两个口径：
  - **old**：`page.extract_text()`（现状，pdfplumber 默认 `y_tolerance=3`）
  - **new**：`src/pdf_lines.py::reconstruct_lines()`（按文本矩阵基线归并 + 行内按 x 排序）

三个数：
  1. `n_lines`：行数
  2. `chars`：字符数（**应当基本不变** —— 重建只改行的切法与顺序，不增删字符）
  3. `same_multiset`：去掉空白后的**字符多重集是否一致**
     —— 这是"重建有没有丢字/加字"的硬判据

再给出每个文档的**样例页**（差异最大的页），供人眼核对。

输出：`data/eval/line_rebuild_report.json`
"""
import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))

import pdf_lines as PL  # noqa: E402

WS = re.compile(r"\s+")


def squash(s):
    return WS.sub("", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="", help="只对比某个 PDF（文件名片段）")
    ap.add_argument("--samples", type=int, default=2, help="每个文档保留几个差异样例页")
    args = ap.parse_args()

    import pdfplumber
    import warnings
    warnings.filterwarnings("ignore")

    pdfs = sorted(RAW.glob("*.pdf"))
    if args.pdf:
        pdfs = [p for p in pdfs if args.pdf in p.name]

    report = {"docs": []}
    for p in pdfs:
        with pdfplumber.open(p) as pdf:
            n_pages = len(pdf.pages)
            tot_old_lines = tot_new_lines = 0
            tot_old_chars = tot_new_chars = 0
            n_same_multiset = 0
            n_pages_parsed = 0
            samples = []
            per_page = []
            for pg in pdf.pages:
                if not pg.chars:
                    continue
                n_pages_parsed += 1
                old_text = pg.extract_text() or ""
                old_lines = [l for l in old_text.split("\n") if l.strip()]
                new_lines = PL.extract_page_lines(pg)
                tot_old_lines += len(old_lines)
                tot_new_lines += len(new_lines)
                tot_old_chars += len(old_text)
                tot_new_chars += sum(len(l) for l in new_lines)
                old_sq = squash(old_text)
                new_sq = squash("".join(new_lines))
                same = Counter(old_sq) == Counter(new_sq)
                if same:
                    n_same_multiset += 1
                diff = abs(len(old_lines) - len(new_lines))
                per_page.append({"page": pg.page_number, "old": len(old_lines),
                                 "new": len(new_lines), "diff": diff, "same": same})
                samples.append((diff, pg.page_number, old_lines, new_lines))
            samples.sort(reverse=True, key=lambda t: t[0])
            report["docs"].append({
                "file": p.name,
                "n_pages": n_pages,
                "n_pages_parsed": n_pages_parsed,
                "old_lines": tot_old_lines,
                "new_lines": tot_new_lines,
                "old_chars": tot_old_chars,
                "new_chars": tot_new_chars,
                "char_delta": tot_new_chars - tot_old_chars,
                "pages_same_multiset": n_same_multiset,
                "pages_same_multiset_ratio": round(n_same_multiset / max(n_pages_parsed, 1), 4),
                "sample_pages": [
                    {"page": pno, "old": old[:14], "new": new[:14]}
                    for _d, pno, old, new in samples[:args.samples]
                ],
            })
            d = report["docs"][-1]
            print(f"{p.name:42s} 页 {n_pages_parsed:3d} | 行 {tot_old_lines:5d} -> {tot_new_lines:5d} "
                  f"| 字符 {tot_old_chars:7d} -> {tot_new_chars:7d} ({d['char_delta']:+d}) "
                  f"| 字符集一致 {d['pages_same_multiset_ratio']:.2%}")

    out = ROOT / "data" / "eval" / "line_rebuild_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
