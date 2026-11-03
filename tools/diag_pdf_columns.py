"""PDF **版式列结构**检测 —— 在 PDF 层用真实 x 坐标判定"双栏交错"。

**为什么必须在 PDF 层做**（实测 2026-11-02）：
  md 里已经**丢掉了 x 坐标**，只剩按 y 排序后的行。任何基于 md 的列分离都只能是猜：
  - 用"行尾是否完整句"判 → 交错区的右列本来就是残片，判不出来；
  - 用"字符 n-gram 相似度"判 → **目录块假阳性**（目录条目成对出现，
    隔行相似度天然更高；实测目录块差值 +0.146，真交错块只有 +0.017），
    真阳性与假阳性完全颠倒。
  只有 PDF 层的 x 坐标是**确定性的事实**。

**判据**（基于 pdfplumber 的 word 坐标）：
  1. 取每页所有 word 的 x0，聚类成若干列（按 x 间隙切）；
  2. 若某页存在**两个列簇**且**页宽利用率**表明是并排双栏
     （第二簇起始 x 明显靠右、且两簇在 y 上**交错出现**），判为双栏页；
  3. 双栏页上，若两列的 y 区间**相互交错**（不是上下分区），
     则按行抽取必然产生交错文本 —— 这正是 md 里看到的现象。

**用途**：
  - 给出"受影响文档与页数"的**可核对事实**；
  - 为"要不要修抽取层重跑"这个决策提供依据（PRD 3.1 策略 4 已就同类问题倾向不重跑）。

输出：`data/eval/pdf_column_report.json`
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


def cluster_columns(words, page_width, gap_ratio=0.04):
    """把 word 的 x0 聚成列簇。返回 [(x_start, x_end, n_words), ...]"""
    if not words:
        return []
    xs = sorted(w["x0"] for w in words)
    gap = page_width * gap_ratio
    clusters = []
    cur_start = xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x - prev > gap:
            clusters.append((cur_start, prev))
            cur_start = x
        prev = x
    clusters.append((cur_start, prev))
    out = []
    for a, b in clusters:
        n = sum(1 for w in words if a <= w["x0"] <= b)
        if n >= 3:
            out.append((a, b, n))
    return out


def analyse_page(page):
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if len(words) < 20:
        return None
    pw = page.width
    cols = cluster_columns(words, pw)
    if len(cols) < 2:
        return None
    # 只取覆盖宽度前 85% 的簇，滤掉页脚/页码这类窄簇
    main = [c for c in cols if c[1] - c[0] > pw * 0.08]
    if len(main) < 2:
        return None
    main.sort(key=lambda c: c[0])
    left, right = main[0], main[1]

    # 取右簇的 word，看它们和左簇的 word 在 y 上是否交错
    def ys(c):
        return sorted(w["top"] for w in words if c[0] <= w["x0"] <= c[1])
    ly, ry = ys(left), ys(right)
    if len(ly) < 8 or len(ry) < 8:
        return None

    # 交错度：把两簇的 y 合并排序，看是否频繁 LR 交替
    merged = sorted([(y, "L") for y in ly] + [(y, "R") for y in ry])
    seq = "".join(m[1] for m in merged)
    # 统计 LR / RL 交替次数（理想交错是 LRLRLR）
    transitions = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
    alternation = transitions / max(len(seq) - 1, 1)
    # 理想交错的交替率接近 1；上下分区接近 0
    return {
        "n_words": len(words),
        "n_columns": len(main),
        "left_x": round(left[0], 1), "right_x": round(right[0], 1),
        "left_words": left[2], "right_words": right[2],
        "alternation": round(alternation, 3),
        "interleaved": alternation >= 0.85,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "pdf_column_report.json"))
    args = ap.parse_args()
    if pdfplumber is None:
        raise SystemExit("需要 pdfplumber：pip install pdfplumber")

    docs = []
    for pdf in sorted(RAW.glob("*.pdf")):
        pages = []
        with pdfplumber.open(str(pdf)) as doc:
            for i, page in enumerate(doc.pages, 1):
                r = analyse_page(page)
                if r:
                    r["page_no"] = i
                    pages.append(r)
        if not pages:
            continue
        inter = [p for p in pages if p["interleaved"]]
        docs.append({
            "doc_id": pdf.stem,
            "n_pages_scanned": len(doc.pages),
            "n_multicol_pages": len(pages),
            "n_interleaved_pages": len(inter),
            "interleaved_page_ratio": round(len(inter) / max(len(pages), 1), 3),
            "pages": pages,
        })
        print(f"{pdf.stem:32s} 多栏页 {len(pages):3d}/{len(doc.pages):3d}"
              f"  交错页 {len(inter):3d}  交错率 {len(inter) / max(len(pages), 1):.1%}")

    tot_inter = sum(d["n_interleaved_pages"] for d in docs)
    tot_multi = sum(d["n_multicol_pages"] for d in docs)
    report = {"docs": docs,
              "total_multicol_pages": tot_multi,
              "total_interleaved_pages": tot_inter}
    pathlib.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\n多栏页合计 {tot_multi}，其中交错页 {tot_inter}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
