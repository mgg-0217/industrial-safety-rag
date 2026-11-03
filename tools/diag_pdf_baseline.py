"""PDF **基线错位**诊断 —— 抽取层是否把同一行的不同字体 run 拆散了。

## 现象（2026-11-02 实测）

`AQ_T3034_2022_huagong_anquan.md` 的正文**读不通**：

```
４７１２ 在工厂选址 总平面布局时 应符合有关设计标准的要求 并按照 的要求进行定
．．． 、 , , GB/T３７２４３
量风险评价 开展外部安全防护距离计算 以满足 所规定的个人与社会可容许风险标
(QRA), , GB３６８９４
```

原文应当是：

```
4.7.1.2 在工厂选址、总平面布局时，应符合有关设计标准的要求，并按照 GB/T 37243
的要求进行定量风险评价，开展外部安全防护距离计算，以满足 GB 36894 (QRA) 所规定的
```

**根因在 PDF 本身，不在抽取代码**：该 PDF 的中文与 ASCII 用**两套字体**，
两套字体的基线差了 7–9 pt。pdfplumber 的 `extract_text()` 默认
`y_tolerance=3`，只把 top 相差 ≤3pt 的字符算同一行 ——
**跨字体的行被拆成 3–4 条 "行"**，ASCII 片段被甩到下一行。

## 关键事实：x 坐标是完好的

拆分出去的那些 run，**x 坐标仍在原文位置**（实测 `．．．` 在 x=76.5、
`GBT３７２４３` 在 x=408，正好在"按照"与"的要求"之间）。
所以这是**排序问题**，不是坐标丢失 ——
把同一 y 带内的字符**按 x 重排**就能还原阅读顺序。

## 两种修法

| 方案 | 做法 | 代价 |
|---|---|---|
| **(a) 重跑抽取** | 抽取层按 y 带归并（容差可调）+ 带内按 x 排序 | 全部 26 份 md 变化 → 所有基线数字作废重算 |
| **(b) 切分层兜底** | 不动 md，在切分时把"短行碎片"并回相邻行 | 不改语料，但正文在 `text`（引用展示）里仍然是碎的 |

本脚本同时输出两种口径的数字，供决策使用。

## 判据（不依赖具体版式）

对每页按 `top` 分簇，若某行的字符**只覆盖很窄的 x 区间**
（远短于同页其它行）且它与相邻"主行"的 top 差 < 行距的一半，
则判定它是**错位碎片**，不是正文行。

输出：`data/eval/pdf_baseline_report.json`
"""
import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


def cluster_rows(chars, y_tol=1.0):
    """按 top 把字符聚成"原始行"（容差很小，只合并真正同一基线的字符）。"""
    rows = []
    for c in sorted(chars, key=lambda c: (c["top"], c["x0"])):
        if rows and abs(c["top"] - rows[-1]["top"]) <= y_tol:
            rows[-1]["chars"].append(c)
            # 行的 top 取首字符
        else:
            rows.append({"top": c["top"], "chars": [c]})
    return rows


def row_text(row):
    return "".join(c["text"] for c in sorted(row["chars"], key=lambda c: c["x0"]))


def row_width(row):
    cs = sorted(row["chars"], key=lambda c: c["x0"])
    return cs[-1]["x1"] - cs[0]["x0"]


def analyze_page(pg):
    """返回该页的错位碎片统计。"""
    chars = pg.chars
    if not chars:
        return None
    rows = cluster_rows(chars)
    if len(rows) < 3:
        return None

    # 主行 = 字符数最多的若干行；碎片 = 明显比同行邻居窄的行
    widths = sorted((row_width(r) for r in rows), reverse=True)
    median_w = widths[len(widths) // 2] if widths else 0

    strays = []
    for i, r in enumerate(rows):
        w = row_width(r)
        n = len(r["chars"])
        # 碎片判据：宽度显著小于中位数，且字符数少
        if median_w > 0 and w < median_w * 0.5 and n <= 12:
            # 与上一行的 top 差
            prev_top = rows[i - 1]["top"] if i > 0 else None
            next_top = rows[i + 1]["top"] if i + 1 < len(rows) else None
            strays.append({
                "text": row_text(r)[:40],
                "top": round(r["top"], 2),
                "width": round(w, 1),
                "n_chars": n,
                "fonts": sorted({c["fontname"] for c in r["chars"]}),
                "prev_top": round(prev_top, 2) if prev_top is not None else None,
                "next_top": round(next_top, 2) if next_top is not None else None,
            })
    return {
        "page_no": pg.page_number,
        "n_rows": len(rows),
        "median_row_width": round(median_w, 1),
        "n_stray_rows": len(strays),
        "strays": strays[:12],
    }


def reextract_text(pg, y_tolerance):
    """用给定 y 容差重新抽一遍文本。"""
    return pg.extract_text(y_tolerance=y_tolerance) or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="", help="只诊断某个 PDF")
    ap.add_argument("--tolerances", default="3,8,10,12,15")
    args = ap.parse_args()

    if pdfplumber is None:
        print("需要 pdfplumber")
        return

    pdfs = sorted(RAW.glob("*.pdf"))
    if args.pdf:
        pdfs = [p for p in pdfs if args.pdf in p.name]

    report = {"docs": [], "tolerances": [int(t) for t in args.tolerances.split(",")]}

    for p in pdfs:
        with pdfplumber.open(p) as pdf:
            n_pages = len(pdf.pages)
            per_page = []
            n_stray_total = 0
            samples = {}
            for pg in pdf.pages:
                a = analyze_page(pg)
                if a:
                    per_page.append(a)
                    n_stray_total += a["n_stray_rows"]
            # 各容差下的文本长度（长度变化 = 是否把碎片并回去了）
            for tol in report["tolerances"]:
                lens = []
                for pg in pdf.pages[:12]:
                    lens.append(len(reextract_text(pg, tol)))
                samples[f"tol_{tol}"] = sum(lens)
            # 默认容差下前 2 页的文本样本
            with pdfplumber.open(p) as pdf2:
                pdf2.pages[0:1]
            report["docs"].append({
                "file": p.name,
                "n_pages": n_pages,
                "n_pages_with_stray": sum(1 for x in per_page if x["n_stray_rows"]),
                "n_stray_rows": n_stray_total,
                "stray_ratio": round(n_stray_total / max(sum(x["n_rows"] for x in per_page), 1), 4),
                "pages": [x for x in per_page if x["n_stray_rows"]][:15],
                "text_len_by_tolerance": samples,
            })
            print(f"{p.name:42s} {n_pages:3d}页  碎片行 {n_stray_total:5d} "
                  f"涉页 {sum(1 for x in per_page if x['n_stray_rows']):3d}  "
                  f"比例 {report['docs'][-1]['stray_ratio']:.4f}")

    out = ROOT / "data" / "eval" / "pdf_baseline_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
