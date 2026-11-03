"""**基线容差扫参** —— 用 OCR 真值选参数。

`src/pdf_lines.py` 的 `baseline_tol` 是个需要定的参数，不能靠猜：

- **太小**（如 1.0）：同一行里不同字体的基线差（实测 1.5–8pt）会把行拆开 ——
  回到"碎片行"的老问题；
- **太大**（如 10）：表格里相邻两行的行距（实测 9–12pt）会被并成一行 ——
  表格行会被串起来，条款号会串行。

判据用**独立信号**：把页渲染成位图做 OCR，得到"页面真实内容"，
再比较不同容差下重建文本与它的字符相似度。OCR 只看像素，
不经过 pdfplumber 的任何逻辑，所以它是可信的外部裁判。

OCR 每页**只跑一次**并缓存，然后对多个容差复用 —— 扫参不重复付出 OCR 成本。

输出：`data/eval/baseline_tol_sweep.json`
"""
import argparse
import difflib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import pdf_lines as PL  # noqa: E402
import normalize as N  # noqa: E402

WS = re.compile(r"\s+")


def squeeze(s):
    return WS.sub("", N.normalize(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=(
        "AQ_T3034_2022_huagong_anquan.pdf:4,6,10,20,33;"
        "mem_xianshi_shouce.pdf:20,38;"
        "mem_2019_78_zhimengdaoze.pdf:5,30"))
    ap.add_argument("--tols", default="1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import pdfplumber
    from rapidocr_onnxruntime import RapidOCR
    from validate_rebuild_vs_ocr import ocr_text

    tols = [float(t) for t in args.tols.split(",")]
    engine = RapidOCR()
    tmp_dir = ROOT / "data" / "_diag"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第一遍：OCR 每页一次，缓存真值 ----
    truths = {}
    for spec in args.pages.split(";"):
        fname, pgs = spec.split(":")
        path = RAW / fname
        if not path.exists():
            continue
        with pdfplumber.open(path) as pdf:
            for pg_no in [int(x) for x in pgs.split(",")]:
                pg = pdf.pages[pg_no - 1]
                if not pg.chars:
                    continue
                img = tmp_dir / f"sweep_{path.stem}_p{pg_no}.png"
                if not img.exists():
                    pg.to_image(resolution=args.dpi).save(img)
                truths[(fname, pg_no)] = squeeze(ocr_text(img, engine))
        print(f"OCR 完成 {fname} {pgs}")

    # ---- 第二遍：扫容差 ----
    scores = {t: [] for t in tols}
    per_page = []
    for spec in args.pages.split(";"):
        fname, pgs = spec.split(":")
        path = RAW / fname
        if not path.exists():
            continue
        with pdfplumber.open(path) as pdf:
            for pg_no in [int(x) for x in pgs.split(",")]:
                pg = pdf.pages[pg_no - 1]
                if not pg.chars:
                    continue
                truth = truths[(fname, pg_no)]
                row = {"file": fname, "page": pg_no, "scores": {}}
                for t in tols:
                    lines = PL.extract_page_lines(pg, baseline_tol=t)
                    s = difflib.SequenceMatcher(None, truth, squeeze("\n".join(lines))).ratio()
                    row["scores"][str(t)] = round(s, 4)
                    scores[t].append(s)
                per_page.append(row)
                best = max(row["scores"], key=row["scores"].get)
                print(f"{fname:38s} p{pg_no:<3d} 最佳容差 {best}  得分 {row['scores'][best]:.4f}")

    summary = {str(t): round(sum(v) / len(v), 4) for t, v in scores.items()}
    best_tol = max(summary, key=summary.get)
    print("\n== 各容差的平均相似度 ==")
    for t in tols:
        mark = " <== 最优" if str(t) == best_tol else ""
        print(f"  tol={t:<5} {summary[str(t)]:.4f}{mark}")

    out = ROOT / "data" / "eval" / "baseline_tol_sweep.json"
    out.write_text(json.dumps({
        "summary": summary,
        "best_tol": float(best_tol),
        "pages": per_page,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
