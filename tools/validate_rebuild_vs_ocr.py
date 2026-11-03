"""**独立验证**：重建后的文本 vs 页面渲染图的 OCR 结果。

## 为什么需要这一步

"重建出来的文本看着对"是**目视判断**，不可复核。
这里用一个**完全独立的信号**做验证：

    PDF 渲染成位图  →  OCR 识别  →  与抽取文本比字符相似度

OCR 只看像素，**不经过 pdfplumber 的任何逻辑**，
所以它是一份与"抽取层"独立的"地面真值"。

- 若 **new 的相似度显著高于 old** → 重建确实把文本拉回了页面真实内容；
- 若两者相近 → 重建没有改善（或者页面本来就没问题）；
- 若 new 反而更低 → 重建破坏了顺序。

## 口径

- OCR 结果按检测框的 (y, x) 排序后拼接（跟阅读顺序一致）；
- 双方都做归一化：全角→半角、去掉所有空白、去掉页眉页脚行；
- 相似度用 `difflib.SequenceMatcher` 的字符级 ratio。

## 输出

`data/eval/rebuild_vs_ocr.json`：每页的 old/new 相似度与样例。
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

import pdf_lines as PL  # noqa: E402
import normalize as N  # noqa: E402

WS = re.compile(r"\s+")


def squeeze(s):
    """归一化：全角转半角 + 去所有空白。"""
    return WS.sub("", N.normalize(s))


def ocr_text(img_path, engine, band_h=700, overlap=60):
    """OCR 一张图，返回识别文本（按检测框位置排序）。

    🔴 **必须按条带切**：rapidocr 的检测默认 `limit_side_len=736, limit_type='min'`，
    整页 A4（200dpi ≈ 1654×2339）会被缩到 736 宽，10pt 的字只剩 ~3px ——
    **检测不到任何文字**（实测 `dt_boxes is None`，返回 None）。
    切成 700px 高的横条后每条都在原生分辨率附近，检测才正常。

    条带之间留 `overlap` 的重叠，避免行被切在中间漏字。
    """
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    items = []
    y = 0
    while y < H:
        y2 = min(y + band_h, H)
        band = img.crop((0, y, W, y2))
        tmp = str(img_path).replace(".png", "_band.png")
        band.save(tmp)
        res, _ = engine(str(tmp))
        if res:
            for box, txt, score in res:
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                items.append((y + min(ys), min(xs), txt))
        if y2 >= H:
            break
        y = y2 - overlap
    items.sort(key=lambda t: (round(t[0] / 8), t[1]))
    return "\n".join(t[2] for t in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="AQ_T3034_2022_huagong_anquan.pdf:6;10;20",
                    help="PDF:页码(从1起)，多页用逗号、多个 PDF 用分号")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default="data/eval/rebuild_vs_ocr.json")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import pdfplumber
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    tmp_dir = ROOT / "data" / "_diag"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in args.pages.split(";"):
        fname, pgs = spec.split(":")
        path = RAW / fname
        if not path.exists():
            print("跳过（文件不存在）：", fname)
            continue
        with pdfplumber.open(path) as pdf:
            for pg_no in [int(x) for x in pgs.split(",")]:
                pg = pdf.pages[pg_no - 1]
                if not pg.chars:
                    continue
                img = tmp_dir / f"ocr_{path.stem}_p{pg_no}.png"
                pg.to_image(resolution=args.dpi).save(img)

                truth = ocr_text(img, engine)
                old = pg.extract_text() or ""
                new = "\n".join(PL.extract_page_lines(pg))

                t_s = squeeze(truth)
                o_s = squeeze(old)
                n_s = squeeze(new)
                r_old = difflib.SequenceMatcher(None, t_s, o_s).ratio()
                r_new = difflib.SequenceMatcher(None, t_s, n_s).ratio()
                results.append({
                    "file": fname, "page": pg_no,
                    "ocr_chars": len(t_s),
                    "old_chars": len(o_s), "new_chars": len(n_s),
                    "sim_old": round(r_old, 4), "sim_new": round(r_new, 4),
                    "delta": round(r_new - r_old, 4),
                })
                print(f"{fname:38s} p{pg_no:<3d} OCR {len(t_s):5d}字 | "
                      f"相似度 old {r_old:.4f} -> new {r_new:.4f}  ({r_new - r_old:+.4f})")

    out = ROOT / args.out
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if results:
        print(f"\n平均：old {sum(r['sim_old'] for r in results)/len(results):.4f} "
              f"-> new {sum(r['sim_new'] for r in results)/len(results):.4f}")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
