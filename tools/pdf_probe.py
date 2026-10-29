"""PDF 探针：判定每份 PDF 是文本型还是扫描件(图片型)，并抽取文本与表格统计。

这是 RAG 项目"图片型 PDF"需求的第一个环节：必须能区分
  - 文本型 PDF  -> 直接抽取
  - 扫描件/图片型 PDF -> 必须走 OCR 路线
输出 probe_report.json 供语料清单使用。
"""
import json
import pathlib
import sys
import warnings

import pdfplumber

warnings.filterwarnings("ignore")

TEXT_LAYER_CHARS_PER_PAGE = 50  # 低于此值视为疑似无文本层(扫描件)


def probe_pdf(path: pathlib.Path) -> dict:
    info = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "pages": 0,
        "chars_total": 0,
        "chars_per_page": [],
        "image_pages": 0,
        "table_pages": 0,
        "table_count": 0,
        "kind": "unknown",
        "text_head": "",
        "has_formula_like": False,
    }
    with pdfplumber.open(path) as pdf:
        info["pages"] = len(pdf.pages)
        all_text = []
        for pg in pdf.pages:
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            all_text.append(txt)
            info["chars_per_page"].append(len(txt.strip()))
            if len(txt.strip()) < TEXT_LAYER_CHARS_PER_PAGE:
                info["image_pages"] += 1
            try:
                tables = pg.extract_tables() or []
            except Exception:
                tables = []
            if tables:
                info["table_pages"] += 1
                info["table_count"] += len(tables)
        joined = "\n".join(all_text)
        info["chars_total"] = len(joined.strip())
        info["text_head"] = joined.strip()[:1200]
        # 粗略判断是否含公式/符号（用真 PDF 文本层的 U+2212 ∕ 等）
        info["has_formula_like"] = any(ch in joined for ch in "±≤≥∫√∑×÷")

    if info["pages"] == 0:
        info["kind"] = "empty"
    elif info["chars_total"] == 0:
        info["kind"] = "scanned_no_text"
    elif info["image_pages"] / info["pages"] > 0.5:
        info["kind"] = "mixed_mostly_image"
    elif info["image_pages"] > 0:
        info["kind"] = "mixed_text_and_image"
    else:
        info["kind"] = "text_based"
    return info


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "probe_report.json")
    results = []
    for p in sorted(src.glob("*.pdf")):
        try:
            r = probe_pdf(p)
        except Exception as e:
            r = {"file": p.name, "error": str(e), "kind": "error"}
        results.append(r)
        print(f"{r['file']:42s} {r.get('kind','?'):22s} pages={r.get('pages','?'):>4} "
              f"chars={r.get('chars_total','?'):>7} img_pages={r.get('image_pages','?'):>4} "
              f"tables={r.get('table_count','?')}")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
