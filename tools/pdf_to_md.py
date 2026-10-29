"""PDF -> Markdown 抽取（保留页码锚点与表格结构）。

与 pdf_probe.py 配套：
  - 文本型 PDF  -> 直接抽取，表格转 markdown 表格
  - 扫描件(无文本层) -> 不抽取，标记 needs_ocr=true，保留原文件给 OCR 链路用
"""
import json
import pathlib
import sys
import warnings

import pdfplumber

warnings.filterwarnings("ignore")

TEXT_LAYER_MIN = 50  # 每页去空白字符数下限，低于此视为无文本层


def cell_text(cell):
    if cell is None:
        return ""
    t = str(cell).replace("\n", " ").replace("|", "/").strip()
    return " ".join(t.split())


def table_to_md(table):
    rows = [[cell_text(c) for c in row] for row in table]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join([" --- "] * width) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def convert(path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    report = {"file": path.name, "needs_ocr": False, "pages": 0, "chars": 0,
              "tables": 0, "out": None}
    chunks = []
    with pdfplumber.open(path) as pdf:
        report["pages"] = len(pdf.pages)
        for i, pg in enumerate(pdf.pages, 1):
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            try:
                tables = pg.extract_tables() or []
            except Exception:
                tables = []
            if not txt.strip() and not tables:
                report["needs_ocr"] = True
                continue
            body = [f"\n<!-- page:{i} -->", ""]
            if txt.strip():
                body.append(txt.strip())
                report["chars"] += len(txt.strip())
            for t in tables:
                md = table_to_md(t)
                if md:
                    body.append("")
                    body.append(md)
                    report["tables"] += 1
            chunks.append("\n".join(body))
    if report["needs_ocr"] and report["chars"] == 0:
        report["out"] = None
        report["kind"] = "scanned_no_text"
        return report
    meta = [
        "---",
        f"source_file: {path.name}",
        "source_path: " + str(path.resolve().as_uri()),
        f"pdf_pages: {report['pages']}",
        f"pdf_tables: {report['tables']}",
        "extract_method: pdfplumber_text_layer",
        "language: zh-CN",
        "---",
        "",
    ]
    out = out_dir / (path.stem + ".md")
    out.write_text("\n".join(meta) + "\n".join(chunks), encoding="utf-8")
    report["out"] = out.name
    report["kind"] = "text_based"
    return report


def main():
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(src.glob("*.pdf")):
        try:
            r = convert(p, out)
        except Exception as e:
            r = {"file": p.name, "kind": "error", "error": str(e)}
        results.append(r)
        print(f"{r['file']:42s} {r.get('kind','?'):18s} pages={r.get('pages','?'):>4} "
              f"chars={r.get('chars','?'):>7} tables={r.get('tables','?'):>3} -> {r.get('out') or '(OCR needed)'}")
    (out.parent / "pdf_extract_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
