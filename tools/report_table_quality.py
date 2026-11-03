"""**表块级**的表头质量报告 —— M3 验收要求的"各文档伪表块数"。

## 为什么是"块"级而不是"chunk"级

策略 4 按行切分，一张表会变成多个 chunk，chunk 级统计看不出
"这张表本身是不是真表"。M3 验收标准写的是
「表格表头须经启发式校验，伪表标 `is_pseudo=true` 且不重复首行
（**实测统计各文档的伪表块数**）」—— 所以这里统计的是**表块**。

## 为什么要统计

抽取层 `pdf_to_md.py::table_to_md()` **无条件**把 pdfplumber 的 `rows[0]`
当表头写出、并**无条件注入** `| --- |` 分隔行 ——
所以"有分隔行"是合成产物（必然成立、零信息量），
"首行像表头"才有信息量，而它对相当一部分块是假的。

`src/chunking.py::header_looks_like_header()` 负责判别。
本脚本把它的判定结果**落成可核对的表**：
每份文档多少表块、多少真表头、多少伪表、伪表的首行长什么样。

## 输出

- `data/eval/table_quality.json` —— 机读
- `data/eval/table_quality.md` —— 人读（含伪表首行样例，供人眼抽查）
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "data" / "md"
sys.path.insert(0, str(ROOT / "src"))

import chunking as C  # noqa: E402


def scan_doc(path):
    """扫一份 md，返回表块统计。"""
    blocks = C.parse_md(path, source_format="pdf_document")
    tables = [b for b in blocks if b.kind == "table"]
    n_key_value = 0
    rows = []
    for b in tables:
        kv = C.is_key_value_table(b.rows)
        if kv:
            n_key_value += 1
        first = " | ".join(b.header) if b.header else ""
        rows.append({
            "page_no": b.page_no,
            "n_data_rows": len(b.rows),
            "header": b.header,
            "header_like": b.header_like,
            "is_pseudo": b.is_pseudo,
            "is_key_value": kv,
            "first_row": first[:120],
        })
    n_real = sum(1 for r in rows if r["header_like"] and not r["is_key_value"])
    n_pseudo = sum(1 for r in rows if r["is_pseudo"] and not r["is_key_value"])
    return {
        "n_table_blocks": len(tables),
        "n_header_like": n_real,
        "n_pseudo": n_pseudo,
        "n_key_value": n_key_value,
        "pseudo_ratio": round(n_pseudo / max(len(tables), 1), 4),
        "blocks": rows,
    }


def main():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    docs = manifest if isinstance(manifest, list) else manifest.get("documents", manifest)
    fmts = {d["doc_id"]: (d.get("source_format") or "pdf_document") for d in docs}

    report = {"docs": {}, "totals": {}}
    md_lines = ["# 表头质量报告（表块级）", "",
                "抽取层无条件把首行当表头写出，所以「有分隔行」是合成产物；",
                "这里统计的是**首行像不像表头**（`header_looks_like_header` 的判定）。", ""]
    md_lines += ["| 文档 | 表块 | 真表头 | 伪表 | 键值对 | 伪表率 |",
                 "|---|---|---|---|---|---|"]
    tot_blocks = tot_real = tot_pseudo = tot_kv = 0
    for path in sorted(MD.glob("*.md")):
        st = scan_doc(path)
        if st["n_table_blocks"] == 0:
            continue
        report["docs"][path.stem] = st
        tot_blocks += st["n_table_blocks"]
        tot_real += st["n_header_like"]
        tot_pseudo += st["n_pseudo"]
        tot_kv += st["n_key_value"]
        md_lines.append(
            f"| `{path.stem}` | {st['n_table_blocks']} | {st['n_header_like']} | "
            f"{st['n_pseudo']} | {st['n_key_value']} | {st['pseudo_ratio']:.0%} |")
    report["totals"] = {
        "n_table_blocks": tot_blocks, "n_header_like": tot_real,
        "n_pseudo": tot_pseudo, "n_key_value": tot_kv,
        "pseudo_ratio": round(tot_pseudo / max(tot_blocks, 1), 4),
    }
    md_lines += ["", f"**合计**：{tot_blocks} 表块 / 真表头 {tot_real} / "
                 f"伪表 {tot_pseudo}（{tot_pseudo / max(tot_blocks, 1):.0%}）/ "
                 f"键值对 {tot_kv}", ""]

    # 伪表首行样例 —— 供人眼抽查启发式判得对不对
    md_lines += ["## 伪表首行样例（人工抽查用）", ""]
    for doc_id, st in report["docs"].items():
        samples = [b for b in st["blocks"] if b["is_pseudo"] and not b["is_key_value"]][:3]
        if not samples:
            continue
        md_lines.append(f"### `{doc_id}`（伪表 {st['n_pseudo']} 块）")
        md_lines.append("")
        for b in samples:
            md_lines.append(f"- p{b['page_no']}：{b['first_row'][:100]}")
        md_lines.append("")

    (ROOT / "data" / "eval" / "table_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "data" / "eval" / "table_quality.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"表块 {tot_blocks} | 真表头 {tot_real} | 伪表 {tot_pseudo} "
          f"({tot_pseudo / max(tot_blocks, 1):.0%}) | 键值对 {tot_kv}")
    print("-> data/eval/table_quality.md")


if __name__ == "__main__":
    main()
