"""PDF 抽取的**行交错**检测 —— M3 的前置诊断。

**现象**（2026-11-02 实测发现）：PDF 抽出的正文是**按 y 坐标交错混排**的。
双栏 / 侧注版式被 pdfplumber 按行合并，左右两栏的文字逐行交替：

    700 企业危险化学品特种作业人员应具备高中或者相当于
    701 19 定》（国家安全监管总局令第30号）第四          <- 右栏（引用标准）
    702 高中及以上文化程度，能力应满足安全生产要求。      <- 回到左栏
    703 条                                                <- 右栏

左列原本是"企业危险化学品特种作业人员应具备高中或者相当于高中及以上文化程度，
能力应满足安全生产要求。"，被右栏切成了三段。

**为什么必须先量化它**：M3 的切分策略全都在这个文本上跑。
词袋检索（BM25）不看行序，**检索效果基本不受影响**；
但受影响的有：
  1. chunk 的 `text` 不通顺 —— 第 6 节的引用展示直接给人看这段；
  2. **跨页合并判定失真** —— 3.2 的互斥情形表判"页 N 末尾是否完整句"
     靠"行尾是否以 。！？ 结尾"，交错会让段落中间的行也以句号结尾；
  3. 标题识别 —— 标题本身是完整行，仍然能识别，但标题下的正文是碎片。

**检测方法**（不依赖具体版式）：对每个连续正文块，尝试把行**按奇偶分成两列**，
比较"以句读结尾的行占比"：
  - 交错块：奇偶分开后，每列的行都更完整 → 占比显著上升；
  - 正常块：奇偶分开会打乱句子 → 占比不升甚至下降。
这个判据只依赖"句子完整性"这个语言学信号，不依赖 pdfplumber 的内部结构。

输出：`data/eval/line_interleave_report.json` + 人读摘要。
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD_DIR = ROOT / "data" / "md"

PAGE_MARK = re.compile(r"<!--\s*page:\d+\s*-->")
SENT_END = "。！？；"

# 判定"像正文"的最小长度。太短的行（页码、残留）不参与。
MIN_LINE = 4


def read_blocks(path):
    """把 md 切成 (页号, 行列表) 的连续正文块。表格行单独成块。"""
    text = path.read_text(encoding="utf-8")
    # 去掉 front-matter
    if text.startswith("---"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1]
    lines = text.split("\n")
    blocks = []
    cur = []
    cur_page = None
    page = None
    for line in lines:
        s = line.strip()
        m = PAGE_MARK.fullmatch(s)
        if m:
            if cur:
                blocks.append((cur_page, cur))
                cur = []
            page = int(m.group(0).strip("<!--> page:"))
            cur_page = page
            continue
        if not s:
            if cur:
                blocks.append((cur_page, cur))
                cur = []
            continue
        if s.startswith("|"):
            if cur:
                blocks.append((cur_page, cur))
                cur = []
            blocks.append((cur_page, [line]))
            continue
        if len(s) >= MIN_LINE:
            cur.append(line)
    if cur:
        blocks.append((cur_page, cur))
    return blocks


def sent_ratio(lines):
    """以句读结尾的行占比。"""
    if not lines:
        return 0.0
    n = sum(1 for l in lines if l.strip().endswith(tuple(SENT_END)))
    return n / len(lines)


def is_interleaved(lines):
    """判断一个块是否是交错的双栏。

    判据：把行按奇偶分成两列后，**两列各自的句读结尾率都明显高于原块**。
    只看"原块"会被"整块都是完整句"的正常段落骗过去。
    """
    if len(lines) < 6:
        return False, 0.0
    base = sent_ratio(lines)
    left = lines[0::2]
    right = lines[1::2]
    lr, rr = sent_ratio(left), sent_ratio(right)
    gain = min(lr, rr) - base
    # 提升要够明显才认（0.15 是经验阈值，见下方自检）
    return gain >= 0.15, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "line_interleave_report.json"))
    args = ap.parse_args()

    docs = []
    for p in sorted(MD_DIR.glob("*.md")):
        if p.parent.name == "_stale":
            continue
        blocks = read_blocks(p)
        n_blocks = len(blocks)
        inter = []
        gains = []
        for i, (page, lines) in enumerate(blocks):
            if lines and lines[0].strip().startswith("|"):
                continue          # 表格块单独统计
            flag, gain = is_interleaved(lines)
            if flag:
                inter.append({"block": i, "page": page, "n_lines": len(lines),
                              "gain": round(gain, 3)})
                gains.append(gain)
        # 受影响的正文字符占比
        body_lines = [l for pg, ls in blocks
                      for l in ls if not l.strip().startswith("|")]
        aff_lines = sum(b["n_lines"] for b in inter)
        docs.append({
            "doc_id": p.stem,
            "n_blocks": n_blocks,
            "n_interleaved_blocks": len(inter),
            "interleaved_char_ratio": round(aff_lines / max(len(body_lines), 1), 4),
            "blocks": inter,
        })

    total_blocks = sum(d["n_blocks"] for d in docs)
    total_inter = sum(d["n_interleaved_blocks"] for d in docs)
    worst = sorted(docs, key=lambda d: -d["interleaved_char_ratio"])[:6]

    report = {
        "total_blocks": total_blocks,
        "total_interleaved_blocks": total_inter,
        "block_ratio": round(total_inter / max(total_blocks, 1), 4),
        "docs": docs,
    }
    pathlib.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    print(f"块总数 {total_blocks}，交错块 {total_inter}"
          f"（{total_inter / max(total_blocks, 1):.1%}）\n")
    print("%-30s %8s %8s %10s" % ("doc_id", "块数", "交错块", "字符占比"))
    for d in sorted(docs, key=lambda x: -x["interleaved_char_ratio"]):
        if d["n_interleaved_blocks"]:
            print("%-30s %8d %8d %9.1f%%" % (
                d["doc_id"][:30], d["n_blocks"], d["n_interleaved_blocks"],
                d["interleaved_char_ratio"] * 100))
    print(f"\n最严重的 6 份：{[d['doc_id'] for d in worst]}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
