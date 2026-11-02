"""qa_pairs 题干改写 —— PRD 策略 6 的前提。

**为什么必须改写**（实测 2026-11-02）：
  `data/eval/qa_pairs.csv` 的 842 条题干里 **774 条是机器模板**：
      《文档名》中，{条款语义短语}是怎么规定的？
  这种题干有三重问题：
    1. **不像用户问的** —— 它是条款切片的副产品，不构成问句；
    2. **检索时 IDF 失效** —— 模板词"《…》中…是怎么规定的？"在 842 条里高度重复，
       BM25 里这些词的 IDF 趋近 0，检索退化成只靠几个残字匹配；
    3. **答非所问的强风险** —— tail 是从条款里截的短语，
       相邻条款共享同一短语时会产出**字面完全相同**的题干
       （实测 5 组，跨索引侧/评测侧泄漏）。

  PRD 3.1 策略 6 把 QA 对的 `question + answer` 直接建成 chunk 参与检索，
  所以**题干质量就是这条检索路的上限** —— 不改写，策略 6 测不出东西。

**两个阶段**（增量落盘，可反复跑续做）：
  --stage backfill   把 `test_set.jsonl` 里已经改写好的题干回填进来（不调 LLM）
  --stage rewrite    LLM 批量改写剩下的

**回填靠行序对齐**（已实测：qa_pairs.csv 的 842 行与 candidates.jsonl 逐行
`qid` / `doc_id` / `clause_no` / `answer` **全部一致**，无错位），
所以按行号取 test_set 的 `src_qid` 就能对上，不依赖 qid 唯一性
（qid 有 7 组重复，按 qid 建索引会撞）。

**产物**：`data/eval/qa_pairs_rewritten.csv`
  原 `question` 列保留不动（审计用），改写结果写在 `question_rewritten` 列，
  并用 `rewrite_source` 标来源（backfill / llm / none）。
"""
import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import re
import sys
import time
from difflib import SequenceMatcher

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm_client as L  # noqa: E402

QA_PAIRS = ROOT / "data" / "eval" / "qa_pairs.csv"
OUT_CSV = ROOT / "data" / "eval" / "qa_pairs_rewritten.csv"
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"

SYSTEM_PROMPT = (
    "你是工业安全规程领域的评测集专家。你要把机器模板生成的粗糙题干"
    "改写成**用户真的会问出口的自然问题**。"
    "你必须严格依据给定的原文作答，不得引入原文没有的信息。"
)

REWRITE_PROMPT = """下面是一条从工业安全规程文档里切出来的「问—答」对。

【当前题干（机器模板，不可直接用）】
{question}

【原文答案】
{answer}

【所在文档】{doc_title}（条款 {clause_no}）
{extra}

请把题干改写成一个**用户可能真的会问出口的自然问题**。

硬性要求：
1. 只依据上面的原文答案，不得引入原文没有的信息；
2. 问句必须指向答案里的**具体规定**（具体条款、具体要求、具体责任主体、具体步骤），
   不许问"怎么规定的"这种空壳问题；
3. 保留文档名，使问句能独立成立；🔴 **文档名必须原样照抄，一个字都不能改、
   不能加、不能删**（实测 LLM 会把"福建泉州中化学天辰(泉州)…"写成"中化学中化学天辰…"）；
4. 🔴 **不要在问句里照抄答案原文的整句**（会造成答案泄漏），但关键的数值、
   限值、单位、条款号可以保留 —— 评测要用数值锚点判分，改动数字等于改坏评测；
5. 一句话问句，不超过 60 字。

只输出 JSON：{{"question": "改写后的问题"}}
"""


def fix_doc_name(q, doc_title):
    """把问句里被改坏的文档名换回原名。

    实测 LLM 复制长中文文档名时会**吞字或加字**
    （"福建泉州中化学天辰(泉州)…" -> "福建泉州中化学中化学天辰…"）。
    这不影响语义，但会让"同一文档的题"失去统一标识，
    后面按文档名聚类做人工核对时会散架，所以这里统一换回原文。

    判据：问句里 `《…》` 的内容与 doc_title 归一化后相似度 >= 0.75 才替换，
    避免把正常提及别的文档名的问句误改。
    """
    if not q or not doc_title:
        return q
    if doc_title in q:
        return q

    def norm(s):
        return re.sub(r"[\s（）()《》“”\"'、,，.。:：\-—_]", "", s)

    title_n = norm(doc_title)
    if not title_n:
        return q

    def repl(m):
        got = m.group(1)
        got_n = norm(got)
        if not got_n:
            return m.group(0)
        if got_n in title_n or title_n in got_n:
            return "《" + doc_title + "》"
        if SequenceMatcher(None, got_n, title_n).ratio() >= 0.75:
            return "《" + doc_title + "》"
        return m.group(0)

    return re.sub(r"《([^《》]{4,})》", repl, q)


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f)), f


def read_qa_rows():
    """读 qa_pairs.csv；**已改写的部分从产物取回**。

    🔴 踩过的坑：最初这里只读原始 csv，于是 `rewrite` 阶段每次都从零开始，
    `backfill` 阶段写进去的 198 条被整个覆盖掉（实测：跑完 rewrite 后
    842 条的 question_rewritten 全空）。**产物必须是累积的，不能每次从零覆盖。**
    对齐方式用**行序**而不是 qid —— qid 有 7 组重复，按 qid 建索引会撞。
    """
    with QA_PAIRS.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if OUT_CSV.exists():
        with OUT_CSV.open(encoding="utf-8-sig", newline="") as f:
            prev = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            if i < len(prev) and prev[i].get("question_rewritten"):
                r["question_rewritten"] = prev[i]["question_rewritten"]
                r["rewrite_source"] = prev[i].get("rewrite_source") or "llm"
    return rows, fields


def write_out(rows, fields):
    if "question_rewritten" not in fields:
        fields = fields + ["question_rewritten"]
    if "rewrite_source" not in fields:
        fields = fields + ["rewrite_source"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def stage_backfill():
    """把 test_set.jsonl 里已改写好的题干按行序回填。"""
    rows, fields = read_qa_rows()
    test_rows = [json.loads(l) for l in TEST_SET.open(encoding="utf-8") if l.strip()]

    # src_qid -> 改写后的 question
    by_src = {}
    for t in test_rows:
        q = (t.get("question") or "").strip()
        src = t.get("src_qid")
        if q and src:
            by_src.setdefault(src, q)

    n = 0
    for i, r in enumerate(rows):
        if i >= len(test_rows):
            break
        src = test_rows[i].get("src_qid")
        q = by_src.get(src)
        if q:
            r["question_rewritten"] = q
            r["rewrite_source"] = "backfill"
            n += 1
        else:
            r.setdefault("question_rewritten", "")
            r.setdefault("rewrite_source", "none")
    write_out(rows, fields)
    print(f"回填 {n} / {len(rows)} 条题干")
    left = sum(1 for r in rows if not r.get("question_rewritten"))
    print(f"仍待改写：{left} 条")
    return left


def stage_rewrite(limit, workers):
    rows, fields = read_qa_rows()
    todo = [i for i, r in enumerate(rows) if not r.get("question_rewritten")]
    todo = todo[:limit]
    print(f"待改写 {len(todo)} 条（并发 {workers}）")

    def do(i):
        r = rows[i]
        extra = ""
        if r.get("must_include_numbers"):
            anchors = r["must_include_numbers"]
            if isinstance(anchors, str) and anchors:
                extra = (f"【数值锚点（必须原样保留，不得改动数字）】{anchors}")
        prompt = REWRITE_PROMPT.format(
            question=r.get("question", ""),
            answer=(r.get("answer") or "")[:2500],
            doc_title=r.get("doc_title", ""),
            clause_no=r.get("clause_no", ""),
            extra=extra,
        )
        try:
            raw = L.call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=1500)
            obj = L.parse_json(raw, default={})
            q = (obj.get("question") or "").strip()
        except Exception as e:
            return i, None, str(e)
        if not q:
            return i, None, "空输出"
        q = fix_doc_name(q, r.get("doc_title", ""))
        return i, q, None

    ok, fail = 0, 0
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, q, err in ex.map(do, todo):
            if q:
                rows[i]["question_rewritten"] = q
                rows[i]["rewrite_source"] = "llm"
                ok += 1
            else:
                fail += 1
                errors.append({"row": i, "qid": rows[i].get("qid"), "error": err})
    write_out(rows, fields)
    print(f"改写成功 {ok} 条，失败 {fail} 条 -> {OUT_CSV.name}")
    if errors:
        err_path = ROOT / "data" / "eval" / "rewrite_errors.jsonl"
        with err_path.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"失败明细 -> {err_path.name}")


def stage_postfix():
    """对已改写的题干统一修正文档名（LLM 复制长中文名会吞字/加字）。"""
    rows, fields = read_qa_rows()
    n = 0
    for r in rows:
        q = r.get("question_rewritten")
        if not q:
            continue
        fixed = fix_doc_name(q, r.get("doc_title", ""))
        if fixed != q:
            r["question_rewritten"] = fixed
            n += 1
    write_out(rows, fields)
    print(f"修正文档名：{n} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["backfill", "rewrite", "postfix"], required=True)
    ap.add_argument("--limit", type=int, default=35, help="rewrite 阶段单次最多改写多少条")
    ap.add_argument("--workers", type=int, default=int(os.getenv("M2_WORKERS", "3")))
    args = ap.parse_args()
    if args.stage == "backfill":
        stage_backfill()
    elif args.stage == "postfix":
        stage_postfix()
    else:
        stage_rewrite(args.limit, args.workers)


if __name__ == "__main__":
    main()
