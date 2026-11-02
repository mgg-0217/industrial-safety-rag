"""评测候选池普查 —— PRD M2 第一步（机器初筛）。

候选池是机器按规则切条款生成的（842 条），直接拿来评测会污染指标。
PRD 7.2 给了 11 条质检标准，其中可自动判定的部分在这里落成可执行检查。

**本脚本只做「发现问题」，不做「修问题」**：
输出 audit_flags.json（机读，供后续 LLM 批量改写取用）+ audit_report.md（人读清单）。
修题干、标 key_point 属于 M2 第二步（LLM 批量改写）与第三步（人工核对）。

可自动判定的质检项（对应 PRD 7.2 表格）：
  R-02 数值题必须带锚点        -> must_include_numbers 非空
  R-04 题干完整性              -> 短题干 / 省略号 / 悬空标点 / 缺字残句
  R-06 陷阱题有唯一正答        -> 同问法跨文档的答案冲突（版本链候选）
  R-07 每题标注时间口径        -> time_mode + query_time 缺失
  R-08 每题标注期望版本        -> expect_doc_id 缺失
  R-09 每题标注 key_points      -> key_points 缺失
  R-10 标识符唯一              -> qid 重复
  R-11 去重                    -> 同 doc_id + clause_no 的多题
  R-01 题干不泄漏答案          -> 题干含答案锚点数值

判不出来的（必须人工/LLM 核对，脚本只标「需复核」）：
  R-03 答案唯一 / R-05 切题正确 —— 依赖语义理解

输出：
  data/eval/audit_flags.json   机读标记
  data/eval/audit_report.md    人读清单（按文档分组）
"""
import argparse
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "data" / "eval" / "candidates.jsonl"

# 题干被抽取截断的典型痕迹：连续省略号（"…急救援............是怎么规定的？"）
RE_ELLIPSIS = re.compile(r"\.{3,}|…{2,}")
# 悬空标点：句首/句尾的逗号顿号（"中，急预案向…" 缺字后残留）
RE_DANGLING = re.compile(r"^[,，、；;]|[,，、；;]$")
# 连续三个以上汉字后紧跟的半个词——用于捕捉"应急预案"被拆成"急预案"这类缺字
RE_MISSING_CHAR = re.compile(r"[，,]\s*[一-龥]{1,3}(?=预案|处置|措施|流程|要求)")
# 题干里出现被吞字的常见形态：单字动词孤立（"向…"/"的…是多少"）
MIN_QUESTION_LEN = 30  # PRD 7.2：短于 30 字符属高风险区


def load(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_question_integrity(q):
    """R-04 题干完整性：返回问题类型列表（可叠加）。"""
    issues = []
    q = q.strip()
    if len(q) < MIN_QUESTION_LEN:
        issues.append("short_question")
    if RE_ELLIPSIS.search(q):
        issues.append("ellipsis_truncation")
    if RE_DANGLING.search(q):
        issues.append("dangling_punctuation")
    if RE_MISSING_CHAR.search(q):
        issues.append("missing_char")
    # 题干以问号结尾但前一个字是标点/空白
    if q and q[-1] in "，,。 ":
        issues.append("truncated_tail")
    return issues


def _anchor_numbers(value):
    """把锚点值拆成数字列表。

    锚点在 jsonl 里是 list（["28.5 m", "4 mm"]），但在 qa_pairs.csv 里是 ";" 分隔字符串，
    且带空格（"19.5 %"）。精确字符串匹配会漏掉题干里写成 "19.5%" 的情况，
    所以统一按**数字值**比对。
    """
    if isinstance(value, str):
        value = [v for v in value.split(";") if v.strip()]
    out = []
    for item in (value or []):
        for m in re.findall(r"\d+(?:\.\d+)?", str(item)):
            out.append(m)
    return out


def check_answer_leakage(q, row):
    """R-01 题干不泄漏答案：题干直接抄了答案里的锚点数值。

    按数字值比对，不按字符串——题干写 "19.5%"、锚点写 "19.5 %" 也是泄漏。
    """
    qnums = set(re.findall(r"\d+(?:\.\d+)?", q))
    leaked = []
    for n in _anchor_numbers(row.get("must_include_numbers")):
        if n in qnums:
            leaked.append(n)
    return sorted(set(leaked))


def check_clause_derived(q, row):
    """检测题干是否由 clause_no 字符串切片而来（**主因**，不是"少几个字"）。

    实测根因：候选生成模板是「《文档名》中，{clause_no 片段}是怎么规定的？」，
    而 clause_no 是切条款时取标题的前若干字符，切点落在词中间就会得到：
      - "《…》中，擅自离开是怎么规定的？"（原文 "2.3.3作业期间不得擅自离开…"）
      - "《…》中，"先通风是怎么规定的？"（引号被切在中间）
      - "器主装置的限值是多少？"（丢了 "（五）主要装"）
    这类题**不能靠补字修好，必须让 LLM 看 gold_answer 全文重新组织问法**。

    判据：题干去掉书名号包裹的文档名后，剩余片段与 clause_no 有 >= 4 字的公共子串。
    """
    body = re.sub(r"《[^》]*》", "", q).strip()
    clause = (row.get("clause_no") or "").strip()
    if len(body) < 4 or len(clause) < 4:
        return False
    # 公共子串长度
    best = 0
    for i in range(len(body)):
        for j in range(i + best + 1, len(body) + 1):
            if body[i:j] in clause:
                best = j - i
            else:
                break
    return best >= 4


def build_flags(rows):
    """逐条打标记。返回 (flags_list, stats)。

    🔴 标记按**行索引**存，不按 qid：候选池有 7 个重复 qid，
    用 qid 做 key 会让后写入的行覆盖前一行，导致前面的题**丢标记**
    （实测症状：missing_* 命中数 835 而不是 842，差 7 行 = 重复 qid 数）。
    """
    flags = []
    for r in rows:
        f = collections.OrderedDict()
        q = r.get("question", "") or ""

        integrity = check_question_integrity(q)
        if integrity:
            f["integrity"] = integrity

        # R-02 数值题锚点
        if r.get("question_type") == "数值题" and not (r.get("must_include_numbers") or []):
            f["empty_number_anchor"] = True

        # R-01 题干泄漏答案数值
        leaked = check_answer_leakage(q, r)
        if leaked:
            f["leaked_numbers"] = leaked

        # 🔴 主因：题干由 clause_no 切片而来，切点落在词中间 -> 必须整句重写
        if check_clause_derived(q, r):
            f["clause_derived_question"] = True

        # R-07 / R-08 / R-09 M2 必须补的标注
        for field, key in (("time_mode", "missing_time_mode"),
                           ("query_time", "missing_query_time"),
                           ("expect_doc_id", "missing_expect_doc_id"),
                           ("key_points", "missing_key_points")):
            if not r.get(field):
                f[key] = True

        # R-05 / R-03 只能靠语义判断，标为待复核
        f["needs_human_review"] = True

        flags.append(f)

    # R-10 qid 重复
    qid_count = collections.Counter(r["qid"] for r in rows)
    dup_qids = {q for q, c in qid_count.items() if c > 1}
    for i, r in enumerate(rows):
        if r["qid"] in dup_qids:
            flags[i]["duplicate_qid"] = True

    # R-11 同 doc_id + clause_no 多题
    clause_count = collections.Counter((r["doc_id"], r["clause_no"]) for r in rows)
    for i, r in enumerate(rows):
        if clause_count[(r["doc_id"], r["clause_no"])] > 1:
            flags[i].setdefault("same_clause_multi_question", True)

    # R-06 同问法跨文档 -> 答案冲突（版本链候选），只标不改
    by_norm_q = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = re.sub(r"\s+", "", r["question"])
        by_norm_q[key].append(i)
    for group in by_norm_q.values():
        if len(group) > 1 and len({rows[i]["doc_id"] for i in group}) > 1:
            doc_ids = [rows[i]["doc_id"] for i in group]
            for i in group:
                flags[i]["cross_doc_same_question"] = doc_ids

    stats = collections.Counter()
    for f in flags:
        for k, v in f.items():
            if v is True or (isinstance(v, list) and v):
                stats[k] += 1
    stats["_total_questions"] = len(rows)
    stats["_total_docs"] = len({r["doc_id"] for r in rows})
    stats["_dup_qids"] = len(dup_qids)
    return flags, stats


def write_report(rows, flags, stats, out_path):
    by_doc = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_doc[r["doc_id"]].append((i, r))

    lines = []
    lines.append("# 评测候选池普查报告\n")
    lines.append("生成脚本：`tools/audit_eval_candidates.py`（PRD 7.2 机器初筛）\n")
    lines.append("> 本报告只列问题，不做修改。修改属 M2 第二步（LLM 批量改写）与第三步（人工核对）。\n")

    lines.append("\n## 一、总览\n")
    lines.append(f"- 候选总数：**{stats['_total_questions']}**（事实题 "
                 f"{sum(1 for r in rows if r['question_type'] == '事实题')} + 数值题 "
                 f"{sum(1 for r in rows if r['question_type'] == '数值题')}）")
    lines.append(f"- 覆盖文档：**{stats['_total_docs']}** 份")
    lines.append(f"- 唯一 qid：**{stats['_total_questions'] - stats['_dup_qids']}** 个"
                 f"（重复 {stats['_dup_qids']} 个）")
    lines.append(f"- 存在至少一处问题的题：**{sum(1 for f in flags if any(k != 'needs_human_review' for k in f))}** 条\n")

    lines.append("\n## 二、根因（普查发现，非推测）\n")
    lines.append("**题干的问题不是「少了几个字」，而是生成模板把 `clause_no` 当正文切了一段。**\n")
    lines.append("候选生成模板形如「《文档名》中，{clause_no 片段}是怎么规定的？」，"
                 "而 `clause_no` 是切条款时取标题的前若干字符，**切点落在词中间**就会得到：\n")
    lines.append("| 实际题干 | clause_no 原文 | 问题 |")
    lines.append("|---|---|---|")
    lines.append("| `《…》中，擅自离开是怎么规定的？` | `2.3.3作业期间不得擅自离开，全过程掌握作业者…` | 切点丢掉了整条编号与前半句 |")
    lines.append("| `《…》中，“先通风是怎么规定的？` | `3.1按照”先通风、再检测、后作业”的原则…` | 引号被切在中间，问法不成句 |")
    lines.append("| `器主装置的限值是多少？` | `（五）主要装置的限值…` | 丢了「（五）主要装」，主语残缺 |")
    lines.append("| `在19.5%～23%以下的限值是多少？` | 数值题 | **直接抄了答案数值**（R-01 泄漏） |")
    lines.append("\n> 结论：这类题**补字修不好**，必须让 LLM 读 `gold_answer` 全文重新组织问法（见 M2 第二步）。\n")

    lines.append("\n## 三、问题分布（按质检项）\n")
    lines.append("| 质检项 | 命中题数 | 占比 | 说明 |")
    lines.append("|---|---|---|---|")
    desc = {
        "integrity": ("R-04 题干完整性", "题干被抽取截断 / 短题干 / 含省略号 / 悬空标点"),
        "clause_derived_question": ("R-04 题干完整性", "题干由 clause_no 切片而来，切点落在词中间"),
        "empty_number_anchor": ("R-02 数值题锚点", "数值题 must_include_numbers 为空"),
        "leaked_numbers": ("R-01 题干泄漏答案", "题干直接出现答案锚点数值"),
        "missing_time_mode": ("R-07 时间口径", "缺 time_mode（M2 必须补）"),
        "missing_query_time": ("R-07 时间口径", "缺 query_time（M2 必须补）"),
        "missing_expect_doc_id": ("R-08 期望版本", "缺 expect_doc_id（M2 必须补）"),
        "missing_key_points": ("R-09 关键信息点", "缺 key_points（M2 必须补）"),
        "duplicate_qid": ("R-10 标识符唯一", "qid 重复，主键有歧义"),
        "same_clause_multi_question": ("R-11 去重", "同 doc + 同 clause_no 有多题"),
        "cross_doc_same_question": ("R-06 唯一正答", "同问法跨文档出现，答案可能冲突"),
    }
    total = stats["_total_questions"]
    for key, (name, note) in desc.items():
        n = stats.get(key, 0)
        if n:
            lines.append(f"| {name} | {n} | {n / total:.1%} | {note} |")
    lines.append("\n> 四个 `missing_*` 是 **M2 必须补齐的新字段**，命中数 = 全部 "
                 f"{total} 条，属预期，非缺陷。\n")

    lines.append("\n## 三、按文档分组的问题清单\n")
    for doc_id in sorted(by_doc):
        doc_rows = by_doc[doc_id]
        bad = [(i, r) for i, r in doc_rows
               if any(k != "needs_human_review" for k in flags[i])]
        if not bad:
            continue
        lines.append(f"\n### `{doc_id}`（{len(doc_rows)} 条中 {len(bad)} 条有问题）\n")
        lines.append("| qid | 问题 | 说明 |")
        lines.append("|---|---|---|")
        for i, r in sorted(bad, key=lambda x: x[1]["qid"]):
            f = flags[i]
            notes = []
            for k, label in desc.items():
                if f.get(k):
                    v = f[k]
                    extra = f" → {v}" if isinstance(v, list) and k in ("leaked_numbers", "cross_doc_same_question") else ""
                    notes.append(f"{label}{extra}")
            lines.append(f"| `{r['qid']}` | {r['question'][:40]} | {'；'.join(notes)} |")

    lines.append("\n\n## 四、待人工核对项\n")
    lines.append(f"- 全部 {total} 条都需要**人工逐条核对**（PRD 7.2 明确：机器生成的题看起来对，"
                 "实际问法有歧义或答案不唯一，指标因此失真）。")
    lines.append("- R-03 答案唯一 / R-05 切题正确 脚本判不了，只标 `needs_human_review`。")
    lines.append("- 首批 300 条按 PRD 7.2 排期（约 5 小时人工核对）。\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--out-json", default=str(ROOT / "data" / "eval" / "audit_flags.json"))
    ap.add_argument("--out-md", default=str(ROOT / "data" / "eval" / "audit_report.md"))
    args = ap.parse_args()

    rows = load(pathlib.Path(args.candidates))
    flags, stats = build_flags(rows)

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        [{"row_index": i, "qid": r["qid"], "flags": f}
         for i, (r, f) in enumerate(zip(rows, flags))],
        ensure_ascii=False, indent=1), encoding="utf-8")

    write_report(rows, flags, stats, pathlib.Path(args.out_md))

    print(f"候选 {len(rows)} 条 / 文档 {stats['_total_docs']} 份")
    print(f"重复 qid: {stats['_dup_qids']}")
    for k in ("integrity", "empty_number_anchor", "leaked_numbers",
              "duplicate_qid", "same_clause_multi_question", "cross_doc_same_question"):
        print(f"  {k}: {stats.get(k, 0)}")
    print(f"-> {out_json}")
    print(f"-> {args.out_md}")


if __name__ == "__main__":
    main()
