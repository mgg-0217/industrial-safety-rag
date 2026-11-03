"""从真实语料自动构造评测题候选池 + 导出 QA 型 CSV。

对应 PRD 的评测方案要求（docs/PRD.md 第 7 节，五类必测题）：
  1. 事实题     —— 条款级问答，测忠实度与引用正确率        ← 本脚本产出
  2. 数值题     —— 答案是具体数字/区间，测数值保真（LLM 不得改写）  ← 本脚本产出
  3. 版本陷阱题 —— 问 2023 年答 2025 版即判错            ← 需人工/LLM 构造
  4. 拒答/澄清题 —— 语料里根本没有的设备/条款，必须拒答      ← 需人工/LLM 构造
  5. 跨文档推理题 —— 同一台设备的规程要求 + 历史事故对比    ← 需人工/LLM 构造

本脚本只产出前两类（事实题 803 + 数值题 39），后三类是待补的空白。
另注意：本脚本产出的候选池**就是评测候选池**，若用作 QA-pair 索引
（PRD 第 3 节策略 6），必须与评测用子集严格不相交，否则评测作废。

输出：
  data/eval/candidates.jsonl  —— 候选题（含证据锚点）
  data/eval/qa_pairs.csv      —— QA 型 CSV（第6种切分策略的输入格式）
"""
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import normalize as N  # noqa: E402
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "data" / "md"
OUT = ROOT / "data" / "eval"
OUT.mkdir(parents=True, exist_ok=True)

# 条款切分：中文法规常用「第X条」「第X章」「一、」「（一）」「1.」
CLAUSE_RE = re.compile(
    r"(?:^|\n)\s*(?P<clause>"
    r"第[一二三四五六七八九十百零〇\d]+条"      # 第一条
    r"|第[一二三四五六七八九十\d]+章"          # 第三章
    r"|[一二三四五六七八九十]+、\s*\S{0,30}"   # 一、标题
    r"|（[一二三四五六七八九十]+）"             # （一）
    r"|\d+[\.、]\s*\S{0,30}"                  # 1. 标题
    r")\s*",
)
PAGE_RE = re.compile(r"<!--\s*page:(\d+)\s*-->")

#: 目录行特征：**连续 4 个以上的点**（前导点 + 页码）。
#: 目录条目不是条款 —— 见 `main()` 里的过滤说明。
TOC_LINE_RE = re.compile(r"[.…]{4,}")

# 数值 + 单位（用于数值保真锚定）
NUM_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*"
    r"(%|‰|MPa|kPa|Pa|bar|m³|m3|m²|m2|mm|cm|m|km/h|m/s|rpm|℃|°C|Hz|kHz|MHz|"
    r"V|A|kW|W|kVA|MVA|N·m|Nm|kg|mg/m³|ppm|mg|L|h|分钟|小时|秒|天|次|年|月|"
    r"级|类|项|条|人|台|套|个|卷|册|米|吨|t)",
    re.I,
)

# 判定"工程量"的单位：只有这些单位才配叫"限值/数值"，其余（人/条/项/年）多半是履历或计数
ENGINEERING_UNITS = {
    "%", "‰", "MPa", "kPa", "Pa", "bar", "m³", "m3", "m²", "m2", "mm", "cm", "m",
    "km/h", "m/s", "rpm", "℃", "°C", "Hz", "kHz", "MHz", "V", "A", "kW", "W",
    "kVA", "MVA", "N·m", "Nm", "kg", "mg/m³", "ppm", "mg", "L", "小时", "分钟", "米", "吨", "t",
}

# 数值区间（常出现在安全规程里，是最容易保真失败的形态）
RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*[~～\-—至]\s*(\d+(?:\.\d+)?)\s*%|"
    r"(\d+(?:\.\d+)?)\s*[~～]\s*(\d+(?:\.\d+)?)\s*(?:MPa|KPa|℃|%|m³|m3|ppm|小时|分钟|年|天)"
)

# 表格行（PDF 抽取后以 | 开头）——必须与正文分离，否则条款被表格截断
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

QUESTION_TEMPLATES = {
    "事实题": [
        "《{title}》中，{topic}是怎么规定的？",
        "{doc_short}对{topic}有什么要求？",
        "关于{topic}，{doc_short}的要求是什么？",
    ],
    "数值题": [
        "{topic}的限值是多少？",
        "{doc_short}规定{topic}应当控制在什么范围？",
        "{topic}的判定标准是什么？",
    ],
}


def load_manifest():
    mf = ROOT / "data" / "manifest.json"
    if not mf.exists():
        return {}
    return {e["doc_id"]: e for e in json.loads(mf.read_text(encoding="utf-8"))}


def strip_tables(text: str) -> str:
    """剥离 markdown 表格行，避免表格把条款正文截断成碎片。"""
    kept = []
    for ln in text.split("\n"):
        if TABLE_LINE_RE.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def body_with_map(raw: str):
    """返回 (工作文本, 归一化坐标 -> **原始 md 坐标** 的映射表)。

    工作文本 = 去 front-matter -> 去表格行 -> 归一化。
    映射表把每一步的偏移变化都串起来。

    🔴 **为什么必须有这个映射**：候选题是在**归一化文本**上切条款的，
    而 chunk 的 ``char_start/char_end`` 是**原始 md** 的偏移。
    两者坐标系不同（去表格行会移位，归一化会改长度），
    没有映射就无法把"题目出自原文的哪一段"落到具体 chunk 上 ——
    而 **gold chunk 的定义恰恰依赖这个**（PRD 7.3 的 Recall@k 才有意义）。
    """
    m = re.match(r"^---.*?\n---\n", raw, flags=re.S)
    if m:
        fm_len = m.end()
        body = raw[fm_len:]
    else:
        fm_len = 0
        body = raw
    base = list(range(fm_len, len(raw)))
    kept, kept_base = [], []
    pos = 0
    for ln in body.split("\n"):
        seg = len(ln) + 1
        if not TABLE_LINE_RE.match(ln):
            kept.append(ln)
            kept_base.extend(base[pos:pos + seg])
        pos += seg
    stripped = "\n".join(kept)
    norm, noff = N.normalize_with_map(stripped)
    raw_map = [kept_base[i] if i < len(kept_base) else None for i in noff]
    return norm, raw_map


def split_clauses(text: str):
    """把正文切成 (clause_no, body, start, end) 列表。

    start/end 是条款在**传入文本**里的字符区间（用于定位 gold chunk）。
    """
    marks = []
    for m in CLAUSE_RE.finditer(text):
        marks.append((m.start(), m.group("clause").strip()))
    if not marks:
        return []
    out = []
    for i, (pos, clause) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if len(body) < 40:
            continue
        out.append((clause, body, pos, end))
    return out


def page_of(text: str, pos: int) -> int | None:
    pages = PAGE_RE.findall(text)
    if not pages:
        return None
    before = text[:pos]
    last = PAGE_RE.findall(before)
    return int(last[-1]) if last else int(pages[0])


def extract_numbers(body: str) -> list[str]:
    """抽取可作为保真锚点的工程量：带工程单位，且排除年份等伪数值。"""
    out = []
    for num, unit in NUM_RE.findall(body):
        if unit not in ENGINEERING_UNITS:
            continue
        try:
            v = float(num)
        except ValueError:
            continue
        if 1900 <= v <= 2100 and unit in {"年"}:
            continue
        out.append(f"{num} {unit}")
    return sorted(set(out))


def topic_of(clause: str, body: str) -> str:
    """从条款标题里取主题词；标题太长时回退到正文中的义务性短语。"""
    t = re.sub(r"\s+", " ", clause)  # 跨行断裂要先合并
    t = re.sub(r"^(第[一二三四五六七八九十百零〇\d]+[条章]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[\.、])\s*", "", t)
    t = t.strip(" 、.()（）")
    # 定义式标题（"锅炉：利用各种燃料…"）取冒号前的名词部分
    if "：" in t or ":" in t:
        head = re.split(r"[：:]", t)[0].strip()
        if 2 <= len(head) <= 18:
            return head
    bad_tail = re.compile(r"(为|是|的|和|与|或|在|及|以|等|将|对|把|从)$")
    if 3 <= len(t) <= 18 and not re.search(r"[。；，、]", t) and not bad_tail.search(t):
        return t
    m = re.search(r"(?:应|须|必须|禁止|严禁|不得|按照|依据)([^，。；、]{3,16})", body)
    if m:
        cand = m.group(1).strip()
        if not bad_tail.search(cand):
            return cand
    return t[:16]


def main():
    manifest = load_manifest()
    rows, cands = [], []
    seen = set()
    for md_path in sorted(MD.glob("*.md")):
        doc_id = md_path.stem
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        # 🔴 **切条款之前先归一化**（2026-11-02 加）。
        # 起因：PDF 字体把 `4.1.2` 映射成**全角** `４．１．２`
        # （渲染出来是正常的半角，是 ToUnicode CMap 的映射问题），
        # 而 `CLAUSE_RE` 的 `[\\.、]` 只认**半角**点 →
        # `AQ_T3034` 切出 **0 条**条款，整份文档进不了评测池。
        # 归一化还顺带把位号破折号统一成半角（`19—V—4215` → `19-V-4215`），
        # 与 manifest 里人工标注的位号形态一致 —— 否则题干里的位号
        # 与 manifest 对不上，检索时"位号精确匹配"会失效（K-12）。
        # 实测影响面：26 份里只有 6 条现有条款的文本会变，
        # 且**全都是变好**（破折号/全角% → 半角），其余不变。
        #
        # 2026-11-03：改用 body_with_map() —— 同时拿到**原始 md 坐标**的映射，
        # 让每条候选题都能定位到原文的字符区间（M5 的 gold chunk 靠它定义）。
        body, raw_map = body_with_map(raw)
        meta = manifest.get(doc_id, {})
        title = meta.get("title", doc_id)
        doc_short = title if len(title) <= 14 else title[:14]
        hazards = meta.get("hazard_tags") or []
        equips = meta.get("equipment_tags") or []
        for clause, cbody, c_start, c_end in split_clauses(body):
            # 跳过明显是页眉页脚的碎块
            if len(cbody) > 3000:
                cbody = cbody[:3000]
            numbers = extract_numbers(cbody)
            rngs = RANGE_RE.findall(cbody)
            topic = topic_of(clause, cbody)
            if not topic or len(topic) < 2:
                continue
            # 🔴 **目录条目不是条款**（2026-11-02 加）。
            # 带前导点的目录行（"4.10 设备完好性管理 …… 11"）也是"条款号 + 正文"，
            # 条款切分会把**第一次出现**的条款号（目录里的那个）当成条款开头，
            # 于是这一条的正文就是那行目录 —— 题目问"第4.10条是什么内容"，
            # 答案却是"设备完好性管理 …… 11"，**纯噪声**。
            # 实测 1124 条候选里有 31 条是这种（AQ_T3034 11 条 / mem_xianshi 20 条）。
            # 真正的条款在文档后部还有一份（同样的条款号、真正的正文），
            # 过滤掉目录行之后保留的就是它。
            if TOC_LINE_RE.search(cbody):
                continue
            # 履历/名单类条款不适合出题：大量年份+姓名，无工程语义
            if re.search(r"(中共党员|现任|曾任|学历|职称|联系电话)", cbody) and len(numbers) <= 2:
                continue
            key = (doc_id, clause, topic)
            if key in seen:
                continue
            seen.add(key)
            pos = body.find(clause)
            page = page_of(body, max(pos, 0))
            is_numeric = bool(numbers) or bool(rngs)
            qtype = "数值题" if is_numeric else "事实题"
            tmpl = QUESTION_TEMPLATES[qtype]
            q = tmpl[0].format(title=title, topic=topic, doc_short=doc_short)
            rec = {
                "qid": f"{doc_id}__{clause}".replace(" ", ""),
                "doc_id": doc_id,
                "doc_title": title,
                "clause_no": clause,
                "page_no": page,
                "question_type": qtype,
                "question": q,
                "gold_answer": cbody[:600],
                "must_include_numbers": numbers[:12],
                "hazard_tags": hazards,
                "equipment_tags": equips,
                "source_url": meta.get("source_url", ""),
                "doc_no": meta.get("doc_no", ""),
                "version": meta.get("version", ""),
                "status": meta.get("status", ""),
                # 🔴 条款在**原始 md** 里的字符区间 ——
                # M5 的 gold chunk = 覆盖这个区间的 chunk（PRD 7.3 的 Recall@k）。
                "clause_char_start": (raw_map[c_start]
                                      if c_start < len(raw_map) else None),
                "clause_char_end": (raw_map[c_end - 1] + 1
                                    if 0 < c_end <= len(raw_map) else None),
            }
            cands.append(rec)
            rows.append({
                "qid": rec["qid"], "question": q, "answer": rec["gold_answer"],
                "doc_id": doc_id, "doc_title": title, "clause_no": clause,
                "page_no": page if page else "", "doc_no": meta.get("doc_no", ""),
                "version": meta.get("version", ""), "status": meta.get("status", ""),
                "question_type": qtype,
                "must_include_numbers": ";".join(numbers[:12]),
                "hazard_tags": ";".join(hazards), "equipment_tags": ";".join(equips),
                "source_url": meta.get("source_url", ""),
                "clause_char_start": (raw_map[c_start]
                                      if c_start < len(raw_map) else ""),
                "clause_char_end": (raw_map[c_end - 1] + 1
                                    if 0 < c_end <= len(raw_map) else ""),
            })

    with (OUT / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (OUT / "qa_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"候选题 {len(cands)} 条 -> {OUT/'candidates.jsonl'}")
    print(f"QA CSV {len(rows)} 行 -> {OUT/'qa_pairs.csv'}")
    for k, v in Counter(c["question_type"] for c in cands).most_common():
        print(f"  {k}: {v}")
    print(f"含数值锚点的题: {sum(1 for c in cands if c['must_include_numbers'])}")
    print(f"覆盖文档数: {len(set(c['doc_id'] for c in cands))}")
    print(f"有页码锚点的题: {sum(1 for c in cands if c['page_no'])}")


if __name__ == "__main__":
    main()
