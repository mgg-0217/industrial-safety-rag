"""题集自动质检 —— PRD M2 第三步之前的机器自检。

在**精修后的题集**（data/eval/test_set.jsonl）上再跑一遍 PRD 7.2 的质检标准。
`tools/audit_eval_candidates.py` 查的是候选池，这里查的是成品——
两者查的不是一回事：候选池的问题是"题干残缺"，成题集的问题是
"改写引入的新问题"（题干泄漏答案、key_points 编不出来、时间口径填错）。

🔴 最关键的一条是 **key_point 可达性**：
  PRD 7.3 的 `key_point_coverage` = 命中数 / 总数，
  分子靠"模型答案里有没有这个信息点"判，分母靠 key_points 列表长度。
  如果 key_points 本身**在来源文档正文里就找不到**，
  那这个指标的分母就是假的（永远命中不了），整条评测链路作废。
  所以这条必须在跑任何指标之前查。

🔴 **判据查的字段在 2026-11-06 修正过**（第五轮 §2.2）：
  原来查 `gold_answer` —— 那是 M2 时代按 `clause_no` 机械切片的产物，
  而 key_point 现在来自检索证据 → 字段对不上，
  实测把 **46 条**判成不可达，其中 **40 条是"真原文，只是不在遗留切片里"**
  （判据问题，不是数据坏）。
  → 事实题改查**来源文档正文**（`src/eval_align.py::qc_source`）。
  ⚠️ 对齐流程仍用 `alignment_source`（对齐要的是"那个条款"，不是整篇正文）。

输出：
  data/eval/check_report.md    人读清单
  data/eval/check_flags.json   机读标记（可直接喂给人工核对的抽样脚本）
"""
import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
REPORT_MD = ROOT / "data" / "eval" / "check_report.md"
REPORT_JSON = ROOT / "data" / "eval" / "check_flags.json"

sys.path.insert(0, str(ROOT / "src"))
import eval_align as A  # noqa: E402  按题型分源的对齐逻辑（与 fix_key_points 共用）
import versioning as V  # noqa: E402  版本区间判定（三态）用于陷阱题唯一正答检查

MANIFEST = ROOT / "data" / "manifest.json"

MIN_QUESTION_LEN = 20
RE_ELLIPSIS = re.compile(r"\.{3,}|…{2,}")

# 量词 → 该量的单位。用来查**数值题的题干问的量与锚点单位是否对得上**。
# 实测踩过：题干问「氮气吹扫压力限值」，锚点却是 "1000 kg" / "24 小时" / "71 ℃"，
# 限值根本不在锚点里——数值保真判定会直接判错。这类错脚本判不出来，
# 但「题干量词 ↔ 锚点单位」这一层能查出来。
QUANTITY_UNITS = {
    "压力": ["MPa", "kPa", "Pa", "bar", "mpa", "kpa"],
    "重量": ["kg", "t", "吨", "公斤", "千克"],
    "质量": ["kg", "t", "吨", "公斤", "千克"],
    "温度": ["℃", "°C", "摄氏度"],
    "长度": ["mm", "cm", "m", "米", "毫米", "厘米"],
    "高度": ["mm", "cm", "m", "米", "毫米", "厘米"],
    "距离": ["mm", "cm", "m", "米", "毫米", "厘米"],
    "时间": ["h", "min", "s", "小时", "分钟", "秒", "天", "年", "月"],
    "时长": ["h", "min", "s", "小时", "分钟", "秒"],
    "浓度": ["mg/m³", "mg/m3", "%", "ppm", "mol", "g/m³"],
    "流量": ["m³/h", "L/min", "m3/h", "L/min", "m³", "L"],
    "速度": ["m/s", "km/h", "r/min", "Hz"],
    "电压": ["V", "kV", "伏"],
    "电流": ["A", "mA", "安"],
    "电阻": ["Ω", "MΩ", "兆欧"],
    "功率": ["kW", "W", "MW"],
    "存量": ["t", "m³", "m3", "%", "kg"],
    "数量": ["人", "次", "个", "台", "件", "套"],
    "比例": ["%", "‰"],
}


def load(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_for_match(text):
    """匹配用的归一化：去掉所有空白 + 统一标点 + 去全角。

    key_points 是 LLM 写的短语，和原文的差异通常在标点与空格上
    （"由纪工委进一步调查处理" vs "由纪工委进一步调查处理。"）。
    """
    if not text:
        return ""
    t = re.sub(r"\s+", "", text)
    table = str.maketrans({
        "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
        "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "－": "-", "～": "~", "％": "%", "：": ":",
    })
    t = t.translate(table)
    return t.lower()


#: key_point 可达性判据的版本号。
#: **v1** = 查 `gold_answer` + 三级判据（含"字符集覆盖 >= 0.8"）
#: **v2** = 查**来源文档正文** + 按源类型分级（长文本禁用"字符集覆盖"级）
#: 🔴 跨判据版本的数字不能直接比（K-61）。
REACHABILITY_JUDGE = "v2"


def check_key_point_reachable(kp, answer, allow_char_coverage=True):
    """key_point 是否能在源文本里找到依据。

    判据（从宽到严，逐级放宽）：
      1. 归一化后**完全包含**
      2. 归一化后**前缀包含**（kp 的前 85%，允许 LLM 尾部多写几个字）
      3. 字符集覆盖率 >= 0.8（LLM 可能把语序调整了）

    🔴 **第 3 级只能用在短源上**（v2 实测）：
    源换成整篇文档正文后，"字符集覆盖 0.8"**几乎必然成立**
    （高频字把覆盖率拉满）→ 实测把 **63 条真·跨文档 key_point** 判成可达
    （假阴性：它们是应急预案之间的通用套话，根本不在出处文档里）。
    → 长源（`doc_md`）只走第 1、2 级。

    返回 (是否可达, 用了哪一级)
    """
    if not answer:
        return False, "no_answer"
    a = norm_for_match(answer)
    k = norm_for_match(kp)
    if not k:
        return False, "empty_kp"
    if k in a:
        return True, "exact"
    # 允许 kp 比原文多几个字（如 LLM 补了"的"）
    if len(k) > 6 and k[:int(len(k) * 0.85)] in a:
        return True, "prefix"
    if allow_char_coverage:
        kp_chars = set(k)
        if kp_chars and len(kp_chars & set(a)) / len(kp_chars) >= 0.8:
            return True, "coverage"
    return False, "unreachable"


def check_record(r):
    """返回该题的标记字典。"""
    flags = {}
    q = (r.get("question") or "").strip()
    ans = r.get("gold_answer") or ""
    kps = r.get("key_points") or []

    # 题干完整性
    if len(q) < MIN_QUESTION_LEN:
        flags["question_too_short"] = True
    if RE_ELLIPSIS.search(q):
        flags["ellipsis_in_question"] = True

    # 题干泄漏答案
    if ans:
        # 数值锚点泄漏（锚点在 jsonl 里是 list，在 csv 里是 ";" 分隔字符串）
        anchors_raw = r.get("must_include_numbers") or []
        if isinstance(anchors_raw, str):
            anchors_raw = [a for a in anchors_raw.split(";") if a.strip()]
        for item in anchors_raw:
            for n in re.findall(r"\d+(?:\.\d+)?", str(item)):
                if n and n in q:
                    flags["leaked_number"] = n
                    break
        # key_point 整句泄漏（题干里直接抄了答案原句）
        for kp in kps:
            kn = norm_for_match(kp)
            if len(kn) >= 8 and kn in norm_for_match(q):
                flags["leaked_key_point"] = kp
                break

    # key_point 可达性
    # 🔴 对齐源按题型分（见 src/eval_align.py）：版本陷阱题的答案在
    #    `expect_doc_id` 指向的文档正文里，跨文档推理题在两份文档正文里，
    #    用 gold_answer 查必然判成「不可达」——那是判据用错字段，不是数据坏。
    unreachable = []
    levels = []
    source, mode = A.qc_source(r)
    if mode == A.REFUSAL_CRITERIA:
        # 拒答题的 key_point 是判分标准（「是否声明查不到」），不是原文断言，
        # 它的正确答案就是「语料里没有」，压根不指向任何原文 —— 不做字面判定。
        pass
    else:
        # 🔴 长源（整篇文档正文）禁用「字符集覆盖」级 —— 见函数 docstring
        allow_cc = (mode != "doc_md")
        for kp in kps:
            ok, lvl = check_key_point_reachable(kp, source, allow_cc)
            levels.append(lvl)
            if not ok:
                unreachable.append(kp)
        if unreachable:
            flags["unreachable_key_points"] = unreachable
    if kps and not ans and r.get("question_type") not in ("版本陷阱题", "跨文档推理题"):
        flags["no_gold_answer"] = True
    # 🔴 非拒答题缺 key_points → `key_point_coverage` **无分母**，该题在覆盖率指标里
    #    会被**静默跳过**（实测 1 条跨文档推理题 `t_5310767fe3d2`）。
    #    拒答题的 key_point 是判分标准，本来就不是覆盖率口径 → 排除。
    if not kps and r.get("question_type") != "拒答题":
        flags["no_key_points"] = True
    if kps and len(kps) > 4:
        flags["too_many_key_points"] = len(kps)

    # 时间口径
    if not r.get("time_mode"):
        flags["missing_time_mode"] = True
    if not r.get("query_time"):
        flags["missing_query_time"] = True
    if r.get("time_mode") == "as_of" and not str(r.get("query_time") or "").endswith("-12-31"):
        flags["as_of_not_year_end"] = r.get("query_time")

    # 期望版本
    if r.get("question_type") == "版本陷阱题" and not r.get("expect_doc_id"):
        flags["missing_expect_doc_id"] = True

    # 数值题锚点
    if r.get("question_type") == "数值题":
        anchors = r.get("must_include_numbers")
        if isinstance(anchors, str):
            anchors = [a for a in anchors.split(";") if a.strip()]
        if not anchors:
            flags["empty_number_anchor"] = True
        else:
            # 题干问的量 ↔ 锚点单位是否对得上
            # 🔴 别写 `[q for q in QUANTITY_UNITS if q in q]`：
            #    推导式的 q 遮蔽了外层的题干变量，`q in q` 恒真，量词集合会变成全集。
            q_text = r.get("question") or ""
            q_quant = [quant for quant in QUANTITY_UNITS if quant in q_text]
            if q_quant:
                all_anchor_text = " ".join(str(a) for a in anchors)
                needed = set()
                for quant in q_quant:
                    needed.update(QUANTITY_UNITS[quant])
                present = set()
                for u_list in QUANTITY_UNITS.values():
                    for u in u_list:
                        if u in all_anchor_text:
                            present.add(u)
                # 题干问的量，其单位在锚点里一个都没有 -> 量纲对不上
                if needed and not (needed & present):
                    flags["quantity_unit_mismatch"] = {
                        "question_quantities": q_quant,
                        "anchors": anchors,
                    }

    return flags


def check_trap_uniqueness(rec, manifest_by_id):
    """版本陷阱题的「唯一正答」机器判定。

    PRD 7.2 的第 6 条：「陷阱题有唯一正答 —— 版本陷阱题不能两个版本都合理」。

    这条**可以机器判**，判据是版本区间：
      - `as_of` 口径：期望命中的文档必须在该时点**生效**，
        且同一条版本链上的**其它版本在该时点必须不生效**（否则两个版本都合理）；
      - `current` 口径：期望命中的文档必须在 `query_time` 生效，
        且链上被它取代的旧版本必须**已失效**（否则会召回过期版本）。

    🔴 区间判定必须是三态（True / False / UNKNOWN）：
    生效日未知的文档不能当成"永远现行"（`"" <= "2026-11-02"` 恒真的坑）。
    UNKNOWN 时**不判失败**，只记为"无法判定"，交给人工核对。
    """
    qid_expect = rec.get("expect_doc_id")
    if not qid_expect:
        return None
    when = rec.get("query_time")
    if not when:
        return None
    doc = manifest_by_id.get(qid_expect)
    if doc is None:
        return {"reason": "expect_doc_id 不在 manifest 里"}

    state = V.is_effective(doc, when)
    if state is None:
        return {"reason": "生效日未知，无法判定（交人工）", "doc": qid_expect}

    # 同链其它版本
    others = []
    for d in manifest_by_id.values():
        if d.get("doc_id") == qid_expect:
            continue
        if d.get("superseded_by") == qid_expect or d.get("superseded_by") is None:
            # 只看与该文档直接相关的版本对（新版取代旧版）
            if d.get("doc_id") and (d.get("superseded_by") == qid_expect
                                    or doc.get("superseded_by") == d.get("doc_id")):
                others.append(d)
    conflicts = []
    for o in others:
        st = V.is_effective(o, when)
        if st is True:
            conflicts.append(o["doc_id"])
    if conflicts:
        return {"reason": "同链其它版本在该时点也生效 -> 两个版本都合理",
                "conflicts": conflicts, "doc": qid_expect}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(TEST_SET))
    args = ap.parse_args()

    rows = load(pathlib.Path(args.test_set))
    print(f"题集：{len(rows)} 条")

    # qid 唯一性
    qid_count = collections.Counter(r["qid"] for r in rows)
    dup_qids = {q for q, c in qid_count.items() if c > 1}

    # manifest：版本陷阱题的唯一正答检查要用
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest = manifest.get("documents", manifest)
    manifest_by_id = {d["doc_id"]: d for d in manifest}

    flags_list = []
    for r in rows:
        f = check_record(r)
        if r["qid"] in dup_qids:
            f["duplicate_qid"] = True
        # PRD 7.2 第 6 条：陷阱题有唯一正答（机器可判）
        if r.get("question_type") == "版本陷阱题":
            prob = check_trap_uniqueness(r, manifest_by_id)
            if prob:
                f["trap_not_unique"] = prob
        # PRD 7.2 第 9 条：事实题必须标 key_points
        if r.get("question_type") == "事实题" and not (r.get("key_points") or []):
            f["missing_key_points"] = True
        flags_list.append(f)

    # PRD 7.2 第 11 条：去重
    #  - 题干字面重复（归一化后）-> 必合并
    #  - 同题干不同答案 -> 答案不唯一，直接判失败
    qnorm = collections.defaultdict(list)
    for r, f in zip(rows, flags_list):
        qnorm[norm_for_match(r.get("question") or "")].append((r, f))
    for key, items in qnorm.items():
        if not key:
            continue
        if len(items) > 1:
            answers = {norm_for_match((r.get("gold_answer") or "")[:200]) for r, _ in items}
            for r, f in items:
                if len(answers) > 1:
                    f["ambiguous_answer"] = sorted(
                        x.get("qid") for x, _ in items)
                else:
                    f["duplicate_question"] = sorted(
                        x.get("qid") for x, _ in items)

    stats = collections.Counter()
    for f in flags_list:
        for k, v in f.items():
            if v is True or (isinstance(v, list) and v):
                stats[k] += 1
    stats["_total"] = len(rows)

    # ---------- 机读输出 ----------
    payload = [{"qid": r["qid"], "question_type": r.get("question_type"),
                "question": r.get("question"), "flags": f}
               for r, f in zip(rows, flags_list)]
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    # ---------- 人读输出 ----------
    L = []
    L.append("# 题集质检报告\n")
    L.append("生成脚本：`tools/check_test_set.py`（PRD 7.2 质检标准，成题集版）\n")
    L.append(f"题集：{len(rows)} 条\n")
    L.append("\n## 一、问题分布\n")
    L.append("| 质检项 | 命中 | 说明 |")
    L.append("|---|---|---|")
    desc = {
        "unreachable_key_points": ("🔴 key_point 不可达",
            "key_point 在来源文档正文里找不到 → key_point_coverage 的分母是假的"),
        "leaked_number": ("题干泄漏数值", "题干出现答案锚点数值"),
        "leaked_key_point": ("题干泄漏答案原句", "题干整句抄了 key_point"),
        "no_gold_answer": ("缺 gold_answer", "非合成类题目必须有原文答案"),
        "no_key_points": ("缺 key_points", "key_point_coverage 无分母 → 该题会被静默跳过"),
        "question_too_short": ("题干过短", f"短于 {MIN_QUESTION_LEN} 字符"),
        "ellipsis_in_question": ("题干含省略号", "疑似截断"),
        "missing_time_mode": ("缺 time_mode", "7.3 的过期版本指标依赖它"),
        "missing_query_time": ("缺 query_time", "同上"),
        "as_of_not_year_end": ("as_of 不是年末口径", "PRD 5.3 要求 Y-12-31"),
        "missing_expect_doc_id": ("缺 expect_doc_id", "版本陷阱题无法判版本对错"),
        "empty_number_anchor": ("数值题空锚点", "无法做数值保真判定"),
        "quantity_unit_mismatch": ("🔴 量纲对不上",
            "题干问的量与锚点单位不是一回事（数值保真必判错）"),
        "duplicate_qid": ("qid 重复", "主键有歧义"),
        "too_many_key_points": ("key_point 超过 4 个", "PRD 7.3 规定 1–4 个"),
        "trap_not_unique": ("🔴 陷阱题正答不唯一",
            "期望版本在该时点不生效，或同链其它版本也生效 -> 两个版本都合理"),
        "missing_key_points": ("事实题缺 key_points", "key_point_coverage 的分母为 0"),
        "duplicate_question": ("题干字面重复", "语义重复的题应合并（PRD 7.2 第 11 条）"),
        "ambiguous_answer": ("🔴 同题干不同答案",
            "同一问题在语料里有多个答案 -> 答案不唯一"),
    }
    for k, (name, note) in desc.items():
        n = stats.get(k, 0)
        if n:
            L.append(f"| {name} | {n} | {note} |")
    L.append("")

    clean = sum(1 for f in flags_list if not f)
    L.append(f"\n**无标记：{clean} / {len(rows)} 条"
             f"（{clean / max(len(rows), 1):.1%}）**\n")

    # 逐条列出有问题的
    L.append("\n## 二、有标记的题（人工核对的重点）\n")
    L.append("| qid | 题型 | 问题 |")
    L.append("|---|---|---|")
    for r, f in zip(rows, flags_list):
        if not f:
            continue
        notes = []
        for k, (name, _note) in desc.items():
            if f.get(k):
                v = f[k]
                extra = f" → {v}" if isinstance(v, list) else (
                    f" → {v}" if isinstance(v, str) else "")
                notes.append(f"{name}{extra}")
        q = (r.get("question") or "").replace("|", "/")
        L.append(f"| `{r['qid']}` | {r.get('question_type')} | {q[:30]} | {'；'.join(notes)} |")

    # 分题型统计
    L.append("\n\n## 三、分题型覆盖\n")
    L.append("| 题型 | 条数 | 占比 |")
    L.append("|---|---|---|")
    by_type = collections.Counter(r.get("question_type") for r in rows)
    for t, n in by_type.most_common():
        L.append(f"| {t} | {n} | {n / len(rows):.1%} |")
    L.append(f"\n目标（PRD 7.1 五类必测题）：事实题 / 数值题 / 版本陷阱题 / 拒答题 / 跨文档推理题\n")
    missing_types = {"事实题", "数值题", "版本陷阱题", "拒答题", "跨文档推理题"} - set(by_type)
    if missing_types:
        L.append(f"🔴 **尚缺的题型**：{'、'.join(sorted(missing_types))}\n")
    else:
        L.append("✅ 五类必测题已全部覆盖\n")

    L.append("\n## 四、PRD 7.2 质检标准逐条通过情况\n")
    L.append("> PRD 9 节 M2 的验收标准要求「11 条质检标准**逐条记录**通过情况」。"
             "下表把 PRD 7.2 的 11 条标准逐条对上机器判据与命中数。\n")
    L.append("| # | PRD 7.2 标准 | 机器判据 | 命中 | 结论 |")
    L.append("|---|---|---|---|---|")
    std = [
        ("题干不泄漏答案", "题干含锚点数值 / 含 key_point 原句",
         ["leaked_number", "leaked_key_point"]),
        ("答案唯一", "同题干却给出不同 gold_answer",
         ["ambiguous_answer"]),
        ("数值题必须带锚点", "`must_include_numbers` 非空",
         ["empty_number_anchor"]),
        ("题干完整性", f"题干 ≥ {MIN_QUESTION_LEN} 字、无省略号",
         ["question_too_short", "ellipsis_in_question"]),
        ("切题正确", "辅助信号：key_point 在原文可达",
         ["unreachable_key_points"]),
        ("陷阱题有唯一正答", "版本区间三态判定 + 同链冲突检查",
         ["trap_not_unique"]),
        ("每题标注时间口径", "`time_mode` / `query_time` 非空；as_of 须为年末",
         ["missing_time_mode", "missing_query_time", "as_of_not_year_end"]),
        ("每题标注期望版本", "版本陷阱题有 `expect_doc_id`",
         ["missing_expect_doc_id"]),
        ("每题标注 key_points", "事实题有 1–4 个 key_point",
         ["missing_key_points", "too_many_key_points"]),
        ("标识符唯一", "`qid` 全局唯一",
         ["duplicate_qid"]),
        ("去重", "题干字面重复（同条款多题属正常，角度不同）",
         ["duplicate_question"]),
    ]
    for i, (name, crit, keys) in enumerate(std, 1):
        hits = {k: stats.get(k, 0) for k in keys}
        total = sum(hits.values())
        verdict = "✅ 通过" if total == 0 else f"⚠️ {total} 条待处理"
        hit_txt = "、".join(f"{k}×{n}" for k, n in hits.items() if n) or "—"
        L.append(f"| {i} | {name} | {crit} | {hit_txt} | {verdict} |")
    L.append("\n> 「答案唯一」「切题正确」这两条**机器只能给信号，不能给结论** —— "
             "最终要靠人工核对（PRD 7.2 明确「人工核对不能省」）。\n")

    L.append("\n## 五、下一步\n")
    L.append("- 全部题 `review_status=pending`，**人工逐条核对未做，不得用于报指标**（PRD 7.2）。")
    L.append("- 人工核对重点：本报告第二节列出的题。\n")

    REPORT_MD.write_text("\n".join(L), encoding="utf-8")

    print("\n=== 问题分布 ===")
    for k, (name, _n) in desc.items():
        n = stats.get(k, 0)
        if n:
            print(f"  {name}: {n}")
    print(f"\n无标记 {clean}/{len(rows)} 条")
    print(f"题型分布：{dict(by_type)}")
    print(f"-> {REPORT_MD}")
    print(f"-> {REPORT_JSON}")


if __name__ == "__main__":
    main()
