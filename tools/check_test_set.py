"""题集自动质检 —— PRD M2 第三步之前的机器自检。

在**精修后的题集**（data/eval/test_set.jsonl）上再跑一遍 PRD 7.2 的质检标准。
`tools/audit_eval_candidates.py` 查的是候选池，这里查的是成品——
两者查的不是一回事：候选池的问题是"题干残缺"，成题集的问题是
"改写引入的新问题"（题干泄漏答案、key_points 编不出来、时间口径填错）。

🔴 最关键的一条是 **key_point 可达性**：
  PRD 7.3 的 `key_point_coverage` = 命中数 / 总数，
  分子靠"模型答案里有没有这个信息点"判，分母靠 key_points 列表长度。
  如果 key_points 本身**在 gold_answer 里就找不到**，
  那这个指标的分母就是假的（永远命中不了），整条评测链路作废。
  所以这条必须在跑任何指标之前查。

输出：
  data/eval/check_report.md    人读清单
  data/eval/check_flags.json   机读标记（可直接喂给人工核对的抽样脚本）
"""
import argparse
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
REPORT_MD = ROOT / "data" / "eval" / "check_report.md"
REPORT_JSON = ROOT / "data" / "eval" / "check_flags.json"

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


def check_key_point_reachable(kp, answer):
    """key_point 是否能在 gold_answer 里找到依据。

    判据（从宽到严，逐级放宽）：
      1. 归一化后**完全包含**
      2. 归一化后**子串包含**（去掉标点空格后）
      3. 字符集覆盖率 >= 0.8（LLM 可能把语序调整了）
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
    unreachable = []
    levels = []
    for kp in kps:
        ok, lvl = check_key_point_reachable(kp, ans)
        levels.append(lvl)
        if not ok:
            unreachable.append(kp)
    if unreachable:
        flags["unreachable_key_points"] = unreachable
    if kps and not ans and r.get("question_type") not in ("版本陷阱题", "跨文档推理题"):
        flags["no_gold_answer"] = True
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(TEST_SET))
    args = ap.parse_args()

    rows = load(pathlib.Path(args.test_set))
    print(f"题集：{len(rows)} 条")

    # qid 唯一性
    qid_count = collections.Counter(r["qid"] for r in rows)
    dup_qids = {q for q, c in qid_count.items() if c > 1}

    flags_list = []
    for r in rows:
        f = check_record(r)
        if r["qid"] in dup_qids:
            f["duplicate_qid"] = True
        flags_list.append(f)

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
            "key_point 在 gold_answer 里找不到 → key_point_coverage 的分母是假的"),
        "leaked_number": ("题干泄漏数值", "题干出现答案锚点数值"),
        "leaked_key_point": ("题干泄漏答案原句", "题干整句抄了 key_point"),
        "no_gold_answer": ("缺 gold_answer", "非合成类题目必须有原文答案"),
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

    L.append("\n## 四、下一步\n")
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
