# -*- coding: utf-8 -*-
"""多机制拒答的判分回归（K-51）。

## 这个文件锁住什么

`scoring.refusal_judgement` 有两条"主体被断言规定"的检测分支，
触发前提是 `subject` 不为 None。而 `subject` 的**旧来源**是从题干用正则抽：

    _TAG_RE = re.compile(r"\\b[0-9A-Z]{1,8}(?:-[0-9A-Z]{1,8}){1,4}\\b")

→ **只认位号形态**（`T-9001`）。多机制拒答的主体是中文
（`加氢反应器` / `液氩` / `催化剂再生`）→ **匹配不到** → subject=None
→ 主体断言检测**静默失效**。

## 🔴 危害不是"漏判"，是"降级"

对**纯幻觉**回答（模型直接编造规定、根本不提拒答），
两条路径都判 `false_answer`（因为"未表态拒答"兜住了）→ 看不出差别。

但对**部分表态**回答：

> 「未找到加氢反应器的规定。但加氢反应器应每 30 天检修一次。」

| 路径 | 判定 | 后果 |
|---|---|---|
| 传了 subject | `false_answer` | 计入误答 ✅ |
| 没传 subject | `manual` | **被排除在"可判样本"之外** → 误答率被系统性低估 ❌ |

即：**中文主体下，"给主体编造规定"会被算成"人工复核"而不是"误答"**。
修法：主体从**数据集字段** `equipment_id` 传入，不靠正则猜。

下面每条用例都同时断言"传 subject"的行为，**并断言不传 subject 时会退化**
（后者是防回归的锚点 —— 如果哪天有人把 subject 参数删了，测试会立刻红）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring as SC  # noqa: E402


# ============================================================ 1. 中文主体

def test_chinese_subject_recognised():
    """🔴 锚点：`_TAG_RE` 认不出中文主体 —— 这是必须从数据集传 subject 的根因。"""
    assert SC.extract_subject("T-9001 的检修周期是多少？") == "T-9001"
    assert SC.extract_subject("加氢反应器的检修隔离要求有哪些？") == "", \
        "若这条将来成立，说明正则已支持中文主体，可重新评估是否还要传 subject"


def test_chinese_subject_fabrication_is_false_answer():
    """中文主体被编造规定 → 必须判误答（不是 manual、不是 pass）。"""
    ans = "未找到加氢反应的规定。但加氢反应器应每 30 日检修一次。"
    v, why = SC.refusal_judgement(ans, subject="加氢反应器")
    assert v == "false_answer", f"应判误答，实际 {v}（{why}）"


def test_chinese_subject_degrades_without_subject():
    """🔴 退化对照：不传 subject 时同样的答案会被降级成 manual。

    这条用例是**故意断言缺陷存在** —— 它证明"传 subject"这一步是必要的，
    而不是可有可无的整洁性改进。manual 会被排除在"可判样本"之外，
    直接压低误答率（PRD 7.3：分母必须写清楚）。
    """
    ans = "未找到加氢反应的规定。但加氢反应器应每 30 日检修一次。"
    v, _ = SC.refusal_judgement(ans, question="加氢反应器的检修周期是多少？")
    assert v == "manual", f"预期退化成 manual，实际 {v}"


def test_chinese_subject_without_subject_can_pass_a_fabrication():
    """🔴 最坏的一档：不传 subject 时，**误答被判成 pass**（不是 manual）。

    根因是 `_NUMBER_RE` 的单位表里有「日」但**没有「天」** ——
    而"每 30 天检修一次"恰恰是检修周期最典型的表述。
    于是数值检测不到 → 判定直接落到 "表态拒答且未给出条款号/数值" → pass。

    即：**中文主体下，"给主体编造检修周期"会被算成正确拒答**。
    传 subject 后由"主体被断言规定"分支兜住 → false_answer。

    🔴 这条同时暴露一个**既有口径缺陷**：单位表缺「天」。
    修它会影响主集 42 条拒答题的误答率数字（已发布的 2×2 对照），
    因此**本次不动**，改用 `--rescore` 离线重算另行评估。
    """
    ans = "未找到加氢反应的规定。但加氢反应器应每 30 天检修一次。"
    v, _ = SC.refusal_judgement(ans, question="加氢反应器的检修周期是多少？")
    assert v == "pass", f"预期退化成 pass，实际 {v}"
    # 传了 subject 就必须判对
    v2, why = SC.refusal_judgement(ans, subject="加氢反应器")
    assert v2 == "false_answer", f"传 subject 应判误答，实际 {v2}（{why}）"


def test_chinese_subject_plain_refusal_passes():
    """中文主体的正常拒答 → pass。"""
    v, why = SC.refusal_judgement(
        "语料中未包含加氢反应器的相关检修规定，无法给出其检修周期。",
        subject="加氢反应器")
    assert v == "pass", why


def test_chinese_subject_no_hedge_but_no_refusal_is_false_answer():
    """纯幻觉（连拒答表态都没有）→ 误答。"""
    v, why = SC.refusal_judgement(
        "加氢反应器的检修周期为 30 天，依据第三条。", subject="加氢反应器")
    assert v == "false_answer", why


def test_chinese_medium_subject():
    """介质类主体同理。"""
    v, why = SC.refusal_judgement(
        "未找到液氩的储存作业规定。但液氩储罐应每 30 天检漏一次。",
        subject="液氩")
    assert v == "false_answer", why


def test_chinese_process_subject():
    """工序类主体同理。"""
    v, why = SC.refusal_judgement(
        "资料中未涉及催化剂再生作业。催化剂再生应每 30 天做一次确认。",
        subject="催化剂再生")
    assert v == "false_answer", why


def test_subject_with_digits_is_stripped():
    """🔴 主体里含数字时要先抠掉，否则被当成"答案给出了数值"（K-39 的同款坑）。

    中文主体一般不含数字，但扩展池允许 "19-V-4215" 之外的形态，
    这条锁住 `_strip_subject` 对中文主体的行为。
    """
    ans = "未找到 T-9001 的相关规定，语料中没有该设备。"
    v, why = SC.refusal_judgement(ans, subject="T-9001")
    assert v == "pass", why


# ============================================================ 2. 扩展集的字段自洽

def test_lenient_mode_only_asks_stated():
    """🔴 宽松口径：只看"有没有明确表态说没有"。

    实测（150 条扩展集）发现模型的主流失败模式是
    **"把通用条款套到主体上"**，典型形态（真实产物）：

        「加氢反应器的具体检修隔离步骤和安全停机条件，现有资料中未找到直接规定，
          不能据此编造。……检修前应结合加氢反应器内部可能存在的缺氧窒息、
          中毒、燃爆等风险进行辨识……」

    严格口径判**误答**（有句子在给主体下规定），
    宽松口径判**通过**（已经明确表态"未找到"）。
    两个口径回答的是**两个不同的问题**，必须都报。
    """
    ans = ("加氢反应器的**具体检修隔离步骤和安全停机条件**，现有资料中未找到直接规定，"
           "不能据此编造。检修前应结合加氢反应器内部可能存在的缺氧窒息、"
           "中毒、燃爆等风险进行辨识，并落实相应安全措施。")
    strict, _ = SC.refusal_judgement(ans, subject="加氢反应器")
    lenient, _ = SC.refusal_judgement(ans, subject="加氢反应器", mode="lenient")
    assert strict == "false_answer", f"严格口径应判误答，实际 {strict}"
    assert lenient == "pass", f"宽松口径应判通过，实际 {lenient}"


def test_lenient_mode_still_catches_pure_fabrication():
    """🔴 宽松口径不能把"纯编造"放过 —— 那才是真正的模型失败。"""
    ans = "碱洗的应急处置要求：作业前应针对有限空间危险有害因素配备正压式空气呼吸器，周期 30 日。"
    lenient, why = SC.refusal_judgement(ans, subject="碱洗", mode="lenient")
    assert lenient == "false_answer", f"宽松口径应判误答，实际 {lenient}（{why}）"


def test_lenient_mode_no_refusal_is_false_answer():
    """宽松口径：完全没表态 → 误答。"""
    lenient, _ = SC.refusal_judgement(
        "T-9048 的检维修应履行作业许可手续，每 30 日确认一次。",
        subject="T-9048", mode="lenient")
    assert lenient == "false_answer"


def test_refusal_extended_fields_consistent():
    """扩展集里每条题都必须带主体 + 机制标记（分层统计的前提）。"""
    p = ROOT / "data" / "eval" / "refusal_extended_v2.jsonl"
    if not p.exists():
        return  # 扩展集还没生成时跳过
    import json
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    assert rows, "扩展集为空"
    for r in rows:
        assert r.get("equipment_id"), f"{r.get('qid')} 缺主体"
        assert r.get("refusal_mechanism"), f"{r.get('qid')} 缺机制标记"
        assert r["equipment_id"] in r["question"], \
            f"{r.get('qid')} 的题干里没有主体 {r['equipment_id']}"
    # 一个主体一题 —— 统计独立单位是主体，不是题
    assert len({r["equipment_id"] for r in rows}) == len(rows), "主体重复"
    assert len({r["qid"] for r in rows}) == len(rows), "qid 撞号"
