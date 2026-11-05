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
import answering as AN  # noqa: E402


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

def test_global_boundary_covers_later_restating_sentence():
    """🔴 边界声明是**全局**的，不只是那一句。

    实测形态（`SYSTEM_V2` 下 T-9042 的真实输出）：
      「资料中**没有设备 T-9042 的专门规定**……」
      「以下为**通用要求，其适用性不等同于对设备 T-9042 的专门规定**：」
      「**因此，T-9042 检维修至少应按通用流程办理作业许可**、……」

    第三句含主体、含断言谓语，但它在**复述通用要求**，不是给 T-9042 编造规定。
    只看单句的判据会把它误判成误答（假阳性）。
    """
    ans = ("资料中**没有设备 T-9042 的专门规定**，无法针对该设备确定其专属许可手续。\n"
           "以下为**通用要求，其适用性不等同于对设备 T-9042 的专门规定**：\n"
           "1. 装置区施工和检维修作业应纳入作业许可管理范围。\n"
           "2. 因此，T-9042 检维修至少应按通用流程办理作业许可、审批前条件确认。")
    v, why = SC.refusal_judgement(ans, subject="T-9042")
    assert v == "bounded", f"应判 bounded（复述通用要求），实际 {v}（{why}）"


def test_global_boundary_does_not_excuse_real_fabrication():
    """🔴 但**真编造**仍要抓住：给出具体数值/条款号的不能放过。"""
    ans = ("资料中没有 T-9042 的专门规定。\n"
           "以下为通用要求，其适用性不等同于对 T-9042 的专门规定。\n"
           "**T-9042 的检修周期为 30 日，依据第三条执行。**")
    v, why = SC.refusal_judgement(ans, subject="T-9042")
    assert v == "false_answer", f"应判误答（真编造），实际 {v}（{why}）"


def test_no_boundary_still_fails():
    """🔴 没有边界声明时，含主体的断言句仍然判误答（口径没被普遍放宽）。"""
    ans = ("资料中没有 T-9042 的专门规定。\n"
           "因此，T-9042 检维修至少应按通用流程办理作业许可。")
    v, why = SC.refusal_judgement(ans, subject="T-9042")
    assert v == "false_answer", f"应判误答，实际 {v}（{why}）"


def test_absence_statement_detected():
    """🔴 缺失声明：主体必须**在同一句里**，否则代码兜底会漏兜。"""
    ok, sent = SC.has_subject_absence_statement(
        "资料中没有 T-9041 的专门规定，因此未找到其要求。", "T-9041")
    assert ok, sent
    # 🔴 主体不在同一句 → 不算
    ok2, _ = SC.has_subject_absence_statement(
        "资料中没有相关专门规定。T-9041 是某装置。", "T-9041")
    assert not ok2


def test_chunks_mention_subject():
    """🔴 判据：召回的 chunk 正文里**有没有提到该主体**。"""
    chunks = [{"text": "企业应建立作业许可管理制度。"},
              {"text": "T-9041 属于装置区检维修作业。"}]
    assert AN.chunks_mention_subject(chunks, "T-9041") is True
    assert AN.chunks_mention_subject(chunks, "T-9999") is False
    assert AN.chunks_mention_subject(chunks, None) is None


def test_attribution_fallback_appended_when_model_silent():
    """🔴 模型没说清楚 → 代码追加归因声明。"""
    need, fb, _poll = AN.attribution_fallback(
        "结论：设备 T-9041 的检维修应履行作业许可手续。",
        [{"text": "企业应建立作业许可管理制度。"}], "T-9041")
    assert need, "模型给主体下了规定却没说这是通用的 → 应追加"
    assert "T-9041" in fb and "不构成" in fb


def test_attribution_fallback_skips_when_model_already_clear():
    """🔴 模型自己说清楚了 → **不重复追加**。"""
    # 已经标注了适用性边界
    need, _fb, _poll = AN.attribution_fallback(
        "资料中没有 T-9041 的专门规定。以下为通用要求，"
        "其适用性不等同于对 T-9041 的专门规定。",
        [{"text": "企业应建立作业许可管理制度。"}], "T-9041")
    assert not need
    # 已经明说查不到（但没加边界声明）
    need2, _fb2, _poll2 = AN.attribution_fallback(
        "资料中未找到 T-9041 的相关管理规定。",
        [{"text": "企业应建立作业许可管理制度。"}], "T-9041")
    assert not need2


def test_attribution_fallback_skips_when_chunk_mentions_subject():
    """🔴 chunk 里提到了该主体 → 那是**关于它**的内容，不该说"全是通用规定"。"""
    need, _fb, _poll = AN.attribution_fallback(
        "结论：设备 T-9041 的检维修应履行作业许可手续。",
        [{"text": "T-9041 属于装置区检维修作业，应纳入作业许可管理。"}],
        "T-9041")
    assert not need


def test_attribution_fallback_needs_subject():
    """🔴 没有主体 → 不启用（事实题/数值题没有单一主体概念）。"""
    need, fb, poll = AN.attribution_fallback("任意答案", [{"text": "x"}], None)
    assert not need and not fb and not poll


def test_boundary_declaration_detected():
    """🔴 适用性边界声明：必须**同一个句子**里同时有主体和强边界词。"""
    ok, sent = SC.has_boundary_declaration(
        "以下为通用要求，其适用性不等同于对 T-9041 的专门规定：", "T-9041")
    assert ok, sent
    # 🔴 主体不在同一句 → 不算（模型在别处随口加一句不算数）
    ok2, _ = SC.has_boundary_declaration(
        "以下为通用要求。资料中没有 T-9041 的专门规定。", "T-9041")
    assert not ok2
    # 只有弱边界词 → 不算
    ok3, _ = SC.has_boundary_declaration(
        "以下为通用要求，适用于 T-9041。", "T-9041")
    assert not ok3


def test_bounded_is_not_false_answer():
    """🔴 `bounded`：表态了 + 显式做了归因隔离 → 不算"给不存在的装置编造规定"。"""
    ans = ("资料中**没有 T-9041 的专门规定**，因此未找到其具体要求。\n"
           "以下为**通用要求，其适用性不等同于对 T-9041 的专门规定**：\n"
           "1. 企业应建立设备预防性维护管理程序。")
    v, why = SC.refusal_judgement(ans, subject="T-9041")
    assert v == "bounded", f"应判 bounded，实际 {v}（{why}）"


def test_boundary_does_not_excuse_real_fabrication():
    """🔴 边界声明**不能豁免**真正的归因污染。

    模型如果一边标注边界、一边仍用「结论：<主体> 应……」给主体下规定，
    那还是误答 —— 边界声明只是把声明放在了错误的位置。
    """
    ans = ("资料中没有 T-9041 的专门规定。\n"
           "**结论：设备 T-9041 的检维修应履行作业许可手续**，并实行一事一审批。\n"
           "（以上通用要求，其适用性不等同于对 T-9041 的专门规定）")
    v, why = SC.refusal_judgement(ans, subject="T-9041")
    assert v == "false_answer", f"应判误答，实际 {v}（{why}）"


def test_bounded_does_not_change_legacy_verdicts():
    """🔴 兼容性：旧的答案（没有边界声明）判定**必须完全不变**。

    这是两批可比的前提 —— 新增判定态不能让历史批次的数字漂移。
    """
    # 旧口径的典型误答：没有边界声明
    v, _ = SC.refusal_judgement(
        "T-9001 的检修周期为 30 天，依据第三条。", subject="T-9001")
    assert v == "false_answer"
    # 旧口径的典型通过
    v2, _ = SC.refusal_judgement(
        "未找到 T-9001 的相关规定，语料中没有该设备。", subject="T-9001")
    assert v2 == "pass"


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


# ============================================================ 4. 污染句定位
#
# 🔴 2026-11-05：判据第三轮修正（子串陷阱 + 中文引号 + 条件式收窄）

def test_absence_statement_is_not_pollution():
    """🔴 **缺失声明句不是污染** —— 模型在承认"没规定"，不是在下规定。

    实测形态（system_v2 真实输出）：
      「**结论：资料中没有"聚丙烯装置"的专门规定**」
    → "没有"后面紧跟全角引号 `“` → 旧字符类不含引号 → 否定句被判成肯定句。
    """
    for ans in (
        '**结论：资料中没有“聚丙烯装置”的专门规定**',
        '资料中没有“氯碱装置”的专门规定',
        '资料中没有“加氢反应”这一主体的专门规定',
        '现有资料缺少T-9079属于何种设备或装置的说明',
    ):
        poll = SC.find_pollution_sentences(ans, "聚丙烯装置" if "聚丙烯" in ans
                                           else ("氯碱装置" if "氯碱" in ans
                                                 else ("加氢反应" if "加氢" in ans else "T-9079")))
        assert not poll, f"缺失声明被当成污染：{ans} → {poll}"


def test_ownership_clarification_is_not_pollution():
    """🔴 **澄清归属**不是污染 —— 模型在说"这条不是针对它的"。"""
    cases = [
        ("但该原文针对游乐设施，不属于乙烯裂解装置的专门要求", "乙烯裂解装置"),
        ("该模板仅适用于受限空间作业，不是螺杆压缩机专门要求", "螺杆压缩机"),
        ("但该模板仅是受限空间模板，不是电除尘器的专门规定", "电除尘器"),
    ]
    for ans, subj in cases:
        poll = SC.find_pollution_sentences(ans, subj)
        assert not poll, f"澄清归属被当成污染：{ans} → {poll}"


def test_real_pollution_still_detected():
    """🔴 收紧判据**不能**把真污染放过。"""
    cases = [
        ("T-9052开展检维修作业时，原则上应办理作业许可", "T-9052"),
        ("综上，T-9058开展检维修时，应按企业作业许可及检维修管理规定执行", "T-9058"),
        ("T-9066属于“设备”，检维修作业应纳入作业许可管理", "T-9066"),
        ("检修前应结合加氢反应器内部可能存在的缺氧窒息、中毒、燃爆等风险进行辨识",
         "加氢反应器"),
    ]
    for ans, subj in cases:
        poll = SC.find_pollution_sentences(ans, subj)
        assert poll, f"真污染被放过：{ans}"


def test_emergency_substring_not_assertion():
    """🔴 子串陷阱：「**应急**预案」里的"应"不是断言谓语。"""
    ans = "这些内容需要由乙烯裂解装置适用的专项应急预案、操作规程和检维修管理制度确定"
    poll = SC.find_pollution_sentences(ans, "乙烯裂解装置")
    assert not poll, f"“应急”的“应”被当成断言谓语：{poll}"


def test_pollution_finder_matches_judgement():
    """🔴 判分与兜底**必须用同一套逻辑**（否则口径分裂）。"""
    anss = [
        ("结论：设备 T-9041 的检维修应履行作业许可手续，并实行一事一审批。",
         "T-9041"),
        ("资料中没有 T-9041 的专门规定。", "T-9041"),
        ("以下为通用要求，其适用性不等同于对 T-9041 的专门规定："
         "T-9041 应按通用流程办理作业许可。", "T-9041"),
    ]
    for ans, subj in anss:
        poll = SC.find_pollution_sentences(ans, subj)
        verdict, _ = SC.refusal_judgement(ans, subject=subj)
        expect = "false_answer" if poll else verdict
        assert verdict == expect, (
            f"判定与污染句定位不一致：{verdict} vs {bool(poll)} — {ans}")


# ============================================================ 5. 污染句划除

def test_mark_pollution_strikes_through():
    """🔴 污染句被就地划掉（用户看到的输出里它不再是一个正常结论）。"""
    ans = ("资料中没有 T-9041 的专门规定。\n"
           "结论：设备 T-9041 的检维修应履行作业许可手续。")
    poll = SC.find_pollution_sentences(ans, "T-9041")
    assert poll
    marked = AN.mark_pollution(ans, poll)
    # 🔴 划线覆盖**句子本体**；句末标点属于分隔符，不在句子里（切句规则）
    assert "~~结论：设备 T-9041 的检维修应履行作业许可手续~~" in marked
    # 划掉之后，剩余文本里没有污染句了
    assert not SC.find_pollution_sentences(
        SC.strip_marked_pollution(marked), "T-9041")


def test_mark_pollution_no_nested_strike():
    """🔴 同一句话出现两次 → 不能产生 `~~~~s~~~~`（嵌套删除线）。"""
    ans = "T-9041 应办理作业许可。T-9041 应办理作业许可。"
    poll = SC.find_pollution_sentences(ans, "T-9041")
    marked = AN.mark_pollution(ans, poll)
    assert "~~~~" not in marked, marked
    assert marked.count("~~") == 4, marked


def test_strip_marked_pollution_roundtrip():
    """🔴 剔除划线部分后，原文其余部分逐字保留。"""
    ans = "前言。\n结论：设备 T-9041 的检维修应履行作业许可手续。\n后记。"
    poll = SC.find_pollution_sentences(ans, "T-9041")
    marked = AN.mark_pollution(ans, poll)
    stripped = SC.strip_marked_pollution(marked)
    assert "前言。" in stripped and "后记。" in stripped
    assert "应履行作业许可手续" not in stripped
