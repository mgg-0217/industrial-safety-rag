# -*- coding: utf-8 -*-
"""生成层单元测试（PRD M6 / M7 的验收要求）。

M6 的验收标准原文：

    「`as_of` / `current` 两口径可切换；
      **版本卡由代码拼接**（单测断言版本卡字符串出现在最终答案里，
      且**不在**模型输出里）；
      **过期 chunk 不进上下文的端到端断言**」

所以第 2、3 节是本文件的核心：
- 版本卡必须出现在**最终答案**里，且**不在模型原始输出**里
  （它不是模型"自觉说"出来的，是代码拼的）。
- 过期 chunk 必须**根本没进上下文**（不是"进了但标了警告"）。

跑法：
    python -m pytest tools/test_version_card.py -v
    python tools/test_version_card.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import answering as AN      # noqa: E402
import citation as CIT      # noqa: E402
import scoring as SC        # noqa: E402
import version_card as VC   # noqa: E402

import json                  # noqa: E402


# ============================================================ 夹具

def _chunk(doc_id, text, eff_from, eff_to, eff_to_source=None, title=None,
           doc_no=None, version=None, note=None, norm_text=None, chunk_id=None):
    if eff_to_source is None:
        eff_to_source = "explicit" if eff_to else "open"
    snap = {
        "doc_id": doc_id, "title": title or doc_id, "doc_no": doc_no or "",
        "version": version or "", "effective_from": eff_from,
        "effective_to": eff_to, "effective_to_source": eff_to_source,
        "superseded_by": None, "status": "有效",
        "source_url": "https://example.gov.cn/doc", "extract_quality": "ok",
    }
    if note:
        snap["note"] = note
    return {
        "chunk_id": chunk_id or f"{doc_id}::c0",
        "doc_id": doc_id, "strategy": "test", "text": text,
        "norm_text": norm_text if norm_text is not None else text,
        "char_start": 0, "char_end": len(text), "page_no": 3,
        "clause_no": "第三条", "section_path": [], "table": None,
        "quality": "ok", "strategy_meta": {}, "doc_snapshot": snap,
    }


#: 现行版 / 已废止版（用真实版本链：榆林 2023 版被 2025 版取代）
CHUNKS = [
    _chunk("doc_old", "旧版条款：动火作业应办理作业票。", "2023-10-06",
           "2025-01-09", title="榆林市危险化学品企业动火和受限空间作业安全管理制度",
           doc_no="榆政办发〔2023〕26号", version="2023版",
           note="被2025版明示废止；作为'过期版本召回'负样本与冲突检测样本",
           chunk_id="doc_old::c0"),
    _chunk("doc_new", "现行条款：动火作业应办理作业票并设专人监护。", "2025-01-09",
           None, title="榆林市危险化学品企业动火和受限空间作业安全管理制度",
           doc_no="榆政办发〔2025〕1号", version="2025版",
           note="第十九条明示原〔2023〕26号废止，可作版本治理正样本",
           chunk_id="doc_new::c0"),
]

MANIFEST = {
    "doc_old": {"doc_id": "doc_old", "title": "榆林市危险化学品企业动火和受限空间作业安全管理制度",
                "doc_no": "榆政办发〔2023〕26号", "version": "2023版"},
    "doc_new": {"doc_id": "doc_new", "title": "榆林市危险化学品企业动火和受限空间作业安全管理制度",
                "doc_no": "榆政办发〔2025〕1号", "version": "2025版"},
}


def _fake_call(text):
    """假模型：永远返回固定文本。"""
    def call(prompt, system=None, **kw):
        call.last_prompt = prompt
        return text
    return call


# ============================================================ 1. 版本卡措辞

def test_status_wording_by_mode():
    """同一份数据，两种口径 → 两种措辞（PRD 5.4）。"""
    snap_old = CHUNKS[0]["doc_snapshot"]
    snap_new = CHUNKS[1]["doc_snapshot"]
    assert VC.status_text(snap_new, "2026-11-03", "current") == "现行有效"
    assert "该时点生效" in VC.status_text(snap_old, "2024-06-01", "as_of")
    assert VC.status_text(snap_old, "2026-11-03", "cross_version") == "历史版本（已废止）"
    # 生效日未知 → 三态里的 UNKNOWN
    snap_unknown = dict(snap_new, effective_from="")
    assert "未标注" in VC.status_text(snap_unknown, "2026-11-03", "current")


def test_supersede_text():
    """废止关系：谁取代了谁。"""
    snap_old = dict(CHUNKS[0]["doc_snapshot"], superseded_by="doc_new")
    t = VC.supersede_text(snap_old, MANIFEST)
    assert "2025-01-09" in t and "榆政办发〔2025〕1号" in t
    assert VC.supersede_text(CHUNKS[1]["doc_snapshot"], MANIFEST) == "本版本未被取代"


def test_note_missing_writes_weibiao():
    """没有 note 就写"未标注"，**不写"无变动"**（后者是断言，不是事实）。"""
    assert VC.note_text({"doc_id": "x"}) == "未标注"
    assert VC.note_text({"doc_id": "x", "note": "  "}) == "未标注"
    assert VC.note_text({"note": "有说明"}) == "有说明"


def test_cards_one_per_document():
    """多版本命中 → 每个文档各一张卡，不合并（PRD 5.4）。"""
    cards = VC.build_cards(CHUNKS, manifest=MANIFEST, when="2026-11-03",
                           mode="cross_version")
    assert len(cards) == 2, cards
    assert "榆政办发〔2025〕1号" in cards[1]
    assert "榆政办发〔2023〕26号" in cards[0]


# ============================================================ 2. 版本卡由代码拼接

def test_version_card_in_final_answer_not_in_model_output():
    """🔴 M6 验收核心：版本卡出现在**最终答案**里，且**不在模型输出**里。

    这是"代码拼"与"模型自觉说"的区别 ——
    后者一旦模型忘了说，版本信息就没了，而且**测不出来**。
    """
    model_out = "动火作业应办理作业票并设专人监护。[[C2]]"
    call = _fake_call(model_out)
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST, top_k=5)
    r = ans.answer("动火作业有什么要求？", when="2026-11-03", mode="current",
                   call=call)

    card = r["version_cards"][0]
    assert card, "应该生成版本卡"
    # 最终答案里有
    assert card.split("\n")[0] in r["rendered"]
    assert "现行有效" in r["rendered"]
    # 模型原始输出里**没有**
    assert card.split("\n")[0] not in r["answer_text"]
    assert "现行有效" not in r["answer_text"]
    assert "废止关系" not in r["answer_text"]


def test_version_card_appears_even_if_model_says_nothing():
    """模型一个字都不说，版本卡照样在 —— 这正是"代码拼"的意义。"""
    call = _fake_call("")
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST)
    r = ans.answer("动火作业有什么要求？", when="2026-11-03", call=call)
    assert "现行有效" in r["rendered"]
    assert r["answer_text"] == ""


# ============================================================ 3. 过期 chunk 不进上下文

def test_expired_chunk_not_in_context_current_mode():
    """🔴 M6 验收核心：过期 chunk **根本没进上下文**（PRD 5.5）。

    不是"进了但标了警告" —— 是压根不在。
    """
    call = _fake_call("答案 [[C1]]")
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST)
    r = ans.answer("动火作业有什么要求？", when="2026-11-03", mode="current",
                   call=call)
    ids = [c["chunk_id"] for c in r["ctx_chunks"]]
    assert "doc_old::c0" not in ids, f"过期 chunk 进了上下文：{ids}"
    assert "doc_new::c0" in ids


def test_expired_chunk_in_context_under_as_of():
    """as_of(2024) 口径下旧版**当时是现行**，应当正常召回（PRD 5.5 第 2 条）。"""
    call = _fake_call("答案")
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST)
    r = ans.answer("动火作业有什么要求？", when="2024-06-01", mode="as_of",
                   call=call)
    ids = [c["chunk_id"] for c in r["ctx_chunks"]]
    assert "doc_old::c0" in ids, f"as_of 口径下旧版应召回：{ids}"
    assert "doc_new::c0" not in ids, "as_of 口径下当时尚未生效的新版不该在场"
    # 版本卡措辞也要跟着口径变
    assert "该时点生效" in r["rendered"]


def test_cross_version_keeps_both():
    """cross_version 口径：两版都要在场（PRD 5.2）。"""
    call = _fake_call("答案")
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST)
    r = ans.answer("动火作业作业票怎么办？", when="2026-11-03", mode="cross_version",
                   call=call)
    ids = [c["chunk_id"] for c in r["ctx_chunks"]]
    assert "doc_old::c0" in ids and "doc_new::c0" in ids, ids
    assert len(r["version_cards"]) == 2


# ============================================================ 4. 引用由代码生成

def test_citation_card_uses_raw_text_not_norm_text():
    """🔴 引用卡片必须用**原文轨**（PRD 6.2）。

    归一化后的文本里全角数字被转成半角、位号破折号被改了，
    展示给用户必须一字不改。
    """
    c = _chunk("doc_x", "原文：19—V—4215 出口管道", "2025-01-09", None,
               norm_text="原文：19-V-4215 出口管道")
    card = CIT.build_card(1, c)
    assert "19—V—4215" in card
    assert "19-V-4215" not in card.replace("原文：19—V—4215", "")


def test_invalid_citation_counted_not_dropped():
    """模型编了不存在的编号 → 标记 + 计数，**不静默丢弃**（PRD 6.1）。"""
    call = _fake_call("第一条 [[C1]]，第二条 [[C9]]。")
    ans = AN.Answerer(CHUNKS, embedder=None, manifest=MANIFEST)
    r = ans.answer("动火作业有什么要求？", when="2026-11-03", call=call)
    st = r["cite_stats"]
    assert st["n_refs"] == 2
    assert st["n_valid"] == 1
    assert st["n_invalid"] == 1
    assert st["invalid_ids"] == [9]
    assert "引用无效" in r["rendered"]
    assert "C9" in r["rendered"]


def test_citation_card_contains_required_fields():
    """引用卡片字段：条款 + 页码 + 文号 + 版本状态 + URL（PRD 6.2）。"""
    c = CHUNKS[1]
    card = CIT.build_card(1, c)
    for kw in ("[C1]", "榆政办发〔2025〕1号", "2025版", "条款：第三条",
               "第 3 页", "现行有效", "example.gov.cn"):
        assert kw in card, f"引用卡片缺 {kw}：\n{card}"


def test_page_text_rules():
    """页码：跨页块给区间，网页来源写"无页码"，**不编页码**（PRD 6.2）。"""
    assert CIT.page_text({"page_no": 3}) == "第 3 页"
    assert CIT.page_text({"page_no": None,
                          "strategy_meta": {"cross_page": [4, 5]}}) == "第 4–5 页"
    assert "无页码" in CIT.page_text({"page_no": None, "strategy_meta": {}})


# ============================================================ 5. 拒答判定

def test_refusal_pass_cases():
    """PRD 7.1 判定表第 1、2 行。"""
    v, why = SC.refusal_judgement("未找到 T-9001 的相关规定，语料中没有该设备。",
                                  subject="T-9001")
    assert v == "pass", why
    v, why = SC.refusal_judgement(
        "语料中无 T-9001 的相关规定；如需其检修周期，请补充该设备手册。",
        subject="T-9001")
    assert v == "pass", why


def test_refusal_false_answer_cases():
    """PRD 7.1 判定表第 3、4 行：编造规定 / 未表态。"""
    v, why = SC.refusal_judgement("T-9001 的检修周期为 30 天，依据第三条。",
                                  subject="T-9001")
    assert v == "false_answer", why
    v, why = SC.refusal_judgement("设备检修应按第三条执行，周期 30 天。",
                                  subject="T-9001")
    assert v == "false_answer", why


def test_refusal_mixed_goes_to_manual():
    """PRD 7.1 第 5 行：混合 → 人工复核，**不自动判**。"""
    v, why = SC.refusal_judgement(
        "未找到 T-9001 的规定。类似地，压力容器检修按第三条执行，周期 30 天。",
        subject="T-9001")
    assert v == "manual", why


# ============================================================ 5b. 拒答判定的回归（K-39）

#: K-39 修掉的**误报**：这些是模型**真实写出来**的正确拒答，
#: 旧版判分把它们判成"误答"，因为"提到主体 + 句中无旧清单里的否定词"就等于
#: "给主体下了规定" —— 而模型写的是"不能确定/未给出/还需补充/语料中无"。
#: 每一句都是从真实运行产物里摘出来的原文。
_REFUSAL_REGRESSION = [
    # 「未给出」—— 旧清单里没有这个词
    ("资料仅要求对设备定期巡回检查并建立定期检查记录，但未给出 T-9001 的具体检修周期或间隔。",
     "T-9001"),
    # 「不能据现有资料确定」—— 旧清单里没有"不能"
    ("因此，不能据现有资料确定 T-9001 的具体执行要求。", "T-9001"),
    # 「语料中无 + 空格 + 主体」—— 否定词与后续词之间隔着空格
    ("语料中无 T-9001 的相关规定；如需其检修周期，请补充该设备手册。", "T-9001"),
    # 「若…涉及…」—— 条件句，是"带明确限定"地引用通用条款（PRD 7.1 第 2 行）
    ("结论：资料中未找到设备 T-9001 的具体检修周期要求。"
     "若 T-9001 检维修涉及受限空间等特殊作业，还应执行相应特殊作业安全规范。", "T-9001"),
    # 「还应执行…」的完整版：条件句 + 后续说明
    ("结论：资料中未找到设备 T-9001 的具体检修周期要求。"
     "若 T-9001 检维修涉及受限空间等特殊作业，还应执行相应特殊作业安全规范；"
     "现有片段未完整提供相关要求。", "T-9001"),
]


def test_refusal_correct_answers_not_misjudged_as_false():
    """🔴 K-39 回归：**正确的拒答不能被判成误答**。

    旧版判分在真实运行里把 2/2 条标准拒答判成误答 ——
    拒答误答率会因此虚高到 1.0，看起来像"模型完全不会拒答"，
    而实际上模型答得完全正确。
    """
    for text, subject in _REFUSAL_REGRESSION:
        v, why = SC.refusal_judgement(text, subject=subject)
        assert v in ("pass", "manual"), f"正确拒答被判成 {v}：{why}\n{text}"


def test_refusal_still_catches_fabrication():
    """🔴 K-39 回归：放宽否定词之后，**编造规定仍要被抓住**。

    放宽判定最怕的就是"把误报修成漏报" ——
    所以正向、反向两组用例必须同时锁住。
    """
    for text in [
        "T-9001 的检修周期为 30 天，依据第三条。",
        "T-9001 应按《设备检修规程》执行，每季度检修一次。",
        "T-9001 不得带病运行，检修周期为 30 天。",
    ]:
        v, why = SC.refusal_judgement(text, subject="T-9001")
        assert v == "false_answer", f"编造的规定没被抓住：{v} {why}\n{text}"


def test_refusal_number_inside_tag_is_not_content():
    """🔴 K-39：位号 `T-9001` 里的数字**不算**"给出了具体数值"。

    旧版数值正则的单位是可选的 → 任何数字都算 →
    "T-9001" 和列表序号 "1." 都被当成具体数值 → 正确拒答被推进"人工复核"。
    """
    v, why = SC.refusal_judgement(
        "未找到 T-9001 的规定。1. 一般要求 2. 检维修许可应一事一审批。",
        subject="T-9001")
    assert v == "pass", why


def test_refusal_dodging_answer_is_false_answer():
    """🔴 K-39：全文级表态检测必须**严**，否则"未表态"永远判不出来。

    "未表态 + 给出别处条款"是 PRD 7.1 表格第 4 行 → 误答。
    若句级的宽口径被误用到全文级，"其他内容未提及"这种顺口一提
    就会让这条漏判。
    """
    v, why = SC.refusal_judgement("设备检修应按第三条执行，周期 30 天。",
                                  subject="T-9001")
    assert v == "false_answer", why
    v, why = SC.refusal_judgement(
        "设备检修应按第三条执行，周期 30 天，其他内容未提及。",
        subject="T-9001")
    assert v == "false_answer", why


# ============================================================ 6. 判分口径

def test_key_point_matching_uses_match_norm():
    """🔴 K-35：key_point 匹配必须走 `match_norm`（标点折叠 + 去空白）。

    逐字匹配只有 45% 命中 —— 全角冒号和半角冒号会被当成两个串。
    """
    ans = "处理建议：由泉港石化进一步调查处理。"
    kps = ["处理建议:由泉港石化"]
    hit, total, detail = SC.key_point_hits(ans, kps)
    assert total == 1 and hit == 1, detail


def test_numeric_matching_normalizes_units():
    """数值比对要归一化（PRD 7.3）：`0.35MPa` / `0.35 MPa` 同一个串。"""
    hit, total, detail = SC.numeric_hits("设计压力为 0.35 MPa。", "0.35MPa")
    assert (hit, total) == (1, 1), detail
    hit, total, detail = SC.numeric_hits("设计压力为 0.35 MPa。", "19.5 %;21 %")
    assert (hit, total) == (0, 2), detail


def test_wilson_bounds():
    p, lo, hi = SC.wilson(30, 50)
    assert lo < p < hi
    assert SC.wilson(0, 0) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} 条通过")
    sys.exit(0 if passed == len(fns) else 1)
