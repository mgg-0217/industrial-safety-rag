# -*- coding: utf-8 -*-
"""版本关系模块的自检 —— **真实语料 + 构造样例**双轨。

真实语料能验证的：榆林版本对的推导、正文明示废止的解析、生效日与人工标注的冲突。
真实语料**验证不了**的（库里只有 1 条链长 2 的链）：链长 ≥ 3、中间版本缺失、
显式优先于推导——这些用**构造样例**覆盖。

跑法：
    python tools/test_versioning.py          # 直接出报告
    pytest tools/test_versioning.py -v       # 也兼容 pytest
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.versioning import (  # noqa: E402
    SOURCE_DECLARED,
    SOURCE_DERIVED,
    SOURCE_DOC_DATE,
    SOURCE_EXPLICIT,
    SOURCE_OPEN,
    SOURCE_UNKNOWN,
    apply_version_fields,
    derive_chain_fields,
    is_effective,
    is_unknown,
    parse_effective_date,
    parse_explicit_supersessions,
)

MD_DIR = ROOT / "data" / "md"
MANIFEST = ROOT / "data" / "manifest.json"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_docs() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def by_id(docs: list[dict]) -> dict:
    return {d["doc_id"]: d for d in docs}


def make_doc(doc_id, title, doc_type, effective_from, doc_no="", **kw):
    """构造一份最小的文档记录。"""
    d = {
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "doc_no": doc_no,
        "effective_from": effective_from,
        "effective_to": None,
        "effective_to_source": SOURCE_OPEN,
        "effective_from_source": SOURCE_DOC_DATE,
    }
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# 1. 从正文抽施行日
# ---------------------------------------------------------------------------

def check_1_parse_effective_date():
    cases = [
        ("本制度自2023年10月6日起施行，有效期5年。", "2023-10-06"),
        ("本预案自2024年3月1日起实施。", "2024-03-01"),
        ("自 2023-10-06 起施行。", "2023-10-06"),
        ("2023-04-01 实施", "2023-04-01"),
        # ISO 紧凑型，全角数字 + 全角 G（AQ/T 3034 封面的写法）
        ("２０２２Ｇ１０Ｇ０１发布 ２０２３Ｇ０４Ｇ０１实施", "2023-04-01"),
        # 没有具体日期的写法 → 抽不出（不猜）
        ("本制度自印发之日起施行。", None),
        ("本预案自发布之日起实施。", None),
    ]
    problems = []
    for text, expect in cases:
        got, _ = parse_effective_date(text)
        if got != expect:
            problems.append((text[:30], expect, got))
    return problems


# ---------------------------------------------------------------------------
# 2. 从正文抽明示废止
# ---------------------------------------------------------------------------

def check_2_parse_supersessions():
    text = (
        "第十九条  本制度自印发之日起施行。原《榆林市危险化学品企业动火和受限空间"
        "作业安全管理制度》（榆政办发〔2023〕26号）同时废止。"
    )
    got = parse_explicit_supersessions(text)
    expect_no = "榆政办发〔2023〕26号"
    problems = []
    if len(got) != 1 or got[0]["doc_no"] != expect_no:
        problems.append(("文号抽取", expect_no, got))
    # 只提标题不发文号的 → 抽不出来（不靠标题猜版本）
    only_title = "原《某某管理办法》同时废止。"
    if parse_explicit_supersessions(only_title):
        problems.append(("只提标题时不应抽出", "[]", parse_explicit_supersessions(only_title)))
    return problems


# ---------------------------------------------------------------------------
# 3. 三态区间判定
# ---------------------------------------------------------------------------

def check_3_is_effective():
    problems = []

    doc = make_doc("a", "T", "制度", "2023-01-01")
    doc["effective_to"] = "2024-01-01"
    doc["effective_to_source"] = SOURCE_DERIVED

    if is_effective(doc, "2023-06-01") is not True:
        problems.append(("区间内应生效", True, is_effective(doc, "2023-06-01")))
    if is_effective(doc, "2022-06-01") is not False:
        problems.append(("生效前不应生效", False, is_effective(doc, "2022-06-01")))
    if is_effective(doc, "2024-06-01") is not False:
        problems.append(("失效后不应生效", False, is_effective(doc, "2024-06-01")))
    # 边界：左闭右开
    if is_effective(doc, "2023-01-01") is not True:
        problems.append(("左边界应生效", True, is_effective(doc, "2023-01-01")))
    if is_effective(doc, "2024-01-01") is not False:
        problems.append(("右边界不应生效", False, is_effective(doc, "2024-01-01")))

    # 🔴 未知生效日 → 必须是 None（三态），不是 True
    unk = make_doc("u", "T", "制度", None)
    unk["effective_from"] = ""
    unk["effective_from_source"] = SOURCE_UNKNOWN
    r = is_effective(unk, "2026-10-30")
    if r is not None:
        problems.append(("未知生效日必须返回 None 而非 True/False", None, r))

    # 至今未失效（effective_to=None 且 source=open）
    open_doc = make_doc("o", "T", "制度", "2023-01-01")
    open_doc["effective_to"] = None
    open_doc["effective_to_source"] = SOURCE_OPEN
    if is_effective(open_doc, "2099-01-01") is not True:
        problems.append(("open 文档应持续生效", True, is_effective(open_doc, "2099-01-01")))

    # 未知的几种形态都要认
    for v in ("", "-", None):
        if not is_unknown(v):
            problems.append(("未知形态未识别", True, v))

    return problems


# ---------------------------------------------------------------------------
# 4. 版本链推导（构造样例：真实语料只有链长 2）
# ---------------------------------------------------------------------------

def check_4_derive_chain():
    problems = []

    # 4.1 链长 3
    chain = [
        make_doc("v1", "某制度", "安全管理制度", "2021-01-01"),
        make_doc("v2", "某制度", "安全管理制度", "2022-01-01"),
        make_doc("v3", "某制度", "安全管理制度", "2023-01-01"),
    ]
    fields, _ = derive_chain_fields(chain)
    if fields["v1"]["effective_to"] != "2022-01-01":
        problems.append(("v1 失效日", "2022-01-01", fields["v1"]))
    if fields["v2"]["effective_to"] != "2023-01-01":
        problems.append(("v2 失效日", "2023-01-01", fields["v2"]))
    if fields["v1"]["superseded_by"] != "v2":
        problems.append(("v1 继任", "v2", fields["v1"]))
    if fields["v3"]["effective_to"] is not None:
        problems.append(("v3 至今未失效", None, fields["v3"]))
    if fields["v3"]["effective_to_source"] != SOURCE_OPEN:
        problems.append(("v3 来源", SOURCE_OPEN, fields["v3"]))

    # 4.2 中间版本缺失（v2 不在库里）→ v1 的失效日推到 v3，
    #     这是"库是真实版本链子集"的必然结果（PRD 2.6 有说明）
    sparse = [
        make_doc("v1", "某制度", "安全管理制度", "2021-01-01"),
        make_doc("v3", "某制度", "安全管理制度", "2023-01-01"),
    ]
    fields2, _ = derive_chain_fields(sparse)
    if fields2["v1"]["effective_to"] != "2023-01-01":
        problems.append(("缺中间版本时 v1 失效日", "2023-01-01", fields2["v1"]))
    if fields2["v1"]["effective_to_source"] != SOURCE_DERIVED:
        problems.append(("推导值来源须为 derived", SOURCE_DERIVED, fields2["v1"]))

    # 4.3 不同 doc_type 不同名 → 不并成一条链
    mixed = [
        make_doc("x1", "同名文件", "安全管理制度", "2021-01-01"),
        make_doc("x2", "同名文件", "应急预案", "2022-01-01"),
    ]
    fields3, _ = derive_chain_fields(mixed)
    if "x1" in fields3:
        problems.append(("doc_type 不同不应并链", "不并链", fields3))

    return problems


# ---------------------------------------------------------------------------
# 5. 真实语料：榆林版本对
# ---------------------------------------------------------------------------

def check_5_real_corpus():
    problems = []
    docs = load_docs()
    new_docs, conflicts = apply_version_fields(docs, MD_DIR)
    idx = by_id(new_docs)

    old = idx.get("yulin_2023_old")
    new = idx.get("yulin_donghuo_xianshi")
    if old is None or new is None:
        return ["语料里找不到榆林版本对"]

    # 5.1 正文明示的施行日优先于人工标注的 effective_date
    if old["effective_from"] != "2023-10-06":
        problems.append(("2023 版生效日（正文为准）", "2023-10-06", old["effective_from"]))
    if old["effective_from_source"] != SOURCE_DECLARED:
        problems.append(("生效日来源", SOURCE_DECLARED, old["effective_from_source"]))

    # 5.2 冲突要被记录下来（不是悄悄改）
    hit = [c for c in conflicts
           if c.get("doc_id") == "yulin_2023_old"
           and c.get("type") == "effective_date_mismatch"]
    if not hit:
        problems.append(("生效日冲突未记录", "有记录", conflicts))
    elif hit[0]["recorded"] != "2023-12-07":
        problems.append(("冲突记录里的原值", "2023-12-07", hit[0]["recorded"]))

    # 5.3 明示废止：2023 版被 2025 版取代
    if old["superseded_by"] != "yulin_donghuo_xianshi":
        problems.append(("2023 版继任", "yulin_donghuo_xianshi", old["superseded_by"]))
    if old["effective_to_source"] != SOURCE_EXPLICIT:
        problems.append(("废止关系来源", SOURCE_EXPLICIT, old["effective_to_source"]))
    if old["effective_to"] != new["effective_from"]:
        problems.append(("失效日应等于继任生效日", new["effective_from"], old["effective_to"]))

    # 5.4 2025 版至今未失效
    if new["effective_to"] is not None or new["effective_to_source"] != SOURCE_OPEN:
        problems.append(("2025 版应至今有效", (None, SOURCE_OPEN),
                         (new["effective_to"], new["effective_to_source"])))

    # 5.5 明示废止的双向记录
    if "yulin_2023_old" not in (new.get("supersedes") or []):
        problems.append(("2025 版未记录其废止对象", ["yulin_2023_old"], new.get("supersedes")))

    return problems


# ---------------------------------------------------------------------------
# 6. 未知生效日的文档（真实语料里的 guide_jiaxian）
# ---------------------------------------------------------------------------

def check_6_unknown_effective():
    problems = []
    docs = load_docs()
    new_docs, _ = apply_version_fields(docs, MD_DIR)
    idx = by_id(new_docs)

    d = idx.get("guide_jiaxian_guanli_zhidu")
    if d is None:
        return ["语料里找不到 guide_jiaxian_guanli_zhidu"]
    if d["effective_from_source"] != SOURCE_UNKNOWN:
        problems.append(("未知生效日来源", SOURCE_UNKNOWN, d["effective_from_source"]))
    if is_effective(d, "2026-10-30") is not None:
        problems.append(("未知生效日必须三态返回 None", None,
                         is_effective(d, "2026-10-30")))

    # 全库不该有"生效日未知却被当成永远现行"的文档
    bad = [x["doc_id"] for x in new_docs
           if is_effective(x, "2099-01-01") is True
           and x.get("effective_from_source") == SOURCE_UNKNOWN]
    if bad:
        problems.append(("未知生效日却被判为现行", [], bad))

    return problems


# ---------------------------------------------------------------------------
# 7. 幂等性：跑两遍结果必须完全一致
# ---------------------------------------------------------------------------

def check_7_idempotent():
    problems = []
    docs = load_docs()
    a, _ = apply_version_fields(docs, MD_DIR)
    b, _ = apply_version_fields(a, MD_DIR)
    if json.dumps(a, ensure_ascii=False, sort_keys=True) != \
       json.dumps(b, ensure_ascii=False, sort_keys=True):
        problems.append(("两次运行结果不一致", "一致", "不一致"))
    return problems


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 正文施行日抽取", check_1_parse_effective_date),
    ("2. 明示废止解析", check_2_parse_supersessions),
    ("3. 三态区间判定", check_3_is_effective),
    ("4. 版本链推导（构造样例）", check_4_derive_chain),
    ("5. 真实语料·榆林版本对", check_5_real_corpus),
    ("6. 未知生效日处置", check_6_unknown_effective),
    ("7. 幂等性", check_7_idempotent),
]


def run_report() -> int:
    failed = 0
    for name, fn in CHECKS:
        problems = fn()
        if problems:
            failed += 1
            print(f"[FAIL] {name}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"[ OK ] {name}")
    print()
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} 组通过")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# pytest 入口
# ---------------------------------------------------------------------------

def test_parse_effective_date():
    assert not check_1_parse_effective_date()


def test_parse_supersessions():
    assert not check_2_parse_supersessions()


def test_is_effective():
    assert not check_3_is_effective()


def test_derive_chain():
    assert not check_4_derive_chain()


def test_real_corpus_yulin_pair():
    assert not check_5_real_corpus()


def test_unknown_effective_date():
    assert not check_6_unknown_effective()


def test_idempotent():
    assert not check_7_idempotent()


if __name__ == "__main__":
    sys.exit(run_report())
