#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**本轮新工具的单测**（溯源一致性 / 配对比较 / 噪声检测 / 假称检测）。

覆盖四个最容易出错的地方：
1. **kp 归源切分**（在出处文档内 vs 跨文档）—— 切错会让所有溯源数字失真；
2. **McNemar 精确检验**的 p 值 —— 手算核对；
3. **导航噪声检测**的假阳性 —— ��文不能被判成噪声；
4. **「假称有规定」检测** —— 必须与「未找到」表态区分开。
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROV = _load("t_prov", "tools/rescore_provenance.py")
PAIRED = _load("t_paired", "tools/compare_refusal_paired.py")
BOIL = _load("t_boil", "tools/diag_boilerplate_impact.py")
CLAIM = _load("t_claim", "tools/diag_corpus_claim.py")


# ---------------------------------------------------------------- 1. kp 归源

def test_nz_removes_whitespace_and_zero_width():
    assert PROV.nz("甲 乙\t丙") == "甲乙丙"
    assert PROV.nz("﻿正文") == "正文"


def test_kp_split_in_source_vs_cross():
    """出处文档里的 kp 与别文档的 kp 必须被分到不同桶。"""
    mds = {"docA": PROV.nz("第一条规定内容在这里。第二条也有内容。"),
           "docB": PROV.nz("完全不同的另一份文件的内容。")}
    src = "docA"
    kps = ["第一条规定内容在这里。", "完全不同的另一份文件的内容。"]
    in_src = [k for k in kps if PROV.nz(k) in mds.get(src, "")]
    cross = [k for k in kps if k not in in_src]
    assert in_src == ["第一条规定内容在这里。"]
    assert cross == ["完全不同的另一份文件的内容。"]


def test_kp_split_empty_source_doc_yields_all_cross():
    mds = {"docA": PROV.nz("内容")}
    kps = ["不在任何文档里的片段"]
    assert [k for k in kps if PROV.nz(k) in mds["docA"]] == []
    assert kps == [k for k in kps if PROV.nz(k) not in mds["docA"]]


# ------------------------------------------------------- 2. McNemar 精确检验

def test_mcnemar_symmetric_no_discordance():
    p, note = PAIRED.mcnemar_exact(0, 0)
    assert p == 1.0
    assert "无不一致" in note


def test_mcnemar_known_values():
    """手算核对：X~Bin(b+c, 0.5) 的双侧精确 p。

    🔴 注意分母是**不一致对数 b+c**，不是总题数 ——
    这是 McNemar 检验的定义，写错会让 p 值凭空变好。
    """
    # b=6, c=3 → n=9，P(X≤3)=(1+9+36+84)/512=0.2539，双侧 0.5078
    p, note = PAIRED.mcnemar_exact(6, 3)
    assert p == pytest.approx(0.5078, abs=1e-4)
    assert "9 对" in note
    # b=3, c=0 → n=3，P(X≤0)=1/8=0.125，双侧 0.25
    assert PAIRED.mcnemar_exact(3, 0)[0] == pytest.approx(0.25, abs=1e-4)
    # b=8, c=0 → n=8，双侧 2*(1/256)=0.0078
    p2, _ = PAIRED.mcnemar_exact(8, 0)
    assert p2 == pytest.approx(0.0078, abs=1e-4)
    # 对称性
    p3, _ = PAIRED.mcnemar_exact(0, 3)
    assert p3 == pytest.approx(0.25, abs=1e-9)


def test_wilson_ci_bounds():
    lo, hi = PAIRED.wilson(3, 150)
    assert 0.0 < lo < 0.02 < hi < 0.1
    assert PAIRED.wilson(0, 0) is None


# --------------------------------------------------------- 3. 导航噪声检测

def test_boilerplate_detects_nav_tokens_and_provinces():
    text = "正文开始\n网站标识码: bm34000002\n北京\n正文结束"
    hit = BOIL.boiler_lines_for(text)
    assert 1 in hit and 2 in hit
    assert 0 not in hit and 3 not in hit


def test_boilerplate_does_not_flag_prose():
    """🔴 正文不能被判成噪声（假阳性 = 误报语料脏）。"""
    text = ("有限空间作业事故应急救援工作坚持“以人为本”的理念和“安全第一、预防为主”的方针；\n"
            "生产经营单位应当建立健全安全生产责任制。\n")
    hit = BOIL.boiler_lines_for(text)
    assert hit == set()


def test_cross_doc_repeats_flags_shared_nav():
    docs = {"a.md": ["登录", "注册", "版权所有"], "b.md": ["登录", "注册", "正文段落"]}
    reps = BOIL.cross_doc_repeats(docs)
    assert "登录" in reps and "注册" in reps
    assert "正文段落" not in reps


def test_cross_doc_repeats_ignores_long_lines():
    """长句正文即使在两篇里重复，也不该被当成导航。"""
    long = "生产经营单位应当建立健全安全生产责任制并落实全员安全生产责任制。"
    docs = {"a.md": [long], "b.md": [long]}
    assert BOIL.cross_doc_repeats(docs) == set()


# ---------------------------------------------------- 4.「假称有规定」检测

def test_corpus_claim_detects_claim():
    ans = "资料中有合成氨装置检维修作业许可的明确规定。"
    hits = [m.group(0) for p in CLAIM.CLAIM_PATTERNS for m in p.finditer(ans)]
    assert hits, "应识别出『假称资料里有规定』"


def test_corpus_claim_ignores_absent_statement():
    ans = "现有资料中未找到合成氨装置检维修作业许可的明确规定，无法据此给出答案。"
    hits = [m.group(0) for p in CLAIM.CLAIM_PATTERNS for m in p.finditer(ans)]
    has_absent = any(p.search(ans) for p in CLAIM.ABSENT_PATTERNS)
    # 有"未找到"表态 → 不算纯假称
    assert has_absent


def test_corpus_claim_absent_patterns_cover_variants():
    for s in ["未找到相关规定", "资料中未提供具体数值", "没有检索到相关条款"]:
        assert any(p.search(s) for p in CLAIM.ABSENT_PATTERNS), s
