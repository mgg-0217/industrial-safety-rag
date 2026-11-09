#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**数值归一化与锚点抽取**的回归测试。

🔴 **为什么要有这一组测试**（2026-11-09 一晚上挖出 4 个缺陷）：
1. `NUM_RE` 的单位分支**顺序** → "10mg/m3" 被切成 "10 m"（K-81）
2. `ENGINEERING_UNITS` 白名单比较**没做小写化** → "KPa" 被丢掉
3. 前置断言用 `\\w`（**Python 的 `\\w` 含中文**）→ "压力为35～50KPa" 整段漏抽
4. `fold_unit` 的 `(?<=\\d)` 对**复合单位**不成立 → "mg/m³" 折不掉

四个缺陷的共同点：**单例测试看着都对，只有"组合起来"才暴露**。
→ 这里逐条钉死。

跑：``python -m pytest tools/test_numeric_normalization.py -v``
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import normalize as N  # noqa: E402
import scoring as SC  # noqa: E402
from build_eval_candidates import extract_numbers  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 单位折叠：ASCII ↔ 中文 ↔ 上标变体
# ---------------------------------------------------------------------------

def test_ascii_and_chinese_units_match():
    """语料写 ASCII（"0.5MPa"），模型答中文（"0.5兆帕"）→ 必须命中。"""
    assert SC.numeric_hits("压力为0.5兆帕", ["0.5 MPa"])[0] == 1
    assert SC.numeric_hits("压力为0.5MPa", ["0.5 MPa"])[0] == 1
    assert SC.numeric_hits("长度5米", ["5 m"])[0] == 1
    assert SC.numeric_hits("时间24小时", ["24 小时"])[0] == 1


def test_superscript_and_ascii_are_equivalent():
    """``m³`` 与 ``m3`` 是同一个单位 —— 锚点写哪种都得能命中。"""
    assert SC.numeric_hits("10000m³", ["10000 m3"])[0] == 1
    assert SC.numeric_hits("10000m3", ["10000 m³"])[0] == 1
    assert N.match_num_norm("10 mg/m³") == N.match_num_norm("10 mg/m3")


def test_composite_unit_is_folded():
    """复合单位整体折（"mg/m3" → "毫克/立方米"），不能被拆成 "m3"。"""
    assert N.normalize_number("10 mg/m3") == "10毫克/立方米"
    assert SC.numeric_hits("浓度10mg/m3", ["10 mg/m3"])[0] == 1


def test_range_separator_unified():
    """语料写 "35～50KPa"，模型答 "35-50KPa" → 必须命中。"""
    assert SC.numeric_hits("压力为35-50KPa", ["35～50 KPa"])[0] == 1
    assert SC.numeric_hits("压力为35～50KPa", ["35～50 KPa"])[0] == 1


def test_negative_numbers_not_broken():
    """负数（"-75 %"）不能被范围分隔符的归一化误伤。"""
    assert SC.numeric_hits("浓度-75%", ["-75 %"])[0] == 1
    assert N.normalize_number("-75 %") == "-75%"


# ---------------------------------------------------------------------------
# 2. 锚点抽取
# ---------------------------------------------------------------------------

def test_extract_composite_unit_not_truncated():
    """🔴 K-81："10mg/m3" 不能被切成 "10 m"。"""
    out = extract_numbers("10mg/m3接触6h")
    assert "10 mg/m3" in out, out
    assert "10 m" not in out, out


def test_extract_range_value():
    """范围值必须**整段**抽出来（"35～50KPa" → "35～50 KPa"）。"""
    out = extract_numbers("正常操作压力为35～50KPa")
    assert "35～50 KPa" in out, out
    assert "50 KPa" not in out, out


def test_extract_handles_chinese_prefix():
    """🔴 Python 的 ``\\w`` 含中文 → 前置断言必须用 ASCII 类。"""
    out = extract_numbers("正常操作压力为35～50KPa")
    assert out, out
    out2 = extract_numbers("压力为0.5MPa")
    assert "0.5 MPa" in out2, out2


def test_extract_unit_case_insensitive():
    """🔴 白名单比较必须先小写化（原文 "KPa" vs 白名单 "kPa"）。"""
    assert "0.5 KPa" in extract_numbers("限值为0.5 KPa")


def test_extract_skips_pseudo_numbers():
    """年份、条款序号不是工程量。"""
    assert extract_numbers("第35条") == []
    assert extract_numbers("2023年") == []
    assert extract_numbers("第十三条") == []


def test_extract_keeps_original_separator():
    """锚点必须保留**原文分隔符**（否则溯源时逐字不可达）。"""
    out = extract_numbers("压力为35～50KPa")
    assert any("～" in x for x in out), out
