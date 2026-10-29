# -*- coding: utf-8 -*-
"""归一化模块的自检 —— 用真实语料验证，不是构造样例。

四组验证：
  1. 全角数字归一化后归零
  2. **设备位号能否被 manifest 标注的位号召回**（核心验证）
  3. 中文破折号有没有被误伤
  4. 归一化是否改变了原文长度以外的语义（抽查对照）

跑法：
    python tools/test_normalize.py          # 直接出报告
    pytest tools/test_normalize.py -v       # 也兼容 pytest
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.normalize import (  # noqa: E402
    build_chunk_text,
    is_equipment_tag,
    normalize,
    tag_key,
)

MD_DIR = ROOT / "data" / "md"
MANIFEST = ROOT / "data" / "manifest.json"

_FULLWIDTH_DIGIT = re.compile(r"[０-９Ａ-Ｚａ-ｚ]")


# ---------------------------------------------------------------------------
def load_docs() -> list[tuple[str, str]]:
    """返回 [(doc_id, 正文)]，正文已剥掉 front-matter。"""
    out = []
    for path in sorted(MD_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        body = raw.split("---", 2)[-1] if raw.startswith("---") else raw
        out.append((path.stem, body))
    return out


def check_1_fullwidth(docs) -> tuple[int, int]:
    """全角数字/字母归一化后应归零。返回 (归一化前数量, 归一化后数量)。"""
    before = after = 0
    for _, body in docs:
        before += len(_FULLWIDTH_DIGIT.findall(body))
        after += len(_FULLWIDTH_DIGIT.findall(normalize(body)))
    return before, after


def check_2_equipment_tags(docs) -> tuple[int, int, int, list[str], list[str]]:
    """核心验证：manifest 里人工标注的位号，在正文中能否被找回。

    归一化前用原样匹配，归一化后用归一化文本匹配。位号在原文里是
    全角破折号，所以不归一化必然漏召 —— 这正是要验证的。

    注意：manifest 的 equipment_tags 里目前混着少量**中文设备类别名**
    （如「厂内机动车辆」），那不是位号，不参与位号匹配，单独统计。
    返回 (总标注数, 归一化前命中, 归一化后命中, 未命中列表, 中文类别列表)
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    doc_map = dict(docs)

    total = hit_before = hit_after = 0
    misses: list[str] = []
    chinese: list[str] = []
    for entry in manifest:
        tags = entry.get("equipment_tags") or []
        if not tags:
            continue
        doc_id = entry["doc_id"]
        body = doc_map.get(doc_id)
        for tag in tags:
            total += 1
            if not is_equipment_tag(tag):
                chinese.append(f"{doc_id}: {tag}")
                continue
            if body is None:
                continue
            if tag in body:
                hit_before += 1
            norm = normalize(body)
            squashed = re.sub(r"[-_]", "", norm).upper()
            if tag in norm or tag_key(tag) in squashed:
                hit_after += 1
            else:
                misses.append(f"{doc_id}: {tag}")
    return total, hit_before, hit_after, misses, chinese


def check_3_dash_untouched(docs) -> tuple[int, int]:
    """中文破折号（——）和顿号、书名号不该被动。返回 (前, 后)。"""
    before = after = 0
    for _, body in docs:
        before += body.count("——")
        after += normalize(body).count("——")
    return before, after


def check_4_tag_pool_coverage(docs) -> list[tuple[str, str, int]]:
    """位号池的覆盖情况：每个标注位号在全语料里出现多少次。

    出现次数为 0 的，说明标注了但正文抽不出（抽取质量问题）；
    出现次数远大于标注文档数的，说明这个位号在别的文档里也提到了
    （跨文档关联，是好事）。
    返回 [(doc_id, tag, 出现次数)]，只列出现次数为 0 的。
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tagged_docs: dict[str, str] = {}
    for entry in manifest:
        for tag in entry.get("equipment_tags") or []:
            if is_equipment_tag(tag):
                tagged_docs[tag] = entry["doc_id"]

    norm_docs = [(doc_id, normalize(body)) for doc_id, body in docs]
    zero: list[tuple[str, str, int]] = []
    for tag, doc_id in tagged_docs.items():
        n = sum(norm.count(tag) for _, norm in norm_docs)
        if n == 0:
            zero.append((doc_id, tag, 0))
    return zero


def check_5_double_text(docs) -> None:
    """双轨文本：原文必须一字不动。"""
    for doc_id, body in docs:
        text, norm = build_chunk_text(body)
        assert text == body, f"{doc_id}: 原文被改动了"
        assert norm != body or not body.strip(), f"{doc_id}: 归一化副本与原文相同（应至少折叠空白）"


# ---------------------------------------------------------------------------
def main() -> int:
    docs = load_docs()
    print(f"语料：{len(docs)} 份\n" + "=" * 62)

    print("\n[1] 全角数字/字母归一化")
    b, a = check_1_fullwidth(docs)
    print(f"    归一化前 {b} 个  →  归一化后 {a} 个")
    ok1 = a == 0
    print(f"    {'✅ 通过' if ok1 else '❌ 未归零'}")

    print("\n[2] 设备位号召回（manifest 人工标注 → 正文查找）")
    total, hb, ha, misses, chinese = check_2_equipment_tags(docs)
    n_tags = total - len(chinese)
    print(f"    标注总数 {total}，其中位号形态 {n_tags} 个，中文设备类别名 {len(chinese)} 个")
    print(f"    归一化前命中 {hb} 个 ({100 * hb / max(1, n_tags):.1f}%)")
    print(f"    归一化后命中 {ha} 个 ({100 * ha / max(1, n_tags):.1f}%)")
    if misses:
        print("    归一化后仍未命中：")
        for m in misses:
            print(f"      - {m}")
    if chinese:
        print("    ⚠️ equipment_tags 里混入了中文设备类别名（不是位号，需 PRD 裁决归属）：")
        for c in chinese:
            print(f"      - {c}")
    ok2 = not misses
    print(f"    {'✅ 归一化后位号全部命中' if ok2 else '❌ 仍有未命中'}")

    print("\n[3] 中文破折号是否被误伤")
    db, da = check_3_dash_untouched(docs)
    print(f"    '——' 归一化前 {db} 个  →  归一化后 {da} 个")
    ok3 = db == da
    print(f"    {'✅ 未被误伤' if ok3 else '❌ 被改动了'}")

    print("\n[4] 位号池覆盖（标注了但正文抽不出的）")
    zero = check_4_tag_pool_coverage(docs)
    if zero:
        for doc_id, tag, _ in zero:
            print(f"      - {doc_id}: {tag}")
        print(f"    ⚠️ {len(zero)} 个标注位号在正文中找不到（抽取质量问题，不阻塞）")
    else:
        print("    ✅ 全部标注位号都能在正文里找到")

    print("\n[5] 双轨文本：原文是否保持不动")
    try:
        check_5_double_text(docs)
        print("    ✅ 原文全部保持不动")
        ok5 = True
    except AssertionError as e:
        print(f"    ❌ {e}")
        ok5 = False

    print("\n" + "=" * 62)
    all_ok = ok1 and ok2 and ok3 and ok5
    print("总判定：" + ("✅ 全部通过" if all_ok else "❌ 有未通过项"))
    return 0 if all_ok else 1


# --- pytest 兼容 ------------------------------------------------------------
def test_fullwidth_gone():
    before, after = check_1_fullwidth(load_docs())
    assert before > 1000, "语料里应该存在大量全角数字"
    assert after == 0


def test_equipment_tags_all_found():
    docs = load_docs()
    total, hb, ha, misses, chinese = check_2_equipment_tags(docs)
    assert total - len(chinese) > 0, "manifest 里应该有带位号的文档"
    assert not misses, f"归一化后仍未命中：{misses}"


def test_dash_untouched():
    before, after = check_3_dash_untouched(load_docs())
    assert before > 0
    assert before == after


def test_double_text_keeps_original():
    check_5_double_text(load_docs())


if __name__ == "__main__":
    sys.exit(main())
