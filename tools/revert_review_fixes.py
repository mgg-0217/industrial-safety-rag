#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""撤销 `fix_review_findings.py` 写下的溯源修正（按 `review_note` 还原旧值）。

🔴 **为什么要撤销**：
溯源修正是**基于当时的 key_point** 做的定位。
而 key_point 后来又被 `fix_key_points.py` 重对齐过（指向变了）→
那批修正就**不再成立**，必须先还原，等 key_point 定稿后再重新定位。

用法：
    python tools/revert_review_fixes.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"


def main():
    rows = [json.loads(l) for l in TEST_SET.open(encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        note = r.get("review_note") or ""
        if "溯源条款由" in note and "自动更正为" in note:
            old = note.split("溯源条款由 ")[1].split(" 自动更正为")[0].strip()
            r["clause_no"] = old
            del r["review_note"]
            n += 1
    with TEST_SET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已还原 {n} 条溯源修正")


if __name__ == "__main__":
    main()
