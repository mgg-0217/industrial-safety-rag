# -*- coding: utf-8 -*-
"""修正题集里**时间口径与出处版本矛盾**的题（2026-11-03，M5 实测发现）。

## 背景：两处矛盾，都是检索层测出来的

### 矛盾一 · 出处版本已过期，题目却标 `current` 口径（13 道事实题）

这 13 道题的出处是 `yulin_2023_old`（榆林 2023 版，2025-01-09 已废止），
但 `time_mode` 标的是 `current`。
后果：**现行口径下旧版被版本过滤正确地滤掉，这些题在语料里根本答不了**
（实测事实题召回从 0.837 掉到 0.767，差的正是这 13 道）。

处置：改成 `as_of(2023-12-31)` ——
它们问的就是"2023 版当时是怎么规定的"，这正是 PRD 5.2 的 `as_of` 口径
（"复盘'2023 年那会儿适用的是哪版'"）。
版本卡会如实写明"该版本已于 2025-01-09 被替代"（PRD 5.2 的口径措辞规则）。

### 矛盾二 · "比较两版"的问题，`current` 口径把旧版滤掉了（25 道跨文档题）

这 25 道题问"两份榆林制度有何差异"，**两个版本都要在场**。
但 `current` 口径只留现行版、`as_of` 口径只留当时生效的那版 ——
两种口径都答不了"比较两版"（实测跨文档题召回只有 **0.24**）。

处置：新增 **`cross_version` 口径**（PRD 5.2 同步 amend）——
该口径**不做版本过滤**，由版本卡负责标注哪一版是现行、哪一版是历史。

## 为什么用工具改而不是手改

306 条题要人工核对（还没做），任何改动都必须：
1. **可复跑**（同样的规则再跑一次，结果不变）；
2. **可审计**（打印每一条改了什么、为什么）；
3. **不碰内容**（只改 `time_mode` / `query_time` 两个字段，题干与答案一个字不动）。

## 跑法

    python tools/fix_time_modes.py            # 改并落盘
    python tools/fix_time_modes.py --dry-run  # 只看会改什么
"""
import argparse
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"

#: 出处版本已过期、但题目标的是 current 口径的文档（及其当时的有效时刻）
EXPIRED_SOURCE_AS_OF = {
    "yulin_2023_old": "2023-12-31",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in TEST_SET.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    changes = []
    for r in rows:
        qt = r.get("question_type")
        tm = r.get("time_mode")
        # 矛盾一：出处版本已过期，标 current 的事实题 → as_of
        src = r.get("doc_id")
        if (tm == "current" and qt in ("事实题", "数值题")
                and src in EXPIRED_SOURCE_AS_OF):
            changes.append((r["qid"], qt, "current", "as_of",
                            f"出处 {src} 已过期，current 口径下不可答"))
            r["time_mode"] = "as_of"
            r["query_time"] = EXPIRED_SOURCE_AS_OF[src]
            continue
        # 矛盾二：跨版本对比题 → cross_version
        if qt == "跨文档推理题" and tm != "cross_version":
            changes.append((r["qid"], qt, tm, "cross_version",
                            "问两版差异，两版都要在场；current 口径会滤掉旧版"))
            r["time_mode"] = "cross_version"

    print(f"题集 {len(rows)} 条，待改 {len(changes)} 条：")
    by = Counter((c[1], c[2], c[3]) for c in changes)
    for (qt, a, b), n in sorted(by.items()):
        print(f"  {qt:12s} {a:14s} -> {b:14s} {n:3d} 条")
    print("  样例：")
    for c in changes[:5]:
        print(f"    {c[0]}  {c[1]}  {c[2]}->{c[3]}  ({c[4]})")

    if args.dry_run:
        print("\n--dry-run，未落盘")
        return
    with TEST_SET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已写入 {TEST_SET.relative_to(ROOT)}")
    print("新的 time_mode 分布：",
          dict(Counter(r.get("time_mode") for r in rows)))


if __name__ == "__main__":
    main()
