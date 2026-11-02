# -*- coding: utf-8 -*-
"""给 manifest.json 补版本关系字段。

新增 6 个字段（每份文档一份）：

  effective_from          区间判定用的生效日（正文明示 > 人工标注）
  effective_from_source   declared / doc_date / unknown
  effective_to            失效日（None = 至今未失效）
  effective_to_source     explicit / derived / open / unknown
  superseded_by           继任版本的 doc_id
  supersedes              本版本明示废止的旧版本 doc_id 列表

为什么要有 source 字段：**推导值不等于明示值**。
`effective_to` 可能是"库里可见的下一个版本的生效日"（derived），
中间版本缺失时会晚于真实废止日（PRD 2.6）。带上来源标记，
消费方才能区分"文件自己说的"和"我们推出来的"。

跑法：
    python tools/build_version_fields.py             # 写回 data/manifest.json
    python tools/build_version_fields.py --dry-run   # 只报告，不写
    python tools/build_version_fields.py --report    # 打印冲突报告
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.versioning import apply_version_fields  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.json"
MD_DIR = ROOT / "data" / "md"

NEW_FIELDS = [
    "effective_from",
    "effective_from_source",
    "effective_to",
    "effective_to_source",
    "superseded_by",
    "supersedes",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--report", action="store_true", help="打印版本关系全表")
    args = ap.parse_args()

    docs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    new_docs, conflicts = apply_version_fields(docs, MD_DIR)

    # ---- 冲突报告 ----
    if conflicts:
        print(f"⚠️ 发现 {len(conflicts)} 处冲突（以正文为准，人工标注保留在原字段）：")
        for c in conflicts:
            print(f"  [{c['type']}] {c['doc_id']}")
            print(f"      人工标注：{c['recorded']}   正文明示：{c['declared']}")
            print(f"      原文：{c['quote']}")
            print(f"      处置：{c['resolution']}")
    else:
        print("✅ 无冲突")

    # ---- 统计 ----
    from collections import Counter
    src_from = Counter(d.get("effective_from_source") for d in new_docs)
    src_to = Counter(d.get("effective_to_source") for d in new_docs)
    print()
    print("生效日来源：", dict(src_from))
    print("失效日来源：", dict(src_to))

    chains = [d for d in new_docs if d.get("superseded_by")]
    print(f"有继任版本的文档：{len(chains)}")
    for d in chains:
        print(f"  {d['doc_id']} → {d['superseded_by']}"
              f"（{d['effective_from']} ~ {d['effective_to']}，来源 {d['effective_to_source']}）")

    if args.report:
        print()
        print("全表：")
        for d in new_docs:
            print(f"  {d['doc_id']:38s} {str(d.get('effective_from')):12s} "
                  f"{str(d.get('effective_to')):12s} "
                  f"[{d.get('effective_from_source')}/{d.get('effective_to_source')}]")

    if args.dry_run:
        print("\n(dry-run，未写文件)")
        return 0

    # ---- 写回 ----
    MANIFEST.write_text(
        json.dumps(new_docs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n✅ 已写回 {MANIFEST.relative_to(ROOT)}（{len(new_docs)} 份文档，"
          f"新增 {len(NEW_FIELDS)} 个字段）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
