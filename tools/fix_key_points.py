"""key_point 规范化 —— 把 LLM 写的 key_point 落回原文片段。

**为什么必须做**（实测数据）：
  PRD 7.3 的 `key_point_coverage` = 命中数 / 总数，是"可复核"的指标。
  可复核的前提是：每个 key_point 能用**字面包含**在原文中找到。
  但 LLM 写的 key_point 是**概括性改写**，三类问题：

    A. 数值题带了标签前缀        "数值锚点：5 m"      -> 数字对，前缀是噪声
    B. 措辞被改写                "连接对象为浮顶与罐体"
                                 原文 "应将浮顶与罐体…作电气连接"  -> 语义对，字面不匹配
    C. 幻觉（原文里根本没有）    "50 %" / "80 %"       -> 必须剔除

  按原样用，`key_point_coverage` 的分母就是假的——分母里的东西永远命中不了。

**修法（全部本地，不调 LLM）**：
  1. 数值题：key_point 直接取 `must_include_numbers`（人工/规则提取的保真锚点）。
     PRD 7.1 明确数值题用"数值精确匹配"判分，锚点本就是判分依据，
     让 LLM 再概括一遍只会引入改写风险。
  2. 事实题：用最长公共子串把 kp 对齐到原文的具体片段，
     对齐成功就用**原文片段**替换（这样字面包含判定成立）。
  3. 对齐失败的（幻觉）：从 key_points 剔除，并标记 orphan_key_points；
     若剔完为空，该题标 needs_manual_review，不进指标。

输入/输出都落盘，原文件备份为 .bak（test_set.jsonl 尚未入库，可从候选池重跑恢复）。
"""
import argparse
import json
import pathlib
import re
import shutil
from difflib import SequenceMatcher

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"

# LLM 给数值题写的标签前缀
LABEL_PREFIX = re.compile(
    r"^\s*(数值锚点|干扰数值|正确锚点|答案数值|关键数值|数值)\s*[:：]\s*")
# 归一化：去空白 + 统一标点
TABLE = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
    "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "－": "-", "～": "~", "％": "%",
})


def flat(s):
    return re.sub(r"\s+", "", s or "").translate(TABLE).lower()


def best_span(kp, ans):
    """kp 在 ans 里的最长公共连续片段（返回原文片段，可能为空）。"""
    if not kp or not ans:
        return ""
    sm = SequenceMatcher(None, kp, ans, autojunk=False)
    blocks = sm.get_matching_blocks()
    best = max(blocks, key=lambda b: b.size)
    if best.size == 0:
        return ""
    return ans[best.b:best.b + best.size]


def align_key_point(kp, ans, min_len=4):
    """把 kp 对齐到原文片段。

    判据：**只要公共片段 >= min_len 个字就算对齐成功**，用该片段替换 kp。

    🔴 为什么不用"公共片段 >= kp 长度的 60%"这种比例门槛：
    实测 93 个 kp 因此被误剔，而它们**语义全是对的**，只是 LLM 做了概括：
        "开展设备管道试压、吹扫等准备"  <- 原文含 "管道试压、吹扫"（7/13 = 54%）
        "紧抓重点环节并采取断然措施"    <- 原文含 "紧抓重点环节"（6/13 = 46%）
    剔掉它们会丢掉真正的关键信息点，还会让 20 条题的 key_points 变成空。
    片段短一点不要紧——**只要它是原文里真实存在的片段，字面包含判定就成立**，
    这才是 PRD 要的"可复核"。

    只有公共片段为空（真幻觉，如 "50 %"）才剔除。
    """
    kp_f = flat(kp)
    ans_f = flat(ans)
    if not kp_f or not ans_f:
        return None, ""
    # 先去掉标签前缀再比
    kp_clean = LABEL_PREFIX.sub("", kp).strip()
    kp_f = flat(kp_clean)
    span = best_span(kp_f, ans_f)
    if not span:
        return None, ""
    if len(span) >= min_len:
        return span, span
    return None, span


def normalize_anchor(a):
    """锚点规范化：去标签前缀、去空格、统一单位空格。"""
    a = LABEL_PREFIX.sub("", str(a)).strip()
    a = re.sub(r"\s*([%°a-zA-Z³²/·])\s*$", r"\1", a)  # "50 %" -> "50%"
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(TEST_SET))
    ap.add_argument("--backup", action="store_true", default=True)
    ap.add_argument("--from-backup", action="store_true",
                    help="先从 .bak 恢复再改（脚本幂等性有疑问时的安全做法）")
    args = ap.parse_args()

    path = pathlib.Path(args.test_set)
    bak = path.with_suffix(".jsonl.bak")
    if args.from_backup:
        if not bak.exists():
            raise SystemExit(f"找不到备份 {bak}")
        shutil.copy2(bak, path)
        print(f"已从 {bak.name} 恢复")

    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    print(f"题集：{len(rows)} 条")

    if args.backup and not args.from_backup:
        shutil.copy2(path, bak)
        print(f"备份 -> {bak.name}")

    stats = {"numeric_replaced": 0, "fact_aligned": 0,
             "fact_orphaned": 0, "orphan_kps": 0, "emptied": 0}
    for r in rows:
        ans = r.get("gold_answer") or ""
        kps = r.get("key_points") or []
        qtype = r.get("question_type")

        if qtype == "数值题":
            anchors = r.get("must_include_numbers") or []
            if isinstance(anchors, str):
                anchors = [a for a in anchors.split(";") if a.strip()]
            norm = []
            for a in anchors:
                a = normalize_anchor(a)
                if a and a not in norm:
                    norm.append(a)
            if norm:
                if norm != kps:
                    stats["numeric_replaced"] += 1
                r["key_points"] = norm[:4]
                r["key_point_source"] = "must_include_numbers"
            continue

        # 事实题 / 跨文档 / 版本陷阱：把 kp 对齐回原文片段
        aligned, orphans = [], []
        for kp in kps:
            span, raw_span = align_key_point(kp, ans)
            if span:
                aligned.append(span)
                if flat(span) != flat(kp):
                    stats["fact_aligned"] += 1
            else:
                orphans.append({"key_point": kp, "best_common": raw_span})
        if orphans:
            stats["fact_orphaned"] += 1
            stats["orphan_kps"] += len(orphans)
            r["orphan_key_points"] = orphans
        if aligned:
            r["key_points"] = aligned[:4]
            r["key_point_source"] = "aligned_to_source"
        else:
            stats["emptied"] += 1
            r["key_points"] = []
            r["needs_manual_review"] = True

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n数值题改用保真锚点：{stats['numeric_replaced']} 条")
    print(f"事实题 kp 对齐到原文：{stats['fact_aligned']} 个")
    print(f"剔���的对齐失败 kp：{stats['orphan_kps']} 个"
          f"（涉及 {stats['fact_orphaned']} 条题）")
    print(f"key_points 被清空的题：{stats['emptied']} 条（已标 needs_manual_review）")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
