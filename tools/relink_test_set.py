"""题集 → 候选池的**重新挂接**（候选池重建后跑）。

## 为什么需要这个脚本

2026-11-02 重跑抽取层后，`build_eval_candidates.py` 重新生成了候选池
（842 → 1124 条）。候选的 `qid` 是 `doc_id__clause_no` 拼接，
而 `clause_no` 来自正文切分 —— **正文一变，qid 就变**。

实测有 **4 条**题集的 `src_qid` 因此悬空（找不到对应候选）：

| 旧 clause_no | 新 clause_no | 原因 |
|---|---|---|
| `1.硫化氢（H` | `1.硫化氢（H2S）` | 下标 `2` 被基线重建归位 |
| `2.甲烷（CH` | `2.甲烷（CH4）` | 同上 |
| `5.磷化氢（PH` | `5.磷化氢（PH3）` | 同上 |
| `2.管道MP—41009—200—S1B—H1` | `2.管道MP-41009-200-S1B-H1` | 破折号归一化 |

**题干与数值锚点都没变**（下标归位不影响正文语义），所以只需改挂接关系。

## 匹配规则

同文档内，新 clause_no 的**归一化形式**以旧 clause_no 的归一化形式为前缀
（旧的被截断，新的补全了）→ 唯一命中。多于一个命中时报错，不猜。

跑法：
    python tools/relink_test_set.py            # 改写并落盘
    python tools/relink_test_set.py --dry-run  # 只报告
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import normalize as N  # noqa: E402

TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
CANDIDATES = ROOT / "data" / "eval" / "candidates.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cands = load_jsonl(CANDIDATES)
    by_qid = {c["qid"]: c for c in cands}
    by_doc = {}
    for c in cands:
        by_doc.setdefault(c["doc_id"], []).append(c)

    tests = load_jsonl(TEST_SET)

    fixed = []
    unresolved = []
    for t in tests:
        src = t.get("src_qid")
        if not src:
            continue
        if src in by_qid:
            # 还活着，但 clause_no 可能已变（候选池里被改写过）
            new = by_qid[src]
            if t.get("clause_no") != new.get("clause_no"):
                fixed.append((t, {"clause_no": new["clause_no"],
                                  "src_qid": src, "_why": "clause_no 文本变化"}))
                t["clause_no"] = new["clause_no"]
            continue
        # 悬空：找同文档下"新 clause 以旧 clause 为前缀"的候选
        doc = t.get("doc_id")
        old_norm = N.normalize(src.split("__", 1)[-1])
        hits = [c for c in by_doc.get(doc, [])
                if N.normalize(c["clause_no"]).startswith(old_norm)]
        if len(hits) == 1:
            new = hits[0]
            fixed.append((t, {"clause_no": new["clause_no"], "src_qid": new["qid"],
                              "_why": "下标/破折号归位后重挂"}))
            t["clause_no"] = new["clause_no"]
            t["src_qid"] = new["qid"]
        elif not hits:
            unresolved.append((t, "同文档下找不到前缀匹配"))
        else:
            unresolved.append((t, f"前缀匹配到 {len(hits)} 个候选，不猜"))

    print(f"题集 {len(tests)} 条 | 需重挂 {len(fixed)} 条 | 无法重挂 {len(unresolved)} 条")
    for t, info in fixed:
        print(f"  [重挂] {t['qid']}  {info['_why']}")
        print(f"         clause_no -> {info['clause_no']}")
    for t, why in unresolved:
        print(f"  [未解] {t['qid']}  {why}  src_qid={t.get('src_qid')}")

    if unresolved:
        print("\n⚠️ 有无法重挂的题，未写盘（避免造出悬空引用）")
        return 1
    if not args.dry_run:
        TEST_SET.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in tests) + "\n",
            encoding="utf-8")
        print(f"\n-> {TEST_SET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
