"""诊断：key_point "不可达"到底是真不可达，还是判据太死。

分三类：
  A. 数字在答案里、只是短语不匹配（标签前缀 / 措辞改写） -> 判据问题
  B. key_point 里的数字在答案里找不到                    -> 真不可达
  C. 答案里根本没有该数字，但 key_point 表述正确         -> 答案被截断等问题
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "data" / "eval" / "test_set.jsonl").open(encoding="utf-8")]
flags = json.load((ROOT / "data" / "eval" / "check_flags.json").open(encoding="utf-8"))
bad_qids = {e["qid"] for e in flags if "unreachable_key_points" in e["flags"]}

def flat(s):
    return re.sub(r"\s+", "", s or "")

cat = {"A_数字在答案里": [], "B_数字找不到": [], "C_无数字": []}
for r in rows:
    if r["qid"] not in bad_qids:
        continue
    ans = flat(r.get("gold_answer") or "")
    f = next(e["flags"] for e in flags if e["qid"] == r["qid"])
    for kp in f["unreachable_key_points"]:
        nums = re.findall(r"\d+(?:\.\d+)?", kp)
        if not nums:
            cat["C_无数字"].append((r["qid"], r.get("question_type"), kp, ""))
            continue
        hits = [n for n in nums if flat(n) in ans]
        if hits and len(hits) == len(nums):
            cat["A_数字在答案里"].append((r["qid"], r.get("question_type"), kp, ",".join(hits)))
        elif hits:
            cat["B_数字找不到"].append((r["qid"], r.get("question_type"), kp,
                                        "部分命中:" + ",".join(hits)))
        else:
            cat["B_数字找不到"].append((r["qid"], r.get("question_type"), kp, "全未命中"))

for k, v in cat.items():
    print(f"{k}: {len(v)}")
print()

# C 类：打印答案原文，判定是"匹配 bug"还是"LLM 幻觉"
if "--detail" in __import__("sys").argv:
    print("===== C 类：答案原文 =====")
    by_q = {r["qid"]: r for r in rows}
    for qid, qt, kp, _ in cat["C_无数字"][:6]:
        r = by_q[qid]
        ans = (r.get("gold_answer") or "")
        print(f"\n--- kp: {kp}")
        print(f"    答案({len(ans)}字): {ans[:220]}")
    print()
    print("===== B 类：答案原文 =====")
    for qid, qt, kp, extra in cat["B_数字找不到"][:3]:
        r = by_q[qid]
        ans = (r.get("gold_answer") or "")
        print(f"\n--- kp: {kp}  ({extra})")
        print(f"    答案({len(ans)}字): {ans[:300]}")
    print()
else:
    for k, v in cat.items():
        print(f"===== {k} =====")
        for qid, qt, kp, extra in v[:8]:
            print(f"  [{qt}] {kp}   ({extra})")
        print()
