# 人工核对队列

生成脚本：`tools/review_eval_set.py`
题集：data/eval/test_set.candidate2.jsonl（306 条）

| 结论 | 条数 | 说明 |
|---|---|---|
| `auto_pass` | 306 | 机器核对项全过 |
| `needs_human` | 0 | 有机器判不了/判软的疑点 |
| `likely_bad` | 0 | **机器判出明确问题**，建议直接改题 |

→ **人工只需要看下面 0 条**（而不是全部 306 条）。

---

