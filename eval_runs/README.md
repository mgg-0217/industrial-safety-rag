# eval_runs/ —— 评测落盘（带时间戳，不覆盖）

PRD 7.5 的约定：**每次重跑生成新的时间戳文件，不覆盖旧批次**，
所以这里能看到"评测口径一路怎么改的"。

---

## ⚠️ 当前权威批次

### 生成层（M6/M7，2026-11-04）

| 文件 | 内容 |
|---|---|
| `20261104_0950_generation_retrieval_main.answers.jsonl` | **检索组(fixed)** 306 题逐题明细（模型原始回答 + 引用 + 版本卡） |
| `20261104_1039_generation_retrieval_cite.answers.jsonl` | **检索组(structural)** 306 题逐题明细（引用定位对照） |
| `20261104_1000_generation_closed_book_b1.answers.jsonl` | **闭卷 B1a** 306 题逐题明细 |
| `20261104_1032_generation_full_doc_b1.answers.jsonl` | **全文 B1b** 264 题逐题明细（42 条拒答题无所属文档，skip） |
| `20261104_1045_generation_compare.json` | 🔴 **四组统一口径对比表** —— 报告里的数字全部出自这里 |

> 🔴 **不要直接横向比较各批次自己的 `.json`**：
> 它们是在不同代码版本下跑的（单位折不折、软匹配加没加），
> 数字口径不同。**一律用 `tools/compare_generation.py` 离线重算后再比**：

```bash
python tools/compare_generation.py --batches \
  "检索组(fixed)=eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl,\
检索组(structural)=eval_runs/20261104_1039_generation_retrieval_cite.answers.jsonl,\
闭卷B1a=eval_runs/20261104_1000_generation_closed_book_b1.answers.jsonl,\
全文B1b=eval_runs/20261104_1032_generation_full_doc_b1.answers.jsonl"
```

### 检索层（M5/M4，2026-11-03）

| 文件 | 内容 |
|---|---|
| `20261103_1912_chunk_<策略>.json` | 6 份，每份含 4 组基线的全部指标 |
| `20261103_1912_baseline_<基线>.json` | 13 份，按基线切分 |
| `20261103_1912_common_subset.json` | 共同子集对照（含"六策略交集为空"这一事实） |

解读见 `docs/检索基线与切分策略对比.md`。

### 延迟基准（PRD 7.6）

`20261104_1024_latency_fixed.json` / `20261104_1049_latency_structural.json` /
`20261104_1049_latency_clause.json` —— 分阶段计时，
用 `tools/bench_latency.py` 复现。

---

## 之前的批次：**数字不可用，但留着是为了留痕**

> ⚠️ **这四批只在本地保留、没有被提交进仓库**（它们是"口径错了"的历史现场，
> 不是权威结果）。要复现"修 bug 前后的差异"需本地重跑；
> 仓库里留下的是 `20261103_1912_*`（权威）与本文件的说明。

| 批次 | 为什么不能用 |
|---|---|
| `20261103_1001_*` | 修 K-32 之前跑的：chunk 快照缺 `effective_to_source`，`as_of` 口径下 26 份文档被误排除 |
| `20261103_1003_*` | 同上；且只有 B2′ 的 α 网格，没有 V 基线 |
| `20261103_1907_*` | 修了 K-32，但"不可评"的题只在开了版本过滤的配置下才被排除（K-33）→ 策略 4 的 B2 召回被压到 0.077（实际 0.500） |
| `20261103_1910_*` | 修了 K-33，但共同子集取的是六策略交集（= 空集），分组口径不对 |
| `20261104_1000_generation_full_doc_b1.*` | 🔴 **跑到 83/306 就中断**，汇总 json 根本没写出来 —— 已 gitignore，**别当"全文基线只有 83 题"** |

> 🔴 **别拿这四个检索层批次的数字对外说**。
> 它们唯一的价值是：能看出"修 K-32/K-33 前后，同一套题的召回从 0.077 变成 0.500"，
> 这本身就是"评测口径错了会造出假结论"的证据。

---

## 不入库的批次（`.gitignore`）

n ≤ 12 的**冒烟试跑**、失败的额度探测、连通性探测（`tools/probe_llm.py`）。
信息量为零，留在目录里只会让人误以为是"另一个口径的权威批次"。
**文件本身保留在本地**，没删。
