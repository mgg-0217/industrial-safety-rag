# 工业设备维修与安全规程问答 RAG

面向制造业的**工业设备维修 + 安全规程问答** RAG 项目。当前阶段：**PRD 已定稿，M1（版本建模）、M2（评测集精修，自动化部分 + 拒答机制扩容）、M3（chunk pipeline）、M5 检索基线 + M4 策略对比、M6/M7/M8（生成层）已完成**。

- **PRD**：`docs/PRD.md`（10 节，含切分策略、检索架构、版本治理、评测方案的字段级规格）
- PRD 生成提示词：`prompts/工业设备维修与安全规程问答-RAG-PRD提示词.md`（自包含，直接整份喂给 LLM 即可产出 `docs/PRD.md`）
- **抽取层决策记录**：`docs/抽取层决策记录.md`（M3 验收要求的"是否重跑抽取层"决策，含 OCR 独立验证）
- **检索基线与切分策略对比**：`docs/检索基线与切分策略对比.md`（M5/M4 实测报告：B2/B2′/B3/V 四组基线 × 6 策略）
- **生成层评测报告**：`docs/生成层评测报告.md`（M6/M7/M8 实测：306 题 × 3 配置 + 引用定位对照 + 延迟基准）
- **拒答机制扩容实验**：`docs/拒答机制扩容实验.md`（M2 补充批次：4 机制 150 条 + 同机制对照组 160 条 + 三分类行为模式）
- **归因隔离与提示词改造**：`docs/归因隔离与提示词改造.md`（M2 第二轮：`SYSTEM_V2` 提示词 + `bounded` 判定态）
- 语料来源与选型说明：`docs/语料来源与选型说明.md`
- 语料清单：`docs/语料清单.md`
- 评测集说明：`docs/评测集说明.md`（M2 产物的口径、题型分布与人工核对清单）
- 踩坑日志：`notes/踩坑.md`（**56 条**，每条按现象→根因→解法→教训）
- 失败样本日志：`notes/失败样本.md`（PRD 7.6 要求，每条按现象→根因→下一步，追加式）

---

## 当前数据

| 指标 | 数值 |
|---|---|
| 真实文档 | **27 份**（中国政府网站公开文件） |
| 正文字符 | **33.7 万** |
| 文档类型 | 15 类（安全管理制度 / 应急预案 / 事故调查报告 / 排查治理导则 / 行业标准 / 作业票表单 …） |
| 危险源标签 | **34 种**（动火、受限空间、中毒窒息、锅炉爆炸、起重机械倾覆、静电、超压、联锁失效 …） |
| 设备位号 | 真实位号（19-V-4215、T-4200、1910-PV-42151、MP-41009-200-S1B-H1 …） |
| 格式形态 | HTML正文 / 文本型PDF / **纯扫描件PDF** / 表格密集型 / 作业票表单 |
| 版本治理 | 含**同文两版**且新版明示废止旧版（榆林〔2025〕1号 废止〔2023〕26号），已建模（`src/versioning.py`） |
| 评测候选 | **1124 条**（1081 事实题 + 43 数值题带保真锚点），覆盖 **17** 份文档，**643** 条带页码锚点 |
| 正式题集 | **306 条**，PRD 7.1 的五类题型**全部齐备**（详见 `docs/评测集说明.md`） |
| 表格块 | **122 块**（真表头 86 / 伪表 32 / 键值对 4），逐文档见 `data/eval/table_quality.md` |
| 切分产物 | **6 种策略**的 chunk 全部产出（`data/chunks/<strategy>.jsonl`），见下表 |
| 检索基线 | **4 组基线**（B2 不过滤 / B2′ 过滤+α网格 / B3 混合 / V 仅向量）× 6 策略全部落盘（`eval_runs/`） |

### 六种切分策略的产物规模（M3 实测，M5 重建后）

| 策略 | 块数 | 均长 | p95 | 表块 | 伪表 | 跨页 | 覆盖文档 |
|---|---|---|---|---|---|---|---|
| 1 固定长度 L=512 O=64 | 752 | 501.4 | 512 | 0 | 0 | 0 | 26 |
| 2 结构感知 | 1946 | 168.8 | 590 | 73 | 16 | 120 | 26 |
| 3 条款级 | 3075 | 106.5 | 355 | 0 | 0 | 120 | 26 |
| 4 表格切分（表头随行，K=8） | 374 | 294.3 | 606 | 374 | 33 | 0 | 5 |
| 5 语义滑窗 L=400 | 782 | 483.4 | 1269 | 0 | 0 | 75 | 26 |
| 6 QA-pair（索引侧） | 803 | 169.9 | 479 | 0 | 0 | 0 | 10 |

> 策略 5 的均长高于目标 L=400：长块集中在**目录页**（无句末标点、无法按句子切）
> 与**表格**（表格是不可分割单元）。这是 PRD 3.1 允许的"块长不均"。
> 策略 6 只覆盖 10 份文档 —— 索引侧是候选池的一半，且只在有条款号的文档上有 QA。

### 检索基线实测（M5，264 道可评题，向量后端 L2）

| 配置 | fixed | structural | clause | semantic | 过期版本误召回 |
|---|---|---|---|---|---|
| B2（BM25，不过滤） | 0.879 | 0.875 | 0.849 | 0.856 | **0.29–0.33** |
| **B2′（BM25 + 版本过滤）** | **0.883** | **0.883** | 0.856 | 0.860 | **0.000** |
| B3（混合 + 过滤） | 0.803 | 0.784 | 0.712 | 0.769 | 0.000 |
| V（仅向量 + 过滤） | 0.724 | 0.701 | 0.633 | 0.670 | 0.000 |

> R@5，Wilson 95% CI 约 ±0.04。
> **三条硬结论**：① 版本过滤**不损召回**（反而 +0.004~+0.008），还把过期版本误召回从三成打到 0；
> ② 本机**没有预训练句向量**，向量路是负贡献，最强配置就是「BM25 + 过滤」（PRD 4.6 的 L4 降级）；
> ③ 六种策略的召回差距（0.856–0.883）**小于置信区间宽度**，只报粗结论，真正拉开差距的是规模（752 vs 3075 块）。
> 详见 `docs/检索基线与切分策略对比.md`。

### 生成层实测（M6 / M7 / M8，306 道题，检索组）

| 指标 | 值 | 口径 |
|---|---|---|
| **过期 chunk 进上下文** | **0 / 306** | 端到端**断言**（PRD 5.5），不为 0 说明版本过滤漏了 |
| **引用有效率** | **1.000**（1153 有效 / 0 无效） | 引用由**代码**生成，模型只输出 `[[Cn]]` 占位符 |
| 有无效引用的题数 | **0 / 306** | |
| **拒答误答率** | 🔴 **0.024 已作废**（见下） | 42 道拒答题，判定见 PRD 7.1 |
| **拒答误答率（4 机制扩容）** | **0.287**（43/150，CI [0.220, 0.364]） | 150 条多机制拒答题，`fixed` 策略；**宽松口径 0.020** |
| **拒答误答率（`system_v2` 提示词）** | **0.160**（24/150，CI [0.110, 0.227]） | 归因污染 26.7% → 14.0%；**明确告知"该主体没有专门规定" 68.7% → 96.7%** |
| 版本陷阱题版本命中 | **1.000**（41/41） | 期望文档在上下文里 |
| 事实题 key_point 覆盖 | 0.430（软 0.7）/ 0.168（严格字面） | 两个口径都报，原因见下 |
| 数值题精确匹配 | **0.506**（fixed）/ **0.557**（structural+top-8） | `must_include_numbers` 逐个比对 |
| 引用可定位率 | 页码 0.423 / **条款号 0.000** | fixed 块 0% 带条款号 → 对照组见报告第 4 节 |
| 条款可定位率（`structural` 对照） | **0.958**（top-5）/ **0.970**（top-8） | 🔴 条款级切分把引用从"指不到条款"提到"95.8% 指得到"，**答案质量持平、版本治理不受影响** → 切分策略的价值在引用定位，不在召回 |
| 嵌入后端 | **L4（纯 BM25）** | PRD 4.6 降级路径，**全链路跑通**（M8） |

> 🔴 **推荐配置 = `structural` 切分 + top-k 5**（2×2 完整对照，实测依据见
> `docs/附录表召回诊断.md`）：
> ① 条款可定位率 **0.000 → 0.958**（数量级差异）；② 答案质量**没有一项变差**
> （事实题严格口径 0.168 = 0.168，数值题严格 0.494 > 0.443，跨文档题严格 0.164 > 0.127）；
> ③ **版本治理完全不受影响**（版本陷阱题版本命中率四组都是 1.000、
> 上下文里的过期 chunk 都是 0）。
> ⚠️ **top-k 没有全局最优**：top-8 让数值题更好（精确匹配 0.506 → **0.557**），
> 但事实题更差、**拒答更差**（fixed+top-8 的误答率 6/42 = 0.143，是四组最差）→
> **默认 top-5，数值/限值类问题用 top-8**。
> 🔴 **横比切分策略必须按字符预算对齐**（六策略块长 107–501，
> 固定 top-k 是在比不同的上下文预算，PRD 3.3 第 7 条）。

> 🔴 **key_point 覆盖率必须看两个口径**：M2 把 key_point 对齐成**原文片段**以保证可复核，
> 但对齐产物很短（中位数 10 字）且带列表符号，导致**答对了却字面不匹配**。
> 实测未命中里 **87% 是口径产物，不是模型答错**（`notes/踩坑.md` K-42，分类见 `notes/失败样本.md`）。
>
> 详见 `docs/生成层评测报告.md`。

> 🔴🔴 **「拒答误答率 0.024」已作废（2026-11-05）** —— 那是**单机制 + 单措辞**测出来的。
> 扩到 150 条 4 种机制后，严格口径误答率 **0.287**（CI [0.220, 0.364]），
> 与 0.024 **完全不在一个区间**。
>
> 更要紧的是**根因**：同机制、同策略、**同一批 40 个主体**，只换题干措辞，
> 误答率从 **0.075 到 0.975（差 13 倍）**——
> **措辞越贴近语料里的条款结构（如"许可手续和确认程序"），
> 模型手里可套的通用条款就越多，归因污染的机会越大。**
> → **"拒答误答率"这个指标在很大程度上测的是"检索命中率"，不是"拒答能力"。**
> → **跨题集的拒答误答率一律不可外推**；同批题内部的对比（题干是控制变量）仍成立。
>
> 同时修正一个说法：模型**没有编造**（宽松口径 0.020 = 3/150 漏表态）。
> 三分类实测：**完全没意识到 2.0% / 表态但归因污染 26.7% / 完全正确 67.3%**。
> → 失败模式是**归因污染**（把通用条款写成主体专属规定），不是幻觉。
> 两者对策不同：幻觉靠"少给上下文"，归因污染靠**生成层的适用性标注约束**。
>
> 详见 `docs/拒答机制扩容实验.md`。
>
> 🔴 **✅ 归因隔离已用提示词治了**（`docs/归因隔离与提示词改造.md`）：
> `src/answering.py::SYSTEM_V2` 加两层约束（句式 + 适用性边界声明），
> 配套新增 `bounded` 判定态。150 条配对实测：
> **严格误答率 0.287 → 0.160（CI 不重叠）**、归因污染 26.7% → 14.0%、
> **明确告知"该主体没有专门规定"68.7% → 96.7%**；
> 且**没有变保守**（宽松误答率不变、事实题 key_point 覆盖持平、引用有效率仍 1.0000）。
> 新提示词下 `fixed` 与 `structural` 误答率**完全相同** → 改进全在生成层。

---

## 目录

```
工业维修安全规程RAG/
├─ data/
│  ├─ raw/              # 原始下载件（HTML/PDF），含 _rejected/ 归档（已 gitignore）
│  ├─ md/               # 统一抽取的 Markdown（含 front-matter 元数据）
│  ├─ manifest.json     # 语料元数据主表（PRD 元数据 schema 的真实实例）
│  ├─ probe_report.json # PDF 探针结果（文本型/扫描件/表格数）
│  ├─ chunks/           # 六种切分策略的产物（每策略一个 jsonl + 规模统计）
│  └─ eval/
│     ├─ candidates.jsonl      # 评测题候选（带证据锚点）
│     ├─ qa_pairs.csv          # QA 型 CSV（策略 6 的输入格式）
│     ├─ qa_pairs_rewritten.csv # 同上 + 改写后的自然语言题干（策略 6 检索路的前提）
│     ├─ test_set.jsonl        # 正式题集（五类题型，key_point 已对齐回原文）
│     ├─ check_report.md       # 题集质检报告（机器自检）
│     ├─ check_flags.json      # 机读标记（喂给人工核对抽样）
│     ├─ table_quality.md      # 逐文档表头质量（真表头/伪表/键值对，M3 验收产物）
│     ├─ qa_index_side.csv     # 策略 6：索引侧 QA（与评测侧严格不相交）
│     └─ qa_eval_side.csv      # 策略 6：评测侧 QA
├─ docs/
│  ├─ PRD.md                 # 产品需求文档（10 节，字段级规格）
│  ├─ 抽取层决策记录.md         # M3 验收产物：重跑抽取层的决策与证据
│  ├─ 检索基线与切分策略对比.md   # M5/M4 实测报告：四组基线 × 六策略
│  ├─ 生成层评测报告.md          # M6/M7/M8 实测报告：四组配置统一口径对比
│  ├─ 附录表召回诊断.md          # M10 实测报告：「多路索引」方案的证伪 + 替代方案
│  ├─ 拒答机制扩容实验.md         # M2 补充批次：4 机制 150 条 + 同机制对照组 + 三分类行为模式
│  ├─ 归因隔离与提示词改造.md      # M2 第二轮：SYSTEM_V2 提示词 + bounded 判定态（归因污染 26.7% → 14.0%）
│  ├─ 评测集说明.md            # M2 产物说明与人工核对清单
│  ├─ 语料清单.md
│  └─ 语料来源与选型说明.md
├─ eval_runs/              # M4/M5/M6 评测落盘（带时间戳，不覆盖，见目录内 README.md）
├─ notes/
│  ├─ 踩坑.md                # 踩坑日志（现象 → 根因 → 解法 → 教训）
│  └─ 失败样本.md             # 生成层失败样本分类（R/D/M）与人工核对清单
├─ prompts/
│  ├─ 工业设备维修与安全规程问答-RAG-PRD提示词.md
│  └─ PRD生成-调用说明.md
├─ src/
│  ├─ normalize.py          # 归一化模块（三轨分离：text / norm_text / tag_key）+ 匹配用归一化
│  ├─ pdf_lines.py          # PDF 基线重建（修抽取层的行序错乱，见抽取层决策记录）
│  ├─ versioning.py         # 版本关系建模（M1）：区间判定三态 + 推导规则
│  ├─ chunking.py           # 切分流水线（M3）：六策略共用框架 + 跨页合并 + 表头校验
│  ├─ bm25.py               # 自研 BM25（M5）：过滤后重算 IDF/avgdl + RRF 融合
│  ├─ embed.py              # 向量后端四级降级（M5）：L1 句向量 → L2 TF-IDF+SVD → L4 纯 BM25
│  ├─ retrieval.py          # 检索器（M5）：版本过滤 → 两路召回 → RRF → Top-K
│  ├─ version_card.py       # 版本卡（M6）：**代码强制拼接**，措辞随口径变
│  ├─ citation.py           # 引用卡片（M7）：**代码生成**，无效引用标记并计数
│  ├─ answering.py          # 问答装配（M6/M7）：检索 → 上下文 → 生成 → 拼装
│  ├─ scoring.py            # 生成层判分（M7）：key_point / 数值 / 拒答 / 引用
│  ├─ eval_align.py         # key_point 对齐源（按题型分源，质检与规范化共用）
│  └─ llm_client.py         # LLM 调用封装（重试 / 429 退避 / JSON 解析 / **凭据脱敏**）
└─ tools/
   ├─ html_to_md.py          # 政府网页正文抽取（含嵌套div配对 + 兜底）
   ├─ pdf_to_md.py           # PDF → Markdown（页码锚点 + 表格结构，**基线重建**）
   ├─ pdf_probe.py           # PDF 类型判定（文本层/扫描件/表格）
   ├─ build_manifest.py      # 生成 manifest.json + 语料清单.md
   ├─ build_eval_candidates.py # 条款切分 → 评测题候选 + QA CSV
   ├─ build_version_fields.py # 回填 effective_from / effective_to / superseded_by
   ├─ audit_eval_candidates.py # 候选池普查（题干质量根因定位）
   ├─ refine_eval_set.py     # M2 主流程：精修 / 合成三类题 / qa_split
   ├─ check_test_set.py      # 题集质检（PRD 7.2 标准）
   ├─ fix_key_points.py      # key_point 对齐回原文
   ├─ build_refusal_pool.py  # 拒答题用的安全假位号池
   ├─ rewrite_qa_questions.py # 策略 6 的题干改写（回填 + LLM 改写）
   ├─ relink_test_set.py     # 候选池重建后，题集 src_qid 的重新挂接
   ├─ fix_time_modes.py      # 修正题集里"时间口径 ↔ 出处版本"矛盾（38 条）
   ├─ build_chunks.py        # M3：跑六种切分策略，产出 chunk + 逐文档统计
   ├─ report_table_quality.py # M3：逐文档表头质量（真表头/伪表/键值对）
   ├─ run_retrieval_eval.py  # M5/M4：检索层评测（四组基线 × 六策略 + 共同子集 + **内容覆盖度**）
   ├─ run_generation_eval.py # M6/M7：生成层评测（检索 / 闭卷 / 给全文 三组配置）
   ├─ diag_table_recall.py   # M10：附录表召回诊断（5 种方案对照 + **等字符预算**）
   ├─ compare_qid_subset.py  # M10：按 qid 交集对比两个批次（分母一致）
   ├─ compare_generation.py  # 四组配置**统一口径**对比（离线重算，不调 API）
   ├─ build_refusal_pool_multitype.py # 多机制拒答**主体池**（双口径存在性 + 语义近邻剔除）
   ├─ synthesize_refusal_multitype.py # 多机制拒答题合成（主体确定性注入，不走 LLM）
   ├─ diag_refusal_near_miss.py # 主体**语义近邻**检测（精确匹配的必要补充）
   ├─ analyze_refusal_stats.py # 拒答统计：分层 + 三分类行为模式 + 聚类 bootstrap CI
   ├─ bench_latency.py      # 检索延迟基准（分阶段计时，PRD 7.6）
   ├─ probe_llm.py          # 端点连通性探测（**不打印凭据**，换 key 后先跑它）
   ├─ validate_rebuild_vs_ocr.py # 抽取层修复的独立验证（渲染页 OCR 当裁判）
   ├─ sweep_baseline_tol.py  # 基线容差扫参（用 OCR 真值选参数）
   ├─ diag_line_interleave.py # 诊断：md 层行交错检测
   ├─ diag_pdf_columns.py    # 诊断：PDF 层版式列结构检测
   ├─ diag_pdf_baseline.py   # 诊断：PDF 基线错位检测
   ├─ diag_line_rebuild.py   # 诊断：基线重建 vs 现状的逐页对比
   ├─ test_normalize.py      # 归一化模块自检
   ├─ test_versioning.py     # 版本建模单元测试（含链长 ≥ 3 的构造样例）
   ├─ test_chunking.py       # 切分 + 基线重建单元测试（36 条，M3 验收要求）
   ├─ test_retrieval.py      # 检索层单元测试（17 条，M5 验收要求）
   ├─ test_version_card.py   # 生成层单元测试（19 条，M6/M7 验收要求）
   └─ test_refusal_multitype.py # 多机制拒答判分回归（13 条，含中文主体与双口径）
```

> 仓库已 git 化（无远端）。`data/raw/` 原文入库而不只入库产物 —— 政府页面会改版失效，
> 重下无法保证与 manifest 记录一致；评测要可复现，语料就必须可复现。

## 重跑数据流水线

```bash
cd D:/ccworkplace/工业维修安全规程RAG
python tools/html_to_md.py data/raw data/md          # HTML → MD
python tools/pdf_to_md.py  data/raw data/md          # PDF  → MD
python tools/pdf_probe.py  data/raw data/probe_report.json
python tools/build_manifest.py                       # 清单 + manifest
python tools/build_eval_candidates.py                # 评测候选 + QA CSV
python tools/build_version_fields.py                 # 版本字段回填（M1）
```

> `pdf_to_md.py` 对 55/59 页的大导则抽表格较慢（数分钟），属正常。

## 重跑评测集（M2）

```bash
python tools/audit_eval_candidates.py                # ① 候选池普查
python tools/refine_eval_set.py --stage select       # ② 抽样（落盘清单，可续跑）
python tools/refine_eval_set.py --stage refine --limit 35 --workers 3   # ③ 改写题干
python tools/refine_eval_set.py --stage synthesize --only version --n-version 12   # ④ 三类新题
python tools/refine_eval_set.py --stage synthesize --only refusal --n-refusal 12
python tools/refine_eval_set.py --stage synthesize --only cross --n-cross 12
python tools/fix_key_points.py                       # ⑤ key_point 对齐回原文
python tools/check_test_set.py                       # ⑥ 质检
python tools/refine_eval_set.py --stage qa_split      # ⑦ 策略 6 切分
python tools/rewrite_qa_questions.py --stage backfill # ⑧ 策略 6 题干回填
python tools/rewrite_qa_questions.py --stage rewrite --limit 50 --workers 3
```

> ⚠️ **并发不能高**：并发 6 会撞 Token Plan 速率限制（实测 37/45 条被 429 打回），
> 并发 3 才稳定；429 的退避要比普通错误长（20s/40s，普通 3/6/12s 等不到额度恢复）。
> 批量调用在 auto 权限模式下可能被分类器拦，最稳的是用 `!` 前缀自己跑。

## 重跑检索基线与策略对比（M5 / M4）

```bash
python tools/fix_time_modes.py                         # 修题集口径矛盾（幂等，38 条）
python tools/build_chunks.py                           # 重建 chunk（改了 DOC_SNAPSHOT_FIELDS 才要重跑）
python tools/run_retrieval_eval.py \
    --strategies fixed,structural,clause,table,semantic,qa \
    --baselines B2,B3,B2prime,V                        # 四组基线
python -m pytest tools/test_retrieval.py -v            # 17 条检索层单测
```

> 产物落 `eval_runs/<时间戳>_*.json`（带时间戳，不覆盖）。
> 解读见 `docs/检索基线与切分策略对比.md`。

## 重跑拒答扩容实验（M2 补充批次）

```bash
python tools/build_refusal_pool_multitype.py --out refusal_pool_multitype_v2.json
python tools/synthesize_refusal_multitype.py \
    --pool refusal_pool_multitype_v2.json \
    --out refusal_extended_v2.jsonl --control \
    --control-out refusal_extended_control_v2.jsonl
python tools/diag_refusal_near_miss.py
python tools/probe_llm.py                    # 🔴 换 key 后先探额度
python tools/run_generation_eval.py --configs retrieval --strategy fixed \
    --questions-file refusal_extended_v2.jsonl --workers 3 --tag refuse_v2_fixed
python tools/run_generation_eval.py --configs retrieval --strategy structural \
    --questions-file refusal_extended_v2.jsonl --workers 3 --tag refuse_v2_structural
python tools/run_generation_eval.py --configs retrieval --strategy fixed \
    --questions-file refusal_extended_control_v2.jsonl --workers 3 --tag refuse_ctrl_fixed
python tools/analyze_refusal_stats.py --batch "..." --out eval_runs/<...>_refusal_analysis.json
python -m pytest tools/test_refusal_multitype.py -v          # 13 条
```

> 主集 `data/eval/test_set.jsonl` **未改动** —— 已发布的 2×2 对照与生成层数字保持可比。
> 扩展集是**独立题集**，分母分开统计。

## 重跑归因隔离实验（M2 补充批次 · 第二轮）

```bash
python tools/probe_llm.py                    # 🔴 换 key 后先探额度
python tools/run_generation_eval.py --configs retrieval --strategy fixed \
    --questions-file refusal_extended_v2.jsonl --workers 3 \
    --system system_v2 --tag refuse_v2sys2_fixed
python tools/run_generation_eval.py --configs retrieval --strategy structural \
    --questions-file refusal_extended_v2.jsonl --workers 3 \
    --system system_v2 --tag refuse_v2sys2_structural
# 副作用检查：主集事实题（旧批次用 --rescore 统一口径后配对比较）
python tools/run_generation_eval.py --configs retrieval --strategy fixed \
    --only-type 事实题 --workers 3 --system system_v2 --tag sys2_fact
python tools/analyze_refusal_stats.py --batch "..." --out eval_runs/<...>_refusal_sys2_analysis.json
python -m pytest tools/test_refusal_multitype.py -v          # 17 条
```

> ⚠️ **改提示词必须同步改判分**（本轮的 `bounded` 判定态），
> 且判分改动要用 `--rescore` 重算历史批次，核对**旧数字逐字不变**——
> 否则两批不可配对比较（`notes/踩坑.md` K-56）。

## 重跑生成层评测（M6 / M7 / M8）

```bash
python tools/probe_llm.py                               # ① 先探端点（不打印凭据）
python tools/run_generation_eval.py --configs retrieval --strategy fixed --tag main
python tools/run_generation_eval.py --configs retrieval --strategy structural --tag cite
python tools/run_generation_eval.py --configs closed_book --tag b1
python tools/run_generation_eval.py --configs full_doc  --tag b1
python tools/compare_generation.py --batches "..."      # ⑤ 四组统一口径对比（离线）
python tools/bench_latency.py --strategy fixed         # ⑥ 分阶段延迟基准
python -m pytest tools/test_version_card.py -v          # 23 条生成层单测
```

> 🔴 **不要直接横向比较各批次自己的 `.json`** —— 它们跑在不同代码版本下，
> 口径不同（单位折不折、软匹配加没加）。一律用 `tools/compare_generation.py`
> 离线重算后再比。详见 `docs/生成层评测报告.md` 第 1 节。

> ⚠️ **并发不能高**：并发 6 会撞 Token Plan 速率限制（实测 37/45 条被 429 打回），
> 并发 3 才稳定；429 的退避要比普通错误长（20s/40s，普通 3/6/12s 等不到额度恢复）。

---

## 已知缺口（诚实清单）

| 缺口 | 现状 | 补齐方式 |
|---|---|---|
| **题集人工核对未做** | 306 条全部 `review_status=pending`；生成层实跑又查出**锚点挂错条款**的一批（`notes/失败样本.md` F-01~F-04） | **用户逐条核对**；重点清单已从"质检报告"升级成"失败样本日志"（含 R/D/M 分类） |
| **`key_point` 覆盖率有两个口径** | 严格字面 0.168 / 软匹配 0.430；实测未命中里 **87% 是口径产物**（K-42） | 两个口径都报 + 阈值敏感性（0.6/0.7/0.8）；口径变更用 `--rescore` 离线重算，不重跑 API |
| **数值锚点会挂错条款** | 有题问的限值语料里**根本没给**，锚点是从别的条款扒来的 —— 模型答"未找到"是对的，却被判全错 | 人工核对时逐条回原文重核锚点（`notes/失败样本.md` F-01/F-02） |
| **附录表类内容召不回** | 🔴 **已实测证伪「多路索引」方案**：RRF 融合 / 合并索引 / 整表块三种变体**全部净负面**（总体 R@5 0.854 → 0.788–0.854，覆盖度 0.871 → 0.784–0.871）。真解法是**主索引换 `structural`**（表格题覆盖度 0.390 → **0.517**）+ 加大 top-k（0.871 → 0.907 @top-8） | 换 `structural` 作主索引（已在生成层同分母验证：条款可定位率 0.000 → **0.968**，数值精确匹配 0.506 → **0.557**）；补跑版本陷阱题的端到端数据 |
| **没有预训练句向量** | `huggingface.co` 超时、`sentence-transformers` 未装 → 退到 L2（TF-IDF+SVD），实测**向量路是负贡献**；生成层跑的是 **L4 纯 BM25** | 用户 `! pip install sentence-transformers`（走腾讯镜像）+ `HF_ENDPOINT=hf-mirror` 后重跑，落 `embed_backend=L1` 的结果 |
| 乱码语料 | `guide_shijiu_shoujiu` 的 gzip **已解压救回**（2026-11-03），但**质量判据还没改**（"非文本 → `failed`"） | 改 `html_to_md.py` 的质量判定 |
| **Word(.docx) 语料** | 公开源几乎没有 | 用你的内部维修手册替换；或造格式等价样例 |
| **厂商设备手册**（参数表/故障码/拆装步骤） | 施耐德/西门子/汇川/英威腾均防盗链或 SPA | 你的内部手册；或 CNIPA 专利公开文本 |
| 纯扫描件 OCR 语料 | 仅 1 份（运城市排查表） | 现有样本已可验证 OCR 链路；可再补 |
| 跨文档题多样性窄 | 25 条跨文档题集中于同一对版本文档 | 扩充危险源标签的跨文档覆盖（语料侧增加同类文档） |
| 策略 4 / 策略 6 口径受限 | 只覆盖 5/26、10/26 份文档，可评题 12 / 4，与主集合**不相交** | 真实系统里作为**多路索引并存**；横比要等语料补齐 |
| **版本链只有 1 条、链长 2** | 版本治理的验证面是"一个二版本链"，不是"同主题多版并存" | 语料侧补同主题多版文档 |
| **拒答题只有 42 条** | ✅ **已扩容**（2026-11-05）：150 条 4 机制 + 160 条同机制对照组 + 24 种题干骨架。`fixed` vs `structural` 误答率 **0.287 vs 0.293，CI 完全重叠 → 原来 1/42 vs 4/42 的差异是噪声** | 详见 `docs/拒答机制扩容实验.md` |
| 🔴 **拒答误答率对题干措辞极度敏感** | 同主体同批次只换问法，误答率 **0.075 ~ 0.975（13 倍）** | 拒答题集要**按"通用条款命中度"分层**设计，而不是按机制数堆量；跨题集数字不可外推 |
| **版本过滤是 O(块数) 线性扫描** | ✅ **已优化**（2026-11-04）：改成按 `doc_id` 缓存判定结果 + `Retriever` 跨查询缓存 → `clause` 策略端到端检索 **5.673ms → 1.470ms（降 74%）**，过滤不再占 66%；正确性实测 4 策略 × 5 组合完全一致 | 已完成，见 `docs/生成层评测报告.md` 5.3 |
| **B1b 全文组不是公平对照** | 它按题目 `doc_id` 投喂全文（= 提前告诉模型答案在哪），且 42 条拒答题不适用（**分母 264 vs 306**） | 只能当**上界**读，不能与检索组横比（PRD 3.3） |
| **四组用不同的系统提示词** | 对比表里的混淆变量，无法消除 | 已在报告 3.4 声明 |

---

## 下一步（按优先级）

1. **人工核对题集**（M2 收尾，**唯一挡住"结论可信"的一步**）：
   按 `notes/失败样本.md` 的 F-01~F-04 逐条回原文重核锚点，
   再按 `data/eval/check_report.md` 第二节过剩余题。
   🔴 **优先级高于调模型** —— 实测显示生成层的失败**大部分不是模型的错**。
2. ✅ **拒答题扩量到 150+ 条**（2026-11-05 完成）：150 条 4 机制 + 160 条同机制对照组。
   `fixed` vs `structural` 误答率 **0.287 vs 0.293，CI 完全重叠** →
   原来 1/42 vs 4/42 的差异是**噪声**。
   🔴 但发现了更根本的问题：**误答率由题干措辞决定**（同主体只换问法，0.075~0.975）。
   → 下一步不是继续扩量，而是**按"通用条款命中度"给拒答题分层**，
   并把"归因污染"做成**可单独度量的指标**。
3. **生成层加"适用性标注"约束**：✅ **已完成**（`docs/归因隔离与提示词改造.md`）
   —— 实测 **26.7% 的拒答题是"表态了但把通用条款套到主体上"**，
   `SYSTEM_V2` 提示词把它治到 **14.0%**（误答率 0.287 → 0.160，CI 不重叠），
   且没有变保守。剩余的残留主要来自**未经资料支持的分类推断**
   （"T-9042 属于装置区检维修作业"）→ 下一轮的靶子。
4. **附录表 / 表格多路索引**（M8 之后的优化项）：🔴 **已实测证伪** ——
   三种变体（RRF 融合 / 合并索引 / 整表块）全部净负面。
   **替代方案：主索引换 `structural`**（块边界 = 标题边界，表天然完整）：
   表格题内容覆盖度 0.390 → 0.517，条款可定位率 0.000 → 0.968。
   详见 `docs/附录表召回诊断.md`。
5. **补 L1 句向量**（需用户装包）后重跑，落 `embed_backend=L1` 的对比数字
6. **补语料**：Word(.docx) 内部手册、厂商参数表、同主题多版文档

> 完整规格见 `docs/PRD.md` 第 9 节里程碑表
> （M0、M1、M3、M4、M5、**M6、M7、M8、M9 已完成**；
> M2 自动化部分已完成、**人工核对待用户做**）。
