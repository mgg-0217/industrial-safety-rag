# 工业设备维修与安全规程问答 RAG

面向制造业的**工业设备维修 + 安全规程问答** RAG 项目。当前阶段：**PRD 已定稿，M1（版本建模）、M2（评测集精修，自动化部分）、M3（chunk pipeline）已完成**。

- **PRD**：`docs/PRD.md`（10 节，含切分策略、检索架构、版本治理、评测方案的字段级规格）
- PRD 生成提示词：`prompts/工业设备维修与安全规程问答-RAG-PRD提示词.md`（自包含，直接整份喂给 LLM 即可产出 `docs/PRD.md`）
- **抽取层决策记录**：`docs/抽取层决策记录.md`（M3 验收要求的"是否重跑抽取层"决策，含 OCR 独立验证）
- 语料来源与选型说明：`docs/语料来源与选型说明.md`
- 语料清单：`docs/语料清单.md`
- 评测集说明：`docs/评测集说明.md`（M2 产物的口径、题型分布与人工核对清单）
- 踩坑日志：`notes/踩坑.md`（31 条，每条按现象→根因→解法→教训）

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

### 六种切分策略的产物规模（M3 实测）

| 策略 | 块数 | 均长 | p95 | 表块 | 伪表 | 跨页 | 覆盖文档 |
|---|---|---|---|---|---|---|---|
| 1 固定长度 L=512 O=64 | 757 | 501.8 | 512 | 0 | 0 | 0 | 26 |
| 2 结构感知 | 1930 | 171.7 | 590 | 73 | 16 | 120 | 26 |
| 3 条款级 | 3068 | 107.6 | 355 | 0 | 0 | 120 | 26 |
| 4 表格切分（表头随行，K=8） | 374 | 294.3 | 606 | 374 | 33 | 0 | 5 |
| 5 语义滑窗 L=400 | 791 | 483.7 | 1228 | 0 | 0 | 75 | 26 |
| 6 QA-pair（索引侧） | 797 | 166.9 | 451 | 0 | 0 | 0 | 9 |

> 策略 5 的均长高于目标 L=400：长块集中在**目录页**（无句末标点、无法按句子切）
> 与**表格**（表格是不可分割单元）。这是 PRD 3.1 允许的"块长不均"。
> 策略 6 只覆盖 9 份文档 —— 索引侧是候选池的一半，且只在有条款号的文档上有 QA。

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
│  ├─ 评测集说明.md            # M2 产物说明与人工核对清单
│  ├─ 语料清单.md
│  └─ 语料来源与选型说明.md
├─ notes/
│  └─ 踩坑.md                # 踩坑日志（现象 → 根因 → 解法 → 教训）
├─ prompts/
│  ├─ 工业设备维修与安全规程问答-RAG-PRD提示词.md
│  └─ PRD生成-调用说明.md
├─ src/
│  ├─ normalize.py          # 归一化模块（三轨分离：text / norm_text / tag_key）
│  ├─ pdf_lines.py          # PDF 基线重建（修抽取层的行序错乱，见抽取层决策记录）
│  ├─ versioning.py         # 版本关系建模（M1）：区间判定三态 + 推导规则
│  ├─ chunking.py           # 切分流水线（M3）：六策略共用框架 + 跨页合并 + 表头校验
│  ├─ eval_align.py         # key_point 对齐源（按题型分源，质检与规范化共用）
│  └─ llm_client.py         # LLM 调用封装（重试 / 429 退避 / JSON 解析）
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
   ├─ build_chunks.py        # M3：跑六种切分策略，产出 chunk + 逐文档统计
   ├─ report_table_quality.py # M3：逐文档表头质量（真表头/伪表/键值对）
   ├─ validate_rebuild_vs_ocr.py # 抽取层修复的独立验证（渲染页 OCR 当裁判）
   ├─ sweep_baseline_tol.py  # 基线容差扫参（用 OCR 真值选参数）
   ├─ diag_line_interleave.py # 诊断：md 层行交错检测
   ├─ diag_pdf_columns.py    # 诊断：PDF 层版式列结构检测
   ├─ diag_pdf_baseline.py   # 诊断：PDF 基线错位检测
   ├─ diag_line_rebuild.py   # 诊断：基线重建 vs 现状的逐页对比
   ├─ test_normalize.py      # 归一化模块自检
   ├─ test_versioning.py     # 版本建模单元测试（含链长 ≥ 3 的构造样例）
   └─ test_chunking.py       # 切分 + 基线重建单元测试（36 条，M3 验收要求）
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

---

## 已知缺口（诚实清单）

| 缺口 | 现状 | 补齐方式 |
|---|---|---|
| **题集人工核对未做** | 306 条全部 `review_status=pending` | **用户逐条核对**；质检报告已标出重点题 |
| **1 份乱码语料** | `guide_shijiu_shoujiu` 的原始文件是没解压的 gzip，md 是二进制乱码，却标成 `extract_quality=ok` | 解压后重抽（内容是真的）；并把质量判据改成"非文本 → `failed`"（K-31） |
| **Word(.docx) 语料** | 公开源几乎没有 | 用你的内部维修手册替换；或造格式等价样例 |
| **厂商设备手册**（参数表/故障码/拆装步骤） | 施耐德/西门子/汇川/英威腾均防盗链或 SPA | 你的内部手册；或 CNIPA 专利公开文本 |
| 纯扫描件 OCR 语料 | 仅 1 份（运城市排查表） | 现有样本已可验证 OCR 链路；可再补 |
| 跨文档题多样性窄 | 25 条跨文档题集中于同一对版本文档 | 扩充危险源标签的跨文档覆盖（语料侧增加同类文档） |
| 数值题锚点幻觉 | 4 条锚点在原文中找不到 | 人工核对时逐条改锚点或删题 |

---

## 下一步（按优先级）

1. **人工核对题集**（M2 收尾）：`data/eval/check_report.md` 第二节列出的重点题
2. **检索基线**（PRD M5）：BM25（自研）+ 向量路 + RRF + 先过滤再排序，跑 B1/B2/B3 三组基线
3. **切分策略对比实验**（PRD M4）：在检索基线上跑 6 种策略，落盘 `eval_runs/<时间戳>_chunk_<策略>.json`

> 完整规格见 `docs/PRD.md` 第 9 节里程碑表
> （M0、M1 已完成；M2 自动化部分已完成、**人工核对待用户做**；M3 已完成）。
