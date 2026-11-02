# 工业设备维修与安全规程问答 RAG

面向制造业的**工业设备维修 + 安全规程问答** RAG 项目。当前阶段：**PRD 已定稿，版本关系建模（M1）与评测集精修（M2）的自动化部分已完成**。

- **PRD**：`docs/PRD.md`（10 节，含切分策略、检索架构、版本治理、评测方案的字段级规格）
- PRD 生成提示词：`prompts/工业设备维修与安全规程问答-RAG-PRD提示词.md`（自包含，直接整份喂给 LLM 即可产出 `docs/PRD.md`）
- 语料来源与选型说明：`docs/语料来源与选型说明.md`
- 语料清单：`docs/语料清单.md`
- 评测集说明：`docs/评测集说明.md`（M2 产物的口径、题型分布与人工核对清单）

---

## 当前数据

| 指标 | 数值 |
|---|---|
| 真实文档 | **27 份**（中国政府网站公开文件） |
| 正文字符 | **34.0 万** |
| 文档类型 | 15 类（安全管理制度 / 应急预案 / 事故调查报告 / 排查治理导则 / 行业标准 / 作业票表单 …） |
| 危险源标签 | **34 种**（动火、受限空间、中毒窒息、锅炉爆炸、起重机械倾覆、静电、超压、联锁失效 …） |
| 设备位号 | 真实位号（19-V-4215、T-4200、1910-PV-42151、MP-41009-200-S1B-H1 …） |
| 格式形态 | HTML正文 / 文本型PDF / **纯扫描件PDF** / 表格密集型 / 作业票表单 |
| 版本治理 | 含**同文两版**且新版明示废止旧版（榆林〔2025〕1号 废止〔2023〕26号），已建模（`src/versioning.py`） |
| 评测候选 | **842 条**（803 事实题 + 39 数值题带保真锚点），覆盖 15 份文档，362 条带页码锚点 |
| 正式题集 | **306 条**，PRD 7.1 的五类题型**全部齐备**（详见 `docs/评测集说明.md`） |

---

## 目录

```
工业维修安全规程RAG/
├─ data/
│  ├─ raw/              # 原始下载件（HTML/PDF），含 _rejected/ 归档（已 gitignore）
│  ├─ md/               # 统一抽取的 Markdown（含 front-matter 元数据）
│  ├─ manifest.json     # 语料元数据主表（PRD 元数据 schema 的真实实例）
│  ├─ probe_report.json # PDF 探针结果（文本型/扫描件/表格数）
│  └─ eval/
│     ├─ candidates.jsonl      # 评测题候选（带证据锚点）
│     ├─ qa_pairs.csv          # QA 型 CSV（策略 6 的输入格式）
│     ├─ qa_pairs_rewritten.csv # 同上 + 改写后的自然语言题干（策略 6 检索路的前提）
│     ├─ test_set.jsonl        # 正式题集（五类题型，key_point 已对齐回原文）
│     ├─ check_report.md       # 题集质检报告（机器自检）
│     ├─ check_flags.json      # 机读标记（喂给人工核对抽样）
│     ├─ qa_index_side.csv     # 策略 6：索引侧 QA（与评测侧严格不相交）
│     └─ qa_eval_side.csv      # 策略 6：评测侧 QA
├─ docs/
│  ├─ PRD.md                 # 产品需求文档（10 节，字段级规格）
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
│  ├─ versioning.py         # 版本关系建模（M1）：区间判定三态 + 推导规则
│  ├─ eval_align.py         # key_point 对齐源（按题型分源，质检与规范化共用）
│  └─ llm_client.py         # LLM 调用封装（重试 / 429 退避 / JSON 解析）
└─ tools/
   ├─ html_to_md.py          # 政府网页正文抽取（含嵌套div配对 + 兜底）
   ├─ pdf_to_md.py           # PDF → Markdown（页码锚点 + 表格结构）
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
   ├─ test_normalize.py      # 归一化模块自检
   └─ test_versioning.py     # 版本建模单元测试（含链长 ≥ 3 的构造样例）
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
| **Word(.docx) 语料** | 公开源几乎没有 | 用你的内部维修手册替换；或造格式等价样例 |
| **厂商设备手册**（参数表/故障码/拆装步骤） | 施耐德/西门子/汇川/英威腾均防盗链或 SPA | 你的内部手册；或 CNIPA 专利公开文本 |
| 纯扫描件 OCR 语料 | 仅 1 份（运城市排查表） | 现有样本已可验证 OCR 链路；可再补 |
| 跨文档题多样性窄 | 25 条跨文档题集中于同一对版本文档 | 扩充危险源标签的跨文档覆盖（语料侧增加同类文档） |
| 数值题锚点幻觉 | 4 条锚点在原文中找不到 | 人工核对时逐条改锚点或删题 |

---

## 下一步（按优先级）

1. **人工核对题集**（M2 收尾）：`data/eval/check_report.md` 第二节列出的重点题
2. **chunk pipeline**（PRD M3–M4）：按 6 种切分策略切成带元数据的 chunk，跑策略对比实验
3. **检索基线**（PRD M5–M8）：BM25（自研）+ 向量路 + RRF + 先过滤再排序，跑 B1/B2/B3 三组基线

> 完整规格见 `docs/PRD.md` 第 9 节里程碑表（M0、M1 已完成，M2 自动化部分已完成）。
