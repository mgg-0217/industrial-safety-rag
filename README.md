# 工业设备维修与安全规程问答 RAG

面向制造业的**工业设备维修 + 安全规程问答** RAG 项目。当前阶段：**PRD 已定稿，真实语料与评测基座已就位**。

- **PRD**：`docs/PRD.md`（10 节，含切分策略、检索架构、版本治理、评测方案的字段级规格）
- PRD 生成提示词：`prompts/工业设备维修与安全规程问答-RAG-PRD提示词.md`（自包含，直接整份喂给 LLM 即可产出 `docs/PRD.md`）
- 语料来源与选型说明：`docs/语料来源与选型说明.md`
- 语料清单：`docs/语料清单.md`

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
| 版本治理 | 含**同文两版**且新版明示废止旧版（榆林〔2025〕1号 废止〔2023〕26号） |
| 评测候选 | **842 条**（803 事实题 + 39 数值题带保真锚点），覆盖 15 份文档，362 条带页码锚点 |

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
│     ├─ candidates.jsonl # 评测题候选（带证据锚点）
│     └─ qa_pairs.csv     # QA 型 CSV（第6种切分策略的输入格式）
├─ docs/
│  ├─ PRD.md                 # 产品需求文档（10 节，字段级规格）
│  ├─ 语料清单.md
│  └─ 语料来源与选型说明.md
├─ notes/
│  └─ 踩坑.md                # 踩坑日志（现象 → 根因 → 解法 → 耗时）
├─ prompts/
│  ├─ 工业设备维修与安全规程问答-RAG-PRD提示词.md
│  └─ PRD生成-调用说明.md
├─ src/
│  └─ normalize.py          # 归一化模块（三轨分离：text / norm_text / tag_key）
└─ tools/
   ├─ html_to_md.py          # 政府网页正文抽取（含嵌套div配对 + 兜底）
   ├─ pdf_to_md.py           # PDF → Markdown（页码锚点 + 表格结构）
   ├─ pdf_probe.py           # PDF 类型判定（文本层/扫描件/表格）
   ├─ build_manifest.py      # 生成 manifest.json + 语料清单.md
   ├─ build_eval_candidates.py # 条款切分 → 评测题候选 + QA CSV
   └─ test_normalize.py      # 归一化模块自检（4 条 pytest）
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
```

> `pdf_to_md.py` 对 55/59 页的大导则抽表格较慢（数分钟），属正常。

---

## 已知缺口（诚实清单）

| 缺口 | 现状 | 补齐方式 |
|---|---|---|
| **Word(.docx) 语料** | 公开源几乎没有 | 用你的内部维修手册替换；或造格式等价样例 |
| **厂商设备手册**（参数表/故障码/拆装步骤） | 施耐德/西门子/汇川/英威腾均防盗链或 SPA | 你的内部手册；或 CNIPA 专利公开文本 |
| 纯扫描件 OCR 语料 | 仅 1 份（运城市排查表） | 现有样本已可验证 OCR 链路；可再补 |
| 拒答题 / 澄清题 | 需人工编写 | 候选池已标出应拒答的证据位置 |

---

## 下一步（按优先级）

1. **版本关系建模**（PRD M1）：给 manifest 补 `effective_from` / `effective_to` / `superseded_by`，
   推导规则与单元测试（链长 ≥ 3 的构造样例）
2. **评测集精修**（PRD M2）：842 条候选 → 正式题集，补齐版本陷阱题 / 拒答题 / 跨文档推理题
3. **chunk pipeline**（PRD M3–M4）：按 6 种切分策略切成带元数据的 chunk，跑策略对比实验
4. **检索基线**（PRD M5–M8）：BM25（自研）+ 向量路 + RRF + 先过滤再排序，跑 B1/B2/B3 三组基线

> 完整规格见 `docs/PRD.md` 第 9 节里程碑表（M0 已完成，M1 起步）。
