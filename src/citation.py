# -*- coding: utf-8 -*-
"""引用卡片：**由代码生成**（PRD 6.1 硬约束）。

## 为什么不让模型生成引用

模型编造引用（引一个不存在的条款号、错一个页码）在长文档问答里是高发错误，
而"引用可核对"恰恰是本项目对用户的核心价值承诺。
让模型自由生成引用 = 把最有价值的承诺交给最不可靠的环节。

→ 模型输出里只允许出现**占位符** ``[[C1]]``，指向上下文中给它的第 1 块；
   **真实引用卡片由代码渲染**。

## 无效引用不静默丢弃

模型输出了上下文里没有的编号（比如自己编了个 ``[[C9]]``）：
- 行内标记成 ``[C9｜引用无效]``；
- 答案末尾追加一行告警，**并计数**（`citation_valid_rate` 的分母就是它）。

🔴 这条同时是**评测指标**：`citation_valid_rate` =
有效引用数 / 引用总数（PRD 7.3）。

## 原文用 `text` 轨，不用 `norm_text` 轨

归一化后的文本里全角数字被转成了半角、位号破折号被改了，
**展示给用户必须一字不改** —— 这就是三轨分离里 `text` 这一轨的职责。
"""
import re

import version_card as VC

#: 模型输出里的引用占位符：`[[C1]]` / `[[C12]]`
CITE_RE = re.compile(r"\[\[C(\d+)\]\]")


# ---------------------------------------------------------------------------
# 页码
# ---------------------------------------------------------------------------

def page_text(chunk):
    """引用卡片里的页码描述。

    - 普通块有 `page_no` → "第 N 页"
    - 跨页块（`page_no` 为 null、`strategy_meta.cross_page` 有值）→ "第 N–M 页"
    - HTML 源（`page_no` 为 null 且无跨页信息）→ "无页码（网页来源）"，**不编页码**
    """
    p = chunk.get("page_no")
    if p is not None:
        return f"第 {p} 页"
    meta = chunk.get("strategy_meta") or {}
    pages = meta.get("cross_page")
    if pages:
        pages = sorted(pages)
        if len(pages) == 1:
            return f"第 {pages[0]} 页"
        return f"第 {pages[0]}–{pages[-1]} 页"
    return "无页码（网页来源）"


def clause_text(chunk):
    """条款标识：优先 `clause_no`，其次 `section_path` 的最后一段，都没有就空。"""
    cn = chunk.get("clause_no")
    if cn:
        return str(cn)
    sp = chunk.get("section_path") or []
    if sp:
        return str(sp[-1])
    return ""


def quality_note(chunk):
    """质量标注。

    `ok` 不显示；`partial` / `ocr_verified` 要显示；
    `failed` / `ocr_required` **不应出现在引用里**（PRD 2.8 直接排除）。
    """
    q = chunk.get("quality")
    if q == "partial":
        return "（该来源抽取不完整）"
    if q == "ocr_verified":
        return "（来自 OCR 识别，可能有识别错误）"
    return ""


# ---------------------------------------------------------------------------
# 卡片
# ---------------------------------------------------------------------------

#: 引用卡片里"原文"的最大展示长度。
#:
#: 🔴 实测踩过：表格块的原文有 2000 多字，整块贴进卡片，答案根本没法读。
#: 但**不能丢溯源** —— 截断处必须给出精确指针（PRD 6.3 的三层溯源）：
#: 答案 → chunk（条款+页码+原文块）→ md 原文（字符区间）→ 原始文件（URL/文件名）。
QUOTE_MAX = 300


def quote_text(chunk):
    """引用卡片里的原文展示（截断 + 精确溯源指针）。

    🔴 截断不是丢信息 —— 后面跟着 `data/md/<doc_id>.md` 的字符区间，
    用户按区间就能取到一字不差的全文。
    """
    text = (chunk.get("text") or "").strip()
    if len(text) <= QUOTE_MAX:
        return text
    cs = chunk.get("char_start")
    ce = chunk.get("char_end")
    did = chunk.get("doc_id") or (chunk.get("doc_snapshot") or {}).get("doc_id")
    if cs is not None and ce is not None:
        tail = (f"……（全文 {len(text)} 字，"
                f"见 data/md/{did}.md 字符区间 [{cs}, {ce})）")
    else:
        tail = f"……（全文 {len(text)} 字）"
    return text[:QUOTE_MAX] + tail


def build_card(cid, chunk, manifest=None, when=None, mode="current"):
    """一张引用卡片（PRD 6.2 的格式）。"""
    snap = chunk.get("doc_snapshot") or {}
    title = snap.get("title") or chunk.get("doc_id") or "（未知标题）"
    doc_no = snap.get("doc_no") or ""
    version = snap.get("version") or ""
    meta = "，".join(x for x in (doc_no, version) if x)

    lines = [f"[C{cid}] 《{title}》" + (f"（{meta}）" if meta else "")]
    loc = clause_text(chunk)
    if loc:
        lines.append(f"     条款：{loc}")
    lines.append(f"     页码：{page_text(chunk)}")
    qn = quality_note(chunk)
    lines.append(f"     原文：「{quote_text(chunk)}」")

    # 状态行：与版本卡同一套措辞（current 现行 / as_of 该时点生效）
    lines.append(f"     状态：{VC.status_text(snap, when, mode)}"
                 f"    来源：{snap.get('source_url') or '—'}"
                 + (f"    质量：{qn}" if qn else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(model_text, ctx_chunks, manifest=None, when=None, mode="current",
           section_title="引用（由系统按检索结果自动生成）"):
    """把模型输出里的占位符渲染成引用，并生成引用区。

    :param model_text: 模型原始输出
    :param ctx_chunks: 上下文块列表（顺序即编号 ``C1..Cn``）
    :returns: ``(渲染后的正文, 统计dict)``

    统计字段：
    - ``n_refs`` 模型输出的引用总数（去重前，按出现次数）
    - ``n_valid`` 指向真实上下文的引用数
    - ``n_invalid`` 指向不存在编号的引用数
    - ``invalid_ids`` 无效编号列表
    - ``cited`` 被有效引用到的 chunk 编号列表
    """
    model_text = model_text or ""
    n_ctx = len(ctx_chunks)
    found = CITE_RE.findall(model_text)

    invalid_ids = []
    valid_ids = []
    for sid in found:
        n = int(sid)
        if 1 <= n <= n_ctx:
            valid_ids.append(n)
        else:
            invalid_ids.append(n)

    # 行内替换：有效 → [C1]；无效 → [C9｜引用无效]
    def _sub(m):
        n = int(m.group(1))
        if 1 <= n <= n_ctx:
            return f"[C{n}]"
        return f"[C{n}｜引用无效]"

    body = CITE_RE.sub(_sub, model_text)

    # 引用区：只列被有效引用到的块（按首次引用顺序）
    cited = []
    for n in valid_ids:
        if n not in cited:
            cited.append(n)
    cards = [build_card(n, ctx_chunks[n - 1], manifest=manifest, when=when,
                         mode=mode) for n in cited]

    stats = {
        "n_refs": len(found),
        "n_valid": len(valid_ids),
        "n_invalid": len(invalid_ids),
        "invalid_ids": sorted(set(invalid_ids)),
        "cited": cited,
    }

    out = body
    if cards:
        out += ("\n\n┌─ " + section_title + " " +
                "─" * max(0, 36 - len(section_title)) + "\n")
        out += "\n".join(cards) + "\n└" + "─" * 46
    if invalid_ids:
        uniq = sorted(set(invalid_ids))
        out += ("\n\n⚠️ 无效引用告警：模型引用了上下文中不存在的编号 "
                + "、".join(f"C{i}" for i in uniq)
                + f"（共 {stats['n_invalid']} 处）—— 这些引用无法核对，已原样标出。")
    return out, stats
