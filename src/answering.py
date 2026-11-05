# -*- coding: utf-8 -*-
"""问答装配层（PRD M6 / M7）：检索 → 上下文 → 生成 → **代码**拼版本卡与引用。

```
question ─┬─ 检索（先过滤再排序）─┬─ 上下文块 C1..Cn（原文轨）
          │                       └─ 版本事实（标题/文号/生效日）
          └─ 无检索基线（B1）─────── 语料子集 / 空

   LLM ──→ 正文（只带 [[Cn]] 占位符）
        └─ 代码 ──→ 引用卡片（PRD 6.1）+ 版本卡（PRD 5.4）
```

## 三条硬约束（都是 PRD 的红线）

1. **版本卡由代码拼接**，不靠 prompt 让模型自觉（PRD 5.4 / 第 8 节红线 8）。
2. **过期 chunk 根本不进上下文**（PRD 5.5）——
   所以不存在"模型引用了过期条款、然后再提示它"这种情况。
3. **引用由代码生成**（PRD 6.1）——
   模型只被允许引用上下文中给它的编号；编出来的编号会被标成"引用无效"并计数。

🔴 prompt 只负责"让模型别乱说"，**不负责正确性** ——
正确性全部由代码侧的过滤、拼装、判分保证。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import citation as CIT      # noqa: E402
import retrieval as RT      # noqa: E402
import scoring as SC        # noqa: E402
import version_card as VC   # noqa: E402


# ---------------------------------------------------------------------------
# 🔴 代码级归因兜底（2026-11-05）
# ---------------------------------------------------------------------------
#
# 背景：M2 第二轮实测发现，"归因污染"（模型说了未找到，却仍把通用条款写成
# 主体专属规定）是拒答场景的主要失败模式。`SYSTEM_V2` 提示词把它治到 14%，
# 但**残留的那部分靠提示词治不了** —— 模型会先做
# 「T-9042 属于装置区检维修作业」这种**未经资料支持的分类推断**，再套用制度。
#
# 🔴 **代码能做的判断，模型就不该扛**：
# 「本次召回的 chunk 里，有没有任何一块**正文提到**这个主体？」
#
# * 提到了 → 那就是**关于该主体**的内容（真实场景）；
# * 没提到 → 那这些 chunk 全是**通用条款**（假主体场景）。
#
# 这个判据**不需要 LLM、也不需要正则猜**，是纯代码事实。
# → 代码据此在答案末尾**强制追加**归因声明，
#    无论模型写的是什么，用户都不会把通用规定误当成该主体的专门规定。
#
# ⚠️ **只在模型没自己说清楚时才追加**（避免重复）：
#   模型已经标注了适用性边界（`SC.has_boundary_declaration`）
#   或已经明说"资料里没有该主体的规定"（`SC.has_subject_absence_statement`）
#   → 都不追加。
#
# ⚠️ **本兜底只对"主体明确"的场景有效**（拒答题）。
#   事实题/数值题没有单一主体概念，主体判定不成立 → 不启用。

ATTRIBUTION_FALLBACK = (
    "⚠️ **归因说明（系统自动追加）**：本次检索到的资料中"
    "**未出现「{subject}」**，因此以上引用均为**通用规定**，"
    "**不构成对「{subject}」的专门规定**。"
)

#: 🔴 **污染句的不予采信声明**（2026-11-05 兜底升级）。
#:
#: 只追加一句声明是**不够的** —— 实测（150 条旧 system 批次）：
#: 追加声明只把严格误答率从 0.287 压到 0.273（**只降 2 条**），
#: 因为模型正文里那句「结论：<主体> 应……」**原样还在**，
#: 末尾补一句"这些是通用规定"是**自相矛盾**的。
#:
#: → 兜底必须**动正文**：把污染句**划掉**并明示"不予采信"。
POLLUTION_NOTICE = (
    "🔴 **不予采信**：答案中有 {n} 句把通用条款表述为「{subject}」的"
    "专门规定，这些句子已被划去，**请以划线部分之外的内容为准**。"
)


def chunks_mention_subject(chunks, subject):
    """召回的 chunk 里有没有任何一块**正文提到**该主体。

    🔴 查的是**原文轨** `text`（不是归一化轨）—— 与展示给模型的内容一致。

    :returns: ``True`` / ``False`` / ``None``（主体为空，无法判定）
    """
    if not subject:
        return None
    for c in chunks or ():
        if subject in (c.get("text") or ""):
            return True
    return False


def mark_pollution(text, pollution):
    """把污染句**就地划掉**（删除线）。

    🔴 **必须用占位符两段式替换**：直接 `replace(s, "~~s~~")` 会在
    同一句话出现两次时产生 `~~~~s~~~~`（嵌套删除线）。

    :param pollution: ``find_pollution_sentences`` 的返回值
    :returns: 标记后的文本（原文没被改动的地方逐字保留）
    """
    out = text or ""
    marks = []
    for i, item in enumerate(pollution or ()):
        s = (item[0] if isinstance(item, (tuple, list)) else item).strip()
        if not s or s not in out:
            continue
        ph = f"\x00POLL{i}\x00"
        out = out.replace(s, ph)
        marks.append((ph, s))
    for ph, s in marks:
        out = out.replace(ph, SC.MARK_POLLUTION_OPEN + s + SC.MARK_POLLUTION_CLOSE)
    return out


def attribution_fallback(answer_text, chunks, subject):
    """代码级归因兜底：**两件事**，各自独立触发。

    1. **污染句划除**（无条件）：只要答案里有"把通用条款当成主体规定"
       的句子，就把它划掉并标注不予采信。
       → 这一步**不依赖**模型有没有说清楚，因为污染句本身就是错的。
    2. **归因声明追加**（有条件）：召回的 chunk 里根本没出现这个主体，
       且模型自己既没做边界声明、也没说"资料里没有该主体的规定"
       → 在末尾补一句系统声明。

    🔴 为什么第 2 步要"模型没说清楚才补"：避免与模型自己的话重复。

    :returns: ``(需要追加声明?, 追加的文本, 污染句列表)``
    """
    if not subject:
        return False, "", []
    pollution = SC.find_pollution_sentences(answer_text, subject)
    mentioned = chunks_mention_subject(chunks, subject)
    if mentioned is not False:
        # 有块提到该主体 → 引用是关于它的内容，不追加声明
        return False, "", pollution
    if SC.has_boundary_declaration(answer_text, subject)[0]:
        return False, "", pollution
    if SC.has_subject_absence_statement(answer_text, subject)[0]:
        return False, "", pollution
    return True, ATTRIBUTION_FALLBACK.format(subject=subject), pollution


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------

def build_context_block(chunks, when=None, mode="current"):
    """把命中的块拼成给模型看的上下文（带 ``[C1]`` 编号）。

    🔴 上下文里给的是 **原文轨**（`text`），不是归一化轨 ——
    展示与核对要用原文，归一化会把全角数字和位号破折号改掉。
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        snap = c.get("doc_snapshot") or {}
        loc = CIT.clause_text(c)
        meta = "，".join(x for x in (snap.get("doc_no"), snap.get("version")) if x)
        head = f"[C{i}] 《{snap.get('title') or c.get('doc_id')}》"
        if meta:
            head += f"（{meta}）"
        lines = [head]
        if loc:
            lines.append(f"    条款：{loc}")
        lines.append(f"    页码：{CIT.page_text(c)}")
        lines.append(f"    原文：{(c.get('text') or '').strip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


SYSTEM = (
    "你是制造业工业安全规程问答助手。你只能依据用户给出的【资料片段】回答问题。\n"
    "规则：\n"
    "1. 依据某个片段得出的结论，后面用 [[C1]] 这样的编号标注来源；多个来源写 [[C1]][[C2]]。\n"
    "2. 只允许引用给出过的编号，**不得编造编号**，也不得写出片段里没有的条款号或数值。\n"
    "3. 如果片段里没有答案，就明确说“未找到相关规定”，并说明缺什么，不要用相邻内容凑答案。\n"
    "4. 用简体中文回答，先给结论再给依据，简洁。"
)

#: 🔴 **适用性标注版**系统提示词（2026-11-05）。
#:
#: 起因：M2 拒答扩容实测（150 条 4 机制）发现**归因污染**才是拒答场景的主要失败模式：
#:
#: | 行为模式 | 占比 |
#: |---|---|
#: | 完全没意识到语料里没有该主体 | 2.0% |
#: | **表态了，但把通用条款写成主体专属规定** | **26.7%** |
#: | 完全正确 | 67.3% |
#:
#: 典型形态（真实产物）：
#:
#: > 「**结论：设备 T-9042 的检维修作业应至少履行以下许可和确认程序**……
#: >   资料未提供 T-9042 的具体检维修方案……无法确认其已实际完成上述手续。」
#:
#: `SYSTEM` 的规则 3 已经说了"不要用相邻内容凑答案"，**但没有约束句式** ——
#: 模型照样用「结论：<主体> 应/须/需要……」把通用条款归给主体。
#:
#: 本版加的是**句式 + 边界声明**两层约束：
#: * **句式**：主体没有专门规定时，不许用"结论：<主体> 应……"下结论；
#: * **边界声明**：要给通用要求，必须**显式标注其适用性不等同于对该主体的专门规定**。
#:
#: ⚠️ **改了提示词就必须同时改判分**，否则会得出"改了没用"或"改了更好"的错误结论。
#: 判分侧对应的新规则见 `src/scoring.py::refusal_judgement`
#: （`mode="strict"` 下新增"显式适用性边界声明"的降级通道）。
SYSTEM_V2 = (
    "你是制造业工业安全规程问答助手。你只能依据用户给出的【资料片段】回答问题。\n"
    "规则：\n"
    "1. 依据某个片段得出的结论，后面用 [[C1]] 这样的编号标注来源；多个来源写 [[C1]][[C2]]。\n"
    "2. 只允许引用给出过的编号，**不得编造编号**，也不得写出片段里没有的条款号或数值。\n"
    "3. 如果片段里没有答案，就明确说“未找到相关规定”，并说明缺什么，不要用相邻内容凑答案。\n"
    "4. 用简体中文回答，先给结论再给依据，简洁。\n"
    "5. **必须分清“该主体的专门规定”和“通用要求”，不得把通用要求写成该主体的规定**：\n"
    "   - 若资料里**没有**提问主体（设备/位号/装置/介质/工序）的**专门规定**，\n"
    "     必须**先**明确说明“资料中没有<主体>的专门规定”。\n"
    "   - 随后若要列出通用要求，必须**显式标注**：\n"
    "     “以下为通用要求，**其适用性不等同于对<主体>的专门规定**”，\n"
    "     并且**不得**使用“结论：<主体> 应/须/需要……”这种把通用条款归给主体的句式。\n"
    "   - 只有资料里**确实有**该主体的专门规定时，才可以对它下结论。\n"
    "6. 对每个给出的通用要求，都要说明它是**一般性规定**，不是针对提问主体的专门要求。"
)

#: 提示词注册表（评测脚本按名字选）。
SYSTEMS = {"system": SYSTEM, "system_v2": SYSTEM_V2}

#: 🔴 基线配置**不能沿用主配置的系统提示词**（K-41）。
#:
#: 主配置的 SYSTEM 说"只能依据【资料片段】回答"，
#: 而闭卷基线（B1a）**根本没有片段** ——
#: 实测下来模型每题都答"未找到相关规定。当前提供的资料片段中没有…"，
#: 于是这一组**测的不是"模型的参数化知识"，而是"模型在没资料时会怎么答"**。
#:
#: 拿这样一组数去和检索组比，差距会被**凭空放大** ——
#: 那不是"检索有用"，是"基线被自己的提示词废掉了"。
#: → 基线必须有自己的系统提示词，且**不许提引用编号**（没有编号可引）。
SYSTEM_NO_CITE = (
    "你是制造业工业安全规程问答助手。\n"
    "规则：\n"
    "1. 依据给到的资料回答问题；资料里没有的内容，就明确说明没有，不要编造条款号或数值。\n"
    "2. 用简体中文回答，先给结论再给依据，简洁。"
)

#: 闭卷基线（B1a）的系统提示词：**问的是模型自己知道什么**。
CLOSED_BOOK_SYSTEM = (
    "你是制造业工业安全规程问答助手。请根据你已有的知识回答问题。\n"
    "规则：\n"
    "1. 知道就回答；不知道就直接说不知道，不要编造条款号或数值。\n"
    "2. 用简体中文回答，先给结论再给依据，简洁。"
)

USER_TEMPLATE = """【问题】
{question}

【资料片段】
{context}

【要求】
按上面的编号引用来源。若片段里没有答案，明确说明未找到相关规定。"""

#: 无编号片段时的 prompt（B1b 全文基线）——
#: 🔴 不能沿用带"按上面的编号引用来源"的那版：
#: 全文基线给的资料**没有 [C1] 编号**，要求它按编号引用，
#: 模型要么编编号、要么干脆不答 —— 两种都污染基线（K-41）。
USER_TEMPLATE_NO_CITE = """【问题】
{question}

【资料片段】
{context}

【要求】
依据资料回答。若资料里没有答案，明确说明未找到相关规定，不要用相邻内容凑答案。"""


def build_prompt(question, context, numbered=True):
    """拼 user prompt。

    :param numbered: 片段是否带 ``[C1]`` 编号。
        闭卷基线（无片段）与全文基线（无编号）都传 ``False``。
    """
    if not context:
        return f"【问题】\n{question}\n\n【要求】\n" \
               "根据你已有的知识回答；若不确定就说不确定，不要编造条款号或数值。"
    tpl = USER_TEMPLATE if numbered else USER_TEMPLATE_NO_CITE
    return tpl.format(question=question, context=context)


# ---------------------------------------------------------------------------
# 回答器
# ---------------------------------------------------------------------------

class Answerer:
    """检索 + 生成 + 装配。

    :param chunks: chunk 列表
    :param embedder: 向量编码器（``None`` = 纯 BM25）
    :param manifest: ``{doc_id: 记录}``，版本卡要用
    :param use_vector: **是否走向量路**。

    🔴 **默认 ``False``**：M5 实测（`docs/检索基线与切分策略对比.md`）
    本机没有可用的预训练句向量，向量路（TF-IDF+SVD）是**负贡献**
    （R@5 0.633–0.724 vs 纯 BM25 的 0.856–0.883），
    PRD 4.6 的 **L4 降级（纯 BM25）就是本项目的实际配置**。
    传了 ``embedder`` 不等于要用向量路 —— 要用得显式 ``use_vector=True``。
    """

    def __init__(self, chunks, embedder=None, manifest=None, top_k=5,
                 alpha=0.5, use_vector=False, system=None,
                 attribution_fallback=False):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.use_vector = use_vector
        self.manifest = manifest or VC.load_manifest()
        self.top_k = top_k
        self.alpha = alpha
        #: 系统提示词（`None` = 默认的 `SYSTEM`；`SYSTEM_V2` = 适用性标注版）
        self.system = system or SYSTEM
        #: 🔴 是否启用**代码级归因兜底**（2026-11-05）。
        #: 生产配置应为 ``True``；评测时要能关掉它，
        #: 才能区分"模型自己做到的"与"代码兜底补上的"。
        self.attribution_fallback = attribution_fallback
        self.retriever = RT.Retriever(self.chunks, embedder=embedder, alpha=alpha)

    # -- 检索 ---------------------------------------------------------------

    def retrieve(self, question, when=None, mode="current",
                 version_filter=True, top_k=None):
        """检索并**强制过滤过期 chunk**（PRD 5.5）。

        🔴 `cross_version` 口径下不过滤 —— 问"两版差异"时两版都要在场。
        """
        ranked = self.retriever.retrieve(
            question, top_k=top_k or self.top_k, when=when, mode=mode,
            version_filter=version_filter,
            use_vector=self.use_vector and self.embedder is not None,
            alpha=self.alpha)
        idxs = [i for i, _s in ranked]
        return [self.chunks[i] for i in idxs]

    # -- 生成 ---------------------------------------------------------------

    def answer(self, question, when=None, mode="current", version_filter=True,
               top_k=None, call=None, extra_context="", subject=None):
        """完整跑一遍：检索 → 上下文 → 生成 → 装配。

        :param call: ``call_llm(prompt, system=...)``，便于测试注入假模型
        :param subject: **提问主体**（如假位号" T-9041"、假装置"加氢反应器"）。
            🔴 只在**主体明确**的场景传（拒答题）——
            代码级归因兜底靠它判断"召回的 chunk 里有没有提到这个主体"。
            事实题/数值题没有单一主体 → 传 ``None``，兜底不启用。
        :returns: 字典，含 ``answer_text``（模型原始正文）、``rendered``（最终答案）、
            ``ctx_chunks``、``cite_stats``、``version_cards``、
            ``attribution_fallback``（代码追加的归因声明，没有则 ``None``）
        """
        if call is None:
            import llm_client
            call = llm_client.call_llm

        ctx = self.retrieve(question, when=when, mode=mode,
                            version_filter=version_filter, top_k=top_k)
        ctx_block = build_context_block(ctx, when=when, mode=mode)
        if extra_context:
            ctx_block = extra_context + "\n\n" + ctx_block
        prompt = build_prompt(question, ctx_block)
        raw = call(prompt, system=self.system)

        fb_text = None
        pollution = []
        raw_for_render = raw
        if self.attribution_fallback and subject:
            need, fb, pollution = attribution_fallback(raw, ctx, subject)
            if need:
                fb_text = fb
            if pollution:
                raw_for_render = mark_pollution(raw, pollution)

        rendered, stats = CIT.render(raw_for_render, ctx, manifest=self.manifest,
                                     when=when, mode=mode)
        # 🔴 代码级归因兜底（2026-11-05）：模型没自己说清楚时，代码强制追加
        if fb_text:
            rendered = rendered + "\n\n" + fb_text
        if pollution:
            rendered = rendered + "\n\n" + POLLUTION_NOTICE.format(
                n=len(pollution), subject=subject)
        cards = VC.build_cards(ctx, manifest=self.manifest, when=when, mode=mode)
        card_block = VC.render_cards(cards)
        final = rendered + ("\n\n" + card_block if card_block else "")
        return {
            "answer_text": raw,
            "rendered": final,
            "ctx_chunks": ctx,
            "ctx_docs": sorted({(c.get("doc_snapshot") or {}).get("doc_id")
                                for c in ctx}),
            "cite_stats": stats,
            "version_cards": cards,
            "attribution_fallback": fb_text,
            "pollution_sentences": [p[0] for p in pollution],
        }


def answer_no_retrieval(question, corpus_text="", call=None, max_chars=20000,
                        system=None):
    """**无检索基线**（PRD 7.4 的 B1）。

    :param corpus_text: 给模型看的语料子集（空 = 闭卷 B1a）
    :param max_chars: 语料子集的最大字符数（34 万字符超上下文，必须截断）
    :param system: 系统提示词；``None`` 时按有没有语料自动选
        （闭卷 → :data:`CLOSED_BOOK_SYSTEM`，有语料 → :data:`SYSTEM_NO_CITE`）
    :returns: 与 :meth:`Answerer.answer` 同构的字典（无引用、无版本卡）

    🔴 **系统提示词必须跟着配置走**（K-41）：
    沿用主配置的"只能依据【资料片段】回答"会让闭卷组每题都答
    "未找到相关规定" —— 那一组数测的就不是"模型自己知道什么"了。
    """
    if call is None:
        import llm_client
        call = llm_client.call_llm
    ctx_text = (corpus_text or "")[:max_chars]
    if system is None:
        system = CLOSED_BOOK_SYSTEM if not ctx_text.strip() else SYSTEM_NO_CITE
    prompt = build_prompt(question, ctx_text, numbered=False)
    raw = call(prompt, system=system)
    return {
        "answer_text": raw,
        "rendered": raw,
        "ctx_chunks": [],
        "ctx_docs": [],
        "cite_stats": {"n_refs": 0, "n_valid": 0, "n_invalid": 0,
                       "invalid_ids": [], "cited": []},
        "version_cards": [],
    }
