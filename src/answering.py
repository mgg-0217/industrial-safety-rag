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
import version_card as VC   # noqa: E402


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
                 alpha=0.5, use_vector=False, system=None):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.use_vector = use_vector
        self.manifest = manifest or VC.load_manifest()
        self.top_k = top_k
        self.alpha = alpha
        #: 系统提示词（`None` = 默认的 `SYSTEM`；`SYSTEM_V2` = 适用性标注版）
        self.system = system or SYSTEM
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
               top_k=None, call=None, extra_context=""):
        """完整跑一遍：检索 → 上下文 → 生成 → 装配。

        :param call: ``call_llm(prompt, system=...)``，便于测试注入假模型
        :returns: 字典，含 ``answer_text``（模型原始正文）、``rendered``（最终答案）、
            ``ctx_chunks``、``cite_stats``、``version_cards``
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

        rendered, stats = CIT.render(raw, ctx, manifest=self.manifest, when=when,
                                     mode=mode)
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
