# -*- coding: utf-8 -*-
"""检索层：版本过滤 → 两路召回 → RRF 融合（PRD 第 4 节）。

## 流程（PRD 4.1 的图）

    query ──过滤条件─┬─ 向量索引 ─┐
                      └─ BM25 索引 ─┴─ RRF 融合 ── Top-K

## 🔴 三条硬约束

1. **先过滤，再排序**（PRD 4.4）。
   版本区间过滤发生在**检索层**，不在生成层 ——
   让模型"自己别用过期条款"的服从率只有 60–70%，正确性不能押在这上面。
   BM25 的 IDF 与 avgdl 都按**过滤后的存活子集**重算（见 `bm25.py`）。

2. **过期 chunk 根本不进上下文**（PRD 5.5）。
   这里过滤掉，它就永远不会出现在候选集里 —— 不是"排序靠后"，是"不存在"。

3. **`status` 只是缓存，判据一律走 `effective_from ≤ t < effective_to`**
   （PRD 2.7）。本模块**不读** `status` 字段。

## 已知生效日未知的文档

`effective_from` 未知的文档，`is_effective()` 返回 **UNKNOWN**（三态，不是 False）。
- `current` 口径：**纳入**（并标注"生效日未标注"，PRD 2.6.1）
- `as_of` 口径：**排除**（无法判定，就不能说它当时有效）
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bm25 as BM  # noqa: E402
import versioning as V  # noqa: E402


# ---------------------------------------------------------------------------
# 版本过滤
# ---------------------------------------------------------------------------

def version_alive(chunks, when, mode="current"):
    """按版本区间算出**存活**的 chunk 下标。

    :param chunks: chunk 列表（每个带 ``doc_snapshot``）
    :param when: 判定时刻 ``YYYY-MM-DD``（PRD 5.1：显式传入，不读系统时间）
    :param mode: ``current`` / ``as_of`` / ``cross_version``（PRD 5.2 的口径）

    :returns: ``(alive_indices, unknown_indices)``
        - ``alive_indices``：可以进候选集的 chunk 下标
        - ``unknown_indices``：生效日未知的 chunk 下标（current 口径下也在 alive 里，
          单列出来是为了让版本卡能标注"生效日未标注"）

    🔴 **不能用字符串比较**：`"" <= "2026-11-02"` 恒真，
    会把生效日未知的文档悄悄当成"永远现行"。
    必须走 `versioning.is_effective()` 的三态判定。

    ### `cross_version` 口径（2026-11-03 补，PRD 5.2 同步 amend）

    问"**两版有什么差异**"时，两个版本都要在场 ——
    但 `current` 口径会把已废止的旧版**整个滤掉**，
    `as_of(t)` 口径则会把当时还没生效的新版滤掉。
    两种口径都答不了"比较两版"的问题。
    实测（clause 策略，跨文档题 25 道）：`current` 口径下召回只有 **0.24**。

    → `cross_version` 口径**不做版本过滤**，
    由**版本卡**负责标注哪一版是现行、哪一版是历史（PRD 5.4）。
    """
    alive = []
    unknown = []
    if mode == "cross_version":
        # 两版都要在场 → 不做版本过滤
        return list(range(len(chunks))), unknown
    for i, c in enumerate(chunks):
        doc = c.get("doc_snapshot") or {}
        st = V.is_effective(doc, when)
        if st is True:
            alive.append(i)
        elif st is None:
            unknown.append(i)
            if mode == "current":
                # PRD 2.6.1：current 口径纳入，版本卡标注"生效日未标注"
                alive.append(i)
            # as_of 口径：无法判定 → 排除
    return alive, unknown


# ---------------------------------------------------------------------------
# 内容覆盖度（PRD 7.3 的 `content_coverage`）
# ---------------------------------------------------------------------------

def interval_union_len(intervals):
    """一组区间并集的总长度。"""
    if not intervals:
        return 0
    xs = sorted(intervals)
    total = 0
    cur_s, cur_e = xs[0]
    for s, e in xs[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def content_coverage(chunks, chunk_ids, gold_start, gold_end):
    """上下文带进了 gold 区间的多少比例（PRD 7.3）。

    :param chunks: 参与检索的 chunk 列表
    :param chunk_ids: top-k 结果的 chunk_id 集合
    :param gold_start / gold_end: gold 区间（**原始 md 的字符偏移**）
    :returns: ``0.0 ~ 1.0``

    🔴 **为什么要有这个指标**：`Recall@k` 是**二值命中** ——
    top-k 里有一块碰到 gold 就算命中。但对表格类题，
    **命中一块碎片 ≠ 能答对**（模型要的是整张表）。
    实测：表格题的 `Recall@5` 有 0.750，而 `content_coverage` 只有 **0.390** ——
    召回到了，却只带进四成内容。

    🔴 **它比 `Recall@k` 多回答了一个问题**：
    "找没找到" vs "**带够了没有**"。
    而且 gold 区间取自**原始 md 的偏移**（不随切分策略变）
    → **跨策略可比**；`Recall@k` 的 gold 要按各策略的 chunk 偏移重定义，
    跨策略比大小不成立。
    """
    if gold_end is None or gold_start is None or gold_end <= gold_start:
        return 0.0
    idset = set(chunk_ids)
    ivs = []
    for c in chunks:
        if c.get("chunk_id") not in idset:
            continue
        a, b = c.get("char_start"), c.get("char_end")
        if a is None or b is None:
            continue
        s, e = max(a, gold_start), min(b, gold_end)
        if e > s:
            ivs.append((s, e))
    if not ivs:
        return 0.0
    return interval_union_len(ivs) / (gold_end - gold_start)


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------

class Retriever:
    """检索器。

    :param chunks: chunk 列表（``data/chunks/<strategy>.jsonl`` 的行）
    :param embedder: 向量编码器；``None`` 表示纯 BM25（PRD 4.6 的 L4 降级）
    """

    def __init__(self, chunks, embedder=None, alpha=0.5, rrf_k=BM.RRF_K,
                 mode="bigram"):
        self.chunks = list(chunks)
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.bm25 = BM.BM25Index([c.get("norm_text") or c.get("text") or ""
                                  for c in self.chunks], mode=mode)
        # partial 文档的 chunk 下标（PRD 2.8 的降权对象）
        self.partial_idx = {i for i, c in enumerate(self.chunks)
                            if c.get("quality") == "partial"}
        self.embedder = embedder
        self.emb = None
        if embedder is not None:
            self.emb = embedder.encode(
                [c.get("norm_text") or c.get("text") or "" for c in self.chunks])

    # -- 向量路 -------------------------------------------------------------

    def vector_search(self, query, alive=None, top_k=10):
        """向量路检索。**先掩码后点积**（PRD 4.4 的实现细节 1）。

        🔴 `effective_to` 为 null 表示"至今未失效"，不能直接参与比较。
        本模块用**掩码**（布尔数组）表达，不做字符串比较，天然没有这个坑。
        """
        if self.embedder is None or self.emb is None:
            return []
        import numpy as np
        q = self.embedder.encode([query])[0]
        mat = np.asarray(self.emb, dtype="float32")
        qn = np.linalg.norm(q) or 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        sims = (mat @ q) / (norms * qn)
        if alive is not None:
            mask = np.zeros(len(self.chunks), dtype=bool)
            mask[list(alive)] = True
            sims = np.where(mask, sims, -2.0)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > -1.5]

    # -- 主流程 -------------------------------------------------------------

    def retrieve(self, query, top_k=10, when=None, mode="current",
                 version_filter=True, use_vector=True, alpha=None,
                 candidate_k=None, use_bm25=True):
        """检索。

        :param query: 查询串（自然语言问题）
        :param top_k: 返回条数
        :param when: 判定时刻 ``YYYY-MM-DD``；``None`` 表示不做版本过滤
        :param mode: 时间口径（``current`` / ``as_of``）
        :param version_filter: 是否做版本过滤（**B2 关掉它，B3 打开**）
        :param alpha: partial 降权系数；``None`` 用实例默认值
        :param use_bm25: 是否走 BM25 路（关掉它就是"仅向量路"的对照组）
        :returns: ``[(chunk_idx, score), ...]``，已按最终分数降序
        """
        alpha = self.alpha if alpha is None else alpha
        alive = None
        if version_filter and when:
            alive, _unknown = version_alive(self.chunks, when, mode)
            if not alive:
                return []
        # 每路多取一些，RRF 融合后再截断（超量取候选，PRD 2.8 的闸需要）
        ck = candidate_k or max(top_k * 4, 50)

        routes = []
        if use_bm25:
            routes.append(self.bm25.search(query, alive=alive, top_k=ck,
                                           alpha=alpha, partial=self.partial_idx))
        if use_vector and self.embedder is not None:
            routes.append(self.vector_search(query, alive=alive, top_k=ck))
        if not routes:
            return []
        fused = BM.rrf_fuse(routes, k=self.rrf_k, top_k=top_k)
        return fused

    # -- 诊断 ---------------------------------------------------------------

    def stats(self):
        s = self.bm25.stats()
        s["n_partial_chunks"] = len(self.partial_idx)
        s["embed_backend"] = ("L4_pure_bm25" if self.embedder is None
                              else self.embedder.name)
        s["alpha"] = self.alpha
        s["rrf_k"] = self.rrf_k
        return s


def load_chunks(path):
    """读 ``data/chunks/<strategy>.jsonl``。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
