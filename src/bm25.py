# -*- coding: utf-8 -*-
"""自研 Okapi BM25（PRD 4.2）+ 中文字符 bigram 分词。

**不用 rank_bm25 之类的第三方库**：本项目要能解释每一个数字，
而且 PRD 4.2 要求"在**过滤后的子集上**重算 IDF 与 avgdl"——
主流库的 `get_scores()` 都是按全局统计量算的，不支持这个语义。

## 公式（PRD 4.2 写死）

    score(q, d) = Σ_{t∈q} IDF(t) · ( f(t,d) · (k1+1) )
                  / ( f(t,d) + k1 · (1 - b + b · |d|/avgdl) )

    IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

参数 `k1 = 1.5`、`b = 0.75`（Robertson & Zaragoza 2009 的常用默认值）。

## 🔴 本模块最重要的一条：过滤先于排序

PRD 4.4 的硬约束：**先按版本区间过滤候选集，再在过滤后的子集上排序**。
这意味着 IDF 与 avgdl 都必须按**存活 chunk** 重算：

- 不能先算全局 BM25 分数再过滤 —— 分数里混着已过期文档的 IDF 与长度归一化，
  不可比；而且全局 top-K 里可能有一半是过期文档，过滤后剩下的不足 K 条。
- `avgdl` 同样要按存活 chunk 算 —— 否则长度归一化用的是全库平均长度。

代价是每次查询都要重算，不能复用预计算的全局分数。
语料只有数千 chunk，这个代价可接受。
"""
import math
import re
from collections import Counter, defaultdict

#: 参数（PRD 4.2 写死）
K1 = 1.5
B = 0.75

#: RRF 的 k（PRD 4.3，论文原值 60）
RRF_K = 60

# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

#: 一个 token 要么是 **ASCII 串**（数字/字母/位号/标准号），要么是 **CJK 串**。
#:
#: 🔴 **ASCII 串必须整串保留**（PRD 4.1 说 BM25 擅长精确串）：
#: 若也按 bigram 切，"19-V-4215" 会变成 "19" / "-v" / "-4" / "42" / "15"，
#: 位号的精确性就没了 —— 而位号正是这个项目最需要精确召回的东西。
#:
#: CJK 串按 bigram 切（PRD 4.2）：「粗产品缓冲罐」→「粗产」「产品」「品缓」「缓冲」「冲罐」。
_TOKEN_RE = re.compile(
    r"[0-9A-Za-z][0-9A-Za-z._%+\-]*"          # ASCII 串：位号/标准号/数值
    r"|[㐀-䶿一-鿿]+"                    # CJK 串
)


def tokenize(text, mode="bigram"):
    """把文本切成检索用的 token 列表。

    :param mode: ``"bigram"``（默认，PRD 4.2）/ ``"jieba"``（对照实验项，
        需要用户自己 ``pip install jieba``）/ ``"unigram"``（对照用）。
    """
    if not text:
        return []
    if mode == "jieba":
        return _tokenize_jieba(text)
    out = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        if s[0].isascii():
            out.append(s.lower())
            continue
        if len(s) == 1:
            out.append(s)
            continue
        if mode == "unigram":
            out.extend(s)
            continue
        for i in range(len(s) - 1):
            out.append(s[i:i + 2])
    return out


def _tokenize_jieba(text):
    """jieba 分词（PRD 4.2 的对照实验项）。未装则明确报错，不静默降级。"""
    try:
        import jieba
    except ImportError as e:  # pragma: no cover - 依赖可选
        raise ImportError(
            "jieba 未安装。要跑分词对比实验，请先：pip install jieba"
        ) from e
    out = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w:
            continue
        if w[0].isascii():
            out.append(w.lower())
        else:
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25 倒排索引（纯 dict，语料规模小到不需要搜索引擎）。

    :param texts: 每个 chunk 的**检索用文本**（用 ``norm_text``，不是原文）
    :param k1: 饱和参数（PRD 4.2 = 1.5）
    :param b: 长度归一化强度（PRD 4.2 = 0.75）
    :param mode: 分词模式
    """

    def __init__(self, texts, k1=K1, b=B, mode="bigram"):
        self.k1 = k1
        self.b = b
        self.mode = mode
        self.texts = list(texts)
        self.N = len(self.texts)
        self.doc_len = [len(t) for t in self.texts]
        self.postings = defaultdict(list)     # term -> [(chunk_idx, tf)]
        self.term_count = 0
        for i, t in enumerate(self.texts):
            toks = tokenize(t, mode)
            self.term_count += len(toks)
            for term, f in Counter(toks).items():
                self.postings[term].append((i, f))
        # 全文统计量（不做版本过滤时用）
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

    def __len__(self):
        return self.N

    # -- 统计量（按存活子集重算） ------------------------------------------

    def stats_on(self, alive):
        """存活子集上的 ``(N, avgdl)``。

        🔴 这是"先过滤再排序"的关键：**IDF 与长度归一化都要按存活集算**，
        不能用全库统计量。
        """
        if alive is None:
            return self.N, self.avgdl
        n = len(alive)
        if n == 0:
            return 0, 0.0
        return n, sum(self.doc_len[i] for i in alive) / n

    # -- 检索 ---------------------------------------------------------------

    def search(self, query, alive=None, top_k=10, alpha=1.0, partial=None,
               min_score=0.0):
        """检索。

        :param alive: 存活 chunk 下标的集合；``None`` 表示不过滤（全部存活）。
        :param top_k: 返回条数
        :param alpha: ``partial`` 文档的降权系数（PRD 2.8，默认 0.5）。
            🔴 降权作用在**本路的排序之前** —— RRF 只用排名不用分数，
            放到融合之后就对这一路完全失效（PRD 4.4 / 2.8 的红字警告）。
        :param partial: ``partial`` 文档的 chunk 下标集合
        :returns: ``[(chunk_idx, score), ...]``，已按分数降序
        """
        qterms = tokenize(query, self.mode)
        if not qterms:
            return []
        alive_set = set(alive) if alive is not None else None
        N, avgdl = self.stats_on(alive_set)
        if N == 0 or avgdl <= 0:
            return []

        scores = defaultdict(float)
        for term in set(qterms):
            postings = self.postings.get(term)
            if not postings:
                continue
            if alive_set is None:
                surv = postings
            else:
                surv = [(i, f) for i, f in postings if i in alive_set]
            df = len(surv)
            if df == 0:
                continue
            # IDF 按**存活集**算：N 是存活数，df 是存活集里的文档频率
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue
            for i, f in surv:
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        # 🔴 partial 降权：**在本路排序之前**乘 α（PRD 2.8 的关键一步）
        if partial and alpha != 1.0:
            for i in list(scores):
                if i in partial:
                    scores[i] *= alpha

        if min_score > 0:
            scores = {i: s for i, s in scores.items() if s >= min_score}
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # -- 诊断 ---------------------------------------------------------------

    def stats(self):
        return {
            "n_chunks": self.N,
            "n_terms": len(self.postings),
            "avgdl": round(self.avgdl, 1),
            "k1": self.k1,
            "b": self.b,
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# RRF 融合（PRD 4.3）
# ---------------------------------------------------------------------------

def rrf_fuse(rankings, k=RRF_K, top_k=10, weights=None):
    """Reciprocal Rank Fusion。

        RRF_score(d) = Σ_{r ∈ rankers} w_r / (k + rank_r(d))

    :param rankings: 每一路的结果 ``[(chunk_idx, score), ...]``（已按分数降序）
    :param k: RRF 常数（PRD 4.3 = 60，论文原值）
    :param weights: 每一路的权重（默认全 1）
    :returns: ``[(chunk_idx, rrf_score), ...]``，按 RRF 分降序

    🔴 **只用排名，不用分数** —— 两路的分数不可比（BM25 无界正值，余弦 [-1,1]）。
    这也正是"partial 降权必须作用在本路排序上"的原因。
    """
    acc = defaultdict(float)
    for ri, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[ri]
        for rank, (idx, _score) in enumerate(ranking, start=1):
            acc[idx] += w / (k + rank)
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
