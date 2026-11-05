# -*- coding: utf-8 -*-
"""检索层单元测试（PRD M5 验收要求）。

M5 的验收标准（PRD 第 9 节）原文：

    「BM25（自研）与向量路两路可跑；RRF 融合可用；
      **先过滤再排序在两路都生效**（单元测试：注入一条过期 chunk，断言它不在结果里）」

所以这里最要紧的是第 3 节：**注入一条过期 chunk，断言它不在结果里**。
断言必须落在**两路都测**（BM25 路 + 向量路）+ **融合后**，
只测融合后的结果是"看起来对"—— 过期 chunk 可能只是被别的 chunk 挤掉了，
根本没经过过滤。**每条断言都配一条反向对照**（关掉过滤它就该出现），
否则测试可能是空的（vacuous）。

跑法：
    python -m pytest tools/test_retrieval.py -v
    python tools/test_retrieval.py          # 不装 pytest 也能跑
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import bm25 as BM          # noqa: E402
import retrieval as RT     # noqa: E402
import versioning as V     # noqa: E402

import numpy as np         # noqa: E402


# ============================================================ 测试夹具

def _chunk(doc_id, text, eff_from, eff_to, quality="ok", eff_to_source=None):
    """造一条最小可用的 chunk（字段对齐 ``data/chunks/*.jsonl``）。

    🔴 `effective_to_source` 必须带上：它区分
    「至今未失效（open）」与「真的不知道（unknown）」——
    少了它，`effective_to=None` 会被判成 UNKNOWN（K-32）。
    """
    if eff_to_source is None:
        eff_to_source = "explicit" if eff_to else "open"
    return {
        "chunk_id": f"{doc_id}::c0",
        "doc_id": doc_id,
        "strategy": "test",
        "text": text,
        "norm_text": text,
        "char_start": 0,
        "char_end": len(text),
        "page_no": 1,
        "quality": quality,
        "strategy_meta": {},
        "doc_snapshot": {
            "doc_id": doc_id,
            "title": doc_id,
            "effective_from": eff_from,
            "effective_to": eff_to,
            "effective_to_source": eff_to_source,
            "superseded_by": None,
            "status": "有效",
        },
    }


#: 三份文档：现行版 / 已废止版 / 生效日未知
CHUNKS = [
    _chunk("doc_old", "动火作业前应进行气体分析并办理动火作业票", "2023-10-06",
           "2025-01-09"),
    _chunk("doc_new", "动火作业前应进行气体分析并办理动火作业票", "2025-01-09",
           None),
    _chunk("doc_unknown", "动火作业前应进行气体分析并办理动火作业票", "", None),
]

WHEN = "2026-11-03"          # 判定时刻（PRD 5.1：显式传入，不读系统时间）
QUERY = "动火作业 气体分析 作业票"


class FakeEmbedder:
    """确定性的假编码器（不依赖模型下载）。

    🔴 文本相同 → 向量相同 → **两路的候选分数完全一样**。
    这样"过期 chunk 有没有被召回"就只可能由**过滤掩码**决定，
    测试才能精确指向"先过滤再排序"这件事本身。
    """

    name = "fake_hash"

    def __init__(self, dim=64):
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for i, ch in enumerate(t or ""):
                v[(ord(ch) * 31 + i) % self.dim] += 1.0
            n = float(np.linalg.norm(v)) or 1.0
            out.append(v / n)
        return np.asarray(out, dtype="float32")


# ============================================================ 1. 版本判定三态

def test_version_alive_three_states():
    """`is_effective` 是**三态**：True / False / **UNKNOWN**（不是布尔）。

    🔴 空串比较是本项目最大的坑之一："" <= "2026-11-03" 恒真，
    会把"生效日未标注"的文档悄悄当成"永远现行"。
    """
    docs = {c["doc_id"]: c["doc_snapshot"] for c in CHUNKS}
    assert V.is_effective(docs["doc_new"], WHEN) is True
    assert V.is_effective(docs["doc_old"], WHEN) is False
    assert V.is_effective(docs["doc_unknown"], WHEN) is None, \
        "生效日未知必须返回 UNKNOWN（None），不能是 True"


def test_version_alive_modes():
    """`current` 纳入生效日未知的文档，`as_of` 排除它（PRD 2.6.1）。

    ⚠️ 已废止的 doc_old 在**两种口径下都不该存活** ——
    `current` 口径要的是"现在还生效"，旧版不在其中。
    """
    alive, unknown = RT.version_alive(CHUNKS, WHEN, mode="current")
    assert set(alive) == {1, 2}, alive
    assert set(unknown) == {2}, unknown

    alive, unknown = RT.version_alive(CHUNKS, WHEN, mode="as_of")
    assert set(alive) == {1}, alive
    assert set(unknown) == {2}, unknown


def test_cross_version_mode_does_not_filter():
    """`cross_version` 口径：两版都要在场 → **不做版本过滤**。

    问"两版有何差异"时，`current` 会把旧版滤掉、`as_of` 会把新版滤掉，
    两种口径都答不了这个问题（PRD 5.2）。
    """
    alive, _ = RT.version_alive(CHUNKS, WHEN, mode="cross_version")
    assert set(alive) == {0, 1, 2}


def test_unknown_effective_date_is_not_infinite_valid():
    """回归：未知生效日的文档在 `as_of` 口径下**必须**被排除。

    （历史 bug：字符串比较让 "" <= t 恒真 → 永远现行）
    """
    alive, _ = RT.version_alive(CHUNKS, WHEN, mode="as_of")
    assert 2 not in alive


# ============================================================ 2. BM25：先过滤再排序

def test_bm25_stats_recomputed_on_filtered_subset():
    """IDF 与 avgdl 必须按**存活子集**重算（PRD 4.2 / 4.4）。

    反向对照：全库统计量与存活子集统计量必须不同，
    否则"重算"这件事根本没发生。
    """
    texts = ["短", "中等长度的文本内容", "很长很长很长很长很长很长很长很长很长很长很长很长"]
    idx = BM.BM25Index(texts)
    n_all, avg_all = idx.stats_on(None)
    n_alive, avg_alive = idx.stats_on([1])
    assert (n_all, n_alive) == (3, 1)
    assert avg_alive != avg_all, "存活子集的 avgdl 必须重算，不能用全库平均"
    # avgdl 按存活集算：只剩 1 条时 avgdl == 那条的长度
    assert abs(avg_alive - len(texts[1])) < 1e-6


def test_bm25_expired_chunk_absent_with_reverse_control():
    """🔴 核心断言：过期 chunk 不在 BM25 路的结果里。

    反向对照：**关掉过滤它就该出现** ——
    否则这条断言可能是"因为它本来就没被召回"而通过的空断言。
    """
    idx = BM.BM25Index([c["norm_text"] for c in CHUNKS])
    alive, _ = RT.version_alive(CHUNKS, WHEN, mode="current")
    alive = [i for i in alive if CHUNKS[i]["doc_id"] != "doc_unknown"]

    filtered = [i for i, _s in idx.search(QUERY, alive=alive, top_k=10)]
    unfiltered = [i for i, _s in idx.search(QUERY, alive=None, top_k=10)]

    assert 0 not in filtered, "已废止版本（doc_old）不该出现在过滤后的结果里"
    assert 0 in unfiltered, "反向对照失败：不过滤时 doc_old 应该被召回"


def test_bm25_filtered_subset_changes_ranking():
    """过滤会**改变排序**，不只是"少几条"。

    这正是"先过滤再排序"与"先排序再过滤"的区别：
    后者是在全局分数里筛，IDF 与长度归一化都混着已废止文档的统计量。
    """
    idx = BM.BM25Index([c["norm_text"] for c in CHUNKS])
    alive = [1]
    f = idx.search(QUERY, alive=alive, top_k=10)
    u = idx.search(QUERY, alive=None, top_k=10)
    assert len(f) == 1 and f[0][0] == 1
    assert len(u) == 3


# ============================================================ 3. 向量路：掩码而非比较

def test_vector_route_masks_expired_chunk():
    """向量路用**掩码**表达过滤（PRD 4.4 的实现细节 1）。

    🔴 这里不能写 `sims[i] = -1 if not effective else sims[i]` 这种**字符串比较**，
    因为 `effective_to` 为 null 表示"至今未失效"，不能直接参与比较。
    """
    emb = FakeEmbedder()
    ret = RT.Retriever(CHUNKS, embedder=emb)
    alive, _ = RT.version_alive(CHUNKS, WHEN, mode="current")
    alive = [i for i in alive if CHUNKS[i]["doc_id"] != "doc_unknown"]

    vec = ret.vector_search(QUERY, alive=alive, top_k=10)
    assert vec, "向量路应至少返回一条"
    assert all(i in alive for i, _s in vec), \
        f"向量路返回了被过滤掉的 chunk：{[i for i, _s in vec if i not in alive]}"
    assert 0 not in [i for i, _s in vec]

    vec_all = ret.vector_search(QUERY, alive=None, top_k=10)
    assert 0 in [i for i, _s in vec_all], "反向对照失败：不过滤时应能召回 doc_old"


def test_fused_result_excludes_expired_chunk():
    """**融合后**也要断言（PRD 5.5：过期 chunk 根本不进上下文）。

    ⚠️ 当前口径下"生效日未知的文档"**是要进结果**的（PRD 2.6.1），
    所以这里断言的是"**已废止**的那条不在"，不是"不在 alive 集合的都出局"。
    """
    emb = FakeEmbedder()
    ret = RT.Retriever(CHUNKS, embedder=emb)

    fused = ret.retrieve(QUERY, top_k=10, when=WHEN, mode="current",
                         version_filter=True, use_vector=True)
    assert fused
    assert 0 not in [i for i, _s in fused], "已废止版本的 chunk 混进了融合结果"
    assert 1 in [i for i, _s in fused], "现行版本的 chunk 应该在结果里"
    assert 2 in [i for i, _s in fused], \
        "current 口径下生效日未知的文档要纳入（PRD 2.6.1）"

    # 反向对照：关掉过滤，过期 chunk 就会回来
    fused_nofilter = ret.retrieve(QUERY, top_k=10, when=WHEN, mode="current",
                                  version_filter=False, use_vector=True)
    assert 0 in [i for i, _s in fused_nofilter]


def test_no_alive_chunk_returns_empty():
    """全库都被过滤掉时，返回空列表而不是报错或回退到全库。"""
    emb = FakeEmbedder()
    ret = RT.Retriever(CHUNKS, embedder=emb)
    r = ret.retrieve(QUERY, top_k=10, when="1990-01-01", mode="as_of")
    assert r == []


# ============================================================ 4. RRF 与 α

def test_rrf_uses_rank_not_score():
    """RRF **只用排名，不用分数**（PRD 4.3）。

    两路的分数不可比（BM25 无界正值，余弦 [-1,1]），
    所以"分数大的一方赢"是错的，"排名靠前的一方赢"才对。
    """
    a = [(10, 999.0), (20, 0.001)]
    b = [(20, 999.0), (30, 0.001)]
    f1 = BM.rrf_fuse([a, b], k=60, top_k=5)
    assert [i for i, _s in f1] == [20, 10, 30], f1
    # 把两路的分数**整体取倒数**（排名不变）→ 融合结果必须一模一样
    a2 = [(i, 1.0 / s) for i, s in a]
    b2 = [(i, 1.0 / s) for i, s in b]
    f2 = BM.rrf_fuse([a2, b2], k=60, top_k=5)
    assert [i for i, _s in f2] == [i for i, _s in f1], \
        "分数变了而名次没变，融合结果就不该变"


def test_alpha_downweight_applies_within_route():
    """🔴 α（partial 降权）必须作用在**本路排序之前**。

    PRD 4.4 / 2.8 的红字警告：放到 RRF 融合**之后**，
    对 BM25 路**完全失效**（RRF 只看排名）。
    """
    chunks = [
        _chunk("d_ok", "动火作业气体分析作业票", "2025-01-09", None, "ok"),
        _chunk("d_partial", "动火作业气体分析作业票", "2025-01-09", None,
               "partial"),
    ]
    idx = BM.BM25Index([c["norm_text"] for c in chunks])
    partial = {1}
    r1 = idx.search(QUERY, alive=None, top_k=5, alpha=1.0, partial=partial)
    r05 = idx.search(QUERY, alive=None, top_k=5, alpha=0.5, partial=partial)
    assert r1[0][0] == r05[0][0] == 0        # 不降权时靠前的还是它
    # 分数层面：partial 那条被乘了 α
    s1 = dict(r1)[1]
    s05 = dict(r05)[1]
    assert abs(s05 - s1 * 0.5) < 1e-9, (s1, s05)
    # 排序层面：α 足够小时 partial 被压到后面
    r02 = idx.search(QUERY, alive=None, top_k=5, alpha=0.2, partial=partial)
    assert [i for i, _s in r02][0] == 0


def test_alpha_after_fusion_would_be_useless():
    """反向验证上面那条坑是真的：**融合后再乘 α 改不了名次**。

    这条测试是"为什么必须作用在本路内"的证据，
    不是为了测某个函数（RRF 本来就没有分数可乘）。
    """
    ranking = [(0, 1.0), (1, 0.9)]
    fused = BM.rrf_fuse([ranking], k=60, top_k=5)
    order_before = [i for i, _s in fused]
    scaled = [(i, s * 0.1) for i, s in fused]
    order_after = [i for i, _s in sorted(scaled, key=lambda kv: -kv[1])]
    assert order_before == order_after, \
        "如果这一条不成立，说明 RRF 用了分数——那 α 放融合后就是有效的"


# ============================================================ 5. 分词

def test_tokenizer_keeps_ascii_intact():
    """位号必须**整串保留**：拆成 bigram 就丢精确性（PRD 4.1/4.2）。"""
    toks = BM.tokenize("设备位号 19-V-4215 与标准 GB 30871-2022")
    assert "19-v-4215" in toks
    assert "gb" in toks and "30871-2022" in toks
    assert "19" not in toks and "4215" not in toks


def test_tokenizer_cjk_bigram():
    """中文按 character bigram 切（PRD 4.2 的默认口径）。"""
    toks = BM.tokenize("压力容器")
    assert toks == ["压力", "力容", "容器"]


def test_empty_query_returns_empty():
    idx = BM.BM25Index([c["norm_text"] for c in CHUNKS])
    assert idx.search("", alive=None, top_k=5) == []
    assert idx.search("   ", alive=None, top_k=5) == []


# ============================================================ 6. 端到端冒烟

def test_retriever_stats():
    emb = FakeEmbedder()
    ret = RT.Retriever(CHUNKS, embedder=emb)
    s = ret.stats()
    assert s["n_chunks"] == 3
    assert s["embed_backend"] == "fake_hash"
    assert s["alpha"] == 0.5
    assert s["rrf_k"] == BM.RRF_K


# ============================================================ 7. 版本过滤的按文档缓存

# 🔴 这一组测的是 2026-11-04 的优化：版本过滤从 **O(块数)** 降到 **O(文档数)**。
#    延迟实测发现 `clause` 策略下 `version_alive` 占端到端检索延迟的 66%
#    —— 比排序本身还贵。根因是它**逐块**判版本区间，
#    而库里只有 26 份文档、同一文档的所有 chunk 共享完全相同的 `doc_snapshot`。
#
#    优化**不能改变判定结果** —— 所以每条断言都是"缓存路径 == 非缓存路径"。

def test_version_alive_cache_matches_nocache():
    """缓存路径与非缓存路径必须给出**完全相同**的存活集。"""
    for when, mode in [("2026-11-02", "current"), ("2026-11-02", "as_of"),
                        ("2023-12-31", "as_of"), ("2026-11-02", "cross_version")]:
        a1, u1 = RT.version_alive(CHUNKS, when, mode)
        cache = {}
        a2, u2 = RT.version_alive(CHUNKS, when, mode, cache=cache)
        a3, u3 = RT.version_alive(CHUNKS, when, mode, cache=cache)  # 再走一次缓存
        assert a1 == a2 == a3, f"{when}/{mode} 的存活集在缓存前后不一致"
        assert u1 == u2 == u3, f"{when}/{mode} 的 unknown 集在缓存前后不一致"


def test_version_alive_cache_key_isolated():
    """不同 (when, mode) 不能串味 —— 缓存键必须把两者都算进去。"""
    cache = {}
    a_cur, _ = RT.version_alive(CHUNKS, "2026-11-02", "current", cache=cache)
    a_asof, _ = RT.version_alive(CHUNKS, "2026-11-02", "as_of", cache=cache)
    a_old, _ = RT.version_alive(CHUNKS, "2023-12-31", "as_of", cache=cache)
    # as_of(2026) 与 as_of(2023) 的存活集**必须不同**（有文档在这期间生效）
    assert a_asof != a_old, "不同 as_of 时点给出了相同的存活集 → 缓存串味了"
    # current 与 as_of(同一时刻) 在本夹具下可以不同（未标注生效日的文档）
    assert len(a_cur) >= len(a_asof)


def test_retriever_reuses_version_cache():
    """`Retriever` 的跨查询缓存：同一 (when, mode) 第二次不再重算。"""
    ret = RT.Retriever(CHUNKS, embedder=None)
    assert len(ret._ver_cache) == 0
    ret.retrieve("动火作业", top_k=2, when="2026-11-02", mode="current",
                 use_vector=False)
    assert len(ret._ver_cache) == 1, "第一次查询后应该有缓存条目"
    cached = ret._ver_cache[("2026-11-02", "current")]
    ret.retrieve("受限空间作业", top_k=2, when="2026-11-02", mode="current",
                 use_vector=False)
    # 缓存对象应当是**同一个**（没有重新算）
    assert ret._ver_cache[("2026-11-02", "current")] is cached
    # current 口径纳入「生效日未知」（is_effective 返回 UNKNOWN）的块（PRD 2.6.1），
    # 只排除明确失效的 —— 所以判据是"不是 False"，不是"是 True"。
    assert cached[0] == [i for i, c in enumerate(CHUNKS)
                         if V.is_effective(c.get("doc_snapshot") or {},
                                           "2026-11-02") is not False]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} 条通过")
    sys.exit(0 if passed == len(fns) else 1)
