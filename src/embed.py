# -*- coding: utf-8 -*-
"""嵌入模型的后端选择 —— PRD 4.6 的四级降级。

| 级别 | 方案 | 本机状态（2026-11-03 实测） |
|---|---|---|
| **L1** | 本地 `bge-small-zh-v1.5`（via hf-mirror） | ❌ `sentence-transformers` 未安装 |
| **L2** | 降维（本实现 = TF-IDF + SVD，**非预训练语义模型**） | ✅ 可用，零下载 |
| **L3** | ModelScope 拉模型 | 未走到（本机无 modelscope 客户端） |
| **L4** | **纯 BM25，无向量路** | ✅ 恒可用 |

## 🔴 评测落盘必须记 `embed_backend`

跨次评测的数字只有**同一后端**才可比（PRD 4.6）。
这是很实际的坑：L2 的 TF-IDF+SVD 与 L1 的句向量是两个完全不同的空间，
混在一起报"向量路召回率"就是自欺。

## L2 的诚实说明

L2 用**字符 n-gram 的 TF-IDF + 截断 SVD**（Latent Semantic Analysis）做稠密表示。
它**不是**预训练语义模型：
- 它的"语义"来自**词项共现**，不是来自语言学知识；
- 它与 BM25 用的是**同一批词项**，只是权重与空间不同 ——
  所以两路的互补性远达不到 PRD 4.1 说的"向量擅长语义近似"。

**用途**：在没有预训练模型可用的环境里，把"两路 + RRF"这条链路**跑通并留下证据**，
以及给 M8 的降级验证提供"有一路近似向量"与"完全没有向量路"两个可比端点。
**不能**用它替代 L1 去论证"语义检索有效"。
"""
import os
import sys


class BaseEmbedder:
    """编码器接口。"""

    name = "base"

    def encode(self, texts):
        raise NotImplementedError


class L2SvdEmbedder(BaseEmbedder):
    """TF-IDF（字符 2-3 gram）+ 截断 SVD 的稠密表示。

    🔴 **这不是预训练语义模型** —— 见模块说明。
    它的作用是把"向量路"这条链路跑通，而不是论证语义检索的有效性。
    """

    name = "L2_tfidf_svd"

    def __init__(self, corpus_texts, dim=256, min_df=2):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                   min_df=min_df, sublinear_tf=True)
        X = self.vec.fit_transform(list(corpus_texts))
        k = int(min(dim, max(2, X.shape[1] - 1)))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.svd.fit(X)
        self.dim = k

    def encode(self, texts):
        X = self.vec.transform(list(texts))
        return self.svd.transform(X)


class L1SentenceTransformer(BaseEmbedder):
    """sentence-transformers 句向量（默认 bge-small-zh-v1.5，走 hf-mirror）。"""

    def __init__(self, model, name="L1_bge_small_zh"):
        self.model = model
        self.name = name

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)


def try_l1(model_name="BAAI/bge-small-zh-v1.5"):
    """尝试 L1。返回 ``(embedder, 说明)``；不可用时 embedder 为 None。

    环境事实（PRD 4.6 实测）：`huggingface.co` 8 秒超时、无本地缓存；
    `hf-mirror.com` 可用。所以默认把 ``HF_ENDPOINT`` 指到 hf-mirror。

    🔴 **顺序坑（2026-11-06 实测）**：
    `huggingface_hub.constants.ENDPOINT` 是在**导入时**从环境变量算出来的，
    所以 `HF_ENDPOINT` 必须在 **import sentence_transformers 之前**设好 ——
    原写法（先 import 再 setdefault）设置得太晚，
    `huggingface_hub` 早就把 endpoint 固化成了 `huggingface.co`
    → 本机 8 秒超时 → **模型根本下不下来**。

    🔴 **Xet 坑（2026-11-06 实测）**：
    `model.safetensors` 的下载会被**重定向到 `cas-bridge.xethub.hf.co`**，
    那个主机在本机**不可达**（`Read timed out`）→ 必须
    `HF_HUB_DISABLE_XET=1` 强制走传统 HTTP 路径。
    两个变量都要在 **import 之前**设好。
    """
    # 🔴 必须**先**设环境变量，再 import（见上面的说明）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None, "sentence-transformers 未安装（需用户自行 pip install）"
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        return L1SentenceTransformer(m, f"L1_{model_name.split('/')[-1]}"), \
            f"加载成功：{model_name}"
    except Exception as e:  # noqa: BLE001
        return None, f"加载失败：{type(e).__name__}: {e}"


def build_embedder(levels=("L1", "L2"), corpus_texts=None, verbose=True):
    """按顺序尝试各级，返回 ``(embedder, backend_name, 说明)``。

    embedder 为 None 表示 **L4：纯 BM25，无向量路**。
    """
    for lv in levels:
        if lv == "L1":
            emb, msg = try_l1()
        elif lv == "L2":
            if corpus_texts is None:
                continue
            try:
                emb = L2SvdEmbedder(corpus_texts)
                msg = f"TF-IDF(char 2-3gram) + SVD({emb.dim} 维)，无下载"
            except Exception as e:  # noqa: BLE001
                emb, msg = None, f"构建失败：{type(e).__name__}: {e}"
        else:
            continue
        if emb is not None:
            if verbose:
                print(f"[embed] {lv} 可用：{msg}")
            return emb, emb.name, msg
        if verbose:
            print(f"[embed] {lv} 不可用：{msg}")
    if verbose:
        print("[embed] 降级到 L4：纯 BM25，无向量路")
    return None, "L4_pure_bm25", "无向量路（PRD 4.6 的 L4）"
