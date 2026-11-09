"""评测题的 key_point 对齐源解析 —— 质检与规范化共用同一套判据。

🔴 **为什么必须按题型分源**（2026-11-02 实测）：
  新合成的 61 条题被质检判成「key_point 不可达」（版本陷阱题 41 + 跨文档推理题 15
  + 数值题 4），但**判据查错了字段**——判据查 `gold_answer`，而：
    - 版本陷阱题的 `gold_answer` 是 `None`：答案在 `expect_doc_id` 指向的文档正文里，
      用 `gold_answer` 查永远是「找不到」；
    - 跨文档推理题的 `gold_answer` 是 LLM 的**概括**，不是原文片段，
      字面包含判定必然不成立。
  换句话说：**不是数据坏，是判据用错字段**。
  同一批数据在两种判据下的结论完全相反 —— 这就是「判据必须与数据类型对齐」的实例。

五类题的 key_point 性质各不相同：

  | 题型 | 对齐源 | 判分含义 |
  |---|---|---|
  | 数值题 | `must_include_numbers` | 数值精确匹配（保真锚点） |
  | 事实题 | `gold_answer`（原文切片） | 原文片段字面包含 |
  | 版本陷阱题 | `expect_doc_id` 指向文档的 md 正文 | 原文片段字面包含 |
  | 跨文档推理题 | `expect_doc_id` + `related_doc_ids` 的 md 正文 | 原文片段字面包含 |
  | 拒答题 | 无（判分标准） | 「是否声明查得到」是行为判据，无法字面包含在原文里 |

拒答题的 key_point 是**判分标准**（"明确表示语料中无该设备的相关规定"），
不是对原文的断言。要求它在原文里字面可达是**逻辑错误**——
它的正确答案就是「语料里没有」，所以它压根不指向任何原文。
这类题单独标记 `key_point_source = "refusal_criteria"`，质检时跳过可达性判据。
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 拒答题的 key_point 是判分标准，不是原文断言
REFUSAL_CRITERIA = "refusal_criteria"


def read_md(doc_id, root=None):
    """读某份文档的 md 正文（去掉页码注释）。"""
    root = pathlib.Path(root) if root else ROOT
    if not doc_id:
        return ""
    p = root / "data" / "md" / (doc_id + ".md")
    if not p.exists():
        return ""
    return re.sub(r"<!--\s*page:\d+\s*-->", "", p.read_text(encoding="utf-8"))


def doc_evidence(rec, root=None, max_chars=120000):
    """版本陷阱题 / 跨文档推理题的原文依据（md 正文）。"""
    root = pathlib.Path(root) if root else ROOT
    ids = []
    for d in (rec.get("expect_doc_id"), *(rec.get("related_doc_ids") or [])):
        if d and d not in ids:
            ids.append(d)
    parts = []
    for d in ids:
        t = read_md(d, root)
        if t:
            parts.append(f"【{d}】\n{t[:max_chars]}")
    return "\n\n".join(parts)


def qc_source(rec, root=None):
    """**质检**用的可达性判定源 —— 与 `alignment_source` **故意不同**。

    🔴 **为什么要分成两个源**（第五轮 §2.2 实测）：
    `alignment_source` 对事实题返回 `gold_answer`，那是 **M2 时代按 `clause_no`
    机械切片**的产物（切点落在词中间，甚至跨两个条款）。
    而 key_point 现在来自**检索证据**（LLM 从原文整句里提炼）——
    两者**字段对不上** → 质检把 46 条判成「key_point 不可达」，
    其中 **40 条是"真原文，只是不在遗留的 gold_answer 里"**，
    属于**判据用错字段**，不是数据坏。

    → 事实题的可达性判定必须查**来源文档正文**。

    ⚠️ 但**对齐流程**（`fix_key_points.py` / `relink_test_set.py`）仍然用
    `alignment_source`：它要的是"把 kp 对齐回**那个条款**"，
    换成整篇正文会把 kp 对到别的条款的相似句上（K-16 的教训）。

    返回 (源文本, 源类型)，源类型取值见 `alignment_source` 的 docstring。
    """
    qtype = rec.get("question_type")
    if qtype == "拒答题":
        return "", REFUSAL_CRITERIA
    if qtype in ("版本陷阱题", "跨文档推理题"):
        return doc_evidence(rec, root), "doc_md"
    # ---- 事实题 / 数值题：来源文档正文 ----
    # 数值题的锚点也来自原文，只是 M2 的 gold_answer 是**切片**，
    # 锚点常常落在切片之外（实测 8 条）→ 同样属于判据问题。
    d = rec.get("doc_id") or rec.get("expect_doc_id") or ""
    txt = read_md(d, root) if d else ""
    if txt:
        return txt, "doc_md"
    # 兜底：出处文档读不到 → 退回遗留切片（宁可宽判，不可漏判）
    return rec.get("gold_answer") or "", "gold_answer"


def anchor_list(rec):
    """数值题的保真锚点列表（`must_include_numbers`，可能是 ";" 分隔字符串）。"""
    anchors = rec.get("must_include_numbers") or []
    if isinstance(anchors, str):
        anchors = [a for a in anchors.split(";") if a.strip()]
    return [a for a in anchors if str(a).strip()]


def alignment_source(rec, root=None):
    """返回 (对齐源文本, 源类型)。源类型决定质检与规范化怎么用这个文本。

    源类型取值：
      "must_include_numbers" —— 数值题，key_point 取保真锚点（见 anchor_list），
                                但**字面可达判定仍查原文**（gold_answer）
      "gold_answer"         —— 事实题，取原文切片
      "doc_md"              —— 版本陷阱题 / 跨文档推理题，取文档正文
      "refusal_criteria"    —— 拒答题，key_point 是判分标准，不做字面可达判定
      "none"                —— 没有可用的源
    """
    qtype = rec.get("question_type")
    if qtype == "数值题":
        # 锚点本身来自原文，所以可达性判定查 gold_answer：
        # 「锚点是不是真的写在这条题的原文里」——这正是数值保真判定的底线。
        return rec.get("gold_answer") or "", "must_include_numbers"
    if qtype == "拒答题":
        return "", REFUSAL_CRITERIA
    if qtype in ("版本陷阱题", "跨文档推理题"):
        return doc_evidence(rec, root), "doc_md"
    return rec.get("gold_answer") or "", "gold_answer"


def needs_alignment(rec):
    """这道题是否需要跑「key_point 对齐回原文」。"""
    return alignment_source(rec)[1] in ("gold_answer", "doc_md")


# ---------------------------------------------------------------- 对齐算法

import re as _re
from difflib import SequenceMatcher

# 归一化用表：统一标点、去空白
TABLE = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "（": "(", "）": ")",
    "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "－": "-", "～": "~", "％": "%",
})

# LLM 给 key_point 加的标签前缀（"数值锚点：5 m"）
LABEL_PREFIX = _re.compile(
    r"^\s*(数值锚点|干扰数值|正确锚点|答案数值|关键数值|数值|原文依据)\s*[:：]\s*")


def flat(s):
    """归一化：去空白 + 统一标点 + 小写。"""
    return _re.sub(r"\s+", "", s or "").translate(TABLE).lower()


def split_segments(text):
    """把长文本切成自然段（对齐时逐段比对，避免长文本上跑 SequenceMatcher）。

    🔴 为什么按段比：文档正文可达 3 万字，
    `SequenceMatcher(None, kp, 长文本)` 在这种规模上是 O(n·m)，单次要秒级；
    而 key_point 真正的依据几乎总是**连续的一段**，按段比对既快又更准
    （跨段拼接出来的「公共片段」在原文里根本不是连续出现的）。
    """
    if not text:
        return []
    segs = [s.strip() for s in _re.split(r"\n\s*\n|(?<=[。；！？])", text)]
    return [s for s in segs if len(s) >= 4]


def best_span(kp, text):
    """kp 与 text 的最长公共连续片段（**归一化形态**，可能为空）。

    🔴 为什么直接在归一化串上取片段，而不是换算回原串的坐标：
    实测按长度比例换算会**偏移**，切出"所有动火作业时应配备能够采集全过程作业影"
    这种末尾被削掉的片段（原文里是"…作业影像的防爆型摄录设备"）——
    比例换算把空白和全角字符的影响算错了。
    而下游的字面包含判定**本来就是在归一化串上做的**
    （`flat(kp) in flat(source)`），所以归一化片段就是判定要的形态，
    换算回原串既无必要也不准。
    """
    kp_f = flat(kp)
    if not kp_f:
        return ""
    best = ""
    for seg in split_segments(text):
        seg_f = flat(seg)
        if not seg_f:
            continue
        sm = SequenceMatcher(None, kp_f, seg_f, autojunk=False)
        block = max(sm.get_matching_blocks(), key=lambda b: b.size)
        if block.size == 0:
            continue
        # 🔴 **2026-11-09 修一个坐标 bug**：
        #    `block.a` 是**kp 里的**位置，`block.b` 才是**原文段里的**位置。
        #    原代码用 `seg_f[block.a:block.a + block.size]` ——
        #    拿 kp 的坐标去切原文段 → **切出来的根本不是公共片段**。
        #    实测：kp「特种设备事故应急救援工作坚持…」的真实公共块只有 4 字
        #    （"特种设备"），却被切成「范围锅炉,压力容器(含气瓶)…」34 字。
        cand = seg_f[block.b:block.b + block.size]
        if len(cand) > len(best):
            best = cand
    return best


#: 🔴 **证据标记**（2026-11-05 实测修正）。
#:
#: `doc_evidence()` 给 LLM 的证据文本每份文档前面加一行 `【<doc_id>】`，
#: 用来告诉模型"这段来自哪份文档"—— 这个设计是对的。
#:
#: 但它会**泄漏进 key_point**：
#:   1. 模型模仿证据格式，把 `【yulin_donghuo_xianshi】` 写进了 key_point；
#:   2. 对齐算法再忠实地把它对到证据的标记行上
#:      → key_point 变成 **`【yulin_do`** 这种碎片（实测跨文档题 15/25 条中招）。
#:
#: → 对齐时必须把标记从**两边**都剥掉：
#:    * `kp` 的**开头**（模型模仿来的标记，常常没有闭合的 `】`）；
#:    * `text` 里**独占一行**的标记（`doc_evidence` 生成的形态）。
_EVIDENCE_MARKER_LINE = re.compile(r"(?m)^[ \t]*【[^】\n]{1,80}】[ \t]*$")
_EVIDENCE_MARKER_HEAD = re.compile(r"^[ \t]*【[^】\n]{0,80}")


def align_key_point(kp, text, min_len=4):
    """把 kp 对齐到原文片段。

    判据：**只要公共片段 >= min_len 个字就算对齐成功**，用该片段替换 kp。

    🔴 为什么不用「公共片段 >= kp 长度的 60%」这种比例门槛：
    实测 93 个 kp 因此被误剔，而它们**语义全是对的**，只是 LLM 做了概括：
        "开展设备管道试压、吹扫等准备"  <- 原文含 "管道试压、吹扫"（7/13 = 54%）
        "紧抓重点环节并采取断然措施"    <- 原文含 "紧抓重点环节"（6/13 = 46%）
    剔掉它们会丢掉真正的关键信息点，还会让 20 条题的 key_points 变成空。
    片段短一点不要紧——**只要它是原文里真实存在的片段，字面包含判定就成立**，
    这才是 PRD 要的「可复核」。

    只有公共片段为空（真幻觉，如 "50 %"）才剔除。

    🔴 **证据标记必须先剥掉**（见 :data:`_EVIDENCE_MARKER_LINE` 的说明）：
    否则模型模仿来的 `【yulin_do` 会被对齐成 `【yulin_do` 这种碎片，
    在原文里根本不存在 → key_point_coverage 的分母全是垃圾。
    """
    kp_clean = LABEL_PREFIX.sub("", str(kp)).strip()
    kp_clean = _EVIDENCE_MARKER_HEAD.sub("", kp_clean).strip()
    kp_f = flat(kp_clean)
    if not kp_f:
        return None, ""
    text = _EVIDENCE_MARKER_LINE.sub("", text or "")
    span = best_span(kp_f, text)
    if span and len(flat(span)) >= min_len:
        return span, span
    return None, span


def normalize_anchor(a):
    """锚点规范化：去标签前缀、去空格、统一单位空格。"""
    a = LABEL_PREFIX.sub("", str(a)).strip()
    return _re.sub(r"\s*([%°a-zA-Z³²/·])\s*$", r"\1", a)


def dedup_key_points(kps):
    """去掉彼此重复 / 高度重叠的 key_point，保留信息量更大的那个。

    🔴 为什么必须去重：`key_point_coverage` = 命中数 / 总数。
    如果两个 key_point 其实是同一句话（一个是另一个的前缀，或重叠度极高），
    那命中其中一个就等于命中两个 —— **分母被灌水，分数虚高**。
    实测踩过：对齐到原文后，同一题的 kp 变成
    「所有动火作业时应配备能够采集全过程作业影像的防爆型摄录设备」
    和「所有动火作业时应配备能够采集全过程作业影」
    —— 后者是前者被削掉尾巴的残片，判分时毫无信息量。

    判据：归一化后互为子串，或一方是另一方的子串（长度 ≥ 8 字）→ 保留长的。
    """
    kept = []
    for kp in kps:
        f = flat(kp)
        if not f:
            continue
        drop = False
        for j, old in enumerate(kept):
            of = flat(old)
            if not of:
                continue
            if f in of or of in f:
                if len(f) <= len(of):
                    drop = True
                else:
                    kept[j] = kp
                break
            shorter, longer = (f, of) if len(f) <= len(of) else (of, f)
            if len(shorter) >= 8 and shorter in longer:
                if len(f) <= len(of):
                    drop = True
                else:
                    kept[j] = kp
                break
        if not drop:
            kept.append(kp)
    return kept
