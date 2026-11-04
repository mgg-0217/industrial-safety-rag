# -*- coding: utf-8 -*-
"""版本卡：由**代码**强制拼接到答案下方（PRD 5.4 硬约束）。

## 为什么是代码拼的，不是模型说的

把"必须说明版本状态"写进提示词，等于把正确性押在模型的服从率上
（实测服从率只有 60–70%）。而版本状态是**结构化事实**
（`effective_from` / `effective_to` / `superseded_by` 都在库里），
模型只需要**复述**它，不该有**决定**它的权力。

→ 模型只负责组织语言，**代码负责拼接事实**。

## 措辞随口径变（同一份数据，两种呈现）

| 字段 | `current` | `as_of(历史)` |
|---|---|---|
| 状态 | 现行有效 | 该时点生效（查询时点 …） |
| 废止关系 | "已于 … 被《…》取代" | 同左，**不加警告措辞** |

🔴 `as_of` 口径下"后来被取代"是**答案该说的历史事实**，不是警告 ——
渲染成"⚠️ 该条款已废止"会误导使用者，以为查到了不该用的东西（PRD 5.2）。

## 三态，不是两态

`effective_from` 未知的文档，`is_effective()` 返回 UNKNOWN：
- `current` 口径：**纳入**，状态写"生效日未标注"
- `as_of` 口径：**排除**（无法判定就不能说它当时有效）
- `cross_version` 口径：纳入，状态写"历史版本 / 现行有效"

## 多版本命中时每个文档各一张卡，不合并（PRD 5.4）。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys_path = str(ROOT / "src")  # noqa: F401

import versioning as V  # noqa: E402


# ---------------------------------------------------------------------------
# manifest 索引
# ---------------------------------------------------------------------------

def load_manifest(root=None):
    """读 manifest.json，返回 ``{doc_id: 文档记录}``。"""
    import json
    root = pathlib.Path(root or ROOT)
    data = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    docs = data if isinstance(data, list) else data.get("documents", data)
    return {d["doc_id"]: d for d in docs}


# ---------------------------------------------------------------------------
# 状态措辞
# ---------------------------------------------------------------------------

def status_text(doc, when=None, mode="current"):
    """版本状态的一句话。**三态**。"""
    st = V.is_effective(doc, when) if when else V.is_effective(doc, "2999-12-31")
    if st is None:
        return "生效日未标注（无法判定生效区间）"
    if mode == "as_of":
        return f"该时点生效（查询时点 {when}）"
    if st:
        return "现行有效"
    # False：已废止 —— 只可能出现在 cross_version 口径下
    return "历史版本（已废止）"


def supersede_text(doc, manifest=None, mode="current"):
    """废止关系一句话。

    🔴 `as_of` 口径下这句话是**历史事实陈述**，不是警告 ——
    措辞与 `current` 口径完全一致，只是语境不同（PRD 5.2 / 5.4）。
    """
    manifest = manifest or {}
    sb = doc.get("superseded_by")
    if not sb:
        return "本版本未被取代"
    to = doc.get("effective_to")
    target = manifest.get(sb) or {}
    title = target.get("title") or sb
    doc_no = target.get("doc_no") or ""
    when_txt = f"于 {to}" if to else ""
    doc_no_txt = f"（{doc_no}）" if doc_no else ""
    return f"本版本{when_txt}被《{title}》{doc_no_txt}取代"


def note_text(doc):
    """变动摘要。

    🔴 没有 `note` 就写"**未标注**"，不写"无变动" ——
    后者是一个断言，不是事实（PRD 5.4）。
    """
    note = doc.get("note")
    if note and str(note).strip():
        return str(note).strip()
    return "未标注"


# ---------------------------------------------------------------------------
# 版本卡
# ---------------------------------------------------------------------------

def build_card(doc_snapshot, manifest=None, when=None, mode="current",
               quality=None):
    """给一个文档生成一张版本卡（纯字符串）。

    :param doc_snapshot: chunk 里的 ``doc_snapshot``（字段与 manifest 文档级一致）
    :param manifest: ``{doc_id: 记录}``，用于把 ``superseded_by`` 解析成标题/文号
    :param when: 判定时刻（``as_of`` 口径要用）
    :param mode: ``current`` / ``as_of`` / ``cross_version``
    :returns: 版本卡文本（多行）
    """
    doc = dict(doc_snapshot or {})
    manifest = manifest or {}
    lines = [
        f"文档：{doc.get('title') or doc.get('doc_id') or '（未知标题）'}",
    ]
    doc_no = doc.get("doc_no") or ""
    eff_from = doc.get("effective_from") or ""
    meta_line = []
    if doc_no:
        meta_line.append(f"文号：{doc_no}")
    if eff_from:
        meta_line.append(f"生效：{eff_from}")
    if meta_line:
        lines.append("    ".join(meta_line))
    lines.append(f"状态：{status_text(doc, when, mode)}")
    lines.append(f"废止关系：{supersede_text(doc, manifest, mode)}")
    lines.append(f"变动摘要：{note_text(doc)}")
    url = doc.get("source_url") or ""
    if url:
        lines.append(f"来源：{url}")
    if quality in ("partial", "ocr_verified"):
        tag = {"partial": "该来源抽取不完整",
               "ocr_verified": "来自 OCR 识别，可能有识别错误"}[quality]
        lines.append(f"质量：{quality}（{tag}）")
    return "\n".join(lines)


def build_cards(chunks, manifest=None, when=None, mode="current"):
    """给命中的 chunk 列表生成版本卡列表。

    🔴 **每个文档一张，不合并**（PRD 5.4）——
    两版都命中时，两张卡都在（`cross_version` 口径下这是常态）。
    """
    manifest = manifest or {}
    cards = []
    seen = set()
    for c in chunks:
        snap = c.get("doc_snapshot") or {}
        did = snap.get("doc_id") or c.get("doc_id")
        if did in seen:
            continue
        seen.add(did)
        cards.append(build_card(snap, manifest=manifest, when=when, mode=mode,
                                quality=c.get("quality")))
    return cards


def render_cards(cards, title="版本状态（由系统按检索结果自动附加）"):
    """把版本卡列表渲染成答案下方的方框块。"""
    if not cards:
        return ""
    out = ["┌─ " + title + " " + "─" * max(0, 40 - len(title))]
    for card in cards:
        for i, line in enumerate(card.split("\n")):
            prefix = "│ " if i == 0 else "│ "
            out.append(prefix + line)
    out.append("└" + "─" * 46)
    return "\n".join(out)
