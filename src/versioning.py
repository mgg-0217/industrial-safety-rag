# -*- coding: utf-8 -*-
"""版本关系建模 —— 把"这份文档某时刻是否现行"从人工标注变成可计算的区间判定。

**为什么需要这个模块**：manifest 里的 `status` 只是一个布尔投影，
判"某时刻 t 生效的是哪一版"需要区间：``effective_from <= t < effective_to``。
区间怎么来？三条来源，优先级从高到低：

1. **正文明示的施行日**（`effective_from_source = "declared"`）
   —— 文件自己写的"自 2023 年 10 月 6 日起施行"，最权威。
2. **正文明示的废止关系**（`effective_to_source = "explicit"`）
   —— 新版正文写"原《…》（文号）同时废止"，这是**明示**的取代关系。
3. **版本链推导**（`effective_to_source = "derived"`）
   —— 同一 `title` + `doc_type` 的文档按生效日排序，前一个的失效日 = 后一个的生效日。

**🔴 三个必须记住的坑**（都是实测踩出来的）：

1. **"生效日"有三个不同的日子**：印发日、发布日、施行（实施）日。
   `AQ/T 3034` 封面写"２０２２Ｇ１０Ｇ０１发布 ２０２３Ｇ０４Ｇ０１实施"——
   发布 2022-10-01，**实施 2023-04-01**，差半年。区间判定必须用**施行日**。
   人工标注的 `effective_date` 有取成发文日的（见 `CONFLICTS`），
   所以本模块**以正文为准**，并在冲突时记录。

2. **"有效期 X 年" ≠ `effective_to`**。
   榆林 2023 版写"自 2023 年 10 月 6 日起施行，**有效期 5 年**"（→ 2028-10-06），
   但它 2025-01-09 就被新版取代了。有效期是"最长的可能失效日"，
   真正的失效日 = 被取代日。**有效期不参与区间推导**，否则会算出一个假的空窗期。

3. **"未知"在 manifest 里是空串，不是 `-`**。
   实测 20 份文档的 `doc_no` 是 `""`，`guide_jiaxian` 的 `version`/`effective_date` 也是 `""`。
   判未知必须同时接受 `""` / `"-"` / `None`。
   更隐蔽的是：**字符串比较不会报错，只会静默给错答案**——
   `"" <= "2026-10-30"` 恒真，于是"生效日未知"的文档会变成"永远现行"。
   所以 `is_effective()` 返回**三态**（True / False / None），而不是布尔。

**三轨分离的延续**（见 `normalize.py`）：本模块只做**日期与关系**的判定，
不碰文本归一化，也不做检索。
"""
from __future__ import annotations

import json
import pathlib
import re
from datetime import date

# ---------------------------------------------------------------------------
# 未知值：manifest 里的三种"没有"
# ---------------------------------------------------------------------------

#: 视为"未知"的所有形态。实测 manifest 用的是空串，但 `-` 与 `None` 也要认。
UNKNOWN_VALUES = ("", "-", None)

#: `effective_from` 的来源标记
SOURCE_DECLARED = "declared"   # 正文明示施行日
SOURCE_DOC_DATE = "doc_date"    # 人工标注的 effective_date（多为发文日）
SOURCE_UNKNOWN = "unknown"     # 未知

#: `effective_to` 的来源标记
SOURCE_EXPLICIT = "explicit"    # 正文明示"同时废止"
SOURCE_DERIVED = "derived"      # 版本链推导
SOURCE_OPEN = "open"            # 至今未失效（无后继版本）
SOURCE_UNKNOWN_TO = "unknown"   # 未知


def is_unknown(value) -> bool:
    """判断一个字段是不是"未知"。三种形态都认：``""`` / ``"-"`` / ``None``。"""
    return value in UNKNOWN_VALUES


def _to_date(value):
    """``YYYY-MM-DD`` → ``date``；未知或格式不对返回 ``None``。"""
    if is_unknown(value):
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 三态区间判定
# ---------------------------------------------------------------------------

#: `is_effective` 的三态返回值
TRUE = True
FALSE = False
UNKNOWN = None


def is_effective(doc: dict, when: str | date) -> bool | None:
    """判断文档在 ``when`` 时刻是否生效。**返回三态**，不是布尔。

    :param doc: 至少含 ``effective_from`` / ``effective_to`` 键的字典
    :param when: 判定时刻（``YYYY-MM-DD`` 字符串或 ``date``）
    :returns:
        - ``True``  该时刻生效
        - ``False`` 该时刻不生效（已失效，或尚未生效）
        - ``None``  **无法判定**——生效日未知，或 ``when`` 本身不合法

    🔴 **不要把 ``None`` 当成 ``False``**：那是"不知道"，不是"不生效"。
    调用方按口径分别处理（见 PRD 2.6.1）：
    ``current`` 口径纳入但标注"生效日未标注"；``as_of`` 口径排除。

    ⚠️ 字符串比较在这里是安全的，因为**先判了未知**：
    `"" <= "2026-10-30"` 恒真这个坑被 `is_unknown` 挡在前面。
    """
    t = when if isinstance(when, date) else _to_date(when)
    if t is None:
        return UNKNOWN

    frm = _to_date(doc.get("effective_from"))
    if frm is None:
        return UNKNOWN          # 生效日未知 → 无法判定

    to = _to_date(doc.get("effective_to"))
    if to is None:
        # effective_to 未知有两种含义：无后继（未失效）或真的不知道。
        # 用 effective_to_source 区分："open" 才是"至今有效"。
        if doc.get("effective_to_source") == SOURCE_OPEN:
            return frm <= t
        if is_unknown(doc.get("effective_to")):
            return UNKNOWN
        return frm <= t

    return frm <= t < to


# ---------------------------------------------------------------------------
# 从正文里抽日期
# ---------------------------------------------------------------------------

#: "自 2023 年 10 月 6 日起施行"
_RE_EFFECT_CN = re.compile(
    r"自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(?:起)?\s*"
    r"(?:施行|实施|执行|生效)"
)
#: "自 2023-10-06 起施行"
_RE_EFFECT_DASH = re.compile(
    r"自\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*日?\s*(?:起)?\s*"
    r"(?:施行|实施|执行|生效)"
)
#: "2023-04-01 实施"（无"自"字）
_RE_EFFECT_BARE = re.compile(
    r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*日?\s*(?:起)?\s*(?:施行|实施|执行|生效)"
)
#: ISO 紧凑型："２０２２Ｇ１０Ｇ０１发布 ２０２３Ｇ０４Ｇ０１实施"
#: —— 数字与分隔符 G 都是全角，必须先归一化再匹配（见 normalize.py）。
#: ⚠️ **只认"实施/施行/生效"，不认"发布"**：发布日 ≠ 生效日，
#:    AQ/T 3034 封面这两个日子差了半年（2022-10-01 发布 / 2023-04-01 实施）。
_RE_EFFECT_COMPACT = re.compile(
    r"(\d{4})\s*[G]\s*(\d{2})\s*[G]\s*(\d{2})\s*(?:实施|施行|生效)"
)


def _normalize_digits(text: str) -> str:
    """全角 → 半角。优先用 `normalize.normalize_digits`，不可用时退回内置表。"""
    try:
        from .normalize import normalize_digits  # type: ignore
    except ImportError:
        try:
            from normalize import normalize_digits  # type: ignore
        except ImportError:
            table = {chr(c): chr(c - 0xFEE0) for c in range(0xFF10, 0xFF1A)}
            table.update({chr(c): chr(c - 0xFEE0) for c in range(0xFF21, 0xFF1B)})
            table.update({chr(c): chr(c - 0xFEE0) for c in range(0xFF41, 0xFF5B)})
            table["Ｇ"] = "G"
            return text.translate(str.maketrans(table))
    out = normalize_digits(text)
    # normalize_digits 不处理全角 G（它不在"数值符号"表里），这里补上。
    return out.replace("Ｇ", "G")


def parse_effective_date(text: str) -> tuple[str | None, str]:
    """从正文里抽"施行/实施日"。返回 ``(日期, 命中的原文片段)``。

    抽不到就返回 ``(None, "")``——**不猜**。

    识别的写法：
    - ``自 2023 年 10 月 6 日起施行``（中文日期）
    - ``自 2023-10-06 起施行``（横杠日期）
    - ``2023-04-01 实施``（无"自"字）
    - ``２０２３Ｇ０４Ｇ０１实施``（ISO 紧凑型，全角，需先归一化）

    ⚠️ "本制度自**印发之日**起施行"这类**没有具体日期**的写法抽不出来，
    返回 ``None``——这是正确的：区间判定需要一个确定的日子，
    这种情况走人工标注的 `effective_date`（来源标 ``doc_date``）。
    """
    if not text:
        return None, ""

    normalized = _normalize_digits(text)

    for pat in (_RE_EFFECT_CN, _RE_EFFECT_DASH, _RE_EFFECT_BARE, _RE_EFFECT_COMPACT):
        m = pat.search(normalized)
        if m:
            y, mo, d = (int(g) for g in m.groups())
            try:
                return date(y, mo, d).isoformat(), m.group(0).strip()
            except ValueError:
                continue
    return None, ""


# ---------------------------------------------------------------------------
# 从正文里抽明示的废止关系
# ---------------------------------------------------------------------------

#: 文号形态：榆政办发〔2023〕26号 / 国办发〔2023〕15号 / 京政发〔2024〕3号
_RE_DOC_NO = re.compile(r"[一-龥]{1,10}〔\d{4}〕\d{1,4}号")
#: "原《…》（文号）同时废止" —— 废止的宾语是**括号里的文号**
_RE_SUPERSEDE = re.compile(
    r"(?:原|原《[^》]*》|原[^，。；\n]{0,30})[^，。；\n]{0,80}?"
    r"[（(]\s*(?P<no>[一-龥]{1,10}〔\d{4}〕\d{1,4}号)\s*[)）]"
    r"[^，。；\n]{0,20}?(?:同时)?(?:予以)?废止"
)


def parse_explicit_supersessions(text: str) -> list[dict]:
    """从正文里抽"本文件废止了哪些旧文件"。返回 ``[{doc_no, quote}]``。

    识别的写法（以榆林 2025 版第十九条为例）::

        第十九条  本制度自印发之日起施行。原《榆林市危险化学品企业动火和受限空间
        作业安全管理制度》（榆政办发〔2023〕26号）同时废止。

    ⚠️ 只认**文号**级的明示废止。**只提标题不发文号的**（"原《某某办法》同时废止"）
    抽不出来——因为标题可能有多个同名版本，靠标题匹配会认错版本。
    这种情况下靠版本链推导兜底，并在冲突时记录。
    """
    if not text:
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for m in _RE_SUPERSEDE.finditer(text):
        no = m.group("no")
        if no in seen:
            continue
        seen.add(no)
        quote = m.group(0).strip()
        out.append({"doc_no": no, "quote": quote})
    return out


# ---------------------------------------------------------------------------
# 版本链
# ---------------------------------------------------------------------------


def group_version_chains(docs: list[dict]) -> dict[tuple, list[dict]]:
    """按 ``(title, doc_type)`` 把文档分成版本链。

    ⚠️ 同一份制度可能有**不同的 doc_type**（人工标注可能不一致），
    所以分组键用 ``(title, doc_type)``，而不是只看 title——
    只看 title 会把"同名但不同类"的文档错并成一条链。
    """
    chains: dict[tuple, list[dict]] = {}
    for d in docs:
        key = (d.get("title", ""), d.get("doc_type", ""))
        chains.setdefault(key, []).append(d)
    return chains


def derive_chain_fields(docs: list[dict]) -> tuple[dict, list[dict]]:
    """推导版本链上的 ``effective_to`` / ``superseded_by``。

    规则（PRD 2.6）：

    1. 同链内按 ``effective_from`` 升序排列；
    2. 非末位的文档，其 ``effective_to`` = 次位的 ``effective_from``；
    3. 非末位的文档，其 ``superseded_by`` = 次位的 ``doc_id``；
    4. 末位文档的 ``effective_to`` = ``None``（至今未失效）。

    :returns: ``(字段字典, 冲突记录列表)``。
        字段字典形如 ``{doc_id: {"effective_to":..., "superseded_by":...,
        "effective_to_source":...}}``。

    ⚠️ **推导值是"库里可见的下一个版本的生效日"**，可能晚于真实废止日
    （库里缺中间版本时）。这个局限在 PRD 2.6 有完整说明，
    所以推导值一律标 ``derived``，不冒充明示值。
    """
    fields: dict[str, dict] = {}
    conflicts: list[dict] = []

    for key, chain in group_version_chains(docs).items():
        if len(chain) < 2:
            continue

        # 生效日已知的排前面；未知的排最后（不参与推导，但保留在链里）
        def _sort_key(d):
            dt = _to_date(d.get("effective_from"))
            return (dt is None, dt or date.max)

        ordered = sorted(chain, key=_sort_key)

        for i, doc in enumerate(ordered):
            if i + 1 < len(ordered):
                nxt = ordered[i + 1]
                frm = _to_date(doc.get("effective_from"))
                nxt_frm = _to_date(nxt.get("effective_from"))
                if frm is None or nxt_frm is None:
                    continue        # 未知生效日的无法推导，跳过
                fields[doc["doc_id"]] = {
                    "effective_to": nxt_frm.isoformat(),
                    "superseded_by": nxt["doc_id"],
                    "effective_to_source": SOURCE_DERIVED,
                }
            else:
                fields[doc["doc_id"]] = {
                    "effective_to": None,
                    "superseded_by": None,
                    "effective_to_source": SOURCE_OPEN,
                }

    return fields, conflicts


# ---------------------------------------------------------------------------
# 主流程：给 manifest 补版本字段
# ---------------------------------------------------------------------------

#: 需要人工确认的冲突（实测发现的）
CONFLICTS: list[dict] = []


def _read_text(md_dir: pathlib.Path, doc: dict) -> str:
    tf = doc.get("text_file")
    if not tf:
        return ""
    p = md_dir / tf
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def apply_version_fields(docs: list[dict], md_dir) -> tuple[list[dict], list[dict]]:
    """给 manifest 的每份文档补齐版本字段。**不修改入参**，返回新列表。

    新增字段：

    ==========================  ==========================================
    ``effective_from``          区间判定用的生效日
    ``effective_from_source``   ``declared`` / ``doc_date`` / ``unknown``
    ``effective_to``            失效日（``None`` = 至今未失效）
    ``effective_to_source``     ``explicit`` / ``derived`` / ``open`` / ``unknown``
    ``superseded_by``           继任版本的 ``doc_id``
    ``supersedes``              本版本明示废止的旧版本 ``doc_id`` 列表
    ==========================  ==========================================

    优先级：**正文明示 > 链上推导**。

    :returns: ``(新文档列表, 冲突记录列表)``
    """
    md_dir = pathlib.Path(md_dir)
    out: list[dict] = []
    conflicts: list[dict] = []

    # ---- 第一遍：生效日（正文优先） + 明示废止关系 ----
    superseded_map: dict[str, str] = {}   # 被废止文号 → 继任 doc_id
    supersedes_map: dict[str, list[str]] = {}  # 继任 doc_id → 被废止 doc_id 列表

    for doc in docs:
        text = _read_text(md_dir, doc)
        declared, quote = parse_effective_date(text)
        recorded = doc.get("effective_date")
        recorded_date = _to_date(recorded)

        new = dict(doc)

        if declared:
            if recorded_date and recorded_date != _to_date(declared):
                conflicts.append({
                    "type": "effective_date_mismatch",
                    "doc_id": doc["doc_id"],
                    "recorded": recorded,
                    "declared": declared,
                    "quote": quote,
                    "resolution": "以正文明示的施行日为准",
                })
            new["effective_from"] = declared
            new["effective_from_source"] = SOURCE_DECLARED
        elif recorded_date:
            new["effective_from"] = recorded
            new["effective_from_source"] = SOURCE_DOC_DATE
        else:
            new["effective_from"] = None
            new["effective_from_source"] = SOURCE_UNKNOWN

        # 明示废止：正文里提到的旧文号 → 映射到库里的 doc
        explicit = parse_explicit_supersessions(text)
        old_ids: list[str] = []
        for item in explicit:
            for other in docs:
                if other["doc_id"] == doc["doc_id"]:
                    continue
                if (other.get("doc_no") or "") == item["doc_no"]:
                    old_ids.append(other["doc_id"])
                    superseded_map[other["doc_id"]] = doc["doc_id"]
        if old_ids:
            supersedes_map[doc["doc_id"]] = old_ids
        new["supersedes"] = old_ids
        out.append(new)

    # ---- 第二遍：失效日（明示优先，其次链上推导） ----
    derived, _ = derive_chain_fields(out)
    by_id = {d["doc_id"]: d for d in out}

    for doc in out:
        did = doc["doc_id"]

        # (a) 明示：本文档被某新版正文点名废止
        if did in superseded_map:
            succ = by_id[superseded_map[did]]
            succ_from = succ.get("effective_from")
            doc["effective_to"] = succ_from
            doc["effective_to_source"] = SOURCE_EXPLICIT
            doc["superseded_by"] = succ["doc_id"]
            continue

        # (b) 链上推导
        if did in derived:
            info = derived[did]
            doc["effective_to"] = info["effective_to"]
            doc["effective_to_source"] = info["effective_to_source"]
            doc["superseded_by"] = info["superseded_by"]
            continue

        # (c) 单版本文档：至今未失效
        doc["effective_to"] = None
        doc["effective_to_source"] = SOURCE_OPEN
        doc["superseded_by"] = None

    return out, conflicts


def build(manifest_path, md_dir=None, out_path=None):
    """读 manifest → 补版本字段 → 写回。返回 ``(新文档列表, 冲突列表)``。

    ⚠️ **写回会重排 manifest.json 的字段顺序**（新增字段排在最后）。
    这是幂等的：同样的输入跑两次，输出完全一致。
    """
    manifest_path = pathlib.Path(manifest_path)
    root = manifest_path.parent.parent
    md_dir = pathlib.Path(md_dir) if md_dir else root / "data" / "md"

    docs = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_docs, conflicts = apply_version_fields(docs, md_dir)

    if out_path:
        out_path = pathlib.Path(out_path)
        out_path.write_text(
            json.dumps(new_docs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return new_docs, conflicts
