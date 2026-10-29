# -*- coding: utf-8 -*-
"""文本归一化 —— 切分前的第一道工序。

两条脏数据是从真实语料里统计出来的，不是猜的（统计脚本见 tools/test_normalize.py）：

1. **全角数字占全部数字的 19.8%**（4563 / 23025），几乎全部来自 PDF 文本层
   —— 导则封面上的 "２０２２Ｇ１０６０１发布"、作业票里的 "２０２２年 ４月 １日"。
   不归一化则问「设计压力 0.35」召不回正文里的「０.３５」。

2. **位号里的全角破折号**：`19—V—4215`(29 处) / `1910—PV—42151`(14 处) /
   `19—V4215`(5 处)，而 manifest 的 equipment_tags 记的是半角连字符
   （`19-V-4215`）。两边对不上，位号检索会整个失效。

**策略：双轨文本。**
- ``text``     —— 原文一字不动。引用展示必须是原文，改了就失去溯源价值。
- ``norm_text`` —— 归一化副本。只用于分词与检索。

🔴 为什么位号不能全局替换破折号：
   正文里 `—` 也当标点破折号用（"设备名称——粗产品缓冲罐"）。全局替换会把
   标点破折号变成连字符，破坏中文排版语义。所以位号归一化是**上下文感知**的：
   只有当破折号两侧都是数字或字母时才当作位号分隔符。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. 全角 → 半角
# ---------------------------------------------------------------------------

# 全角数字 ０-９
_FULLWIDTH_DIGITS = {chr(c): chr(c - 0xFEE0) for c in range(0xFF10, 0xFF1A)}
# 全角大写 A-Z / 小写 a-z
_FULLWIDTH_UPPER = {chr(c): chr(c - 0xFEE0) for c in range(0xFF21, 0xFF3B)}
_FULLWIDTH_LOWER = {chr(c): chr(c - 0xFEE0) for c in range(0xFF41, 0xFF5B)}

#: 全角标点 → 半角。保留中文常规标点（，。；：、！？（）《》〔〕）不转——
#: 它们是正文的一部分，转了会破坏原文可读性，而且它们在分词时都是边界，
#: 转不转对检索没影响。真正必须转的是那些**承载数值/编号**的全角字符。
_FULLWIDTH_SYMBOLS = {
    "．": ".",   # FULLWIDTH FULL STOP —— PDF 表格的行/列分隔符，1211 次
    "－": "-",   # FULLWIDTH HYPHEN-MINUS —— 位号里可能出现
    "＋": "+",   # FULLWIDTH PLUS SIGN
    "％": "%",   # FULLWIDTH PERCENT SIGN —— 数值题相关
    "＝": "=",   # FULLWIDTH EQUALS SIGN
    "×": "×",   # MULTIPLICATION SIGN（本身已是全角形，留着）
    "　": " ",   # IDEOGRAPHIC SPACE —— 全角空格，转半角空格
}

_TRANSLATION = {**_FULLWIDTH_DIGITS, **_FULLWIDTH_UPPER, **_FULLWIDTH_LOWER,
                 **_FULLWIDTH_SYMBOLS}

# 🔴 str.translate() 的表以 **Unicode 码位**为 key，直接传字符串 key 的 dict 是静默失效的
#    （查表永远查不到，translate 原样返回）。必须先经 str.maketrans 转成码位表。
_TRANSLATE_TABLE = str.maketrans(_TRANSLATION)

# ---------------------------------------------------------------------------
# 2. 位号归一化（上下文感知）
# ---------------------------------------------------------------------------

#: 各种"长得像连字符/破折号"的字符
_DASH_CHARS = "-‐‑‒–—―−－"

#: 只有两侧都是数字或字母时，才把破折号视为位号分隔符。
#: 场景： 19—V—4215   1910—PV—42151   T—4200   200—S1
#: 不动：  设备名称——粗产品缓冲罐     （中文破折号，两侧非 ASCII）
#:        第 4 条——自 2025 年起施行
_TAG_DASH = re.compile(
    r"(?<=[0-9A-Za-z])([" + re.escape(_DASH_CHARS) + r"]){1,2}(?=[0-9A-Za-z])"
)

#: 多个破折号连用（19——V）压成一个
_MULTI_DASH = re.compile(r"-{2,}")


def normalize_digits(text: str) -> str:
    """全角数字/字母/数值符号 → 半角。中文标点不动。"""
    return text.translate(_TRANSLATE_TABLE)


def normalize_equipment_tags(text: str) -> str:
    """位号里的破折号 → 半角连字符。仅在两侧是 ASCII 数字/字母时生效。

    这样 `19—V—4215` → `19-V-4215`，而中文破折号 `——` 原样保留。
    """
    text = _TAG_DASH.sub("-", text)
    return _MULTI_DASH.sub("-", text)


def normalize(text: str) -> str:
    """完整归一化：数字 + 位号。按此顺序（先数字后位号）即可。"""
    return normalize_equipment_tags(normalize_digits(text))


def normalize_ws(text: str) -> str:
    """归一化后的空白：连续空白压成一个空格，去掉行首尾空白。"""
    return re.sub(r"[ \t　]+", " ", text).strip()


def build_chunk_text(text: str) -> tuple[str, str]:
    """切分时给每个块生成双轨文本。

    返回 ``(text, norm_text)``：
    - ``text``     原文，供引用展示与溯源
    - ``norm_text`` 归一化 + 空白收敛，供分词与检索
    """
    return text, normalize_ws(normalize(text))


# ---------------------------------------------------------------------------
# 设备位号：不做正则自动抽取
# ---------------------------------------------------------------------------
#
# 🔴 试过用正则从正文里自动抽位号，**误抽率高到不可用**。实测在 26 份语料里
#    抽出 132 种"位号"，其中大部分是别的东西：
#       2024-10-15（日期）      020-83137111（电话号码）
#       GB30871-2022（标准号）   2025-00062（文号）
#       H2O2-C3H6O（化学式）     0551-62999504（传真）
#    真位号只占其中一小部分（19-V-4215 / T-4200 / 1910-PV-42151 …）。
#
#    结论：**能人工标注的就别用正则猜**。位号池以 manifest 的人工标注为准，
#    检索时拿标注位号去归一化正文里做精确匹配 —— 这既准确又能定位到出现位置。
#
#    代价：manifest 里人工标注的位号目前只覆盖 3 份文档。未标注文档里的位号
#    检索不到。补标注是人工成本，优先级低于把检索链路先跑通。

#: 位号形态判定：形如 19-V-4215 的 ASCII 段-段 串。
#: 与 manifest 的 equipment_tags 区分 —— 那个字段里目前还混着少量中文设备
#: 类别名（如「厂内机动车辆」「油气回收装置」），那不是位号，不参与匹配。
_TAG_SHAPE = re.compile(r"^[0-9A-Z]{1,8}(?:-[0-9A-Z]{1,8}){1,4}$")


def is_equipment_tag(value: str) -> bool:
    """判断一个标注值是不是**位号形态**（而非中文设备类别名）。"""
    return bool(_TAG_SHAPE.match(value))


def tag_key(tag: str) -> str:
    """位号的**匹配键**：去掉所有分隔符 + 大写。

    为什么需要这一轨：
        同一台设备在两批语料里有两种写法 ——
            事故调查报告：  19—V—4215   （两处破折号）
            整改落实报告：  19—V4215    （只有一处）
        只把破折号换成半角连字符，两边仍然对不上（19-V-4215 vs 19-V4215）。
        去掉分隔符后统一成 19V4215，才能匹配。

    🔴 三轨分离（每一轨有各自的职责，不能混用）：
        text      原文        —— 引用展示必须是原文
        norm_text 归一化文本  —— 分词与检索
        tag_key   匹配键      —— 位号等位匹配
    """
    return re.sub(r"[-_]", "", tag).upper()


def load_equipment_tags(manifest_path) -> list[str]:
    """从 manifest 读出人工标注的设备位号池（去重保序，只取位号形态的）。

    ``manifest_path`` 可以是 path，也可以是已经 parse 好的 list。
    """
    import json
    import pathlib

    if isinstance(manifest_path, (str, pathlib.Path)):
        data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    else:
        data = manifest_path

    seen: dict[str, None] = {}
    for entry in data:
        for tag in entry.get("equipment_tags") or []:
            if is_equipment_tag(tag):
                seen.setdefault(tag, None)
    return list(seen)


def find_tag_occurrences(norm_text: str, tag: str) -> list[int]:
    """在**已归一化**的文本里找位号的所有出现位置（字符下标）。

    位置可用于 chunk 锚定：把命中位号的 chunk 提到更靠前，或在引用里高亮。

    匹配用 canonical key（去分隔符），因此 `19—V—4215` 与 `19—V4215`
    两种写法都能命中同一个 key。位置是**归一化文本**里的下标。
    """
    key = tag_key(tag)
    if not key:
        return []
    # 把 norm_text 也压成 key 形态再匹配，才能对上分隔符不一致的情况。
    # 代价：下标与原文不对应。这里返回的是 key 形态下的下标，
    # 供 chunk 粗定位用；要精确到原文位置请走原文侧的精确匹配。
    squashed = re.sub(r"[-_]", "", norm_text).upper()
    return [m.start() for m in re.finditer(re.escape(key), squashed)]
