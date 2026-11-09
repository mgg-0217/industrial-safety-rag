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

#: 多个破折号连用（19——V）压成一个。
#:
#: 🔴 **必须带字母数字上下文**，否则会误伤 HTML 注释：
#: `<!-- page:12 -->` 里的 `<!--` 和 `-->` 各含两个连字符，
#: 无约束的 `-{2,}` 会把它们压成 `<!--` → `<!-`，
#: **页码锚点被整片抹掉**（实测 51 个锚点 → 0）。
#: 加上上下文后：`19——V` 仍然压成 `19-V`（两侧是字母数字），
#: 而 `<!--` 的连字符前面是 `!`、后面是 `-`，不匹配。
_MULTI_DASH = re.compile(r"(?<=[0-9A-Za-z])-{2,}(?=[0-9A-Za-z])")


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


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """归一化 + **原文偏移映射**。

    返回 ``(norm_text, offsets)``：
    ``offsets[i]`` = 归一化文本第 i 个字符在**原文**里的下标。

    🔴 **为什么需要它**：候选题是在**归一化文本**上切条款的，
    而 chunk 的 ``char_start/char_end`` 是**原始 md** 的偏移 ——
    两者坐标系不同（归一化会改长度：全角→半角虽然等长，
    但位号破折号会把 `19——V` 压成 `19-V`，长度变了）。
    没有映射就没法把"题目出自哪段原文"落到 chunk 上，
    也就没法定义 gold chunk。

    **不变式**：``normalize_with_map(t)[0] == normalize(t)``
    （有单元测试锁住 —— 映射版与原版必须逐字一致，
    否则同一个 chunk 在两套代码里的归一化文本会不同，检索会自相矛盾）。
    """
    trans = _TRANSLATION
    chars = []
    offs = []
    for i, ch in enumerate(text):
        t = trans.get(ch, ch)
        for c in t:
            chars.append(c)
            offs.append(i)
    s = "".join(chars)
    for pat in (_TAG_DASH, _MULTI_DASH):
        s, offs = _sub_with_map(s, offs, pat, "-")
    return s, offs


def _sub_with_map(s: str, offs: list[int], pattern, repl: str):
    """在带偏移映射的字符串上做正则替换，替换字符记原匹配的**首字符**位置。"""
    out_c = []
    out_o = []
    last = 0
    for m in pattern.finditer(s):
        if m.start() < last:
            continue
        for i in range(last, m.start()):
            out_c.append(s[i])
            out_o.append(offs[i])
        out_c.append(repl)
        out_o.append(offs[m.start()])
        last = m.end()
    for i in range(last, len(s)):
        out_c.append(s[i])
        out_o.append(offs[i])
    return "".join(out_c), out_o


def normalize_ws(text: str) -> str:
    """归一化后的空白：连续空白压成一个空格，去掉行首尾空白。"""
    return re.sub(r"[ \t　]+", " ", text).strip()


# ---------------------------------------------------------------------------
# 3. 匹配用归一化（与检索用的 `norm_text` 分开）
# ---------------------------------------------------------------------------

#: 标点折叠表：全角标点 → 半角。
#:
#: 🔴 **为什么要有这一层**（K-35）：`key_point` 的"字面可达"完全取决于匹配口径 ——
#: 实测同一批 534 个 key_point：
#: 逐字匹配 **45%** / 去空白 **61%** / 再加标点折叠 **84%**。
#: 差的那部分不是数据脏，是"'处理建议：'（全角冒号）"和
#: "'处理建议:'（半角冒号）"被当成了两个不同的串。
#:
#: ⚠️ 这一层**只用于判分**，不用于检索 ——
#: 检索要保留原文的标点与位号形态（`norm_text` 那一轨）。
_MATCH_PUNCT = str.maketrans({
    "：": ":", "，": ",", "、": ",", "；": ";", "（": "(", "）": ")",
    "。": ".", "．": ".", "！": "!", "？": "?", "“": '"', "”": '"',
    "‘": "'", "’": "'", "《": "<", "》": ">", "〈": "<", "〉": ">",
    "【": "[", "】": "]", "—": "-", "－": "-", "～": "~", "〜": "~",
    "℃": "C", "…": ".",
})


def match_norm(text: str) -> str:
    """**匹配用**归一化：标点折叠 + 去空白 + 小写。

    🔴 与检索用的 `norm_text` 是**两套东西**，不能混用：
    检索要保留原文形态（展示给用户要一字不改，检索要靠位号精确匹配），
    判分要的是"这两段话说的是不是同一件事"。

    实测（K-35）：加上这一层，key_point 的字面命中率从 45% → 84%。
    """
    s = normalize(text or "")
    s = s.translate(_MATCH_PUNCT)
    s = re.sub(r"\s+", "", s)
    return s.lower()


#: 单位同义词 → 统一形态（K-43）。
#:
#: 🔴 语料里的数值带的是 **ASCII 单位**（"0.5t"、"500 mm"、"24 小时"、"15m"），
#:    而模型回答时用的是**中文单位**（"0.5吨"、"500毫米"、"15分钟"）。
#:    两者是**同一个数值**，但字面比对判成不同 → `numeric_exact_match_rate`
#:    被**系统性压低**（压低的是口径，不是模型）。
#:
#: → 数值比对前把单位**折成同一种写法**。
#: 🔴 顺序要紧：长单位必须先折（"m3" / "min" / "mm" 要在 "m" 之前），
#:    否则 "m3" 会先被折成 "米3"。
#:
#: 🔴 **2026-11-09 补复合单位**（`mg/m3` 之类）：
#: 语料里写 "10mg/m3"、模型回答写 "10 mg/m³" —— 上标变体与 ASCII 变体
#: **本来就不等价**，而 `fold_unit` 的规则是 `(?<=\d)`（单位前必须是数字），
#: 对 "mg/m3" 里的 'm' 来说前面是 'g'（字母）→ **折不掉** →
#: 锚点写 "mg/m3"、答案写 "mg/m³" → 判定不命中。
#: → ① 先把上标**统一成 ASCII**（`³`→`3`、`²`→`2`）；
#: → ② 复合单位**整串**入表（"mg/m3" 一次折成"毫克/立方米"）。
_UNIT_FOLD = [
    # 复合单位必须**最先**折（"mg/m3" 里含 "m3"，不先折会被拆）
    ("mg/m3", "毫克/立方米"), ("g/m3", "克/立方米"), ("ug/m3", "微克/立方米"),
    # 🔴 **2026-11-09 补全**：原先只有 m/mm/cm/kg/t/min/h/s，
    #    而语料里的压力/功率/频率单位全是 ASCII（"0.5MPa"），
    #    模型回答用中文（"0.5兆帕"）→ **两侧对不上** → 数值题被判不命中。
    #    ⚠️ 顺序要紧：长单位必须在短单位之前（"kpa" 在 "pa" 前、"kw" 在 "w" 前）。
    ("mpa", "兆帕"), ("kpa", "千帕"), ("mva", "兆伏安"), ("kva", "千伏安"),
    ("khz", "千赫"), ("mhz", "兆赫"), ("kw", "千瓦"), ("hz", "赫"),
    ("m3", "立方米"), ("m2", "平方米"),
    # 🔴 平方单位（"25 mm²" / "25 平方毫米" 是同一个量）
    ("mm2", "平方毫米"), ("cm2", "平方厘米"),
    ("mm", "毫米"), ("cm", "厘米"), ("kg", "千克"), ("min", "分钟"),
    ("t", "吨"), ("m", "米"), ("h", "小时"), ("s", "秒"),
]

#: 上标 → ASCII 的**统一化**（在单位折叠之前做）。
#: 🔴 "mg/m³" 与 "mg/m3" 是**同一个单位**的两种写法，
#:    但 `fold_unit` 的 `(?<=\d)` 前置条件对 'm' 不成立（前面是 'g'）→
#:    上标变体折不掉 → 两侧对不上。
#:    所以先统一成 ASCII，复合单位表里只留 ASCII 形态即可。
_SUPERSCRIPT = {"³": "3", "²": "2"}


def _unify_superscript(s: str) -> str:
    return "".join(_SUPERSCRIPT.get(c, c) for c in s)


#: 范围值的分隔符统一（`～` / `~` / `〜` → `-`）。
#: 🔴 **2026-11-09**：语料写 "35～50KPa"，模型回答写 "35-50KPa"
#: → 归一后一个是 "35～50千帕"、一个是 "35-50千帕" → **判不命中**。
#: ⚠️ 只折波浪号，不动普通连字符 —— 负数（"-75%"）不能被误改。
_RANGE_DASH = str.maketrans({"～": "-", "~": "-", "〜": "-"})


def _unify_range_dash(s: str) -> str:
    return s.translate(_RANGE_DASH)


def fold_unit(s: str) -> str:
    """把 ASCII 单位折成中文单位（K-43）。

    🔴 只在**数字之后紧跟**的单位上折 —— 否则 "T-4200" 里的 "t"、
    "5m" 里的 "m" 会被误当成单位。
    """
    out = []
    for i, ch in enumerate(s):
        out.append(ch)
    s = "".join(out)
    # 先折长的
    for a, b in _UNIT_FOLD:
        s = re.sub(r"(?<=\d)" + re.escape(a) + r"(?![a-z0-9])", b, s)
    return s


def normalize_number(s: str) -> str:
    """数值归一化（PRD 7.3：数值比对要归一化）。

    处理：全角→半角、去所有空格、单位大小写统一（``℃`` → ``C``）、
    **上标统一成 ASCII**（``m³`` → ``m3``，见 `_unify_superscript`）、
    **ASCII 单位折成中文单位**（``0.5t`` → ``0.5吨``，K-43）。
    于是 ``0.35MPa`` / ``0.35 MPa`` / ``０．３５ＭＰａ`` / ``0.5t`` / ``0.5吨``
    比对时都是同一个串。
    """
    s = normalize_digits(s or "")
    s = s.replace("℃", "C").replace("°C", "C")
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    s = _unify_superscript(s)
    s = _unify_range_dash(s)
    s = fold_unit(s)
    return s


def match_num_norm(text: str) -> str:
    """**数值比对**用的归一化：`match_norm` + 单位折叠（K-43）。

    🔴 答案侧也必须折单位 ——
    只折 key 侧的话，"答案写 0.5t、key 写 0.5吨"这种组合仍然对不上。

    🔴 **2026-11-09**：也要先统一上标（``m³`` → ``m3``），
    否则锚点写 ``mg/m3``、答案写 ``mg/m³`` → 对不上。
    """
    return fold_unit(_unify_range_dash(_unify_superscript(match_norm(text or ""))))


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
