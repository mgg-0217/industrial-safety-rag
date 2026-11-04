"""切分流水线 —— PRD 3.1 六种策略的共用框架。

**这个模块只做三件事**：解析 → 切分 → 组装。
检索、评测、版本过滤都在别的模块（PRD 第 4、7 节），这里不碰。

## 三条贯穿全模块的硬约束（都来自 PRD，别自己另立一套）

1. **chunk schema 统一**（PRD 3.1）：所有策略共用同一套字段，
   策略特有的写进 `strategy_meta`。`page_no` **只可能是 int 或 null**，
   跨页块记 `null` 并把页码列表写进 `strategy_meta.cross_page`。
2. **跨页合并只有一个权威**（PRD 3.2）：互斥情形表，按序判断、命中即止。
   策略 2 / 3 / 5 一律引用它，**不许各写一套** ——
   chunk 边界直接决定 M4 的六策略对比数字，两套规则会让数字不可复现。
3. **表格的"表头"要自校验**（PRD 3.1 策略 4）：
   抽取层无条件把 pdfplumber 的 `rows[0]` 当表头、**无条件注入** `| --- |` 分隔行，
   所以"有分隔行"是合成产物（必然成立、零信息量），
   "首行像表头"才有信息量 —— 而它对约 1/3 的块是假的。
   **首行不是表头时按"散文表"处理**：不重复首行、`header=null`、`is_pseudo=true`、
   不参与表头签名合并。

## 段落粒度为什么按 `source_format` 分

- `html_webpage`：正文由 `<p>` 边界决定，抽取后**每行就是一个段落**，行间无空行；
- `pdf_document`：段落之间有空行，但**块内**的换行是排版换行，
  同一段会被拆成多行 —— 按空行分块才对。

实测（26 份 md）：HTML 源的空行数全为 0，PDF 源才有空行。
若对 HTML 也按空行分块，整份文档会变成**一个**段落 —— 结构感知切分直接失效。
"""
import pathlib
import re

# ---------------------------------------------------------------- 基础常量

PAGE_MARK = re.compile(r"^<!--\s*page:(\d+)\s*-->$")

# PRD 3.1 策略 2 的标题识别正则（表格行不参与识别）
RE_H1_CN = re.compile(r"^[一二三四五六七八九十百]+、")
RE_H1_ARTICLE = re.compile(r"^第[一二三四五六七八九十百千]+条")
RE_H_PAREN = re.compile(r"^（[一二三四五六七八九十]+）")

# 🔴 数字编号标题：**不能直接用 PRD 写的 `^\d+(\.\d+)*\s`**。
# 实测 26 份 md 里数字开头的行共 2627 行，形态分布是：
#     "1.粗产品缓冲罐19—V—4215"（分隔符是点，后面直接跟中文）—— 381 行，真标题
#     "1、内容"（顿号）                                            —— 262 行，真标题
#     "13 企业应建立…"（数字+空格+正文）                            —— 980 行，**多数是正文**
#     "2024年1月…"（无分隔符）                                     —— 913 行，**是正文不是标题**
# PRD 那条正则（要求数字后有空白）只覆盖到 980+11 行，且**把正文数字当标题**；
# 而"1.粗产品缓冲罐"这种真标题反而匹配不上。
# 所以拆成两条：
RE_H_NUM_STRONG = re.compile(r"^\d+(\.\d+)*[、.．][^\s]")   # 编号 + 分隔符 + 紧跟非空白
RE_H_NUM_WEAK = re.compile(r"^\d+(\.\d+)*\s+\S")           # 编号 + 空白（PRD 原版）
RE_H_NUM = re.compile(r"^\d+(\.\d+)*")

# 弱标题的额外约束：行要短、且不含句读（否则是正文）
WEAK_HEADING_MAX_LEN = 40
SENT_PUNCT = "。！？；：，、"

HEADING_RES = (RE_H1_CN, RE_H1_ARTICLE, RE_H_PAREN)

# 策略 1 的参数（PRD 3.1 写死）
STRATEGY1_LEN = 512
STRATEGY1_OVERLAP = 64

# 策略 4 的参数
STRATEGY4_ROWS = 8
STRATEGY4_ROW_OVERLAP = 1

# 策略 5 的参数
STRATEGY5_TARGET = 400
STRATEGY5_MIN = STRATEGY5_TARGET // 3


def is_heading(line):
    """是否标题行。**表格行永远不算标题**（PRD 策略 2 的已知代价与缓解）。

    「表格里恰好有一行以数字开头」会被误判为标题 ——
    缓解办法就是先识别表格边界，再在非表格行里找标题（本函数只对非表格行调用）。
    """
    s = line.strip()
    if not s or s.startswith("|"):
        return False
    if any(r.match(s) for r in HEADING_RES):
        return True
    if RE_H_NUM_STRONG.match(s):
        # 强标题：编号后紧跟非空白。但 "13.1 m3" 这类正文数值不能算标题，
        # 所以还要看长度 —— 标题短，正文长。
        return len(s) <= 60
    if RE_H_NUM_WEAK.match(s):
        # 弱标题：数字 + 空白。要求短且不含句读，否则是正文行
        return len(s) <= WEAK_HEADING_MAX_LEN and not any(p in s for p in SENT_PUNCT)
    return False


def heading_level(line):
    """标题层级（用于 section_path）。分不出来时返回 0。"""
    s = line.strip()
    if not s or s.startswith("|"):
        return 0
    if RE_H1_CN.match(s) or RE_H1_ARTICLE.match(s):
        return 1
    if RE_H_PAREN.match(s):
        return 2
    if RE_H_NUM_STRONG.match(s) or RE_H_NUM_WEAK.match(s):
        # 3.2.1 比 3.2 深
        return min(s.count(".") + 1, 4)
    return 0


# ---------------------------------------------------------------- 文档解析


class Block:
    """解析后的最小单元：一个段落（多行）或一个表格块。"""

    __slots__ = ("kind", "text", "lines", "page_no", "char_start", "char_end",
                 "rows", "header", "header_like", "is_pseudo", "caption", "pages",
                 "line_pages", "line_offsets", "row_line_idx")

    def __init__(self, kind, text, lines, page_no, char_start, char_end):
        self.kind = kind          # "para" | "table"
        self.text = text
        self.lines = lines
        self.page_no = page_no
        self.char_start = char_start
        self.char_end = char_end
        # 表格相关
        self.rows = []            # **数据行**（不含表头）
        self.header = None        # **首行原样**（未经校验；校验结果在 header_like）
        self.header_like = False  # 首行是否像表头
        self.is_pseudo = False    # 伪表（首行不是表头）
        self.caption = None
        self.pages = [page_no] if page_no is not None else []
        # 🔴 **行级页码**：块被拆成子块时，页码要能精确到行，
        # 否则一个跨页合并的块被拆成 5 个子块、5 个子块全被标成"跨页"。
        # 行级归属才认得出"这个子块的内容其实全在第 N 页"。
        self.line_pages = []
        # 🔴 **行级字符偏移**：每个字符在 md 原文里的位置。
        # PRD 6.3 承诺 char_start/char_end 是"可回到 md 的**精确位置**"，
        # 而块级偏移做不到 —— 块被拆成子块、或跨页合并后，
        # 块级偏移会偏早（子块继承了父块的起点）。
        self.line_offsets = []
        # 表格：数据行 -> 行下标（表格块内）
        self.row_line_idx = []

    def __repr__(self):
        return f"<Block {self.kind} p{self.page_no} {len(self.text)}字>"


def strip_front_matter(text):
    """去掉 md 开头的 YAML front-matter，返回 (正文, 偏移量)。"""
    if not text.startswith("---"):
        return text, 0
    m = re.search(r"\n---\n", text[3:])
    if not m:
        return text, 0
    end = 3 + m.end()
    return text[end:], end


def _is_separator_line(ln):
    """是否是 `| --- |` 分隔行（抽取层无条件注入的合成产物）。"""
    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
    cells = [c for c in cells if c != ""]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def parse_table(lines):
    """解析表格块的行 -> (数据行, 首行)。

    🔴 第 2 行是 `| --- |` 分隔行，**抽取层无条件注入**（合成产物），
    它不代表"这一块有表头"。这里把它剥掉，**首行是否表头另做自校验**。
    """
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return [], []
    header = rows[0]
    data = rows[1:]
    # 剥掉分隔行（只认全由 - 组成的单元格）
    if data and _is_separator_line(lines[1]):
        data = data[1:]
    return data, header


# 表头自校验：首行平均单元格长度的上限（绝对阈值，不是相对数据行）
HEADER_MAX_AVG_LEN = 15


def header_looks_like_header(header, data_rows, rows=None):
    """首行是否**像表头**（PRD 3.1 策略 4 的自校验启发式）。

    判据（对齐 PRD 的口径"首行短且无句读"）：
      1. **键值对表无表头概念**（PRD 明说：`kaiping_zuoyepiao` / `gaoqing` 的
         两块都是"键值对形式，无表头概念"）→ 整表是键值对形式就不认表头；
      2. 首行**不含句读标点**（。！？；：，、）—— 真表头是列名，不是句子；
      3. 首行的**平均单元格长度** ≤ 15 —— 真表头短，散文行长。

    🔴 为什么用**绝对**阈值而不用"比数据行短"：
    实测两种口径差别很大。键值对式的表（如
    `类别 | 文件信息/ | 存放地点 | 更新时间 | 文件负责人持有人/`）
    数据行只有 2–3 字，比表头还短 —— 用相对判据会把**真表头**误判成伪表。
    绝对阈值能同时分開两类：真表头首行均长 3.5–10.3，
    散文首行均长 18.5–36。

    🔴 为什么必须校验：`pdf_to_md.py::table_to_md()` 无条件把 pdfplumber 的
    `rows[0]` 当表头写出。PRD 实测 122 块里只有 78 块（64%）首行真是表头，
    `mem_xianshi_shouce` 的 41 块里只有 11 块像表头（其余 30 块首行是正文散文）——
    不校验就会给 30 个散文块安上假表头，检索时把正文当列名重复一遍。

    ⚠️ **与 PRD 人工判定的差异（实测，如实记录）**：
    本启发式在 `AQ_T3034` 上判 39 块像表头，PRD 的人工判定是 31 块；
    `kaiping` / `gaoqing` 各 1 块（键值对表，已由特判排除 → 0，与 PRD 一致）；
    `guide_liubei_shouce` 的 1 块孤立残留行本启发式判 0（无数据行可比）。
    抽查差异样本后，AQ_T3034 多判的那几块首行**确实是真表头**
    （`序号|具体内容|实施要求`、`项目|完成|姓名|日期|备注`），
    倾向于是 PRD 的人工判定更严。**不为对齐数字而扭曲判据** ——
    误判方向是保守的：宁可漏认表头（只是不重复表头），
    也不把散文当表头重复一遍（那是往检索库里灌噪声）。
    """
    if not header or not data_rows:
        return False
    if rows is not None and is_key_value_table(rows):
        return False
    joined = "".join(header)
    if any(p in joined for p in "。！？；：，、"):
        return False
    avg = sum(len(c) for c in header) / len(header)
    return avg <= HEADER_MAX_AVG_LEN


def _groups_in_row(row):
    """行内**连续非空段**的个数。

    "标签 | | | 值" → 2 段；"序号 | 排查内容 | 排查依据"（内容相邻）→ 1 段。
    """
    n = 0
    prev_empty = True
    for c in row:
        empty = not c.strip()
        if not empty and prev_empty:
            n += 1
        prev_empty = empty
    return n


def is_key_value_table(rows):
    """是否"标签: 值"式的键值对表（作业票类）。

    PRD 3.1 策略 4 写明这类表**不按行切**（切了就看不懂），整表一个 chunk。

    ## 判据：行内内容横向散成**多段**

    "标签 空 空 值" 这种布局里，一行里有 ≥2 段互不相邻的非空内容 ——
    切掉其他行之后，剩下的"标签"和"值"就分开了，读者看不懂。

    ## 🔴 为什么不能只看"非空格占比低"

    PDF 表格的**单元格内容会跨行折行**，pdfplumber 把一个逻辑行
    拆成多条物理行（每行只剩 1–2 个非空格）。

    实测 `mem_2019_78` 的 17 张排查表（`序号|排查内容|排查依据`，50–76 行）：
    每行只有 1–3 列有内容，非空占比 0.40–0.49，各行非空格数差异 2 ——
    用"占比低 + 差异大"会把它们**全部误判成键值对**，
    于是策略 4 对**最表格密集的文档**完全不切行（PRD 的表格切分能力作废）。

    折行表与键值对表的真区别在**横向分布**：
    - 折行表：内容集中在**相邻**列 → 1 段；
    - 键值对表：内容**散在**不相邻的列 → ≥2 段。

    实测对照：排查表 1 段（数据表）、作业票 3–4 段（表单）、
    `gaoqing` 排查表 2–3 段（表单，与 PRD 的人工判定一致）。
    """
    if len(rows) < 3:
        return False
    n_cols = max(len(r) for r in rows)
    if n_cols < 4:
        # 少于 4 列放不下"标签 空 空 值"的散列布局
        return False
    groups = sorted(_groups_in_row(r) for r in rows)
    return groups[len(groups) // 2] >= 2


def parse_md(path, source_format="pdf_document"):
    """把 md 解析成 Block 列表。

    `source_format` 决定段落粒度（见模块 docstring）。
    """
    path = pathlib.Path(path)
    raw = path.read_text(encoding="utf-8")
    text, offset = strip_front_matter(raw)
    return _scan_blocks(text.split("\n"), offset, source_format)


def _scan_blocks(lines, offset, source_format):
    """线性扫描成 Block 列表。"""
    out = []
    cur_page = None
    pos = offset
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        stripped = ln.strip()
        pm = PAGE_MARK.match(stripped)
        if pm:
            cur_page = int(pm.group(1))
            pos += len(ln) + 1
            i += 1
            continue
        if not stripped:
            pos += len(ln) + 1
            i += 1
            continue
        if stripped.startswith("|"):
            start = pos
            tbl = []
            tbl_offs = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                tbl_offs.append(pos)
                pos += len(lines[i]) + 1
                i += 1
            body = "\n".join(tbl)
            b = Block("table", body, tbl, cur_page, start, start + len(body))
            b.line_pages = [cur_page] * len(tbl)
            b.line_offsets = tbl_offs
            data, header = parse_table(tbl)
            b.rows = data
            b.header = header
            b.header_like = header_looks_like_header(header, data, data + [header])
            b.is_pseudo = not b.header_like
            # 数据行 -> 行下标（首行是首行，其后可能是分隔行）
            first_data = 1
            if len(tbl) > 1 and _is_separator_line(tbl[1]):
                first_data = 2
            b.row_line_idx = list(range(first_data, first_data + len(data)))
            out.append(b)
            continue
        # 段落：按 source_format 决定粒度
        start = pos
        if source_format == "html_webpage":
            body = ln
            pos += len(ln) + 1
            i += 1
            body_lines = [ln]
            body_offs = [start]
        else:
            body_lines = []
            body_offs = []
            while i < n and lines[i].strip() and not lines[i].strip().startswith("|") \
                    and not PAGE_MARK.match(lines[i].strip()):
                body_lines.append(lines[i])
                body_offs.append(pos)
                pos += len(lines[i]) + 1
                i += 1
            body = "\n".join(body_lines)
        b = Block("para", body, body_lines, cur_page, start, start + len(body))
        b.line_pages = [cur_page] * len(body_lines)
        b.line_offsets = body_offs
        out.append(b)
    return out


# ---------------------------------------------------------------- 跨页合并

# PRD 3.2 的互斥情形表，**顺序判断、命中即止**。
# 情形 4 必须排在情形 3 之后 —— 标题行通常不以 。！？ 结尾，
# 若把 3 提前，标题+正文跨页会被误判成"完整句"而不合并。
MERGE_ENDS_SENTENCE = tuple("。！？")


def decide_cross_page(prev_block, next_block):
    """判定相邻两块（跨页）是否合并。

    返回 (merge: bool, reason: str)。

    | # | 页 N 末尾 | 页 N+1 开头 | 判定 |
    |---|---|---|---|
    | 1 | 表格边界 | 任意 | 不合并 |
    | 2 | 任意 | 标题行 | 不合并 |
    | 3 | 完整句 | 任意 | 不合并 |
    | 4 | 标题行 | 该标题的下文 | 合并 |
    | 5 | 非完整句 | 非标题 | 合并 |
    """
    # 情形 1：页 N 末尾是表格边界
    if prev_block is not None and prev_block.kind == "table":
        return False, "情形1：表格边界不合并"
    # 情形 2：页 N+1 开头是标题行
    if next_block is not None and next_block.lines and is_heading(next_block.lines[0]):
        return False, "情形2：新章节标题不合并"
    if prev_block is None or next_block is None:
        return False, "无前块"
    # 情形 3：页 N 末尾是完整句
    prev_text = prev_block.text.rstrip()
    if prev_text.endswith(MERGE_ENDS_SENTENCE):
        return False, "情形3：完整句不合并"
    # 情形 4：页 N 末尾是标题行
    if prev_block.lines and is_heading(prev_block.lines[-1]):
        return True, "情形4：标题与其正文合并"
    # 情形 5：非完整句 + 非标题
    return True, "情形5：句子被分页切断，合并"


def merge_cross_page(blocks):
    """按 3.2 的规则把跨页的相邻块合并。

    只在**页码不同**的相邻块之间判定（HTML 源 page_no 全为 null，不合并）。
    合并后 `page_no` 记 null，页码列表写进 `strategy_meta.cross_page`。

    🔴 **顺序很重要**：先判 1、2（表格/新章节），再判 3（完整句），
    最后才是合并。情形 4 必须排在情形 3 之后 —— 标题行通常不以 。！？ 结尾，
    若把 3 提前，"标题+正文跨页"会被误判成"完整句"而不合并。
    """
    if not blocks:
        return []
    out = [blocks[0]]
    for b in blocks[1:]:
        prev = out[-1]
        if prev.page_no is None or b.page_no is None or prev.page_no == b.page_no:
            out.append(b)
            continue
        merge, _reason = decide_cross_page(prev, b)
        if not merge:
            out.append(b)
            continue
        # 合并：文本拼接，页码记 null（不能只报一页）
        prev.text = prev.text + "\n" + b.text
        prev.lines = prev.lines + b.lines
        lp = prev.line_pages or [prev.page_no] * (len(prev.lines) - len(b.lines))
        lb = b.line_pages or [b.page_no] * len(b.lines)
        prev.line_pages = list(lp) + list(lb)
        lo = prev.line_offsets or []
        prev.line_offsets = list(lo) + list(b.line_offsets or [])
        prev.row_line_idx = list(prev.row_line_idx or []) + \
            [i + len(prev.lines) - len(b.lines) for i in (b.row_line_idx or [])]
        prev.char_end = b.char_end
        prev.page_no = None
        prev.pages = sorted({p for p in (prev.pages + b.pages) if p is not None})
    return out


# ---------------------------------------------------------------- 骨架切分

class Skeleton:
    """把 Block 序列整理成"标题 + 其下内容"的骨架。

    策略 2 / 3 都基于它：**先把行级的标题切出来**，再按层级组织 section_path。
    这一步不做，section_path 就是空的，引用也无法指向章节。
    """

    def __init__(self):
        self.items = []       # [{kind, heading, level, blocks, page_no, ...}]
        self.stack = []       # 当前标题栈 [(level, text)]

    def push(self, block):
        """把一个 Block 挂到当前骨架位置上。"""
        if not self.items:
            self.items.append({"heading": None, "level": 0, "blocks": []})
        self.items[-1]["blocks"].append(block)

    def heading(self, level, text, block):
        """遇到标题：开一个新段，并把标题本身作为第一个 block。"""
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        self.items.append({
            "heading": text,
            "level": level,
            "blocks": [block],
            "section_path": [t for _l, t in self.stack],
        })

    def path(self):
        """当前 section_path。"""
        return [t for _l, t in self.stack]


def build_skeleton(blocks):
    """把 Block 序列整理成骨架。

    🔴 段落块可能**内部含标题行**（PDF 抽取的段落是空行分隔的多行块，
    标题和它的正文可能被放进同一块）。所以要先在块内按行拆：
    遇到标题行就断开，剩下的行归入当前标题。
    """
    sk = Skeleton()
    for b in blocks:
        if b.kind == "table":
            sk.push(b)
            continue
        lps = b.line_pages or [b.page_no] * len(b.lines)
        los = b.line_offsets or []
        buf = []       # [(行文本, 页码, md 偏移)]
        for li, ln in enumerate(b.lines):
            lp = lps[li] if li < len(lps) else b.page_no
            off = los[li] if li < len(los) else None
            if is_heading(ln):
                if buf:
                    sk.push(_sub_block(b, buf))
                    buf = []
                lv = heading_level(ln)
                sk.heading(lv, ln.strip(), _sub_block(b, [(ln, lp, off)]))
            else:
                buf.append((ln, lp, off))
        if buf:
            sk.push(_sub_block(b, buf))
    # 没遇到任何标题时，第一项承载全部内容
    return sk.items


def _sub_block(parent, pairs):
    """从父块切出的子块 —— **页码与偏移都按行归属**。

    🔴 三个坑叠在一起：
    ① 跨页合并过的块 `page_no` 是 None、页码只存在 `pages` 里；
       `Block.__init__` 只按 page_no 初始化 pages → 子块的 pages 会变成空列表。
    ② 若子块直接继承**父块**的 pages，一个跨页块被拆成 5 个子块时，
       5 个子块全被标成"跨页" —— 但其中几个的内容其实全在第 N 页。
    ③ 若子块直接继承**父块**的 char_start，标题子块的起点会**偏早**
       （父块起点到标题之间还有别的正文）→ PRD 6.3 的"精确位置"落空。
    → 三项都按**行级**归属。
    """
    lines = [ln for ln, _lp, _off in pairs]
    nb = Block("para", "\n".join(lines), list(lines), parent.page_no,
               parent.char_start, parent.char_end)
    nb.pages = sorted({lp for _ln, lp, _off in pairs if lp is not None})
    nb.line_pages = [lp for _ln, lp, _off in pairs]
    nb.line_offsets = [off for _ln, _lp, off in pairs]
    offs = [(off, _ln) for _ln, _lp, off in pairs if off is not None]
    if offs:
        nb.char_start = offs[0][0]
        nb.char_end = offs[-1][0] + len(offs[-1][1])
    return nb


# ---------------------------------------------------------------- 六种策略


def strategy1_fixed(blocks, doc_ctx):
    """策略 1 · 固定长度（L=512, O=64）。朴素基线。

    🔴 **表格不处理**（PRD 明确）："表格行被拦腰切断，一行会被切成两个 chunk，
    各自半头，答案必然错" —— 这句写在 PRD 的**代价**栏里，
    意思是策略 1 **故意**不处理表格，作为基线缺陷来对比结构策略的收益。
    所以表格文本**照常参与字符切分**，不做任何表格专用处理。
    （曾误读成"表格要跳过"，那样基线反而变干净，就失去对照意义了。）

    🔴 **滑窗必须是"文档级"的**：块长恒为 L、块间重叠 O，
    而不是"每个段落各自切 L 再拼起来"。
    后者会出现两种偏差：① 段落边界处各切一刀 → 边界处多出两个半块；
    ② 拼接时累加 → 实际块长可达 2L（实测 p95=946 > L=512）。
    PRD 对策略 1 的定义是"定长块 + 重叠"，块长就该是 L。
    """
    stream, page_at, off_at = _flatten_blocks(blocks)
    n = len(stream)
    if n == 0:
        return []

    chunks = []
    step = max(STRATEGY1_LEN - STRATEGY1_OVERLAP, 1)
    i = 0
    while i < n:
        piece = stream[i:i + STRATEGY1_LEN]
        if piece.strip():
            meta = {"n_chars": len(piece)}
            pages = [p for p in page_at[i:i + STRATEGY1_LEN] if p is not None]
            if pages:
                # 多数票：PRD 策略 1「page_no = 块内出现过的页码（取多数票）」
                common = max(set(pages), key=pages.count)
                meta["pages"] = sorted(set(pages))
                page = common
            else:
                page = None
            offs = [o for o in off_at[i:i + STRATEGY1_LEN] if o is not None]
            start = offs[0] if offs else None
            end = (offs[-1] + 1) if offs else None
            chunks.append({"text": piece, "page_no": page, "meta": meta,
                           "clause_no": None, "section_path": [],
                           "char_start": start, "char_end": end})
        i += step
    return chunks


def _char_pages(b):
    """块内每个字符的页码（行级）。

    合并块的 `page_no` 是 None，但行级页码还在 —— 用它才能给
    块内的每个字符找到它物理上在哪一页。
    """
    lps = b.line_pages
    if not lps:
        return [b.page_no] * len(b.text)
    out = []
    for i, ln in enumerate(b.lines):
        lp = lps[i] if i < len(lps) else b.page_no
        out.extend([lp] * len(ln))
        out.append(lp)          # 行尾的换行符
    if len(out) < len(b.text):
        out.extend([b.page_no] * (len(b.text) - len(out)))
    return out[:len(b.text)]


def _char_offsets(b):
    """块内每个字符在 md 原文里的偏移（行级）。

    🔴 不能用 `char_start + j`：跨页合并过的块，块内字符与 md 字符
    已经错位（合并时插了 `\\n`，块与块之间还有空行），
    线性外推会越走越偏。必须逐行按真实行首偏移推。
    """
    los = b.line_offsets
    if not los:
        base = b.char_start
        if base is None:
            return [None] * len(b.text)
        return [base + j for j in range(len(b.text))]
    out = []
    for i, ln in enumerate(b.lines):
        base = los[i] if i < len(los) else None
        if base is None:
            out.extend([None] * (len(ln) + 1))
            continue
        out.extend(range(base, base + len(ln)))
        out.append(base + len(ln))     # 行尾的换行符
    if len(out) < len(b.text):
        out.extend([None] * (len(b.text) - len(out)))
    return out[:len(b.text)]


def _md_offset(b, rel):
    """块内相对字符偏移 -> md 原文偏移。"""
    los = b.line_offsets
    if not los:
        return (b.char_start + rel) if b.char_start is not None else None
    pos = 0
    for i, ln in enumerate(b.lines):
        if rel < pos + len(ln) + 1:
            base = los[i] if i < len(los) else None
            if base is None:
                return (b.char_start + rel) if b.char_start is not None else None
            return base + (rel - pos)
        pos += len(ln) + 1
    return (b.char_start + rel) if b.char_start is not None else None


def _line_span(b, rel_start, rel_end):
    """块内相对区间 -> md 原文字符区间（按**行首偏移**算）。

    🔴 不能用 `起点 + 长度`：跨页合并过的块，块内文本与 md 之间
    隔着若干空行（合并时空行被吸收了），线性外推会偏。
    行首偏移是 md 里的真实位置，逐行相加不会错。
    """
    los = b.line_offsets or []
    if not los:
        if b.char_start is None:
            return None, None
        return b.char_start + rel_start, b.char_start + rel_end
    li_start = None
    li_end = None
    pos = 0
    for i, ln in enumerate(b.lines):
        nxt = pos + len(ln) + 1
        if li_start is None and rel_start < nxt:
            li_start = i
        if rel_end <= nxt:
            li_end = i
            break
        pos = nxt
    if li_start is None:
        li_start = 0
    if li_end is None:
        li_end = max(len(b.lines) - 1, 0)
    lo = los[li_start] if li_start < len(los) else None
    hi = (los[li_end] + len(b.lines[li_end])) if li_end < len(los) else None
    if lo is None or hi is None:
        return (b.char_start, b.char_end) if b.char_start is not None else (None, None)
    return lo, hi


def _flatten_blocks(blocks):
    """把块序列摊平成 (字符流, 每字符页码, 每字符在 md 原文里的偏移)。

    块之间用 `\\n` 连接（与 md 里的换行一致）。
    分隔符的偏移记 None —— 它不是原文里的实体字符。
    """
    chars = []
    page_at = []
    off_at = []
    for bi, b in enumerate(blocks):
        if bi:
            chars.append("\n")
            page_at.append(b.page_no)
            off_at.append(None)
        cp = _char_pages(b)
        co = _char_offsets(b)
        for j in range(len(b.text)):
            chars.append(b.text[j])
            page_at.append(cp[j])
            off_at.append(co[j])
    return "".join(chars), page_at, off_at


def strategy2_structural(blocks, doc_ctx):
    """策略 2 · 结构感知（按标题层级切分）。

    `section_path` = 完整标题路径，`clause_no` = 路径最后一段。
    表格块作为**不可分割单元**随其所属章节走。

    🔴 **一个标题 + 它的全部内容 = 一个 chunk**（节级）。
    曾经把标题行单独切成一个 chunk，结果是 4694 个 chunk、平均 70 字 ——
    碎到没法检索（"一、事故基本情况"这种标题单独成块，对召回毫无贡献，
    还会把同一个 chunk 的正文劈成好几份）。
    """
    blocks = merge_cross_page(blocks)
    items = build_skeleton(blocks)
    chunks = []
    for it in items:
        path = it.get("section_path") or []
        clause = path[-1] if path else None
        body = it["blocks"]
        if not body:
            continue
        text_parts = []
        pages = []
        start = None
        end = None
        table_meta = None
        for b in body:
            text_parts.append(b.text)
            pages.extend(b.pages or ([b.page_no] if b.page_no is not None else []))
            if start is None:
                start = b.char_start
            end = b.char_end
            if b.kind == "table" and table_meta is None:
                table_meta = {"table_id": None, "row_start": 0,
                              "row_end": max(len(b.rows) - 1, 0),
                              "n_rows": len(b.rows),
                              "n_cols": max((len(r) for r in b.rows), default=0),
                              "header": None, "is_pseudo": b.is_pseudo,
                              "caption": b.caption, "whole": True}
        text = "\n".join(text_parts)
        if not text.strip():
            continue
        meta = {"n_chars": len(text), "n_blocks": len(body)}
        if len(set(pages)) > 1:
            meta["cross_page"] = sorted(set(pages))
            page = None
        else:
            page = pages[0] if pages else None
        chunks.append({
            "text": text, "page_no": page, "meta": meta,
            "clause_no": clause, "section_path": list(path),
            "char_start": start, "char_end": end,
            "table": table_meta,
        })
    return chunks


def strategy3_clause(blocks, doc_ctx):
    """策略 3 · 条款级。

    与策略 2 的区别：策略 2 按**标题层级**切，策略 3 按**编号单元**切，
    且把连续多个同层小节合并到父级标题下（一条一个 chunk）。

    实现：把骨架里的"标题 + 其下内容"作为一个条款单元，
    编号层取**最具体的那层**（`clause_no` = 编号 + 标题）。
    """
    blocks = merge_cross_page(blocks)
    items = build_skeleton(blocks)
    chunks = []
    for it in items:
        path = it.get("section_path") or []
        heading = it.get("heading")
        body_blocks = it["blocks"]
        if heading is None:
            # 第一个标题之前的内容：按原样切
            for b in body_blocks:
                if b.text.strip():
                    chunks.append(_mk_clause_chunk(b, [], b.text))
            continue
        # 标题本身 + 其后的内容块合并成一个条款 chunk
        text_parts = []
        pages = []
        start = None
        end = None
        for b in body_blocks:
            text_parts.append(b.text)
            # 🔴 跨页合并过的块 page_no 会被置 None，但 pages 列表还在 ——
            # 只看 page_no 会把它的页码整个丢掉，跨页块就统计不出来了。
            pages.extend(b.pages or ([b.page_no] if b.page_no is not None else []))
            if start is None:
                start = b.char_start
            end = b.char_end
        text = "\n".join(text_parts)
        # 最具体的编号：clause_no = 标题原文
        clause = heading
        meta = {"n_chars": len(text), "n_blocks": len(body_blocks)}
        if len(set(pages)) > 1:
            meta["cross_page"] = sorted(set(pages))
            page = None
        else:
            page = pages[0] if pages else None
        chunks.append({
            "text": text, "page_no": page, "meta": meta,
            "clause_no": clause, "section_path": list(path),
            "char_start": start, "char_end": end,
        })
    return chunks


def _mk_clause_chunk(b, path, text):
    """单个块构成的条款 chunk（标题之前的内容）。"""
    meta = {"n_chars": len(text)}
    pgs = b.pages or ([b.page_no] if b.page_no is not None else [])
    if len(set(pgs)) > 1:
        # 跨页合并过的块：page_no 已是 None，页码在 pages 里
        meta["cross_page"] = sorted(set(pgs))
    return {
        "text": text, "page_no": b.page_no, "meta": meta,
        "clause_no": path[-1] if path else None, "section_path": list(path),
        "char_start": b.char_start, "char_end": b.char_end,
    }


def strategy4_table(blocks, doc_ctx):
    """策略 4 · 表格切分（按行切，表头随行）。

    参数：每个 chunk 含 1 行表头 + K=8 行数据，行组之间重叠 1 行。

    🔴 **表头必须自校验**（`header_looks_like_header`）：
    抽取层无条件把 pdfplumber 的 `rows[0]` 当表头，122 块里只有 78 块（64%）
    首行真是表头。伪表按**散文表**处理：不重复首行、`header=null`、
    `is_pseudo=true`、不参与表头签名合并。

    🔴 **作业票类的键值对表不按行切**（切了就看不懂），整表一个 chunk，
    标 `strategy_meta.table_whole = true`。
    """
    chunks = []
    for b in blocks:
        if b.kind != "table":
            continue
        rows = b.rows
        if not rows:
            continue
        if is_key_value_table(rows):
            chunks.append({
                "text": b.text, "page_no": b.page_no,
                "meta": {"n_chars": len(b.text), "table_whole": True,
                         "table_is_pseudo": b.is_pseudo},
                "clause_no": None, "section_path": [],
                "char_start": b.char_start, "char_end": b.char_end,
                "table": {"table_id": None, "row_start": 0,
                          "row_end": max(len(rows) - 1, 0),
                          "n_rows": len(rows), "n_cols": max(len(r) for r in rows),
                          "header": None, "is_pseudo": b.is_pseudo,
                          "caption": b.caption, "whole": True},
            })
            continue
        # 🔴 表头与数据行是**分开存的**（`b.header` / `b.rows`）——
        # 早前这里写成 `header = rows[0]`（把第一条**数据行**当表头），
        # 结果真表头整个丢失、第一条数据行被重复一遍。
        # PRD 3.1 策略 4 要求"表头行原样存进 chunk 的 text 开头"，
        # 行级答案（"导线型号"）必须带表头才成立。
        header = b.header if b.header_like else None
        data = b.rows
        # 伪表：首行是正文，**不能重复**（重复一次就等于把它当成表头用）
        if not data:
            continue
        K = STRATEGY4_ROWS
        step = max(K - STRATEGY4_ROW_OVERLAP, 1)
        ridx = b.row_line_idx or []
        i = 0
        while i < len(data):
            group = data[i:i + K]
            lines = []
            if header:
                lines.append(_row_to_line(header))
            lines.extend(_row_to_line(r) for r in group)
            text = "\n".join(lines)
            meta = {"n_chars": len(text)}
            if b.header_like:
                meta["table_header_sig"] = _header_signature(header)
            # 🔴 偏移要落在**实际切出来的那些行**上，不是整张表 ——
            # PRD 6.3 承诺 char_start/char_end 是"精确位置"。
            c_start, c_end = _row_span(b, ridx, i, len(group), bool(header))
            chunks.append({
                "text": text, "page_no": b.page_no, "meta": meta,
                "clause_no": None, "section_path": [],
                "char_start": c_start, "char_end": c_end,
                "table": {"table_id": None, "row_start": i,
                          "row_end": i + len(group) - 1,
                          "n_rows": len(group),
                          "n_cols": max(len(r) for r in group),
                          "header": header, "is_pseudo": b.is_pseudo,
                          "caption": b.caption, "whole": False},
            })
            if i + K >= len(data):
                break
            i += step
    return chunks


def strategy4b_table_whole(blocks, doc_ctx):
    """策略 4b · **表格整表一块**（PRD 风险 R8 的预案）。

    ## 为什么要有这个策略

    PRD 3.1 策略 4 把表格按行切（1 行表头 + K=8 行数据）。
    实测（M10 诊断）这会**把一张表切碎**：
    表格类题目的"内容覆盖度"（上下文里带进多少 gold 区间）只有 **0.390** ——
    召回了 gold 块也只带进四成表内容，模型看到的是半张表。

    PRD 风险 R8 早就写了预案：
    「若表格类题目召回仍然低，考虑**整表一个 chunk** 作为策略 4b 单独评测」。

    ## 与策略 4 的差别

    | | 策略 4（按行切） | 策略 4b（整表） |
    |---|---|---|
    | 块数 | 多（一张表 N 块） | 一张表 1 块 |
    | 表头 | 每块开头 1 次 | 整表 1 次 |
    | 行级可解释性 | ❌ 块内第 5 行没有列名 | ✅ 列名与数据同行 |
    | 块长 | 均长 294 字符 | **可能上千字符**（长表） |

    🔴 **不是替代策略 4，是并列的对照项** ——
    块长变大会不会拖慢检索、会不会挤掉正文，要实测。
    """
    chunks = []
    for b in blocks:
        if b.kind != "table":
            continue
        rows = b.rows
        if not rows:
            continue
        header = b.header if b.header_like else None
        lines = []
        if header:
            lines.append(_row_to_line(header))
        lines.extend(_row_to_line(r) for r in rows)
        text = "\n".join(lines)
        meta = {"n_chars": len(text), "table_whole": True}
        if b.header_like:
            meta["table_header_sig"] = _header_signature(header)
        ridx = b.row_line_idx or []
        c_start, c_end = _row_span(b, ridx, 0, len(rows), bool(header))
        chunks.append({
            "text": text, "page_no": b.page_no, "meta": meta,
            "clause_no": None, "section_path": [],
            "char_start": c_start, "char_end": c_end,
            "table": {"table_id": None, "row_start": 0,
                      "row_end": max(len(rows) - 1, 0),
                      "n_rows": len(rows),
                      "n_cols": max((len(r) for r in rows), default=0),
                      "header": header, "is_pseudo": b.is_pseudo,
                      "caption": b.caption, "whole": True},
        })
    return chunks


def _row_span(b, ridx, row_start, n_rows, has_header):
    """表格行组在 md 原文里的字符区间（行级精确）。"""
    if not ridx:
        return b.char_start, b.char_end
    los = b.line_offsets or []
    # 表头行也进 chunk 的 text（PRD 3.1 策略 4）→ 偏移区间必须含它
    first = 0 if has_header else ridx[row_start]
    last_i = min(row_start + n_rows - 1, len(ridx) - 1)
    last = ridx[last_i]
    if not los:
        return b.char_start, b.char_end
    lo = los[first] if first < len(los) else None
    hi_line = b.lines[last] if last < len(b.lines) else ""
    hi = (los[last] + len(hi_line)) if last < len(los) else None
    if lo is None or hi is None:
        return b.char_start, b.char_end
    return lo, hi


def _row_to_line(row):
    return "| " + " | ".join(row) + " |"


def _header_signature(header):
    """表头签名（用于跨页表格按表头合并）。

    🔴 只对**校验通过的真表头**有意义 —— 伪表的首行是正文，
    签名无意义（PRD 3.1 策略 4 的第 3 条）。
    """
    import hashlib
    key = "|".join(c.strip() for c in header)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


SENT_ENDS = tuple("。！？；")
# 🔴 PDF 抽取的"句号"不是"。" —— 字体映射把中文句号输出成半角 "." 或全角"．"。
# 实测：全库"。"=2573 处，但 `AQ_T3034`（最大文档）**一个"。"都没有**，
# 它的句末是半角 "."（行尾 477 处）。只认"。"的话，策略 5 在这份文档上
# 一刀都切不开 —— 与"全角条款号"是同一类字体映射问题。
SENT_SOFT_ENDS = tuple(".．")


def _is_sentence_end(text, i):
    """text[i] 是否是句末标点（硬标点，或 PDF 的半角/全角句号）。"""
    ch = text[i]
    if ch in SENT_ENDS:
        return True
    if ch not in SENT_SOFT_ENDS:
        return False
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    # 行尾的点是句号（PDF 排版一行一句的常见形态）
    if nxt == "" or nxt == "\n":
        return True
    # 行内的点：两侧都不是数字才算（"4.15.2" / "19.5 %" 不是句号）
    if prev.isdigit() or nxt.isdigit():
        return False
    return True


def split_sentences(text):
    """按句子边界切句，保留句末标点。

    返回 [(句子文本, 在 text 里的起始偏移)]。

    🔴 **为什么必须切句**：PDF 抽取的 md **没有段落结构** ——
    段落靠空行界定，而 PDF 源的空行是**排版空行**不是段落空行，
    实测 PDF 文档的"段落"块中位长 **884–1229 字（≈整页）**，
    HTML 源才是真段落（中位 82 字）。
    只"合并短段"而从不切长段，策略 5 对 5 份 PDF 就退化成"按页切块"
    （实测 p95=3367 字，目标 L=400 完全失效）。
    PRD 3.1 策略 5 的「语义边界 = 句子边界 + 段落边界」正是为这个场景写的。
    """
    out = []
    cur = []
    start = 0
    for i, ch in enumerate(text):
        cur.append(ch)
        if _is_sentence_end(text, i):
            s = "".join(cur)
            if s.strip():
                out.append((s, start))
            cur = []
            start = i + 1
    s = "".join(cur)
    if s.strip():
        out.append((s, start))
    return out


def strategy5_semantic(blocks, doc_ctx):
    """策略 5 · 语义滑窗。

    先按段落切、再按**句子边界**切成语义单元，然后合并成块：
    目标块长 `L=400`，**当前块不足 `L/3`（133）时继续吞并**，
    否则放不下就开新块；相邻块之间保留最后 1 个单元作为重叠。

    **不做向量聚类的语义切分** —— 本项目的核心机制不依赖向量，
    语义切分若依赖嵌入模型，向量拉不下来时整条链路就废了。

    🔴 合并方向：**向前吞并**（短单元并入当前块），而不是向后找邻居。
    """
    blocks = merge_cross_page(blocks)

    # 单元化：段落内按句子边界切；表格是不可分割单元，整块保留
    # 每个单元自带它**实际覆盖的页码**（行级算）与**原文切片区间**，
    # 合并块的 page_no 是 None，只有行级页码能说清"这句在第几页"。
    # 🔴 单元带 rel_start/rel_end 是为了**用原文切片拼 chunk 文本** ——
    # PRD 3.1 要求 text 是"原文，一字不动"，
    # 若用 "\n".join(句子) 重排，就等于在原文里插了原文没有的换行。
    units = []   # (kind, block, unit_text, rel_start, rel_end, md_off, unit_pages)
    for b in blocks:
        if b.kind == "table":
            units.append(("table", b, b.text, 0, len(b.text), b.char_start,
                          list(b.pages)))
        else:
            cp = _char_pages(b)
            for s, off in split_sentences(b.text):
                seg = cp[off:off + len(s)]
                units.append(("para", b, s, off, off + len(s), _md_offset(b, off),
                              sorted({p for p in seg if p is not None})))

    groups = []
    cur = []
    cur_len = 0
    for u in units:
        kind, b, utext, _rs, _re, _mo, _up = u
        ulen = len(utext)
        if kind == "table":
            if cur:
                groups.append(cur)
                cur = []
                cur_len = 0
            groups.append([u])
            continue
        # PRD：短于 L/3 的段与相邻段合并，直到超过 L
        if cur_len < STRATEGY5_MIN or cur_len + ulen <= STRATEGY5_TARGET:
            cur.append(u)
            cur_len += ulen
            continue
        groups.append(cur)
        cur = [u]
        cur_len = ulen
    if cur:
        groups.append(cur)

    chunks = []
    for gi, grp in enumerate(groups):
        if not grp:
            continue
        # 相邻块保留最后 1 个单元作为重叠
        overlap = None
        if gi > 0:
            prev = groups[gi - 1]
            if prev and prev[-1][0] == "para":
                overlap = prev[-1]

        pages = []
        para_blocks = []
        seq = ([overlap] if overlap else []) + list(grp)
        # 🔴 文本用**原文切片**拼，不用 "\n".join(句子) ——
        # PRD 3.1 要求 text 是"原文，一字不动"，
        # join 会在原文没有的地方插换行，也会让 chunk 与 md 对不上。
        pieces = []
        for idx, u in enumerate(seq):
            kind, b, utext, rs, re_, mo, up = u
            if pieces and pieces[-1][0] is b and rs >= pieces[-1][2]:
                pieces[-1] = (b, pieces[-1][1], re_)   # 同块相邻单元 → 连成一片
            else:
                pieces.append((b, rs, re_))
            # 🔴 页码与偏移只按**新内容**统计：重叠段是从上一块复制来的。
            if not (overlap is not None and idx == 0):
                pages.extend(up)
            if b.kind != "table" and id(b) not in para_blocks:
                para_blocks.append(id(b))
        text = "\n".join(b.text[rs:re_] for b, rs, re_ in pieces)
        # 字符区间按**行首偏移**算（见 _line_span 的说明）。
        # 🔴 **重叠段也要算进区间**：chunk 的文本里确实有它，
        # 漏掉它就会出现"文本里有这行、但这行在区间外"的越界。
        # （页码归属则相反：页码按**新内容**算，引用要指向答案所在页。）
        start = end = None
        for b, rs, re_ in pieces:
            lo, hi = _line_span(b, rs, re_)
            if lo is None:
                continue
            if start is None:
                start = lo
            end = hi

        meta = {"n_chars": len(text), "merged_paras": len(para_blocks),
                "n_units": len(seq)}
        if overlap is not None:
            meta["overlap_head"] = overlap[2]
        if len(set(pages)) > 1:
            meta["cross_page"] = sorted(set(pages))
            page = None
        else:
            page = pages[0] if pages else None
        chunks.append({
            "text": text, "page_no": page, "meta": meta,
            "clause_no": None, "section_path": [],
            "char_start": start, "char_end": end,
        })
    return chunks


STRATEGIES = {
    "fixed": strategy1_fixed,
    "structural": strategy2_structural,
    "clause": strategy3_clause,
    "table": strategy4_table,
    # 策略 4b：整表一块（PRD 风险 R8 的预案，M10 诊断后加）
    "table_whole": strategy4b_table_whole,
    "semantic": strategy5_semantic,
}


# ---------------------------------------------------------------- chunk 组装

# PRD 2.9：chunk 必须携带的文档级字段（14 个）
# 🔴 2026-11-03 补 `effective_to_source`（M5 实测发现的真 bug，见 notes/踩坑.md K-32）：
# `versioning.is_effective()` 靠这个字段区分
# 「至今未失效（open）」与「真的不知道（unknown）」——
# 少了它，`effective_to=None` 会被判成 UNKNOWN，
# 于是 **as_of 口径下 26 份"至今有效"的文档全被排除**，as_of 的题一道都答不了。
DOC_SNAPSHOT_FIELDS = [
    "doc_id", "title", "doc_type", "doc_no", "version", "effective_from",
    "effective_to", "effective_to_source", "superseded_by", "status",
    "publisher", "source_url", "source_format", "extract_quality",
]


def build_doc_snapshot(doc_meta):
    """从 manifest 的文档级记录取 14 个字段。"""
    snap = {}
    for f in DOC_SNAPSHOT_FIELDS:
        snap[f] = doc_meta.get(f)
    return snap


def narrow_labels(chunk_text, doc_meta, norm_cache):
    """把文档级标签**按 chunk 实际出现情况收窄**（PRD 2.3）。

    🔴 位号不能朴素子串匹配：`tag_key` 去掉分隔符而 `norm_text` 保留，
    拿 key 直接 count 归一化文本恒为 0（K-12）。
    所以位号走 `normalize.find_tag_occurrences` 的压缩口径。
    """
    norm = norm_cache.get("norm")
    if norm is None:
        norm = norm_cache["norm"] = _normalized(chunk_text)

    out = {}
    for field in ("hazard_tags", "equipment_type", "equipment_id",
                  "maintenance_type", "ref_standard"):
        vals = doc_meta.get(field) or []
        kept = []
        for v in vals:
            if _tag_in_text(v, norm):
                kept.append(v)
        out[field] = kept
    return out


def _normalized(text):
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import normalize as N
    return N.normalize(N.normalize_ws(text))


def _tag_in_text(tag, norm_text):
    """标签是否出现在 chunk 的归一化文本里。

    位号（形如 19-V-4215）用**压缩分隔符**的口径；
    其它标签（危险源类别、标准号）用归一化子串。
    """
    import re as _re
    t = str(tag).strip()
    if not t:
        return False
    # 位号形态：含分隔符的数字/字母组合
    if _re.match(r"^[0-9A-Za-z]+([\-—－/][0-9A-Za-z]+){1,4}$", t):
        key = _re.sub(r"[\-—－/]", "", t).lower()
        flat = _re.sub(r"[\-—－/]", "", norm_text).lower()
        return key in flat
    return t in norm_text


def assemble(chunk, doc_meta, strategy, chunk_id, norm_cache):
    """把策略产出的原始块组装成 PRD 3.1 的统一 chunk schema。"""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import normalize as N

    text = chunk["text"]
    n_text, n_norm = N.build_chunk_text(text)
    labels = narrow_labels(text, doc_meta, norm_cache)
    meta = dict(chunk.get("meta") or {})

    out = {
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("doc_id"),
        "strategy": strategy,
        "text": n_text,
        "norm_text": n_norm,
        "clause_no": chunk.get("clause_no"),
        "page_no": chunk.get("page_no"),
        "char_start": chunk.get("char_start"),
        "char_end": chunk.get("char_end"),
        "section_path": chunk.get("section_path") or [],
        "table": chunk.get("table"),
        "doc_snapshot": build_doc_snapshot(doc_meta),
        "quality": doc_meta.get("extract_quality"),
        "hazard_tags": labels["hazard_tags"],
        "equipment_type": labels["equipment_type"],
        "equipment_id": labels["equipment_id"],
        "maintenance_type": labels["maintenance_type"],
        "ref_standard": labels["ref_standard"],
        "strategy_meta": meta,
    }
    return out


# ---------------------------------------------------------------- 策略 6

def strategy6_qa(rows, doc_meta_by_id, root):
    """策略 6 · QA-pair 切分。

    🔴 **输入必须是索引侧的子集**（`data/eval/qa_index_side.csv`），
    不能用完整的 `qa_pairs.csv` —— 后者与评测集同源，等于把答案塞进检索库。

    chunk 内容：`text` = 问题 + 答案，`norm_text` = 归一化后的问题 + 答案。

    🔴 `qa_id` 必须重新生成（原 `qid` 有 7 组重复），用
    `qa_<sha1(doc_id|question|answer)[:12]>`。
    """
    import csv
    import hashlib

    path = root / "data" / "eval" / "qa_index_side.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n必须先跑 `python tools/refine_eval_set.py --stage qa_split`")

    chunks = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if not q or not a:
                continue
            doc_id = r.get("doc_id")
            dm = doc_meta_by_id.get(doc_id, {})
            qa_id = "qa_" + hashlib.sha1("|".join(
                [doc_id, q, a]).encode("utf-8")).hexdigest()[:12]
            text = q + "\n" + a
            page = r.get("page_no")
            chunks.append({
                "chunk_id": qa_id,
                "doc_id": doc_id,
                "strategy": "qa",
                "text": text,
                "norm_text": None,      # assemble 里生成
                "clause_no": r.get("clause_no"),
                "page_no": int(page) if page and str(page).strip().isdigit() else None,
                "char_start": None,
                "char_end": None,
                "section_path": [],
                "table": None,
                "doc_snapshot": build_doc_snapshot(dm),
                "quality": dm.get("extract_quality"),
                "hazard_tags": [], "equipment_type": [], "equipment_id": [],
                "maintenance_type": [], "ref_standard": [],
                "strategy_meta": {"qa_id": qa_id, "n_chars": len(text)},
            })
    return chunks
