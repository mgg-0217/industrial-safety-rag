# -*- coding: utf-8 -*-
"""切分模块与基线重建模块的单元测试。

M3 的验收标准（PRD 第 9 节）明确要求：
「单元测试覆盖**表头识别**、**页标记解析**、**跨页情形表**」。

这里多覆盖两块 —— 它们是这次改动的**要害**，不测等于没验证：

4. **基线重建**（`src/pdf_lines.py`）：字体错位会不会被拆行、
   上标会不会被截断、x 排序会不会还原阅读顺序。
5. **归一化**（`src/normalize.py`）：全角数字/位号破折号，
   以及**全角条款号**能否被条款正则匹配（AQ_T3034 进评测池的前提）。

跑法：
    python -m pytest tools/test_chunking.py -v
    python tools/test_chunking.py          # 不装 pytest 也能跑
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chunking as C          # noqa: E402
import normalize as N         # noqa: E402
import pdf_lines as PL        # noqa: E402

import re  # noqa: E402


# ============================================================ 1. 页标记解析

def test_page_marker_parsing():
    """`<!-- page:N -->` 注释是页码锚点（PRD 3.1）。"""
    assert C.PAGE_MARK.match("<!-- page:12 -->")
    assert C.PAGE_MARK.match("<!--page:3-->")
    assert not C.PAGE_MARK.match("<!-- page:x -->")
    assert not C.PAGE_MARK.match("page:12")
    # 页码必须是 int，不能是字符串
    assert int(C.PAGE_MARK.match("<!-- page:7 -->").group(1)) == 7


def test_parse_md_splits_pages_and_blocks(tmp_path=None):
    """parse_md 要把页标记切开，并按 source_format 决定段落粒度。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "t.md"
        p.write_text(
            "---\nsource_file: x.pdf\n---\n\n"
            "<!-- page:1 -->\n\n"
            "第一段第一行\n第二段第一行\n\n"
            "第二段\n\n"
            "<!-- page:2 -->\n\n"
            "第三页的段落\n\n"
            "| a | b |\n| --- | --- |\n| 1 | 2 |\n",
            encoding="utf-8")
        blocks = C.parse_md(p, source_format="pdf_document")
    paras = [b for b in blocks if b.kind == "para"]
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1, f"应识别出 1 个表格块，实得 {len(tables)}"
    # PDF 源：段落按空行分，第一段是 2 行
    assert len(paras) == 3, f"应切出 3 个段落，实得 {len(paras)}"
    assert paras[0].page_no == 1
    assert paras[1].page_no == 1
    assert paras[2].page_no == 2, f"页码应为 2，实得 {paras[2].page_no}"
    assert paras[0].text == "第一段第一行\n第二段第一行"


def test_html_source_paragraph_granularity():
    """HTML 源按行分段（行间无空行），不能整篇并成一段。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "t.md"
        p.write_text("---\nsource_file: x.html\n---\n行一\n行二\n行三\n", encoding="utf-8")
        blocks = C.parse_md(p, source_format="html_webpage")
    paras = [b for b in blocks if b.kind == "para"]
    assert len(paras) == 3, f"HTML 源应每行一段，实得 {len(paras)}"


# ============================================================ 2. 表头识别

def test_header_looks_like_header_true():
    """真表头：短、无句读。"""
    hdr = ["序号", "具体内容", "实施要求"]
    assert C.header_looks_like_header(hdr, [["1", "x", "y"], ["2", "z", "w"]])
    assert C.header_looks_like_header(["项目", "完成", "姓名", "日期", "备注"],
                                      [["a", "b", "c", "d", "e"]])


def test_header_looks_like_header_false_prose():
    """伪表头：首行是正文散文（有句读、长度长）。"""
    hdr = ["企业危险化学品特种作业人员应具备高中或者相当于高中及以上文化程度，能力应满足安全生产要求。"]
    assert not C.header_looks_like_header(hdr, [["短"], ["也短"]])
    hdr2 = ["本文件规定了化工过程安全管理的相关要素和管理要求"]
    assert not C.header_looks_like_header(hdr2, [["短"], ["也短"]])


def test_header_looks_like_header_absolute_threshold():
    """🔴 必须是**绝对**阈值，不能是"比数据行短"。

    键值对式的表，数据行比表头还短 —— 用相对判据会把真表头误判成伪表。
    """
    hdr = ["类别", "文件信息", "存放地点", "更新时间", "文件负责人"]
    data = [["a", "b", "c", "d", "e"]]          # 数据行比表头还短
    assert C.header_looks_like_header(hdr, data)


def test_is_key_value_table():
    """作业票类的键值对表：非空格数差异大、占比低。

    用真实形态（18 列、大量空 padding）造样 —— 键值对表的**列数多**，
    每行只有 2–8 个非空格。用 4 列的小表测会失真。
    """
    def kv_row(*vals):
        row = [""] * 19
        for i, v in enumerate(vals):
            row[i] = v
        return row

    kv = [
        kv_row("作业申请单位", "", "", "甲醇车间", "", "", "作业申请时间",
               "", "", "", "2022年4月1日8时20分"),
        kv_row("作业人", "", "", "xxx", "", "", "监护人", "", "", "", "xxx"),
        kv_row("气体分析", "", "分析项目", "", "", "", "有毒有害气体名称",
               "", "", "", "氧气含量", "", "", "取样分析时间", "", "", "分析部位", "", "分析人"),
    ]
    assert C.is_key_value_table(kv), "作业票形态应识别为键值对表"
    normal = [["序号", "内容", "要求"], ["1", "a", "b"], ["2", "c", "d"], ["3", "e", "f"]]
    assert not C.is_key_value_table(normal)


# ============================================================ 3. 跨页情形表

def _blk(text, kind="para", page=1):
    lines = text.split("\n")
    b = C.Block(kind, text, lines, page, 0, len(text))
    return b


def test_cross_page_case1_table_boundary():
    """情形 1：页 N 末尾是表格 → 不合并。"""
    prev = _blk("| a | b |", kind="table")
    nxt = _blk("正文继续")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert not merge and "情形1" in reason


def test_cross_page_case2_new_heading():
    """情形 2：页 N+1 开头是标题 → 不合并。"""
    prev = _blk("上一段没有句号结尾")
    nxt = _blk("三、事故设备及管道情况")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert not merge and "情形2" in reason


def test_cross_page_case3_complete_sentence():
    """情形 3：页 N 末尾是完整句 → 不合并。"""
    prev = _blk("这句话是完整的。")
    nxt = _blk("下一页的正文")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert not merge and "情形3" in reason


def test_cross_page_case4_heading_with_body():
    """情形 4：页 N 末尾是标题、页 N+1 是它的下文 → 合并。

    🔴 情形 4 **必须排在情形 3 之后**：标题行通常不以 。！？ 结尾，
    若把 3 提前，"标题+正文跨页"会被误判成"完整句"而不合并。
    """
    prev = _blk("三、事故设备及管道情况")
    nxt = _blk("事故发生在2024年某月")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert merge and "情形4" in reason


def test_cross_page_case5_sentence_cut():
    """情形 5：句子被分页切断 → 合并。"""
    prev = _blk("这句话被分页切断")
    nxt = _blk("后面接的是同一句话")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert merge and "情形5" in reason


def test_merge_cross_page_keeps_page_list():
    """合并后 page_no 记 null，页码列表进 strategy_meta（PRD 3.1）。"""
    blocks = [_blk("前半句被切断", page=3), _blk("后半句接上", page=4)]
    merged = C.merge_cross_page(blocks)
    assert len(merged) == 1
    assert merged[0].page_no is None
    assert merged[0].pages == [3, 4]


def test_cross_page_order_case4_before_case3():
    """🔴 顺序回归测试：情形 4 必须先于情形 3 命中。

    标题"四、事故原因分析"不以句读结尾 ——
    若情形 3 提前，它会被判成"完整句"而不合并，标题与正文被拆开。
    """
    prev = _blk("四、事故原因分析")
    nxt = _blk("经调查，原因是设备老化")
    merge, reason = C.decide_cross_page(prev, nxt)
    assert merge and "情形4" in reason


# ============================================================ 4. 基线重建

def _mk_char(text, x0, baseline_top, size=10.0, width=None, page_h=800.0):
    """造一个 pdfplumber 风格的 char（baseline_top = 自上而下的基线 y）。

    `width` 默认等于 `size`（中日韩字形是全角，advance ≈ 字号）；
    西文字形要窄一些，传 `width=size*0.6`。
    """
    w = size if width is None else width
    return {
        "text": text, "x0": x0, "x1": x0 + w, "size": size,
        "top": baseline_top - size * 0.8, "bottom": baseline_top + size * 0.2,
        "upright": True, "matrix": [1, 0, 0, 1, x0, page_h - baseline_top],
    }


def test_baseline_rebuild_merges_font_shifted_runs():
    """🔴 核心用例：同一视觉行被两种字体拆开，按基线应合回一行。

    模拟 AQ_T3034 第 10 页：
      - 中文 run 基线 702.65（换算成自上而下 = page_h - 702.65）
      - ASCII run 基线低了 1.5pt，但 x 还在原位
    """
    H = 800.0
    base = H - 702.65
    chars = [
        _mk_char("在", 118.4, base, page_h=H),
        _mk_char("工", 128.9, base, page_h=H),
        _mk_char("厂", 139.4, base, page_h=H),
        # ASCII run：基线低 1.5pt，x 落在"厂"与下一字之间
        _mk_char("G", 148.0, base + 1.5, width=6.0, page_h=H),
        _mk_char("B", 155.0, base + 1.5, width=6.0, page_h=H),
        _mk_char("厂", 165.0, base, page_h=H),
    ]
    lines = PL.reconstruct_lines(chars, H, baseline_tol=3.0)
    assert len(lines) == 1, f"应合成 1 行，实得 {len(lines)}: {lines}"
    # 关键：GB 应留在同一行里，且按 x 排在"厂"与后面的"厂"之间
    assert "GB" in lines[0], f"ASCII run 应留在同一行，实得 {lines[0]!r}"
    assert lines[0].index("G") < lines[0].index("厂", 3), \
        f"GB 应按 x 排在后一个'厂'之前，实得 {lines[0]!r}"


def test_baseline_rebuild_keeps_separate_lines():
    """行距 15.9pt 时，两行不能被并成一行。"""
    H = 800.0
    base = H - 702.65
    chars = [
        _mk_char("上", 118.4, base, page_h=H),
        _mk_char("行", 128.9, base, page_h=H),
        _mk_char("下", 118.4, base + 15.9, page_h=H),
        _mk_char("行", 128.9, base + 15.9, page_h=H),
    ]
    lines = PL.reconstruct_lines(chars, H, baseline_tol=3.0)
    assert len(lines) == 2, f"应保持 2 行，实得 {len(lines)}: {lines}"
    assert lines[0] == "上行" and lines[1] == "下行"


def test_baseline_rebuild_keeps_superscript():
    """🔴 上标（m³ 的 ³）字号更小、基线更高，要并回主行而不是被截断。

    实测：正文 10.56pt / 上标 6.96pt，基线差 5.03pt。
    """
    H = 800.0
    base = H - 700.0
    chars = [
        _mk_char("m", 200.5, base, size=10.56, page_h=H),
        _mk_char("3", 208.6, base - 5.03, size=6.96, width=4.0, page_h=H),
        _mk_char("，", 212.0, base, size=10.56, page_h=H),
    ]
    lines = PL.reconstruct_lines(chars, H, baseline_tol=3.0)
    assert len(lines) == 1, f"上标应并入主行，实得 {len(lines)} 行: {lines}"
    assert lines[0].startswith("m3"), f"上标应紧跟 m，实得 {lines[0]!r}"


def test_baseline_rebuild_does_not_merge_small_table_rows():
    """🔴 表格里整行都是小字号时，不能被"小字号规则"并成一行。"""
    H = 800.0
    base = H - 700.0
    chars = [
        _mk_char("行", 100.0, base, size=6.0, page_h=H),
        _mk_char("二", 110.0, base + 10.0, size=6.0, page_h=H),
    ]
    lines = PL.reconstruct_lines(chars, H, baseline_tol=3.0)
    assert len(lines) == 2, f"同字号的两行不该合并，实得 {len(lines)}"


def test_baseline_rebuild_skips_rotated():
    """旋转文字（水印）不进正文行。"""
    H = 800.0
    ch = _mk_char("水", 100.0, H - 700.0, page_h=H)
    ch["upright"] = False
    ch["matrix"] = [0, 1, -1, 0, 100.0, 700.0]
    assert PL.reconstruct_lines([ch], H) == []


# ============================================================ 5. 归一化

def test_normalize_fullwidth_digits():
    assert N.normalize("２０２２年") == "2022年"
    assert N.normalize("４．１．２") == "4.1.2"


def test_normalize_tag_dash_context_aware():
    """位号破折号要转，中文破折号不能转。"""
    assert N.normalize("19—V—4215") == "19-V-4215"
    assert N.normalize("设备名称——粗产品缓冲罐") == "设备名称——粗产品缓冲罐"


def test_normalize_keeps_html_page_markers():
    """🔴 归一化**不能**破坏 `<!-- page:N -->` 页码锚点。

    起因：`-{2,}` → `-` 的规则没有上下文，会把 `<!--` / `-->` 的
    双连字符压掉，**51 个页码锚点整片消失**，
    评测候选的 `page_no` 全变 None。
    """
    src = "<!-- page:12 -->\n正文 19——V\n<!-- page:13 -->"
    out = N.normalize(src)
    assert "<!-- page:12 -->" in out, f"页码锚点被破坏: {out!r}"
    assert "<!-- page:13 -->" in out, f"页码锚点被破坏: {out!r}"
    # 位号的多破折号仍然要压
    assert "19-V" in out


def test_fullwidth_clause_matches_clause_regex():
    """🔴 全角条款号必须能被条款正则匹配 —— AQ_T3034 进评测池的前提。

    实测：PDF 字体把 `4.1.2` 映射成全角 `４．１．２`，
    旧正则 `[\\.、]` 只认半角点 → 切不出条款 → 该文档 0 条候选。

    注：`split_clauses` 会丢掉正文短于 40 字的条款，
    所以样例正文必须够长，否则测的是长度门槛而不是匹配。
    """
    from build_eval_candidates import split_clauses  # noqa: E402
    long1 = "企业主要负责人应领导企业贯彻落实以人为本安全第一的安全理念，建立包括以安全愿景目标在内的核心价值观"
    long2 = "企业主要负责人应领导企业制定安全生产方针和目标，并组织全员参与安全教育培训与考核工作"
    body = N.normalize(f"４．１．２　{long1}\n４．１．３　{long2}\n")
    # split_clauses 返回 (clause, body, start, end) —— 2026-11-03 起多了两个
    # 原文坐标（候选题要靠它落到 chunk 上），别再按 2 元组解包
    clauses = [c for c, _b, _s, _e in split_clauses(body)]
    assert len(clauses) == 2, f"应切出 2 条，实得 {len(clauses)}: {clauses}"
    assert clauses[0].startswith("4.1.2")


# ============================================================ 6. 切分策略

def test_strategy1_fixed_length():
    """策略 1：定长 512、重叠 64。"""
    blocks = [_blk("字" * 1200)]
    chunks = C.strategy1_fixed(blocks, {})
    assert len(chunks) >= 3
    assert all(len(c["text"]) <= 512 for c in chunks)


def test_strategy1_is_document_level_sliding_window():
    """🔴 策略 1 的滑窗必须是**文档级**的，不是"每段各切 L 再拼"。

    起因：旧实现按块各切 512 再累加，块长会涨到 2L（实测 p95=946 > 512），
    而且块边界处各切一刀 → 边界多出两个半块。
    PRD 对策略 1 的定义是"定长块 + 重叠"，块长就该是 L。
    """
    # 三个 400 字的段落，跨块边界也必须连续成定长块
    blocks = [_blk("甲" * 400), _blk("乙" * 400), _blk("丙" * 400)]
    chunks = C.strategy1_fixed(blocks, {})
    assert all(len(c["text"]) <= 512 for c in chunks), \
        f"块长超过 L: {[len(c['text']) for c in chunks]}"
    # 块长应当接近 L，而不是 200–400（那说明按块边界切了）
    assert max(len(c["text"]) for c in chunks) >= 400
    # 相邻块要有 64 字重叠
    a, b = chunks[0]["text"], chunks[1]["text"]
    assert a[-64:] == b[:64], "相邻块之间没有 L-64 的重叠"


def test_strategy1_char_offsets_are_traceable():
    """策略 1 的 char_start/char_end 要能回原文核对（PRD 3.1 字段约束）。"""
    blocks = [_blk("甲" * 300), _blk("乙" * 300)]
    chunks = C.strategy1_fixed(blocks, {})
    for c in chunks:
        assert c["char_start"] is not None and c["char_end"] is not None
        assert c["char_end"] > c["char_start"]


def test_strategy5_splits_long_paragraph_at_sentence_boundary():
    """🔴 策略 5 必须能切**长段**，否则对 PDF 源退化成"按页切块"。

    起因：PDF 抽取的 md 没有段落结构（段 = 整页，中位 884–1229 字），
    旧实现只合并短段、从不切长段 → 实测 p95=3367 字，目标 L=400 完全失效。
    PRD 3.1 策略 5 的"语义边界 = 句子边界 + 段落边界"就是为此写的。
    """
    para = "第一句内容在这里。" * 60          # 480 字，60 个句子
    blocks = [_blk(para)]
    chunks = C.strategy5_semantic(blocks, {})
    assert len(chunks) > 1, "长段没有被切开"
    assert max(len(c["text"]) for c in chunks) < 600, \
        f"切开后仍然过长: {max(len(c['text']) for c in chunks)}"


def test_strategy5_keeps_overlap_and_records_merged_paras():
    """策略 5：相邻块保留最后 1 个单元作为重叠，并记 merged_paras。"""
    blocks = [_blk(("甲" * 10 + "。") * 40 + ("乙" * 10 + "。") * 40)]
    chunks = C.strategy5_semantic(blocks, {})
    assert len(chunks) >= 2, "长段应被切成多块"
    head = chunks[1]["meta"]["overlap_head"]
    assert head, "第二个块应带 overlap_head"
    assert chunks[1]["text"].startswith(head), "重叠段应出现在下一块开头"
    assert chunks[0]["text"].endswith(head), "重叠段应是上一块的结尾"
    assert chunks[1]["meta"]["merged_paras"] >= 1


def test_strategy3_keeps_page_list_after_cross_page_merge():
    """🔴 跨页合并过的块 page_no 被置 None，但**页码列表还在**。

    只看 page_no 会把合并块的页码整个丢掉 → 跨页块统计为 0，
    PRD 3.1 策略 3 的"跨页合并按 3.2 互斥情形表实现"就落空了。
    """
    blocks = [_blk("前半段没有句号结尾", page=3), _blk("后半段继续写", page=4)]
    merged = C.merge_cross_page(blocks)
    assert merged[0].page_no is None, "合并块应把 page_no 置 None"
    assert merged[0].pages == [3, 4], f"合并块应保留页码列表，实得 {merged[0].pages}"

    # 放进策略 3 的骨架里，跨页元数据必须仍然能统计出来
    sk = C.build_skeleton(merged)
    chunks = C.strategy3_clause(merged, {})
    assert any((c.get("meta") or {}).get("cross_page") for c in chunks), \
        "策略 3 丢失了跨页元数据"


def test_strategy5_keeps_page_list_after_cross_page_merge():
    """策略 5 同样要保留跨页块的页码列表。"""
    blocks = [_blk("前半段没有句号结尾", page=3), _blk("后半段继续写", page=4)]
    merged = C.merge_cross_page(blocks)
    chunks = C.strategy5_semantic(merged, {})
    assert any((c.get("meta") or {}).get("cross_page") for c in chunks), \
        "策略 5 丢失了跨页元数据"


def test_pdf_halfwidth_period_is_a_sentence_end():
    """🔴 PDF 抽取的"句号"是半角 `.` / 全角`．`，不是"。"。

    实测：`AQ_T3034`（最大文档）**一个"。"都没有**，
    它的句末是半角"."（全库行尾 477 处）。
    只认"。"的话，策略 5 在这份文档上**一刀都切不开** ——
    与"全角条款号"是同一类 PDF 字体映射问题。
    """
    assert C._is_sentence_end("分级管理.\n下一行", 4) is True, "行尾的半角点应是句末"
    assert C._is_sentence_end("分级管理.\n下一行", 4) is True
    assert C._is_sentence_end("4.15.2 变更分类\n", 1) is False, "小数点不是句末"
    assert C._is_sentence_end("浓度为 19.5 %。\n", 5) is False, "小数点不是句末"
    assert C._is_sentence_end("风险。下一句", 2) is True, "句号是句末"
    assert C._is_sentence_end("甲；乙", 1) is True
    # 全角"．"同理
    assert C._is_sentence_end("４．１５．２　变更分类\n", 1) is False


def test_chunk_text_always_inside_char_range():
    """🔴 PRD 6.3 的承诺：char_start/char_end 是"可回到 md 的**精确位置**"。

    逐个策略跑一遍合成语料，断言：
    chunk 文本的**每一行**都能在 md[char_start:char_end] 里原样找到。
    这条一旦破了，"溯源"就只剩下一句口号。
    """
    import tempfile
    md = (
        "---\nsource_file: x.pdf\n---\n\n"
        "<!-- page:1 -->\n\n"
        "4.1.3  企业主要负责人安全培训考核情况，分管生产、安全负责人专业、学历满足情况。\n"
        "4.1.4  企业主要负责人组织学习、贯彻落实国家安全生产法律法规的情况。\n\n"
        "<!-- page:2 -->\n\n"
        "4.1.5  企业主要负责人和各级管理人员在岗在位、带（值）班的情况。\n"
        "4.1.6  安全生产管理体系建设的落实情况。\n\n"
        "| 序号 | 排查内容 | 排查依据 |\n| --- | --- | --- |\n"
        "| 1 | 检查生产责任制 | GB 30871 |\n"
        "| 2 | 检查培训记录 | GB/T 33000 |\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "t.md"
        p.write_text(md, encoding="utf-8")
        blocks = C.parse_md(p, source_format="pdf_document")
        for name, fn in C.STRATEGIES.items():
            chunks = fn(blocks, {})
            for c in chunks:
                if c.get("char_start") is None:
                    continue
                seg = md[c["char_start"]:c["char_end"]]
                for line in c["text"].split("\n"):
                    if line.strip():
                        assert line in seg, (
                            f"策略 {name} 的 chunk 有一行落在区间外：{line[:40]!r}\n"
                            f"  区间=[{c['char_start']},{c['char_end']}]")


def test_wrapped_cell_table_is_not_key_value():
    """🔴 折行表（PDF 单元格内容跨行）**不能**被误判成键值对表。

    起因（2026-11-02）：旧判据是"非空格占比低 + 各行非空格数差异大"，
    而 `mem_2019_78` 的 17 张排查表（`序号|排查内容|排查依据`，50–76 行）
    因为**单元格内容跨行折行**、pdfplumber 把一个逻辑行拆成多条物理行，
    每行只剩 1–3 个非空格 → 被全判成键值对 →
    **策略 4 对最表格密集的文档完全不切行**，PRD 的表格切分能力作废。

    真区别在横向分布：折行表的内容集中在**相邻**列（1 段），
    键值对表的内容**散在**不相邻的列（≥2 段）。
    """
    rows = [["", "1.主要负责人应组织制定符合本企业实际的安全生产", ""],
            ["", "方针和年度安全生产目标；", ""],
            ["7", "", ""],
            ["", "2.环氧乙烷的安全阀及其他泄放设施直排大气的应采", ""],
            ["", "取安全措施。", ""]]
    assert not C.is_key_value_table(rows), "折行表被误判成键值对表"
    # 相邻列的表头确实像表头
    hdr = ["序号", "排查内容", "排查依据"]
    assert C.header_looks_like_header(hdr, rows, rows)


def test_strategy4_pseudo_table_not_repeated():
    """🔴 伪表不能重复首行 —— 那是把正文当表头再用一遍。"""
    rows = ["正文第一行很长很长很长很长很长很长很长", "| --- | --- |",
            "| 第二行 | 内容 |", "| 第三行 | 内容 |"]
    b = C.Block("table", "\n".join(rows), rows, 1, 0, 100)
    b.rows, b.header = C.parse_table(rows)
    b.header_like = False
    b.is_pseudo = True
    chunks = C.strategy4_table([b], {})
    assert chunks
    for c in chunks:
        assert "正文第一行" not in c["text"], "伪表的首行被重复了"
        assert c["table"]["is_pseudo"] is True


def test_strategy4_true_header_repeated():
    """🔴 真表头要随行重复 —— 否则行级答案没有列义。

    这条测的是 2026-11-02 修的 bug：早前 `b.rows` 里存的是
    "去掉表头后的数据行"，而策略 4 又用 `rows[0]` 当表头 ——
    结果**真表头整个丢失**，第一条数据行被当表头重复了一遍。
    """
    rows = ["| 序号 | 内容 |", "| --- | --- |",
            "| 1 | 甲 |", "| 2 | 乙 |"]
    b = C.Block("table", "\n".join(rows), rows, 1, 0, 100)
    b.rows, b.header = C.parse_table(rows)
    b.header_like = True
    b.is_pseudo = False
    chunks = C.strategy4_table([b], {})
    assert chunks
    for c in chunks:
        assert "序号" in c["text"], f"真表头应随行重复，实得 {c['text'][:60]!r}"
        assert c["table"]["header"] == ["序号", "内容"]
    # 第一条数据行不能被当成表头重复，也不能丢
    joined = "\n".join(c["text"] for c in chunks)
    assert "甲" in joined and "乙" in joined


def test_strategy4_key_value_table_whole():
    """键值对表整表一个 chunk，不按行切（切了就看不懂）。"""
    def row(*vals):
        r = [""] * 19
        for i, v in enumerate(vals):
            r[i] = v
        return "| " + " | ".join(r) + " |"

    rows = [
        row("作业申请单位", "", "", "甲醇车间", "", "", "作业申请时间", "", "", "",
            "2022年4月1日8时20分"),
        row("作业人", "", "", "xxx", "", "", "监护人", "", "", "", "xxx"),
        row("气体分析", "", "分析项目", "", "", "", "有毒有害气体名称", "", "", "",
            "氧气含量", "", "", "取样分析时间", "", "", "分析部位", "", "分析人"),
        row("风险辨识结果", "", "", "中毒窒息、触电、火灾爆炸", "", "", "", "", "", "",
            "", "", "", "", "", "", "", "", ""),
    ]
    b = C.Block("table", "\n".join(rows), rows, 1, 0, 100)
    b.rows, b.header = C.parse_table(rows)
    chunks = C.strategy4_table([b], {})
    assert len(chunks) == 1, f"键值对表应整表一块，实得 {len(chunks)}"
    assert chunks[0]["meta"]["table_whole"] is True


# ============================================================ 运行器

def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} 通过 / {failed} 失败 / 共 {passed + failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
