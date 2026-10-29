"""把抓下来的政府网页正文抽取为干净 Markdown（带来源元数据 front-matter）。

用途：为工业维修/安全规程 RAG 项目准备真实语料。
不做通用 HTML 抽取，只保留正文块 + 表格，丢掉导航/脚本/页脚。
"""
import re
import sys
import pathlib
import html as htmllib

DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]

# 常见的正文容器 class/id 关键词
BODY_HINTS = [
    "TRS_Editor", "TRS_UEDITOR", "article-content", "articleContent", "article_content",
    "view-content", "view_content", "zoom", "Zoom", "content-body", "con_txt", "con_zz",
    "detail_content", "detailContent", "news_content", "newsContent", "wzcon", "text_content",
    "b-content", "e_content", "xl_content", "article", "content", "main",
]

BLOCK_END = re.compile(
    r"</(p|div|tr|li|h1|h2|h3|h4|h5|h6|table|section|blockquote|br)\s*>|<\br\s*/?>", re.I
)
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)


def detect_encoding(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


NAV_NOISE = re.compile(
    r"(首页|上一条|下一条|分享|打印|字号|大 中 小|字体|收藏|网站地图|版权所有|主办单位|"
    r"备案|联系我们|返回顶部|版权所有|政府信息公开|无障碍|长者版|繁|简|"
    r"索引号|信息名称|发文机关|发文字号|成文日期|发布日期|信息来源|主题分类|文号)",
)

BODY_HINT_RE = re.compile(
    r"<(div|section|article)\b[^>]*?(?:class|id)\s*=\s*[\"'][^\"']*"
    r"(?:TRS_Editor|TRS_UEDITOR|TRS_HTML|article[-_]?content|content[-_]?body|"
    r"detail[-_]?content|news[-_]?content|zoom|Zoom|con[-_]?txt|wzcon|"
    r"b[-_]?content|xl[-_]?content|txt[-_]?con|main[-_]?text|rich[-_]?text|"
    r"e[-_]?content|art[-_]?con)"
    r"[^\"']*[\"'][^>]*>",
    re.I | re.S,
)


def extract_balanced(text: str, start: int) -> tuple[str, int]:
    """从 start 处开始做标签配对，返回 (完整内容, 结束位置)。解决嵌套 div 被非贪婪截断的问题。"""
    open_re = re.compile(r"<(div|section|article|td|table)\b[^>]*>", re.I)
    close_re = re.compile(r"</(div|section|article|td|table)\s*>", re.I)
    depth = 0
    pos = start
    while pos < len(text):
        mo = open_re.search(text, pos)
        mc = close_re.search(text, pos)
        if not mc:
            break
        if mo and mo.start() < mc.start():
            depth += 1
            pos = mo.end()
            continue
        depth -= 1
        pos = mc.end()
        if depth <= 0:
            return text[start:mc.start()], mc.end()
    return text[start:], len(text)


def pick_body(text: str) -> str:
    """挑正文：优先 id/class 命中正文特征且内容量最大的容器（标签配对，容忍嵌套）；
    否则退回全文清洗。"""
    best, best_len = "", 0
    for m in BODY_HINT_RE.finditer(text):
        tag = m.group(0)
        inner, _ = extract_balanced(text, m.end())
        plain = len(TAG.sub("", inner))
        if plain > best_len:
            best, best_len = inner, plain
    if best_len >= 600:
        return best
    return text


def to_markdown(raw_html: str) -> str:
    s = SCRIPT_STYLE.sub("", raw_html)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<(br|/p|/div|/tr|/li)\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</t[dh]>", " | ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = htmllib.unescape(s)
    s = s.replace(" ", " ").replace("　", " ")
    lines = []
    for raw_line in s.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        line = re.sub(r"^(\|\s*)+\|?\s*$", "", line)  # 清理空表格行
        if line:
            lines.append(line)
    # 折叠连续空行 + 剔除导航噪声行
    out, blank = [], False
    for ln in lines:
        if not ln:
            if not blank:
                out.append("")
            blank = True
            continue
        blank = False
        if len(ln) < 60 and NAV_NOISE.search(ln):
            continue
        if re.fullmatch(r"[\s>＞|／/·、,，]{0,10}(首页|信息公开|政务公开)[\s>＞|／/·、,，a-zA-Z0-9]{0,60}", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


END_MARK = re.compile(
    r"(责任编辑|扫一扫在手机|打印本页|关闭窗口|上一篇|下一篇|附件[:：]|"
    r"分享到|来源[:：]\s*$|发布时间[:：]|扫一扫|访问量|点击量|字体[:：]|"
    r"网站首页|主办单位|承办单位|上一篇|下一篇)",
)

TITLE_STOP = re.compile(
    r"(索引号|信息名称|发文机关|发文字号|成文日期|发布日期|信息来源|主题分类|"
    r"文号|有效期|公开方式|发布时间|发布机构)",
)


def body_slice(lines: list[str]) -> list[str]:
    """从全文清洗结果里定位正文区间：跳过页头元信息行，截到尾部标记前。"""
    start = None
    for i, ln in enumerate(lines):
        if len(ln) < 25:
            continue
        if TITLE_STOP.search(ln) and len(ln) < 80:
            continue
        if NAV_NOISE.search(ln) and len(ln) < 60:
            continue
        start = i
        break
    if start is None:
        return lines
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if END_MARK.search(lines[j]):
            end = j
            break
    return lines[start:end]


def main():
    src_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "md")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.html")):
        text = detect_encoding(p.read_bytes())
        md = to_markdown(pick_body(text))
        mode = "container"
        # 兜底：容器提取太短 -> 全文清洗 + 正文区间定位
        if len(md) < 1200:
            full = to_markdown(text)
            lines = [ln for ln in full.split("\n") if ln.strip()]
            sliced = "\n".join(body_slice(lines))
            if len(sliced) > len(md):
                md, mode = sliced, "full_slice"
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", htmllib.unescape(TAG.sub("", m.group(1)))).strip()
        fm = [
            "---",
            f"source_url: {p.resolve().as_uri()}",
            f"source_file: {p.name}",
            f"html_title: {title[:200]}",
            "doc_type: government_web_policy",
            f"extract_mode: {mode}",
            "language: zh-CN",
            "---",
            "",
        ]
        (out_dir / (p.stem + ".md")).write_text("\n".join(fm) + md, encoding="utf-8")
        print(f"{p.name:45s} -> {p.stem}.md  {len(md):7d} chars  [{mode}]")


if __name__ == "__main__":
    main()
