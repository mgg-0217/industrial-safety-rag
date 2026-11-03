"""PDF 文本行的**基线重建** —— 修抽取层的行序错乱。

## 要修的病

`pdfplumber` 的 `page.extract_text()` 用字符的**字体度量**（ascent/descent）算 bbox，
再按 `top` 排序成行。**当一个 PDF 里混用多套字体、且各字体的 ascent/descent 不一致时，
bbox 的 top 会整体错位** —— 明明在视觉上同一行的字符，被拆成 3–4 条"行"。

实测（`AQ_T3034_2022_huagong_anquan.pdf` 第 10 页）：

| 行 | pdfplumber 的 `top` | **真实基线**（文本矩阵） | 字体 |
|---|---|---|---|
| 中文正文 | 132.30 | **702.65** | FZSSK--GBK1-0 |
| 条款号数字 | 133.90 | **701.39** | E-FZ |
| 标点 | 139.30 | **702.94** | E-BZ |
| 条款号句点 + `GB/T37243` | 140.90 | **701.39** | E-BZ |

按 `top` 分行 → 4 行；按**基线**分行 → 1 行（基线跨度只有 1.55pt，
真实行距是 15.91pt）。分错的后果是正文读不通：

```
４７１２ 在工厂选址 总平面布局时 应符合有关设计标准的要求 并按照 的要求进行定
．．． 、 , , GB/T３７２４３
```

而 PDF **渲染出来是完全正常的**（已用 pypdfium2 渲染 300dpi 核对过），
所以这是**抽取层的病，不是 PDF 的病**，而且是**可修的**。

## 修法

1. 取 `page.chars`；
2. 用**文本矩阵的平移分量** `matrix[5]`（PDF 空间里的基线 y）算出每页每字符的基线；
3. 按基线聚类成行（容差远小于行距）；
4. 行内**按 `x0` 排序**后拼接。

第 4 步是关键：错位出去的 run **x 坐标仍在原位**，
按 x 重排就还原了阅读顺序。

## 为什么不用 `y_tolerance` 参数

`extract_text(y_tolerance=N)` 是**全局单一容差**，在本语料上试过 3/8/10/12/15/20：
N≥8 时会把**相邻两行正文**也并进来（行距 15.9pt，容差 10 会跨行），
输出变成整页字符乱序的大字符串 —— 比原状更糟。
真实行距是**逐页不同**的（正文 15.9pt、目录 12pt、表格 9pt），
全局容差无法同时适配"跨字体错位 8.5pt"和"行距 9pt"这两个尺度。

## 边界

- **只处理 upright 字符**（`upright=True`，即 `matrix` 无旋转）。
  旋转文字（水印、竖排）另作处理或不处理 —— 见 `is_upright`。
- **不负责表格** —— 表格走 `pdfplumber.extract_tables()`，由 `pdf_to_md.py` 负责。
- **不改变字符本身** —— 全角数字、``«»`` 等编码问题由 `src/normalize.py` 处理。
  本模块只解决**行的切分与顺序**。
"""
from __future__ import annotations

import re

#: 基线聚类容差（pt）。真实行距实测 9–16pt，跨字体基线错位 1.5–1.6pt。
#:
#: **取值 3.0 是扫参出来的，不是拍的**：用 OCR 真值（渲染页 → OCR → 比字符相似度）
#: 对 9 个采样页扫 2.0–8.0，得分平台在 **2.0–4.0 全部相同（0.8908）**，
#: ≥5.0 开始掉（0.8785，并行被并成一行），≤1.5 掉（0.8891，行被拆开）。
#: 取平台中部 3.0（也正好是 pdfplumber 自己的默认 `y_tolerance`）。
#: 复现：`python tools/sweep_baseline_tol.py`
DEFAULT_BASELINE_TOL = 3.0

#: 行内插入空格的 x 间隙阈值（pt）。
#: 与 pdfplumber `extract_text()` 的默认 `x_tolerance=3` 保持一致 ——
#: 重建要还原的是**同一份**文本，间距规则也得一致，否则空格数会变。
DEFAULT_X_TOL = 3.0


def is_upright(ch):
    """字符是否正立（无旋转/镜像）。"""
    if "upright" in ch:
        return bool(ch["upright"])
    m = ch.get("matrix")
    if not m:
        return True
    a, b, c, d = m[0], m[1], m[2], m[3]
    return abs(b) < 1e-6 and abs(c) < 1e-6 and a > 0 and d > 0


def char_baseline(ch, page_height):
    """字符基线在"页面自上而下"坐标系里的 y。

    `matrix[5]` 是文本矩阵的平移分量 = 基线在 PDF 空间（原点左下）的 y。
    `page_height - matrix[5]` 换成自上而下的坐标。

    🔴 用 `top`/`bottom` 是错的 —— 那两个值由**字体度量**算出，
    混用字体时会整体错位（见模块 docstring）。
    """
    return page_height - ch["matrix"][5]


def _cluster_baselines(items, tol):
    """按基线值聚类。`items` = [(baseline_top, size, payload), ...]。

    用**单遍聚类**而不是两两比较：按基线排序后，与**当前簇的最后一个**
    比较（不是簇首），这样一串轻微递增的基线不会被拆散。
    """
    if not items:
        return []
    items = sorted(items, key=lambda t: t[0])
    clusters = [[items[0]]]
    for it in items[1:]:
        if it[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return _merge_small_bands(clusters)


def _median_size(band):
    """带内字号中位数。"""
    sizes = sorted(it[1] for it in band if it[1])
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]


#: 小字号判据：字号 ≤ 主带的这个比例 → 认为是上标/下标
SMALL_SIZE_RATIO = 0.8

#: 小字号带并入主带的基线距离上限（相对主带字号）
SMALL_MERGE_RATIO = 0.6


def _merge_small_bands(bands):
    """把**上标/下标**这类"小字号带"并回相邻的主带。

    ## 为什么需要这一步

    `m³` 的上标 `3` 在 PDF 里是**另一个字号**（实测 6.96 vs 正文 10.56 = 0.66×），
    基线也**更高**（实测偏移 5.03pt）——
    比行内字体错位（1.5–1.6pt）大得多，基线容差根本兜不住，
    于是 `100m3` 被切成 `100m` 和孤零零的 `3`。

    ## 判据

    邻接两带里，**字号明显小的那一带**（≤ 主带的 0.8 倍），
    且基线距离 ≤ 主带字号的 0.6 倍 → 并入主带。

    为什么用"字号"而不是"基线距离"当主判据：
    - 纯基线距离无法区分上标与"表格相邻行"（两者距离可能一样）；
    - 字号是**排版意图的直接证据** —— 上标必然用更小的字号，
      而表格相邻行是同字号的。

    ⚠️ 表格里的整行小字**不会**被误并：两带字号相同（比值 1.0），
    不满足 0.8 的条件。
    """
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i in range(len(bands) - 1):
            upper, lower = bands[i], bands[i + 1]
            mu, ml = _median_size(upper), _median_size(lower)
            if not mu or not ml:
                continue
            gap = lower[0][0] - upper[-1][0]
            # 上带是小字（上标）→ 并入下带
            if mu <= SMALL_SIZE_RATIO * ml and gap <= SMALL_MERGE_RATIO * ml:
                bands[i + 1] = lower + upper
                del bands[i]
                changed = True
                break
            # 下带是小字（下标）→ 并入上带
            if ml <= SMALL_SIZE_RATIO * mu and gap <= SMALL_MERGE_RATIO * mu:
                bands[i] = upper + lower
                del bands[i + 1]
                changed = True
                break
    return bands


def _needs_space(prev_ch, ch, x_tol=DEFAULT_X_TOL):
    """两个相邻字符之间要不要插空格。

    判据与 pdfplumber 的 `x_tolerance` 一致：**x 间隙超过阈值就是原文的词间空隙**。
    这条对中文同样成立 —— 作业票这类**键值对表单**的
    "作业申请单位 甲醇车间" 之间的空隙是版式上的真实分隔，
    丢掉空格会让表单挤成一坨（实测 `kaiping_zuoyepiao` 就是这样）。

    🔴 但**不能**对所有大间隙都插：重建后不同字体的 run 会在 x 上**重叠**
    （实测 AQ_T3034 的 "的要求" 与 "GB/T37243" 的 x 区间交叠），
    此时间隙是负数，绝不能插空格。重叠一律直接拼接。
    """
    p, c = prev_ch["text"], ch["text"]
    if not p or not c:
        return False
    gap = ch["x0"] - prev_ch["x1"]
    if gap <= 0:
        return False            # 重叠：直接拼
    return gap > x_tol


def reconstruct_lines(chars, page_height, baseline_tol=DEFAULT_BASELINE_TOL,
                      x_tol=DEFAULT_X_TOL):
    """把字符重建成文本行。返回 `list[str]`。

    只取 upright 字符；旋转文字（水印）会被跳过。
    """
    upright = [c for c in chars if is_upright(c)]
    if not upright:
        return []
    items = [(char_baseline(c, page_height), c.get("size") or 0.0, c)
             for c in upright]
    clusters = _cluster_baselines(items, baseline_tol)

    lines = []
    for cl in clusters:
        cl.sort(key=lambda t: t[2]["x0"])
        buf = []
        prev = None
        for _bl, _sz, ch in cl:
            if prev is not None and _needs_space(prev, ch, x_tol):
                buf.append(" ")
            buf.append(ch["text"])
            prev = ch
        text = "".join(buf)
        if text.strip():
            lines.append(text)
    return lines


def extract_page_lines(page, baseline_tol=DEFAULT_BASELINE_TOL, x_tol=DEFAULT_X_TOL):
    """从一个 pdfplumber Page 抽出文本行（页面级入口）。"""
    try:
        chars = page.chars
    except Exception:
        return []
    if not chars:
        return []
    return reconstruct_lines(chars, page.height, baseline_tol=baseline_tol, x_tol=x_tol)
