# -*- coding: utf-8 -*-
"""生成层四组配置的**统一口径**对比表（PRD 7.3 的硬约束）。

## 为什么需要这个工具

四组配置的批次是**在不同时间、跑在不同代码版本**下跑出来的：

| 批次 | 跑的时候的代码 | 口径 |
|---|---|---|
| `retrieval_main` | 旧判分 | 只有严格匹配 |
| `retrieval_cite` | 新判分 | 严格 + 软匹配 + 单位折叠 |
| `closed_book_b1` | 新判分 | 同上 |
| `full_doc_b1` | 新判分 | 同上 |

🔴 **直接比这四个批次的 json 是错的** —— 数值题的 `numeric_exact_match_rate`
会因为"单位折不折"（K-43）而差出一截，看起来像模型能力差，其实只是口径差。

→ 本工具把**所有批次用当前代码重算一遍**（`rescore`，纯离线、不调 API），
再并排放到一张表里。**凡是要写进报告的数字，都应该来自这张表。**

## 口径说明（每格都带 n / 分子 / 分母）

- `kp_soft`：`key_point` 的**字符二元组覆盖率** ≥ 阈值（默认 0.7）算命中
- `kp_strict`：`key_point` 字面包含算命中
- 两个口径**都报、不合并**（K-42：只报一个 = 偷偷挑对自己有利的那个）
- 拒答：`pass` / `false_answer` / `manual` 三态；`manual` **不进误答率分母**
- 每格带 Wilson 95% 置信区间 —— 样本量小的时候，没有 CI 的点估计会骗人

跑法：
    python tools/compare_generation.py \
        --batches "检索组(fixed)=eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl,闭卷=eval_runs/20261104_1000_generation_closed_book_b1.answers.jsonl"
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import normalize as N            # noqa: E402
import scoring as SC             # noqa: E402
from run_generation_eval import rescore, summarize  # noqa: E402

RUN_DIR = ROOT / "eval_runs"

#: 题型顺序（与报告一致）
QTYPE_ORDER = ["事实题", "数值题", "版本陷阱题", "跨文档推理题", "拒答题"]


def _fmt_ci(ci, digits=3):
    """把 Wilson 区间格式化：``0.430 [0.370,0.491]``（点估计 + 区间）。

    🔴 点估计和区间必须**分开写** —— 曾经写成 ``0.430[0.430,0.370]``，
    读起来像"区间左端等于点估计"，纯属排版误导。
    """
    if not ci:
        return "-"
    pt, lo, hi = ci[0], ci[1], ci[2]
    return f"{pt:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def load_batch(path):
    """读一份 ``*_answers.jsonl``，并把它的**配置名**一起取回来。

    🔴 配置名（`retrieval` / `closed_book` / `full_doc`）**不在逐题明细里**，
    只在同批次的汇总 json 里 —— 少了这一步，
    "版本命中率"这种只对检索组有意义的指标就会被一律标成 n/a。
    """
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / path
    if not p.exists():
        raise SystemExit(f"找不到 {p}")
    records = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    cfg = records[0].get("config") if records else None
    if not cfg:
        sib = pathlib.Path(str(p).replace(".answers.jsonl", ".json"))
        if sib.exists():
            try:
                cfg = json.loads(sib.read_text(encoding="utf-8")).get("config")
            except Exception:
                cfg = None
    if not cfg:
        # 兜底：有没有检索出来的上下文块
        cfg = "retrieval" if any(r.get("ctx_chunk_ids") for r in records) else "?"
    return records, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True,
                    help="逗号分隔：标签=路径（路径指向 *_answers.jsonl）")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="软匹配的二元组覆盖率阈值（默认 0.7）")
    ap.add_argument("--out", default="", help="对比表 json 的输出路径")
    args = ap.parse_args()

    labels, paths = [], []
    for part in args.batches.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, path = part.partition("=")
        labels.append(label.strip())
        paths.append(path.strip())

    tables = []
    for label, path in zip(labels, paths):
        records, cfg = load_batch(path)
        n_raw = len(records)
        records = rescore(records, threshold=args.threshold)
        summary = summarize(records, cfg, "L4_pure_bm25", 0.0, [], None)
        tables.append({
            "label": label,
            "source": path,
            "n_raw": n_raw,
            "summary": summary,
        })

    # ------------------------------------------------------------------
    # 打印 markdown 表
    # ------------------------------------------------------------------
    print(f"\n统一口径对比（阈值 {args.threshold}，全部离线重算，不调 API）\n")

    header = "| 指标 | " + " | ".join(t["label"] for t in tables) + " |"
    sep = "|---" * (len(tables) + 1) + "|"
    print(header)
    print(sep)

    def row(name, getter):
        cells = []
        for t in tables:
            cells.append(getter(t["summary"]))
        print(f"| {name} | " + " | ".join(cells) + " |")

    row("题数 n", lambda s: str(s["n_answered"]))
    row("上下文里的过期 chunk（应为 0）",
        lambda s: str(s["stale_in_context_questions"]))
    row("引用数 / 无效引用",
        lambda s: f"{s['citation']['n_refs']} / {s['citation']['n_invalid']}")
    row("引用有效率", lambda s: _fmt_ci(s["citation"]["citation_valid_rate"]))
    row("**条款可定位率**",
        lambda s: _fmt_ci((s["citation"].get("clause_located_rate"))))
    row("页码可定位率",
        lambda s: _fmt_ci(s["citation"].get("page_located_rate")))

    for qt in QTYPE_ORDER:
        print(f"\n**{qt}**\n")
        print(header)
        print(sep)
        row("n", lambda s, qt=qt: str(s["per_type"].get(qt, {}).get("n", "-")))
        if qt == "拒答题":
            row("误答率（分母=拒答题总数）",
                lambda s: _fmt_ci(s["per_type"].get(qt, {}).get("refusal_false_answer_rate")))
            row("误答率（分母=只算可判样本）",
                lambda s: _fmt_ci(s["per_type"].get(qt, {}).get("refusal_false_answer_rate_decidable")))
            row("判定分布",
                lambda s: str(s["per_type"].get(qt, {}).get("refusal_verdicts")))
        else:
            row("key_point 覆盖（软）",
                lambda s, qt=qt: _fmt_ci(s["per_type"].get(qt, {}).get("key_point_coverage")))
            row("key_point 覆盖（严格）",
                lambda s, qt=qt: _fmt_ci(s["per_type"].get(qt, {}).get("key_point_coverage_strict")))
            row("题目级全命中率",
                lambda s, qt=qt: _fmt_ci(s["per_type"].get(qt, {}).get("key_point_questions_full_hit")))
            if qt == "数值题":
                row("数值精确匹配",
                    lambda s: _fmt_ci(s["per_type"].get(qt, {}).get("numeric_exact_match_rate")))
            row("版本命中率（检索组口径：期望文档在不在上下文里）",
                lambda s, qt=qt: (_fmt_ci(s["per_type"].get(qt, {}).get("version_hit_rate"))
                                  if s.get("config") == "retrieval" else "n/a（无检索上下文）"))

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "soft_threshold": args.threshold,
        "note": "所有数字都由当前代码对各批次离线重算得到（口径统一），"
                "不是各批次 json 里原样的数字。",
        "batches": [
            {
                "label": t["label"],
                "source": t["source"],
                "n_raw": t["n_raw"],
                "summary": t["summary"],
            } for t in tables
        ],
    }
    out = pathlib.Path(args.out) if args.out else \
        RUN_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M')}_generation_compare.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
