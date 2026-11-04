# -*- coding: utf-8 -*-
"""按 **qid 交集**对比两个生成层批次（同一批题、同一口径）。

## 为什么需要这个工具

2026-11-04 的端到端批次（`structural + top-8`）跑到一半**额度耗尽**，
只答了 81/306 题。这批数据**不能直接和 306 题的基线比** ——
分母不同，比出来的差值里混着"哪几题没跑"。

但它**仍然可用**：只要取两个批次的 **qid 交集**，
在**同一批题**上比，分母就一致了。

🔴 **取交集比"各报各的"更重要**：
- 直接比 81 vs 306 → 差值里混着题目难度差异，不是配置差异；
- 取交集后 → 差值才是配置差异。

## 口径

- 两批都用 `run_generation_eval.rescore()` **离线重算**（统一口径，不调 API）；
- 指标与 `compare_generation.py` 一致（软/严格两把尺子、Wilson CI）。

跑法：
    python tools/compare_qid_subset.py \
        --a "fixed+top5=eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl" \
        --b "structural+top8=eval_runs/20261104_1124_generation_retrieval_k8.answers.jsonl"
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from run_generation_eval import rescore, summarize  # noqa: E402

RUN_DIR = ROOT / "eval_runs"


def load(path):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / path
    if not p.exists():
        raise SystemExit(f"找不到 {p}")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return {r["qid"]: r for r in recs}


def fmt(ci, d=3):
    if not ci:
        return "-"
    return f"{ci[0]:.{d}f} [{ci[1]:.{d}f},{ci[2]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="标签=路径")
    ap.add_argument("--b", required=True, help="标签=路径")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    batches = {}
    for spec in (args.a, args.b):
        label, _, path = spec.partition("=")
        batches[label.strip()] = (path.strip(), load(path.strip()))

    (la, ra), (lb, rb) = batches.values()
    common = sorted(set(ra) & set(rb))
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))
    print(f"批次 A = {la}（{len(ra)} 题）")
    print(f"批次 B = {lb}（{len(rb)} 题）")
    print(f"**qid 交集 = {len(common)} 题**"
          f"（A 独有 {len(only_a)}，B 独有 {len(only_b)}）")
    if len(common) < len(ra) or len(common) < len(rb):
        print("→ 取交集后**分母一致**，下面的差值才是配置差异，不是题目难度差异")

    out_rows = []
    for label, (_path, recs) in batches.items():
        sub = [recs[q] for q in common]
        sub = rescore(sub, threshold=args.threshold)
        s = summarize(sub, "retrieval", "L4_pure_bm25", 0.0, [], None)
        out_rows.append((label, s))

    print(f"\n| 指标 | " + " | ".join(l for l, _ in out_rows) + " |")
    print("|---" * (len(out_rows) + 1) + "|")

    def row(name, fn):
        print(f"| {name} | " + " | ".join(fn(s) for _l, s in out_rows) + " |")

    row("n（交集）", lambda s: str(s["n_answered"]))
    row("上下文里的过期 chunk（应为 0）",
        lambda s: str(s["stale_in_context_questions"]))
    row("引用有效率", lambda s: fmt(s["citation"]["citation_valid_rate"]))
    row("条款可定位率", lambda s: fmt(s["citation"].get("clause_located_rate")))
    for qt in ("事实题", "数值题", "版本陷阱题", "跨文档推理题"):
        print(f"\n**{qt}**\n")
        print(f"| 指标 | " + " | ".join(l for l, _ in out_rows) + " |")
        print("|---" * (len(out_rows) + 1) + "|")
        row("n", lambda s, qt=qt: str(s["per_type"].get(qt, {}).get("n", "-")))
        row("key_point 覆盖（软 0.7）",
            lambda s, qt=qt: fmt(s["per_type"].get(qt, {}).get("key_point_coverage")))
        row("key_point 覆盖（严格）",
            lambda s, qt=qt: fmt(s["per_type"].get(qt, {}).get("key_point_coverage_strict")))
        row("题目级全命中率",
            lambda s, qt=qt: fmt(s["per_type"].get(qt, {}).get("key_point_questions_full_hit")))
        if qt == "数值题":
            row("数值精确匹配",
                lambda s: fmt(s["per_type"].get(qt, {}).get("numeric_exact_match_rate")))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "soft_threshold": args.threshold,
        "n_common": len(common),
        "common_qids": common,
        "n_only_a": len(only_a), "n_only_b": len(only_b),
        "batches": [{"label": l, "source": p, "summary": s}
                    for (l, (p, _r)), (_l2, s) in zip(batches.items(), out_rows)],
        "note": "两批都取 qid 交集后离线重算（统一口径），分母一致。",
    }
    out = pathlib.Path(args.out) if args.out else \
        RUN_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M')}_compare_subset.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
