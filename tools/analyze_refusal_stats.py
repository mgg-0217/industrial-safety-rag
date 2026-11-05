# -*- coding: utf-8 -*-
"""拒答题统计分析：分层 + **按主体聚类的稳健 CI**。

## 为什么需要聚类 CI

拒答题的**统计独立单位是"主体"，不是"题"**。

主集那 42 条拒答题的**语义模板只有 1 个**（全是「假位号 X 的检修周期 / 维护项目 /
作业许可」），只换了位号和措辞。

⚠️ **这里要说清楚一件事**：主集那 42 条的**主体各不相同**（T-9001~T-9040），
所以它们**样本层面是独立的** —— Wilson CI 并不会"虚假地窄"。
真正的问题是另一个：

> **42 个独立样本，只覆盖 1 种拒答机制。**
> CI 反映的是「这一种机制下的误答率」，
> 却被当成「系统整体拒答能力」来读 → **单机制外推失效**。

如果只是把假位号从 40 个扩到 150 个，CI 会**继续收窄**（数学上没错，样本确实更多了），
但**真正的不确定性来源是机制差异，不是抽样误差** →
**CI 收窄 ≠ 结论更可信**。

本工具给出两个 CI：

| 口径 | 算法 | 回答的问题 |
|---|---|---|
| `naive` | Wilson(total_false, total_n) | 常规报法（假设题与题独立） |
| `cluster` | **按主体有放回重采样**的 bootstrap 2.5/97.5 分位 | 把同主体题当**一个**样本时的真实不确定性 |

两者之比 `naive_over_cluster_width_ratio`：
* **>1** → naive 更窄 = **虚假精度**（同主体重复时出现）
* **≈1** → 主体本就独立（每主体 1 题）
* **<1** → naive 更宽（低误答率时 Wilson 下界贴 0 的边界效应）

真正要看的不是这两个 CI，而是**分层后的机制间差异**（`mechanism_spread`）。

## 用法

```bash
python tools/analyze_refusal_stats.py \
    --batch "主集fixed=eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl" \
    --batch "扩展fixed=eval_runs/<...>_refuse_fixed.answers.jsonl" \
    --batch "扩展structural=eval_runs/<...>_refuse_structural.answers.jsonl" \
    --out eval_runs/<...>_refusal_analysis.json
```
"""
import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import scoring as SC  # noqa: E402

B = 2000          # bootstrap 次数
SEED = 20261105   # 固定随机种子 —— 结果可复现


def wilson_width(w):
    """Wilson 区间宽度（w = [p, lo, hi] 或 None）。"""
    if not w:
        return None
    return round(w[2] - w[1], 4)


def cluster_bootstrap(groups, n_boot=B, seed=SEED):
    """按主体有放回重采样，给出总体误答率的 2.5/97.5 分位。

    :param groups: ``{主体: [(误答?, ...), ...]}``，每个元素是该主体下各题的误答标记
    :returns: ``(点估计, lo, hi, 宽度)``
    """
    keys = list(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    tot_false = sum(1 for k in keys for x in groups[k] if x)
    tot_n = sum(len(groups[k]) for k in keys)
    point = tot_false / tot_n if tot_n else None

    rates = []
    for _ in range(n_boot):
        f = n = 0
        for _ in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            grp = groups[k]
            for x in grp:
                n += 1
                if x:
                    f += 1
        if n:
            rates.append(f / n)
    rates.sort()
    lo = rates[int(0.025 * len(rates))]
    hi = rates[min(int(0.975 * len(rates)), len(rates) - 1)]
    return (round(point, 4), round(lo, 4), round(hi, 4), round(hi - lo, 4))


def analyze(path):
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    refusals = [r for r in records if r.get("question_type") == "拒答题"]
    if not refusals:
        return None

    def is_false(r):
        return r.get("refusal_verdict") == "false_answer"

    def is_false_lenient(r):
        """宽松口径：只看"有没有明确表态说没有"（K-53）。

        旧批次没有落盘这个字段 → 现算（答案原文在 answers.jsonl 里）。
        """
        v = r.get("refusal_verdict_lenient")
        if v is None:
            subj = r.get("equipment_id") or SC.extract_subject(r.get("question") or "")
            v, _ = SC.refusal_judgement(r.get("answer") or "", subject=subj,
                                        question=r.get("question"),
                                        mode="lenient")
        return v == "false_answer"

    def is_manual(r):
        return r.get("refusal_verdict") == "manual"

    def behaviour_mode(r):
        """🔴 把两个口径**交叉**成三种行为模式（K-53 的产出）。

        两个口径各自都丢信息，交叉起来才能看出模型到底干了什么：

        | 模式 | 严格 | 宽松 | 含义 |
        |---|---|---|---|
        | `1_完全没意识到` | 误答 | **误答** | 压根没意识到语料里没这个主体 → 直接把通用条款当它的规定 |
        | `2_归因污染` | 误答 | 通过 | 说了"未找到/资料未提供"，**但仍把通用条款断言给主体** |
        | `2b_归因隔离` | **bounded** | 通过 | 说了"未找到"，**并且显式标注**"通用要求不等同于该主体的专门规定" |
        | `3_完全正确` | 通过 | 通过 | 明确"上述是通用要求，未找到该主体的专属规定" |
        | `4_人工复核` | 人工复核 | — | 混合形态，单列 |

        只看宽松口径，模式 2 / 2b / 3 会被混成"通过"；
        只看严格口径，模式 1 和 2 会被混成"误答"。

        🔴 `2b`（bounded）是 `SYSTEM_V2` 提示词带来的**新形态**：
        模型既表态了、又做了归因隔离 → 它**不该**被算成"给不存在的装置编造规定"，
        但也不该和"完全正确"混为一谈（它仍然列了一堆通用条款）。
        """
        lv = r.get("refusal_verdict_lenient")
        if lv is None:
            subj = r.get("equipment_id") or SC.extract_subject(r.get("question") or "")
            lv, _ = SC.refusal_judgement(r.get("answer") or "", subject=subj,
                                        question=r.get("question"),
                                        mode="lenient")
        sv = r.get("refusal_verdict")
        if sv == "manual":
            return "4_人工复核"
        if lv == "false_answer":
            return "1_完全没意识到"
        if sv == "bounded":
            return "2b_归因隔离"
        if sv == "false_answer":
            return "2_归因污染"
        return "3_完全正确"

    # -- 主体归组（统计独立单位）------------------------------------------
    groups = defaultdict(list)
    for r in refusals:
        subj = r.get("equipment_id") or SC.extract_subject(r.get("question") or "")
        groups[subj or f"<no-subject:{r.get('qid')}>"].append(is_false(r))

    n = len(refusals)
    n_false = sum(1 for r in refusals if is_false(r))
    verdicts = Counter(r.get("refusal_verdict") for r in refusals)
    n_pass = verdicts.get("pass", 0)
    n_manual = verdicts.get("manual", 0)

    naive = SC.wilson(n_false, n)
    naive_dec = SC.wilson(n_false, n_pass + n_false) if (n_pass + n_false) else None
    cl = cluster_bootstrap(groups)

    # 🔴 三分类行为模式（两个口径交叉）
    modes = Counter(behaviour_mode(r) for r in refusals)
    mode_stats = {}
    for mk in ("1_完全没意识到", "2_归因污染", "2b_归因隔离",
               "3_完全正确", "4_人工复核"):
        c = modes.get(mk, 0)
        if c:
            mode_stats[mk] = {"n": c, "rate": round(c / n, 4),
                              "wilson": SC.wilson(c, n)}

    out = {
        "file": pathlib.Path(path).name,
        "n_questions": n,
        "n_subjects": len(groups),
        "subjects_per_question": round(n / len(groups), 2) if groups else None,
        "verdicts": dict(verdicts),
        "n_false_answer": n_false,
        "n_decidable": n_pass + n_false,
        "naive_wilson": naive,
        "naive_wilson_width": wilson_width(naive),
        # 🔴 宽松口径（K-53）：只看"有没有明确表态说没有"
        "n_false_answer_lenient": sum(1 for r in refusals if is_false_lenient(r)),
        "naive_wilson_lenient": SC.wilson(
            sum(1 for r in refusals if is_false_lenient(r)), n),
        "n_manual": sum(1 for r in refusals if is_manual(r)),
        # 🔴 bounded（`SYSTEM_V2` 提示词下的新形态：表态了 + 显式做了归因隔离）
        #    既不是误答、也不是"完全正确"（它仍然列了通用条款）→ 单列。
        "n_bounded": sum(1 for r in refusals
                          if r.get("refusal_verdict") == "bounded"),
        # 把 bounded 并入"已妥善处理"后的误答率（便于与不含该形态的批次对比）
        "naive_wilson_bounded_as_pass": SC.wilson(
            n_false, n - sum(1 for r in refusals
                             if r.get("refusal_verdict") == "bounded") - n_manual),
        # 🔴 三分类行为模式
        "behaviour_modes": mode_stats,
        "cluster_bootstrap": {
            "point": cl[0], "lo": cl[1], "hi": cl[2], "width": cl[3],
            "n_boot": B, "seed": SEED,
        } if cl else None,
        # 🔴 naive CI 与 cluster CI 的宽度比。
        #    >1 → naive 更窄 = **虚假精度**（同主体重复时出现）
        #    <1 → naive 更宽（低误答率时 Wilson 下界贴 0 导致的边界效应）
        #    ≈1 → 主体本就独立（每主体 1 题）时两者应当接近
        "naive_over_cluster_width_ratio": (
            round(wilson_width(naive) / cl[3], 2)
            if cl and wilson_width(naive) and cl[3] else None),
    }

    # -- 按机制分层 ------------------------------------------------------
    by_mech = defaultdict(list)
    for r in refusals:
        by_mech[r.get("refusal_mechanism") or "unspecified"].append(r)
    mech = {}
    for mk, rs in sorted(by_mech.items()):
        mf = sum(1 for r in rs if is_false(r))
        mf_len = sum(1 for r in rs if is_false_lenient(r))
        m_groups = defaultdict(list)
        for r in rs:
            subj = r.get("equipment_id") or SC.extract_subject(r.get("question") or "")
            m_groups[subj or r.get("qid")].append(is_false(r))
        m_cl = cluster_bootstrap(m_groups)
        mech[mk] = {
            "label": next((x.get("refusal_mechanism_label") for x in rs
                           if x.get("refusal_mechanism_label")), mk),
            "n": len(rs),
            "n_subjects": len(m_groups),
            "n_false_answer": mf,
            "rate": round(mf / len(rs), 4),
            "rate_lenient": round(mf_len / len(rs), 4),
            "wilson": SC.wilson(mf, len(rs)),
            "wilson_lenient": SC.wilson(mf_len, len(rs)),
            "cluster_bootstrap": ({"point": m_cl[0], "lo": m_cl[1],
                                   "hi": m_cl[2], "width": m_cl[3]}
                                  if m_cl else None),
            "verdicts": dict(Counter(x.get("refusal_verdict") for x in rs)),
        }
    out["by_mechanism"] = mech

    # -- 按**题干骨架**分组 ------------------------------------------------
    # 🔴 主集 42 条（假位号 T-9001~T-9040）与对照组 160 条（假位号 T-9041+）
    #    是**同一机制、同一策略**，误答率却差 13 倍（0.024 vs 0.319）。
    #    差异可能来自**题干的具体程度** —— 对照组内部每主体 4 个模板，
    #    主体集合相同、只换问法 → 正好把"模板"这个变量单独隔离出来。
    by_skel = defaultdict(list)
    for r in refusals:
        subj = r.get("equipment_id") or ""
        skel = (r.get("question") or "").replace(subj, "{s}")
        by_skel[skel].append(r)
    skel_stats = {}
    for sk, rs in sorted(by_skel.items(), key=lambda kv: -len(kv[1])):
        sf = sum(1 for r in rs if is_false(r))
        skel_stats[sk] = {
            "n": len(rs),
            "n_subjects": len({r.get("equipment_id") for r in rs}),
            "n_false_answer": sf,
            "rate": round(sf / len(rs), 4),
            "wilson": SC.wilson(sf, len(rs)),
        }
    out["by_question_template"] = skel_stats


    # -- 总体误答率的"机制间差异"检验（不是两两比，是整体分层后的极差）----
    if len(mech) > 1:
        rates = {k: v["rate"] for k, v in mech.items() if v["n"]}
        best = max(rates, key=rates.get)
        worst = min(rates, key=rates.get)
        out["mechanism_spread"] = {
            "best": best, "best_rate": rates[best],
            "worst": worst, "worst_rate": rates[worst],
            "spread": round(rates[best] - rates[worst], 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", required=True,
                    help="名称=路径，可重复")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    for spec in args.batch:
        name, _, path = spec.partition("=")
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = ROOT / path
        if not p.exists():
            print(f"跳过（不存在）：{name} -> {p}")
            continue
        r = analyze(p)
        if r:
            results[name] = r

    print(json.dumps(results, ensure_ascii=False, indent=1))

    if args.out:
        outp = ROOT / args.out if not pathlib.Path(args.out).is_absolute() \
            else pathlib.Path(args.out)
        outp.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"-> {outp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
