# -*- coding: utf-8 -*-
"""生成层评测（PRD M6 / M7 + PRD 7.4 的 B1 基线）。

## 三组配置

| 编号 | 配置 | 对照什么 |
|---|---|---|
| **`retrieval`** | BM25 + **版本过滤** + 引用/版本卡由代码拼 | 本项目配置 |
| **`closed_book`** | **不给任何上下文**，直接问 | 检索是不是真的有用（B1a） |
| **`full_doc`** | 给该题所属文档的**正文全文**（截断 12000 字） | "把语料塞给模型"行不行（B1b） |

🔴 **为什么主配置用纯 BM25 而不是 PRD 写的"混合检索"**：
M5 实测（见 `docs/检索基线与切分策略对比.md`）本机没有可用的预训练句向量，
向量路（TF-IDF+SVD）**是负贡献**（R@5 0.633–0.724 vs BM25 的 0.856–0.883）。
PRD 4.6 的 L4 降级（纯 BM25）就是本项目的实际配置。
再跑一遍混合检索只会浪费 API 额度，检索层的结论已经足够。

## 指标（全部写明 n / 分子 / 分母，PRD 7.3 的硬约束）

| 题型 | 指标 |
|---|---|
| 事实题 | `key_point_coverage`（命中 kp 数 / 总 kp 数）+ 题目级全命中率 |
| 数值题 | `numeric_exact_match_rate`（命中数值数 / 总数值数） |
| 版本陷阱题 | `version_hit_rate`（期望文档在上下文的题数 / 总数）+ key_point 覆盖 |
| 拒答题 | `refusal_false_answer_rate`（误答题数 / 拒答题总数） |
| 跨文档推理题 | `key_point_coverage` |
| 全部 | `citation_valid_rate`（有效引用数 / 引用总数） |

## 端到端断言（PRD 5.5）

每题都检查**上下文里有没有过期 chunk** —— 在 `current` / `as_of` 口径下必须是 0。
这不是"指标"，是**断言**：不为 0 就说明版本治理在链路里漏了。

跑法：
    python tools/run_generation_eval.py --configs retrieval --limit 10   # 冒烟
    python tools/run_generation_eval.py --configs retrieval,closed_book,full_doc
"""
import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import answering as AN      # noqa: E402
import embed as EMB        # noqa: E402
import normalize as N      # noqa: E402
import retrieval as RT     # noqa: E402
import scoring as SC       # noqa: E402
import version_card as VC  # noqa: E402

CHUNK_DIR = ROOT / "data" / "chunks"
EVAL_DIR = ROOT / "data" / "eval"
RUN_DIR = ROOT / "eval_runs"
MD_DIR = ROOT / "data" / "md"


# ---------------------------------------------------------------------------
# 语料全文（B1b 用）
# ---------------------------------------------------------------------------

_MD_CACHE = {}


def doc_body(doc_id, max_chars=12000):
    """取某份文档的正文（去 front-matter），截断到 ``max_chars``。"""
    if doc_id in _MD_CACHE:
        return _MD_CACHE[doc_id]
    p = MD_DIR / f"{doc_id}.md"
    if not p.exists():
        _MD_CACHE[doc_id] = ""
        return ""
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("---"):
        end = raw.find("\n---\n", 3)
        if end != -1:
            raw = raw[end + 5:]
    out = raw[:max_chars]
    _MD_CACHE[doc_id] = out
    return out


# ---------------------------------------------------------------------------
# 单题打分
# ---------------------------------------------------------------------------

def score_question(q, result, ctx_docs):
    """按题型判分，返回一个分数字典。"""
    qt = q.get("question_type")
    ans = result.get("answer_text") or ""
    s = {}

    if qt == "拒答题":
        # 🔴 **主体从数据集传，不靠正则从题干猜**（K-51）：
        #    `extract_subject` 的 `_TAG_RE` 只认位号形态（`T-9001`），
        #    中文主体（"加氢反应器""液氩"）匹配不到 → subject=None →
        #    "主体被断言规定"这条检测**静默失效**。
        #    题集里 `equipment_id` 字段就是主体，直接用。
        v, why = SC.refusal_judgement(ans, subject=q.get("equipment_id"),
                                      question=q.get("question"))
        s["refusal_verdict"] = v
        s["refusal_reason"] = why
        # 🔴 宽松口径（K-53）：只看"有没有明确说资料里没有"，
        #    不追究是否把通用条款套到了主体上。两个口径都要报 ——
        #    只报一个会把"模型不会拒答"和"口径过严"混成一个数字。
        vl, whyl = SC.refusal_judgement(ans, subject=q.get("equipment_id"),
                                        question=q.get("question"),
                                        mode="lenient")
        s["refusal_verdict_lenient"] = vl
        s["refusal_reason_lenient"] = whyl
        s["kp_hit"] = s["kp_total"] = None
        s["num_hit"] = s["num_total"] = None
    else:
        hit, total, detail = SC.key_point_hits(ans, q.get("key_points"))
        s["kp_hit"], s["kp_total"] = hit, total
        s["kp_detail"] = detail
        # 🔴 两个口径都要落盘（K-42）：严格字面包含 vs 含软匹配。
        #    只报一个 = 偷偷挑对自己有利的那个。
        s["kp_hit_strict"] = sum(1 for d in detail if d.get("strict"))
        nh, nt, ndetail = SC.numeric_hits(ans, q.get("must_include_numbers"))
        s["num_hit"], s["num_total"] = nh, nt
        s["num_detail"] = ndetail

    # 版本陷阱题：期望文档在不在上下文里
    exp = q.get("expect_doc_id")
    if exp:
        s["version_hit"] = bool(exp in ctx_docs)
        s["expect_doc_id"] = exp
    return s


def citation_localization(result):
    """**引用可定位率**：被引用到的块里，有多少能指到条款 / 页码。

    🔴 这是 M4 留给 M7 的那个问题：
    「条款级切分的价值在**引用定位**（块边界 = 条款边界），不在召回」。
    召回在 M4 已经证明没有可辨别差异（fixed 0.883 vs clause 0.856，CI 重叠），
    那条款级的价值就要在**生成层的引用卡片**上量化：

    - `fixed` 块 **0%** 带 `clause_no` → 引用卡片指不到"第十条"
    - `structural` 块 **99%** 带 `clause_no`
    - `clause` 块 **63%** 带 `clause_no`

    → 指标 = 带条款号的被引用块数 / 被引用块总数。
    """
    ctx = result.get("ctx_chunks", [])
    cited = (result.get("cite_stats") or {}).get("cited") or []
    n = 0
    with_clause = 0
    with_page = 0
    for i in cited:
        if not (1 <= i <= len(ctx)):
            continue
        c = ctx[i - 1]
        n += 1
        if (c.get("clause_no") or c.get("section_path")):
            with_clause += 1
        if c.get("page_no") is not None:
            with_page += 1
    return {"cited_n": n, "cited_with_clause": with_clause,
            "cited_with_page": with_page}


def stale_in_context(result, when, mode):
    """上下文里有没有"在 query_time 不生效"的 chunk（PRD 5.5 的端到端断言）。

    🔴 `cross_version` 口径不查 —— 那个口径本来就要历史版本在场。
    """
    if mode == "cross_version":
        return False
    for c in result.get("ctx_chunks", []):
        snap = c.get("doc_snapshot") or {}
        import versioning as V
        if V.is_effective(snap, when) is False:
            return True
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def rescore_main(args):
    """离线重算模式：读已落盘的逐题明细，重算判分并重新汇总。

    用法：
        python tools/run_generation_eval.py --rescore eval_runs/20261104_0950_generation_retrieval_main.answers.jsonl
    """
    src = pathlib.Path(args.rescore)
    if not src.exists():
        raise SystemExit(f"找不到 {src}")
    records = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    records = rescore(records, threshold=args.threshold,
                      anchors=load_test_set_anchors(args.test_set))

    # 从同批次的汇总 JSON 里继承元数据（配置 / 嵌入后端 / 切分策略）
    meta = {}
    sibling = src.parent / src.name.replace(".answers.jsonl", ".json")
    if sibling.exists():
        meta = json.loads(sibling.read_text(encoding="utf-8"))

    cfg = records[0].get("config") or meta.get("config", "?")
    backend = meta.get("embed_backend", "?")
    stem = src.name.replace(".answers.jsonl", "") + f"_rescored_t{args.threshold}"
    out_json = src.parent / f"{stem}.json"
    out_answers = src.parent / f"{stem}.answers.jsonl"

    errors = []
    summary = summarize(records, cfg, backend, 0.0, errors, args)
    summary["rescored_from"] = src.name
    summary["soft_threshold"] = args.threshold
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "system_prompt": meta.get("system_prompt"),
        "embed_backend": backend,
        "embed_note": meta.get("embed_note"),
        "chunk_strategy": meta.get("chunk_strategy"),
        "n_questions": meta.get("n_questions", len(records)),
        "n_answered": len(records),
        "n_errors": 0,
        "n_skipped": meta.get("n_skipped", 0),
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    with out_answers.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out_json.name}")
    print(f"-> {out_answers.name}")
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="retrieval",
                    help="逗号分隔：retrieval,closed_book,full_doc")
    ap.add_argument("--strategy", default="fixed",
                    help="用哪套 chunk 当索引（M4 结论：fixed 与 structural 并列最优，"
                         "fixed 块数只有 1/4）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟）")
    ap.add_argument("--workers", type=int, default=3,
                    help="LLM 并发（实测 6 会撞 429，3 才稳）")
    ap.add_argument("--top-k", type=int, default=5, help="检索取几块进上下文")
    ap.add_argument("--only-type", default="", help="只跑某类题型")
    ap.add_argument("--use-vector", action="store_true",
                    help="检索走混合路（默认关：M5 实测向量路是负贡献，"
                         "本项目实际配置是 PRD 4.6 的 L4 纯 BM25）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--questions-file", default="test_set.jsonl",
                    help="题集文件名（相对 data/eval/）。"
                         "拒答扩展集用 refusal_extended.jsonl，"
                         "主集 test_set.jsonl 不动，两者分母分开统计")
    ap.add_argument("--test-set", default=str(ROOT / "data" / "eval" / "test_set.jsonl"),
                    help="题集路径（rescore 时数值锚点以它为准）")
    ap.add_argument("--rescore", default="",
                    help="离线重算模式：给一份已落盘的 *_answers.jsonl，"
                         "按当前口径重算判分并重新汇总（不调 API）")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="软匹配的二元组覆盖率阈值（默认 0.7）")
    ap.add_argument("--system", default="system",
                    help="系统提示词：system（默认）/ system_v2（适用性标注版，"
                         "针对 26.7%% 的归因污染）")
    ap.add_argument("--attribution-fallback", action="store_true",
                    help="启用**代码级归因兜底**：模型没自己标注适用性时，"
                         "代码在答案末尾强制追加归因声明（2026-11-05）")
    args = ap.parse_args()

    if args.rescore:
        rescore_main(args)
        return

    RUN_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    questions = [json.loads(l) for l in
                 open(EVAL_DIR / args.questions_file, encoding="utf-8") if l.strip()]
    if args.only_type:
        questions = [q for q in questions if q.get("question_type") == args.only_type]
    if args.limit:
        questions = questions[:args.limit]

    chunks = RT.load_chunks(CHUNK_DIR / f"{args.strategy}.jsonl")
    manifest = VC.load_manifest()
    embedder = None
    if args.use_vector:
        corpus = [c.get("norm_text") or c.get("text") or "" for c in chunks]
        embedder, backend, msg = EMB.build_embedder(levels=("L1", "L2"),
                                                    corpus_texts=corpus)
        print(f"[embed] {backend}（{msg}）")
    else:
        backend, msg = "L4_pure_bm25", "无向量路（PRD 4.6 的 L4，本项目实际配置）"
        print(f"[embed] {backend}（{msg}）")
    sys_prompt = AN.SYSTEMS.get(args.system)
    if sys_prompt is None:
        raise SystemExit(f"未知的系统提示词：{args.system}（可选：{list(AN.SYSTEMS)}）")
    answerer = AN.Answerer(chunks, embedder=embedder, manifest=manifest,
                            top_k=args.top_k, use_vector=args.use_vector,
                            system=sys_prompt,
                            attribution_fallback=args.attribution_fallback)

    import llm_client

    def do_llm(prompt, system=None):
        return llm_client.call_llm(prompt, system=system, temperature=0.0,
                                    max_tokens=1500, timeout=180, retries=4)

    all_out = {}
    for cfg in configs:
        # 🔴 每个配置用**哪套系统提示词**必须落盘（K-41）——
        # 基线用的是自己的提示词，不是主配置的；不记下来，
        # 别人复现时会以为三组用的是同一个 prompt，差距就无法解释。
        if cfg == "retrieval":
            sys_kind = f"{args.system}（带 [[Cn]] 引用编号）"
        elif cfg == "closed_book":
            sys_kind = "CLOSED_BOOK_SYSTEM（闭卷，不提资料片段）"
        else:
            sys_kind = "SYSTEM_NO_CITE（全文基线，无编号）"
        print(f"\n=== 配置 {cfg}（{len(questions)} 题，提示词：{sys_kind}）===")
        t0 = time.time()
        records = []
        errors = []
        skipped = []

        def work(q):
            try:
                if cfg == "retrieval":
                    r = answerer.answer(q["question"], when=q.get("query_time"),
                                         mode=q.get("time_mode", "current"),
                                         version_filter=True, top_k=args.top_k,
                                         call=do_llm,
                                         subject=q.get("equipment_id"))
                elif cfg == "closed_book":
                    r = AN.answer_no_retrieval(q["question"], corpus_text="",
                                               call=do_llm)
                elif cfg == "full_doc":
                    did = q.get("doc_id")
                    body = doc_body(did) if did else ""
                    if not body:
                        # PRD 7.4：B1b 对"没有单一所属文档"的题不适用，
                        # 记 n/a 而不是拿别的口径凑数 —— 但要**记下来**，
                        # 否则报告里两个口径的分母差多少就看不出来了。
                        skipped.append((q["qid"], "无单一所属文档（B1b 不适用）"))
                        return None
                    r = AN.answer_no_retrieval(q["question"], corpus_text=body,
                                               call=do_llm)
                else:
                    raise ValueError(f"未知配置 {cfg}")
            except Exception as e:  # noqa: BLE001
                errors.append((q["qid"], f"{type(e).__name__}: {e}"))
                return None
            sc = score_question(q, r, r.get("ctx_docs", []))
            rec = {
                "qid": q["qid"], "question_type": q.get("question_type"),
                "question": q.get("question"),
                "equipment_id": q.get("equipment_id"),
                "refusal_mechanism": q.get("refusal_mechanism"),
                "refusal_mechanism_label": q.get("refusal_mechanism_label"),
                "time_mode": q.get("time_mode"), "query_time": q.get("query_time"),
                "answer": (r.get("answer_text") or "")[:2000],
                "rendered": (r.get("rendered") or "")[:4000],
                # 🔴 代码级归因兜底（2026-11-05）：记录是否追加了。
                #    ⚠️ **判分用的是 `answer`（模型原始正文，不含追加）** ——
                #    所以"模型自己做到的"与"代码兜底补上的"是分开的。
                "attribution_fallback": r.get("attribution_fallback"),
                "ctx_chunk_ids": [c["chunk_id"] for c in r.get("ctx_chunks", [])],
                "ctx_docs": r.get("ctx_docs", []),
                "cite_stats": r.get("cite_stats", {}),
                "version_cards": r.get("version_cards", []),
                "stale_in_context": stale_in_context(r, q.get("query_time"),
                                                     q.get("time_mode", "current")),
                **sc,
            }
            rec.update(citation_localization(r))
            return rec

        # 每题的明细**边跑边落** —— 跑到一半崩了也留下已完成的部分
        # （PRD 7.5：评测结果是不可变快照；攒在内存里最后一次性写 = 全丢）
        stem_parts = [ts, "generation", cfg]
        if args.only_type:
            stem_parts.append(args.only_type)
        if args.tag:
            stem_parts.append(args.tag)
        stem = "_".join(stem_parts)
        p2 = _unique_path(RUN_DIR, stem, "answers.jsonl")
        with p2.open("w", encoding="utf-8") as fout:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for i, rec in enumerate(ex.map(work, questions), start=1):
                    if rec is None:
                        continue
                    records.append(rec)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    if i % 20 == 0:
                        print(f"  {i}/{len(questions)}  ({time.time() - t0:.0f}s)")

        dt = time.time() - t0
        summary = summarize(records, cfg, backend, dt, errors, args)
        summary["n_skipped"] = len(skipped)
        summary["system_prompt"] = sys_kind
        all_out[cfg] = summary

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": cfg,
            "system_prompt": sys_kind,
            "system_prompt_name": args.system,
            "attribution_fallback_enabled": args.attribution_fallback,
            "embed_backend": backend,
            "embed_note": msg,
            "chunk_strategy": args.strategy,
            "n_questions": len(questions),
            "n_answered": len(records),
            "n_errors": len(errors),
            "n_skipped": len(skipped),
            "skipped": skipped[:50],
            "errors": errors[:20],
            "seconds": round(dt, 1),
            "summary": summary,
        }
        p = _unique_path(RUN_DIR, stem, "json")
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"-> {p.relative_to(ROOT)}")
        print(f"-> {p2.relative_to(ROOT)}")
        print(json.dumps(summary, ensure_ascii=False, indent=1)[:2000])


def load_test_set_anchors(path=None):
    """读题集里的**当前锚点**（qid → must_include_numbers）。

    🔴 **2026-11-09 新增**：数值题的锚点曾经**只存在于批次内嵌的 `num_detail`** 里，
    于是题集改了锚点（"10 m" → "10 mg/m3"）之后，
    `rescore` 仍然按**批次当时那一版**锚点判分 → **指标与题集脱钩**。
    → 现在以题集为准（题集是唯一事实源）。
    """
    p = pathlib.Path(path) if path else (ROOT / "data" / "eval" / "test_set.jsonl")
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        nums = r.get("must_include_numbers") or []
        if isinstance(nums, str):
            nums = [x for x in nums.split(";") if x.strip()]
        if nums:
            out[r.get("qid")] = nums
    return out


def rescore(records, threshold=0.7, anchors=None):
    """**离线重算**判分（K-42）。

    为什么需要它：判分口径改过一次（加了软匹配），
    但答案文本**不用重跑** —— 逐题明细里存着模型的原始回答和 key_point 清单，
    重算一遍就行。

    🔴 这一点很重要：**口径变更不应该逼着把 API 重跑一遍**，
    否则"改一次指标"就意味着"再烧一遍额度、再等一小时"，
    指标就永远不敢改 —— 于是错的指标会被一直用下去。
    """
    for r in records:
        # 🔴 **拒答判定也要重算**（K-52）：rescore 原来只重算 key_point / 数值，
        #    拒答题的 `refusal_verdict` 是**批次当时的旧口径**，
        #    于是"统一口径对比"对拒答这一项根本不成立 —— 两批数字不可比。
        #    主体优先用数据集字段（K-51），旧批次没有该字段时退回正则提取
        #    （对位号主体等价）。
        if r.get("question_type") == "拒答题":
            subj = r.get("equipment_id")
            if not subj:
                subj = SC.extract_subject(r.get("question") or "")
            v, why = SC.refusal_judgement(r.get("answer") or "", subject=subj,
                                          question=r.get("question"))
            r["refusal_verdict"] = v
            r["refusal_reason"] = why
            vl, whyl = SC.refusal_judgement(r.get("answer") or "", subject=subj,
                                            question=r.get("question"),
                                            mode="lenient")
            r["refusal_verdict_lenient"] = vl
            r["refusal_reason_lenient"] = whyl

        det = r.get("kp_detail") or []
        if not det:
            continue
        a = N.match_norm(r.get("answer") or "")
        for d in det:
            k = N.match_norm(d.get("key_point") or "")
            strict = bool(k) and k in a
            cov = SC.bigram_coverage(k, a) if k else 0.0
            d["strict"] = bool(strict)
            d["cov"] = round(cov, 3)
            d["klen"] = len(k)
            d["hit"] = bool(strict or (len(k) >= 4 and cov >= threshold))
        r["kp_hit"] = sum(1 for d in det if d["hit"])
        r["kp_hit_strict"] = sum(1 for d in det if d["strict"])
        r["kp_total"] = len(det)
        # 数值命中也要重算（K-43：单位形态归一化改了，
        # key 侧折了还不够，**答案侧也得折**）
        # 🔴 **2026-11-09**：锚点以**题集**为准（题集是唯一事实源），
        #    批次内嵌的 `num_detail` 只在题集里查不到时才用。
        if anchors is not None:
            cur = anchors.get(r.get("qid"))
            if cur is not None:
                r["num_detail"] = [{"number": str(x)} for x in cur]
        ndet = r.get("num_detail") or []
        if ndet:
            a_num = N.match_num_norm(r.get("answer") or "")
            for d in ndet:
                n = N.normalize_number(d.get("number") or "")
                d["hit"] = bool(n) and n in a_num
            r["num_hit"] = sum(1 for d in ndet if d["hit"])
            r["num_total"] = len(ndet)
    return records


def _unique_path(run_dir, stem, ext):
    """**绝不覆盖已有文件**（PRD 7.5："带时间戳、不覆盖"）。

    🔴 实测踩过：时间戳只到分钟（``%Y%m%d_%H%M``），
    同一分钟内跑第二批题就会**静默覆盖**第一批的产物 ——
    而且因为文件名合法、程序不报错，覆盖这件事**完全看不出来**。
    （这是 K-15「中间结果只写在输出文件里」的换皮版本，
    但这次撞的不是"重算"，是**文件名只精确到分钟**。）
    """
    p = run_dir / f"{stem}.{ext}"
    i = 2
    while p.exists():
        p = run_dir / f"{stem}-{i}.{ext}"
        i += 1
    return p


def summarize(records, cfg, backend, dt, errors, args):
    """把逐题结果汇总成指标（每项都写明 n / 分子 / 分母）。"""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r)

    out = {
        "config": cfg,
        "embed_backend": backend,
        "n_answered": len(records),
        "n_errors": len(errors),
        "seconds": round(dt, 1),
        "stale_in_context_questions": sum(1 for r in records
                                          if r.get("stale_in_context")),
        "per_type": {},
    }

    for qt, rs in sorted(by_type.items()):
        t = {"n": len(rs)}
        if qt == "拒答题":
            verdicts = Counter(r.get("refusal_verdict") for r in rs)
            n_false = verdicts.get("false_answer", 0)
            n_manual = verdicts.get("manual", 0)
            n_pass = verdicts.get("pass", 0)
            t["refusal_false_answer_rate"] = SC.wilson(n_false, len(rs))
            # 🔴 "人工复核"既不是通过也不是误答 —— 把它算进分母会**压低误答率**。
            # → 单列一个"只算可判样本"的口径（PRD 7.3 的硬约束：分母写清楚）。
            t["refusal_decidable_n"] = n_pass + n_false
            t["refusal_false_answer_rate_decidable"] = SC.wilson(
                n_false, n_pass + n_false) if (n_pass + n_false) else None
            t["refusal_verdicts"] = dict(verdicts)
            # 🔴 **最终输出的归因保障率**（2026-11-05）：
            #    模型自己标注了边界，**或**代码兜底追加了声明
            #    → 用户看到的答案里带适用性说明。
            #    这与"误答率"是两个维度：兜底能保证"输出安全"，
            #    但不能证明"模型学会了"。
            n_fb = sum(1 for r in rs if r.get("attribution_fallback"))
            n_bd = sum(1 for r in rs if r.get("refusal_verdict") == "bounded")
            t["attribution_guarantee_rate"] = SC.wilson(n_fb + n_bd, len(rs))
            t["n_code_fallback"] = n_fb
            t["n_model_boundary"] = n_bd
            # 🔴 宽松口径（K-53）：只看"有没有明确表态说没有"
            lv = Counter(r.get("refusal_verdict_lenient") for r in rs)
            n_lfalse = lv.get("false_answer", 0)
            n_lpass = lv.get("pass", 0)
            t["refusal_false_answer_rate_lenient"] = SC.wilson(n_lfalse, len(rs))
            t["refusal_verdicts_lenient"] = dict(lv)
            # 🔴 **按机制分层**（多机制拒答扩容的产物）。
            #    总体误答率会把"容易的假位号"和"困难的假装置"混在一起，
            #    掩盖掉机制之间的巨大差异 —— 只报总体数字等于没扩容。
            if any(r.get("refusal_mechanism") for r in rs):
                by_mech = defaultdict(list)
                for r in rs:
                    by_mech[r.get("refusal_mechanism") or "unspecified"].append(r)
                mech_stats = {}
                for mk, mrs in sorted(by_mech.items()):
                    mv = Counter(x.get("refusal_verdict") for x in mrs)
                    m_false = mv.get("false_answer", 0)
                    m_manual = mv.get("manual", 0)
                    m_pass = mv.get("pass", 0)
                    label = next((x.get("refusal_mechanism_label")
                                  for x in mrs if x.get("refusal_mechanism_label")),
                                 mk)
                    mech_stats[mk] = {
                        "label": label,
                        "n": len(mrs),
                        "n_false_answer": m_false,
                        "n_manual": m_manual,
                        "n_pass": m_pass,
                        "refusal_false_answer_rate": SC.wilson(m_false, len(mrs)),
                        "refusal_false_answer_rate_decidable": SC.wilson(
                            m_false, m_pass + m_false) if (m_pass + m_false) else None,
                    }
                t["refusal_by_mechanism"] = mech_stats
        else:
            kp_hit = sum(r.get("kp_hit") or 0 for r in rs)
            kp_tot = sum(r.get("kp_total") or 0 for r in rs)
            t["key_point_coverage"] = SC.wilson(kp_hit, kp_tot) if kp_tot else None
            t["key_point_coverage_mean"] = (round(kp_hit / kp_tot, 4)
                                            if kp_tot else None)
            # 🔴 严格字面包含（不含软匹配）—— 两个口径都报（K-42）
            kp_strict = sum(r.get("kp_hit_strict") or 0 for r in rs)
            t["key_point_coverage_strict"] = (SC.wilson(kp_strict, kp_tot)
                                             if kp_tot else None)
            t["key_point_coverage_mean_strict"] = (round(kp_strict / kp_tot, 4)
                                                   if kp_tot else None)
            # 阈值敏感性：软匹配阈值不能挑，选 0.7 之前把三档都算出来
            sens = {}
            for th in (0.6, 0.7, 0.8):
                h = sum(
                    1 for r in rs for d in (r.get("kp_detail") or [])
                    if d.get("strict")
                    or (d.get("klen", 0) >= 4 and d.get("cov", 0) >= th))
                sens[str(th)] = round(h / kp_tot, 4) if kp_tot else None
            t["key_point_coverage_by_threshold"] = sens
            t["key_point_questions_full_hit"] = SC.wilson(
                sum(1 for r in rs
                    if (r.get("kp_total") or 0) > 0
                    and r.get("kp_hit") == r.get("kp_total")), len(rs))
            nh = sum(r.get("num_hit") or 0 for r in rs)
            nt = sum(r.get("num_total") or 0 for r in rs)
            if nt:
                t["numeric_exact_match_rate"] = SC.wilson(nh, nt)
            if any(r.get("expect_doc_id") for r in rs):
                t["version_hit_rate"] = SC.wilson(
                    sum(1 for r in rs if r.get("version_hit")), len(rs))
        out["per_type"][qt] = t

    # 引用指标（全部题加起来）
    n_refs = sum(r["cite_stats"].get("n_refs", 0) for r in records)
    n_valid = sum(r["cite_stats"].get("n_valid", 0) for r in records)
    n_invalid = sum(r["cite_stats"].get("n_invalid", 0) for r in records)
    out["citation"] = {
        "n_refs": n_refs, "n_valid": n_valid, "n_invalid": n_invalid,
        "citation_valid_rate": SC.wilson(n_valid, n_refs) if n_refs else None,
        "questions_with_invalid_citation": sum(
            1 for r in records if r["cite_stats"].get("n_invalid")),
    }

    # 引用可定位率（条款 / 页码）—— M4 留给 M7 的那个问题
    cited_n = sum(r.get("cited_n") or 0 for r in records)
    if cited_n:
        out["citation"]["cited_chunks"] = cited_n
        out["citation"]["cited_with_clause"] = sum(
            r.get("cited_with_clause") or 0 for r in records)
        out["citation"]["cited_with_page"] = sum(
            r.get("cited_with_page") or 0 for r in records)
        out["citation"]["clause_located_rate"] = SC.wilson(
            out["citation"]["cited_with_clause"], cited_n)
        out["citation"]["page_located_rate"] = SC.wilson(
            out["citation"]["cited_with_page"], cited_n)
    return out


if __name__ == "__main__":
    main()
