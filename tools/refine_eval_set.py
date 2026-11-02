"""评测集精修 —— PRD M2 第二步（LLM 批量改写与造题）。

承接 `tools/audit_eval_candidates.py` 的普查结果，把候选池变成正式题集。

**为什么必须让 LLM 重写而不是补字**：普查发现 576/842（68%）的题干是
「《文档名》中，{clause_no 切片}是怎么规定的？」—— 切点落在词中间，
产生 "《…》中，"先通风是怎么规定的？" 这类不成句的题干。
补字修不好，只能读 `gold_answer` 全文重新组织问法。

三个阶段（可分别跑，支持中断续跑）：
  --stage facts        已有候选（事实题 + 数值题）→ 改写题干 + 标注 key_points
  --stage synthesize   新造 3 类题（版本陷阱题 / 拒答题 / 跨文档推理题）
  --stage qa_split     把 qa_pairs.csv 切成策略 6 索引侧 / 评测侧（严格不相交）

**增量落盘**：每条完成立刻 append 到 data/eval/test_set.jsonl，
重跑时按 qid 跳过已完成的条目（PRD 7.2：每批 100 条并落盘进度，中断可续）。

**职责边界（不可越界）**：本脚本产出的题集是**待核对**状态，
字段 `review_status` 一律为 `pending`。人工核对（PRD 7.2 第三步）未完成前，
这些题不得用于报任何指标。
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm_client as L  # noqa: E402

CANDIDATES = ROOT / "data" / "eval" / "candidates.jsonl"
AUDIT_FLAGS = ROOT / "data" / "eval" / "audit_flags.json"
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"
QA_PAIRS = ROOT / "data" / "eval" / "qa_pairs.csv"

# 时间基准日：PRD 5.1 要求 query_time 显式传入，不读系统时间。
# 写死在这里，全批次统一，任何一条题的这个值都能追到同一处。
BASE_DATE = "2026-11-02"

SYSTEM_PROMPT = (
    "你是工业安全规程领域的评测集专家。你的任务是把机器切出来的粗糙条款片段"
    "改写成**可人工核对的评测题**，并标注可复核的结构化字段。"
    "你必须严格依据给定的原文作答，不得引入原文没有的信息。"
)


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done(path):
    """已完成的 qid 集合（支持续跑）。"""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                continue
    return done


def make_qid(doc_id, question):
    """生成唯一 qid。

    🔴 不能用候选池的原 qid：实测 842 行只有 835 个不同 qid，7 个重复
    （根因是 qid = doc_id__clause_no 拼接，同 doc 下 clause_no 重复即撞）。
    """
    h = hashlib.sha1(f"{doc_id}|{question}".encode("utf-8")).hexdigest()[:12]
    return f"t_{h}"


# ---------------------------------------------------------------------------
# 阶段一：改写已有候选
# ---------------------------------------------------------------------------
FACTS_PROMPT = """请把下面这条机器生成的粗糙评测题改写成一条**可人工核对的评测题**，并输出结构化标注。

【文档标题】{doc_title}
【条款号】{clause_no}
【该条款原文（gold_answer）】
{gold_answer}

【机器生成的原题干（有问题，必须重写）】
{question}

【必须遵守的要求】
1. 题干必须**完整、自然、语义明确**，能独立提问，不依赖读者看过原文。
   原题干的毛病通常是「《文档名》中，{条款号片段}是怎么规定的？」这种切出来的半截话，
   你要读懂条款原文后，用**条款真正讲的内容**重新组织问法。
2. 题干**不得泄漏答案**：不得出现答案里的具体数值、具体限值、条款原文整句。
   例：原文写「氧气含量应在19.5%～23%以下」，题干不能出现 19.5% 或 23%。
3. 题干要问**一个**明确的问题，回答应当唯一、可复核。
4. key_points：1–4 个关键信息点，每个是一个**简短的事实短语**（不超过 30 字），
   用于评测时判定答案命中与否。key_points 必须都能在 gold_answer 里找到依据。
5. time_mode：绝大多数事实题填 `current`（问现行规定）；
   若问题本身问的是"某年"的规定，填 `as_of`，并在 query_time 填该年**年末**日期
   （例如问 2023 年 → "2023-12-31"；⚠️ 不要取 1 月 1 日）。
6. query_time：time_mode=current 时填 {base_date}；time_mode=as_of 时填对应年份的年末日期。
7. expect_doc_id：期望命中的文档 id，就填 {doc_id}。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 围栏：
{{"question": "...", "key_points": ["...", "..."], "time_mode": "current", "query_time": "{base_date}", "expect_doc_id": "{doc_id}"}}"""


def refine_one(row, flags):
    """改写单条。返回 (record, error)。"""
    is_numeric = row.get("question_type") == "数值题"
    anchors = row.get("must_include_numbers") or []

    prompt = FACTS_PROMPT.format(
        doc_title=row.get("doc_title", ""),
        clause_no=row.get("clause_no", ""),
        gold_answer=(row.get("gold_answer") or "")[:3000],
        question=row.get("question", ""),
        doc_id=row.get("doc_id", ""),
        base_date=BASE_DATE,
    )
    if is_numeric:
        prompt += (
            "\n【本题是数值题，额外要求】"
            f"\n必须_include 的数值是：{anchors}"
            "\n题干仍然**不能出现这些数值**，只能问「某个参数的限值/规格是多少」这类不指明数字的问法。"
            "\nkey_points 里要明确标出该题的数值锚点（写成数值本身），供数值保真判定使用。"
        )
    try:
        raw = L.call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=1200)
    except Exception as e:
        return None, f"llm_error: {e}"

    obj = L.parse_json(raw, default=None)
    if not isinstance(obj, dict):
        return None, "parse_failed"

    q = (obj.get("question") or "").strip()
    kps = obj.get("key_points") or []
    if isinstance(kps, str):
        kps = [kps]
    kps = [str(k).strip() for k in kps if str(k).strip()]
    if not q or not kps:
        return None, "missing_fields"

    return {
        "qid": make_qid(row["doc_id"], q),
        "src_qid": row.get("qid"),
        "doc_id": row["doc_id"],
        "doc_title": row.get("doc_title"),
        "clause_no": row.get("clause_no"),
        "page_no": row.get("page_no"),
        "question_type": row.get("question_type"),
        "question": q,
        "gold_answer": row.get("gold_answer"),
        "must_include_numbers": anchors,
        "key_points": kps[:4],
        "time_mode": obj.get("time_mode") or "current",
        "query_time": obj.get("query_time") or BASE_DATE,
        "expect_doc_id": obj.get("expect_doc_id") or row["doc_id"],
        "source_url": row.get("source_url"),
        "doc_no": row.get("doc_no"),
        "hazard_tags": row.get("hazard_tags") or [],
        "equipment_tags": row.get("equipment_tags") or [],
        "review_status": "pending",
        "origin": "refined_from_candidate",
    }, None


def select_batch(rows, flags_list, n_facts, n_numeric, seed=20261102):
    """抽样：覆盖全部文档 + 题型均衡 + 优先含问题题（否则测不到改写能力）。

    🔴 必须是**真轮转**（每轮从每份文档各取 1 条），不是"按文档顺序取光"——
    后者会让前几份文档吃光额度，实测只覆盖 10/15 份文档。
    数值题只有 39 条、量小，先单独全量取；事实题再轮转补齐。
    每份文档内优先取有质量问题的（需要改写的），再取干净的。
    """
    rnd = random.Random(seed)
    by_doc = {}
    for i, r in enumerate(rows):
        by_doc.setdefault(r["doc_id"], []).append(i)

    def priority(i):
        f = flags_list[i]
        bad = sum(1 for k in ("integrity", "clause_derived_question", "leaked_numbers",
                              "empty_number_anchor", "duplicate_qid") if f.get(k))
        return (-bad, rnd.random())

    # 1) 数值题：全量取（按优先级排序）
    numeric_idx = sorted(
        [i for i, r in enumerate(rows) if r["question_type"] == "数值题"], key=priority)
    picked_numeric = numeric_idx[:n_numeric]

    # 2) 事实题：真轮转
    queues = {doc: sorted(idxs, key=priority) for doc, idxs in by_doc.items()}
    pos = {doc: 0 for doc in queues}
    picked_facts = []
    while len(picked_facts) < n_facts:
        progressed = False
        for doc in sorted(queues):
            if pos[doc] >= len(queues[doc]):
                continue
            i = queues[doc][pos[doc]]
            pos[doc] += 1
            progressed = True
            if rows[i]["question_type"] == "事实题":
                picked_facts.append(i)
            if len(picked_facts) >= n_facts:
                break
        if not progressed:
            break
    return picked_facts, picked_numeric


# ---------------------------------------------------------------------------
# 阶段二：新造 3 类题
# ---------------------------------------------------------------------------
VERSION_PROMPT = """请基于下面这**唯一一条真实版本链**造版本陷阱题。

【版本链】（真实语料里只有这一条，链长 2）
- 旧版：{old_id}（{old_no}），生效日 {old_from}，被 {new_id} 明示废止
- 新版：{new_id}（{new_no}），生效日 {new_from}，现行有效

【旧版原文片段】
{old_text}

【新版原文片段】
{new_text}

【要造的题型与硬性规则】
造 **{n}** 条版本陷阱题，正负样本成对。必须遵守：
1. 每条题必须能唯一确定期望命中的版本。给出 `time_mode`：
   - 问"现在/现行/目前" → `current`，`query_time` = {base_date}，期望命中**新版**
   - 问"2023 年时" → `as_of`，`query_time` = **"2023-12-31"**（年末口径，⚠️ 严禁取 1 月 1 日，
     榆林 2023 版生效日是 2023-12-07，取 1 月 1 日会落进空档导致题目不成立）
   - 问"该制度现在还有效吗" → `current`，期望命中**新版**
2. `expect_doc_id` 必须明确写期望命中的 doc_id（{old_id} 或 {new_id}）。
3. 题干**不得泄漏**"答案在哪个版本"，但**必须**含足以定位时点的表达
   （如"2023 年"、"现行"、"现行版本"）。
4. key_points 写 1–3 个该版本特有的关键信息点（必须能从对应版本原文找到依据），
   用于判定"答错版本"。
5. 题干里**不得出现**两个版本共有的原文字句，要问版本有差异的地方。

只输出 JSON 数组，不要解释、不要 markdown 围栏：
[{{"question":"...","key_points":["..."],"time_mode":"current","query_time":"{base_date}","expect_doc_id":"{new_id}","note":"设计意图一句话"}}]"""


def synth_version_chains(manifest, n):
    """找版本链（superseded_by 指向另一份文档）。"""
    chains = []
    by_id = {d["doc_id"]: d for d in manifest}
    for d in manifest:
        sup = d.get("superseded_by")
        if sup and sup in by_id:
            chains.append((by_id[sup], d))  # (旧, 新)
    return chains[:n]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_refine(rows, flags_list, indices, out_path, workers, label):
    done = load_done(out_path)
    todo = [i for i in indices if make_qid(rows[i]["doc_id"], rows[i]["question"]) not in done
            and rows[i].get("qid") not in done]
    # 已完成的判定用 src_qid 更稳（改写后 qid 会变）
    done_src = {r.get("src_qid") for r in load_jsonl(out_path) if r.get("src_qid")}
    todo = [i for i in indices if rows[i].get("qid") not in done_src]
    print(f"[{label}] 待处理 {len(todo)} / {len(indices)}（已完成 {len(done_src)} 条跳过）")

    ok, fail = 0, 0
    errors = []
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(refine_one, rows[i], flags_list[i]): i for i in todo}
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                try:
                    rec, err = fut.result()
                except Exception as e:
                    rec, err = None, f"exception: {e}"
                if rec:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()  # 每条落盘，中断可续
                    ok += 1
                else:
                    fail += 1
                    errors.append({"src_qid": rows[i]["qid"], "error": err})
                done_n = ok + fail
                if done_n % 20 == 0:
                    rate = done_n / max(time.time() - t0, 1e-6)
                    print(f"  {done_n}/{len(todo)} 成功 {ok} 失败 {fail}  {rate:.1f} 条/秒")
    print(f"[{label}] 完成：成功 {ok}，失败 {fail}，耗时 {time.time() - t0:.0f}s")
    if errors:
        err_path = out_path.with_suffix(".errors.jsonl")
        with err_path.open("a", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[{label}] 失败明细 -> {err_path}")
    return ok, fail


def qa_split(test_set_path):
    """把 qa_pairs.csv 切成策略 6 的「索引侧」与「评测侧」，严格不相交。

    PRD 3.1 策略 6 的硬约束：
      1. **检索库中不允许出现与评测题 clause_no 相同的 chunk**，命中即判该次运行无效；
      2. qid 不唯一（842 行只有 835 个不同 qid），必须重新生成 qa_id。

    🔴 切分的**顺序**是这里的关键，容易搞反：
      不是「把 842 条对半切」，而是「**先定评测集用到哪些条款，再把���些条款的题排除出索引**」。
    理由：QA 索引是按 **clause_no** 泄漏的（同一条款的题问的是同一件事），
    所以排除的单位必须是条款，不是题。
    实测：842 条分 835 个条款，其中 **828 个条款只有 1 条题**——
    若按「题」对半切，必然有一半条款两侧都有，违反约束 1。

    副作用（必须写进报告）：索引规模由「被评测集占用的条款数」决定，
    不是候选池的一半。PRD 3.1 里「索引规模只有候选池的一半」的说法
    只在「条款与题一一对应」时成立，本语料 835 条款 / 842 题，
    828 个条款单题，所以实际比例由评测集占用的条款数决定。
    """
    import csv

    if not pathlib.Path(test_set_path).exists():
        raise SystemExit(
            f"找不到题集 {test_set_path}\n"
            "必须先跑 --stage facts / synthesize 确定评测集用到哪些条款，"
            "才能切出不相交的索引侧。（顺序反了会漏同条款的题）")

    test_set = load_jsonl(pathlib.Path(test_set_path))
    test_clauses = {(r.get("doc_id", ""), r.get("clause_no", "")) for r in test_set}
    test_clauses.discard(("", ""))
    print(f"评测集 {len(test_set)} 条，覆盖条款 {len(test_clauses)} 组")

    rows = []
    with QA_PAIRS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"qa_pairs.csv：{len(rows)} 行")

    # 索引侧 = 条款未被评测集占用的题；评测侧 = 条款已被占用的题
    index_side = [r for r in rows
                  if (r.get("doc_id", ""), r.get("clause_no", "")) not in test_clauses]
    eval_side = [r for r in rows
                 if (r.get("doc_id", ""), r.get("clause_no", "")) in test_clauses]

    def write(path, items, side):
        fields = list(rows[0].keys())
        if "qa_id" not in fields:
            fields = ["qa_id"] + fields
        if "split_side" not in fields:
            fields = fields + ["split_side"]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            qa_ids = []
            for r in items:
                h = hashlib.sha1("|".join([
                    r.get("doc_id", ""), r.get("question", ""), r.get("answer", "")
                ]).encode("utf-8")).hexdigest()[:12]
                row = dict(r)
                row["qa_id"] = f"qa_{h}"
                row["split_side"] = side
                qa_ids.append(row["qa_id"])
                w.writerow(row)
        return len(set(qa_ids))

    idx_path = ROOT / "data" / "eval" / "qa_index_side.csv"
    evl_path = ROOT / "data" / "eval" / "qa_eval_side.csv"
    n_idx = write(idx_path, index_side, "index")
    n_evl = write(evl_path, eval_side, "eval")

    # 校验：两侧 clause_no 集合必须无交集
    idx_clauses = {(r.get("doc_id", ""), r.get("clause_no", "")) for r in index_side}
    evl_clauses = {(r.get("doc_id", ""), r.get("clause_no", "")) for r in eval_side}
    overlap = idx_clauses & evl_clauses
    idx_q = {r.get("question", "") for r in index_side}
    evl_q = {r.get("question", "") for r in eval_side}

    print(f"\n索引侧：{len(index_side)} 条（唯一 qa_id {n_idx}）-> {idx_path.name}")
    print(f"评测侧：{len(eval_side)} 条（唯一 qa_id {n_evl}）-> {evl_path.name}")
    print(f"索引规模 / 候选池 = {len(index_side) / max(len(rows), 1):.1%}")
    print(f"\n🔴 交集校验：")
    print(f"  同 (doc_id, clause_no) 交集：**{len(overlap)}** 组"
          f"{'（✅ 为 0 才合格）' if not overlap else '（❌ 有重叠）'}")
    print(f"  同 question 交集：{len(idx_q & evl_q)} 条")
    if overlap:
        for o in list(overlap)[:5]:
            print(f"    重叠条款: {o}")
    return len(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["facts", "synthesize", "qa_split"], required=True)
    ap.add_argument("--n-facts", type=int, default=180)
    ap.add_argument("--n-numeric", type=int, default=39)
    ap.add_argument("--workers", type=int, default=int(os.getenv("M2_WORKERS", "6")))
    ap.add_argument("--out", default=str(TEST_SET))
    ap.add_argument("--test-set", default=str(TEST_SET),
                    help="qa_split 阶段用：以哪份题集的条款集合为准绳")
    ap.add_argument("--dry-run", action="store_true", help="只抽样不调 API")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)

    if args.stage == "qa_split":
        qa_split(args.test_set)
        return

    if args.stage == "facts":
        rows = load_jsonl(CANDIDATES)
        audit = json.load(AUDIT_FLAGS.open(encoding="utf-8"))
        flags_list = [e["flags"] for e in audit]
        assert len(flags_list) == len(rows), "audit 与 candidates 行数不一致"

        picked_facts, picked_numeric = select_batch(
            rows, flags_list, args.n_facts, args.n_numeric)
        print(f"抽样：事实题 {len(picked_facts)} + 数值题 {len(picked_numeric)} "
              f"= {len(picked_facts) + len(picked_numeric)}，覆盖文档 "
              f"{len({rows[i]['doc_id'] for i in picked_facts + picked_numeric})} 份")
        if args.dry_run:
            print("\n[dry-run] 不调 API。抽样文档分布：")
            import collections
            for d, n in collections.Counter(
                    rows[i]["doc_id"] for i in picked_facts + picked_numeric).most_common():
                print(f"  {d}: {n}")
            return

        ok1, f1 = run_refine(rows, flags_list, picked_facts, out_path,
                             args.workers, "事实题")
        ok2, f2 = run_refine(rows, flags_list, picked_numeric, out_path,
                             args.workers, "数值题")
        print(f"\n合计：成功 {ok1 + ok2}，失败 {f1 + f2}")


if __name__ == "__main__":
    main()
