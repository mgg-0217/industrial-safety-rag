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
import re
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
   原题干的毛病通常是「《文档名》中，（条款号片段）是怎么规定的？」这种切出来的半截话，
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

    # 🔴 用**改写后**的题干来切分与去重。
    # 理由：策略 6 真正拿去检索的是 chunk = question + answer，
    # 而 842 条原题干里 774 条是机器模板（"《X》中，…是怎么规定的？"）——
    # 模板词高频重复会让 BM25 的 IDF 趋近 0，检索退化。
    # 用模板题干切分等于在给一个不会发生的检索场景做准备。
    rewritten = ROOT / "data" / "eval" / "qa_pairs_rewritten.csv"
    n_rw = 0
    if rewritten.exists():
        with rewritten.open(encoding="utf-8-sig", newline="") as f:
            rw = list(csv.DictReader(f))
        if len(rw) == len(rows):   # 行序必须对齐（已实测与 candidates 逐行一致）
            for base, new in zip(rows, rw):
                q = (new.get("question_rewritten") or "").strip()
                if q:
                    base["question_original"] = base.get("question", "")
                    base["question"] = q
                    n_rw += 1
        else:
            print(f"🔴 行数不一致（{len(rw)} vs {len(rows)}），仍用原题干")
    print(f"采用改写题干：{n_rw} / {len(rows)} 条")

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

    # 🔴 同题干跨侧重复 = 泄漏，比 clause 泄漏更严重（同一道题两边都能命中）。
    # 根因：**不同条款会产出字面完全相同的题干** —— 题干模板是
    # "《文档名》中，{条款语义短语}是怎么规定的？"，而相邻条款常共享同一短语：
    #   clause "1.控制方案变更应办理审批手续；"  -> 题干 tail "办理审批手续"
    #   clause "2.实施特殊作业前，必须办理审批手续。" -> 题干 tail "办理审批手续"
    # 两条款不同、题干一字不差 —— 只按 clause 切分查不出来。
    # 处置：**题干优先**，索引侧里凡是题干与评测侧重复的行一律剔出索引侧
    #（数据不丢：评测侧本来就有一条同样的题）。
    evl_questions = {r.get("question", "") for r in eval_side}
    dup_cross = [r for r in index_side if r.get("question", "") in evl_questions]
    index_side = [r for r in index_side if r.get("question", "") not in evl_questions]
    print(f"剔出与评测侧题干重复的索引行：{len(dup_cross)} 条")

    # 🔴 索引侧**内部**同题干重复同样要剔：索引里有两条一样的 QA，
    # 命中时会被算成两个证据，等于白送召回。
    seen_q, dedup_idx, dup_inner = set(), [], 0
    for r in index_side:
        q = r.get("question", "")
        if q in seen_q:
            dup_inner += 1
            continue
        seen_q.add(q)
        dedup_idx.append(r)
    index_side = dedup_idx
    print(f"剔出索引侧内部同题干重复：{dup_inner} 条")

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


REFUSAL_PROMPT = """请造 **{n}** 条**拒答题**——问语料里根本不存在的设备，考察系统会不会编造。

【语料中真实存在的设备位号】（这些**不能**用，问它们必然能答出来）
{real_ids}

【已验证语料中不存在的假位号】（每个都在 34 万字符全文里精确匹配验证过）
{fake_ids}

【造题规则】
1. 从上面的假位号里取 {n} 个，每个造一条问句，问它的**维修/检维修要求**。
2. 问句要像真实检修人员会问的：
   - "设备（具体位号）的检修周期和作业前准备有哪些要求？"
   - "《某类设备名称》对（具体位号）这类设备的作业许可有什么规定？"
3. **必须带明确的设备位号**（如 T-9001），否则模型可能靠泛泛的设备类别知识作答，测不到拒答。
4. 题干**不得**暗示语料里应该有规定（如"根据规程，…请说明"）——
   要让模型去检索，检不到就应该说"未找到"。
5. `equipment_id` 字段填这条题用的假位号。

只输出 JSON 数组，不要解释、不要 markdown 围栏：
[{{"question":"...","equipment_id":"T-9001"}}]"""


CROSS_PROMPT = """请基于下面同一危险源在多份文档里的内容，造 **{n}** 条**跨文档推理题**。

{corpus}

【造题规则】
1. 每条题必须**至少涉及两份文档**，且答案要靠把两份文档的信息合起来才能答：
   - 同一危险源的**要求差异**（一处要求 A，另一处要求更严的 B）
   - **事故原因与整改要求**的对应关系
   - 同一危险源在**制度类**与**导则类**文档中的不同表述
2. 题干**不能**泄漏答案，也不能只问单份文档能直接答的内容。
3. `primary_doc_id`：答案的主要出处文档；`related_doc_ids`：需要同时参考的其他文档。
4. `gold_answer`：简洁写出正确答案要点（2–4 句）。
5. `key_points`：1–4 个关键信息点，每个都要能在给出的原文片段里找到依据。

只输出 JSON 数组，不要解释、不要 markdown 围栏：
[{{"question":"...","gold_answer":"...","key_points":["..."],"primary_doc_id":"...","related_doc_ids":["..."]}}]"""


def _read_md(doc_id):
    """读某份文档的 md 正文（去掉页码注释）。"""
    for name in (doc_id + ".md",):
        p = ROOT / "data" / "md" / name
        if p.exists():
            return re.sub(r"<!--\s*page:\d+\s*-->", "", p.read_text(encoding="utf-8"))
    return ""


def synthesize(rows, out_path, args):
    """新造 3 类题（PRD 7.1 里候选池为 0 的那三类）。

    `args.only` 可只跑其中一类（version / refusal / cross），便于分批执行。

    版本陷阱题：以唯一的真实版本链（榆林 2023/2025）为核心，正负样本成对。
    拒答题：用已验证的假位号池（tools/build_refusal_pool.py 产��）。
    跨文档推理题：同一危险源标签出现在两份以上文档时，造需要对比的题。
    """
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest = manifest.get("documents", manifest)
    by_id = {d["doc_id"]: d for d in manifest}

    # ---------- 1. 版本陷阱题 ----------
    chains = []
    for d in manifest:
        sup = d.get("superseded_by")
        if sup and sup in by_id:
            chains.append((by_id[sup], d))
    print(f"真实版本链：{len(chains)} 条")
    version_records = []
    only = getattr(args, "only", None)
    for old, new in (chains if only in (None, "version") else []):
        old_text = _read_md(old["doc_id"])[:6000]
        new_text = _read_md(new["doc_id"])[:6000]
        prompt = VERSION_PROMPT.format(
            old_id=old["doc_id"], old_no=old.get("doc_no") or "-",
            old_from=old.get("effective_from") or "-",
            new_id=new["doc_id"], new_no=new.get("doc_no") or "-",
            new_from=new.get("effective_from") or "-",
            old_text=old_text, new_text=new_text,
            n=args.n_version, base_date=BASE_DATE)
        try:
            raw = L.call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=3000)
            objs = L.parse_json(raw, default=[])
            if not isinstance(objs, list):
                objs = [objs] if isinstance(objs, dict) else []
        except Exception as e:
            print(f"  版本链 {old['doc_id']} 调用失败：{e}")
            continue
        if not objs:
            print(f"  版本链 {old['doc_id']} 解析失败，原始输出前 300 字：")
            print(f"    {(raw or '')[:300]!r}")
            continue
        for o in objs:
            q = (o.get("question") or "").strip()
            kps = o.get("key_points") or []
            if isinstance(kps, str):
                kps = [kps]
            if not q:
                continue
            version_records.append({
                "qid": make_qid(o.get("expect_doc_id") or new["doc_id"], q),
                "src_qid": None,
                "doc_id": o.get("expect_doc_id") or new["doc_id"],
                "doc_title": (by_id.get(o.get("expect_doc_id") or new["doc_id"]) or {}).get("title"),
                "clause_no": None,
                "page_no": None,
                "question_type": "版本陷阱题",
                "question": q,
                "gold_answer": None,
                "must_include_numbers": [],
                "key_points": [str(k).strip() for k in kps][:3],
                "time_mode": o.get("time_mode") or "current",
                "query_time": o.get("query_time") or BASE_DATE,
                "expect_doc_id": o.get("expect_doc_id") or new["doc_id"],
                "version_note": o.get("note") or "",
                "review_status": "pending",
                "origin": "synthesized_version_trap",
            })
    print(f"版本陷阱题：{len(version_records)} 条")

    # ---------- 2. 拒答题 ----------
    pool = json.loads((ROOT / "data" / "eval" / "refusal_pool.json").read_text(encoding="utf-8"))
    fake_ids = [s["equipment_id"] for s in pool["safe_fake_ids"]]
    real_ids = [t for t in pool["real_equipment_tags"]
                if re.match(r"^[0-9A-Z]+(-[0-9A-Z]+){1,4}$", t)]
    refusal_records = []
    if fake_ids and real_ids and only in (None, "refusal"):
        n = min(args.n_refusal, len(fake_ids))
        chunk = fake_ids[:n]
        prompt = REFUSAL_PROMPT.format(
            fake_ids=", ".join(chunk), real_ids=", ".join(real_ids),
            n=n, base_date=BASE_DATE)
        try:
            raw = L.call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=3000)
            objs = L.parse_json(raw, default=[])
            if not isinstance(objs, list):
                objs = [objs] if isinstance(objs, dict) else []
        except Exception as e:
            print(f"  拒答题调用失败：{e}")
            objs = []
        for o in objs:
            q = (o.get("question") or "").strip()
            if not q:
                continue
            refusal_records.append({
                "qid": make_qid(o.get("equipment_id") or "refusal", q),
                "src_qid": None,
                "doc_id": None,
                "doc_title": None,
                "clause_no": None,
                "page_no": None,
                "question_type": "拒答题",
                "question": q,
                "gold_answer": ("语料中不存在该设备/条款的相关规定，"
                                "正确行为是明确说明未找到，不得给出具体条款号、条款原文或具体数值。"),
                "must_include_numbers": [],
                "key_points": ["明确表示语料中无该设备的相关规定",
                               "不给出任何具体条款号或数值"],
                "time_mode": "current",
                "query_time": BASE_DATE,
                "expect_doc_id": None,
                "equipment_id": o.get("equipment_id"),
                "review_status": "pending",
                "origin": "synthesized_refusal",
            })
    print(f"拒答题：{len(refusal_records)} 条")

    # ---------- 3. 跨文档推理题 ----------
    tag_docs = {}
    for d in manifest:
        for t in (d.get("hazard_tags") or []):
            tag_docs.setdefault(t, []).append(d["doc_id"])
    multi = {t: ds for t, ds in tag_docs.items() if len(ds) >= 2}
    print(f"跨文档候选危险源标签：{len(multi)} 个（出现在 >= 2 份文档）")
    cross_records = []
    if multi and only in (None, "cross"):
        tags = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:6]
        desc = []
        for t, ds in tags:
            texts = []
            for did in ds[:3]:
                txt = _read_md(did)
                # 截取含该标签的上下文
                idx = txt.find(t)
                if idx >= 0:
                    texts.append(f"【{did}】…{txt[max(0, idx - 300):idx + 900]}…")
            if texts:
                desc.append(f"### 危险源「{t}」出现在文档 {ds}\n" + "\n".join(texts))
        prompt = CROSS_PROMPT.format(n=args.n_cross, corpus="\n\n".join(desc))
        try:
            raw = L.call_llm(prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=3000)
            objs = L.parse_json(raw, default=[])
            if not isinstance(objs, list):
                objs = [objs] if isinstance(objs, dict) else []
        except Exception as e:
            print(f"  跨文档题调用失败：{e}")
            objs = []
        for o in objs:
            q = (o.get("question") or "").strip()
            kps = o.get("key_points") or []
            if isinstance(kps, str):
                kps = [kps]
            if not q:
                continue
            cross_records.append({
                "qid": make_qid("cross", q),
                "src_qid": None,
                "doc_id": o.get("primary_doc_id"),
                "doc_title": None,
                "clause_no": None,
                "page_no": None,
                "question_type": "跨文档推理题",
                "question": q,
                "gold_answer": o.get("gold_answer") or "",
                "must_include_numbers": [],
                "key_points": [str(k).strip() for k in kps][:4],
                "time_mode": "current",
                "query_time": BASE_DATE,
                "expect_doc_id": o.get("primary_doc_id"),
                "related_doc_ids": o.get("related_doc_ids") or [],
                "review_status": "pending",
                "origin": "synthesized_cross_doc",
            })
    print(f"跨文档推理题：{len(cross_records)} 条")

    # ---------- 落盘 ----------
    all_records = version_records + refusal_records + cross_records
    done_src = {r.get("src_qid") for r in load_jsonl(out_path) if r.get("src_qid")}
    done_qids = {r.get("qid") for r in load_jsonl(out_path)}
    fresh = [r for r in all_records if r["qid"] not in done_qids]
    with out_path.open("a", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n新增 {len(fresh)} 条 -> {out_path}")
    print(f"（题集现有 {len(load_jsonl(out_path))} 条）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "refine", "synthesize", "qa_split"],
                    required=True)
    ap.add_argument("--n-facts", type=int, default=160)
    ap.add_argument("--n-numeric", type=int, default=39)
    ap.add_argument("--workers", type=int, default=int(os.getenv("M2_WORKERS", "6")))
    ap.add_argument("--out", default=str(TEST_SET))
    ap.add_argument("--test-set", default=str(TEST_SET),
                    help="qa_split 阶段用：以哪份题集的条款集合为准绳")
    ap.add_argument("--batch", default=str(ROOT / "data" / "eval" / "selected_batch.json"))
    ap.add_argument("--limit", type=int, default=20,
                    help="refine 阶段单次最多处理多少条（分批落盘，可反复跑续做）")
    ap.add_argument("--dry-run", action="store_true", help="只抽样不调 API")
    ap.add_argument("--n-version", type=int, default=40, help="版本陷阱题数量")
    ap.add_argument("--n-refusal", type=int, default=30, help="拒答题数量")
    ap.add_argument("--n-cross", type=int, default=31, help="跨文档推理题数量")
    ap.add_argument("--only", choices=["version", "refusal", "cross"],
                    help="只合成其中一类（分批执行用）")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)

    if args.stage == "qa_split":
        qa_split(args.test_set)
        return

    rows = load_jsonl(CANDIDATES)
    audit = json.load(AUDIT_FLAGS.open(encoding="utf-8"))
    flags_list = [e["flags"] for e in audit]
    assert len(flags_list) == len(rows), "audit 与 candidates 行数不一致"

    if args.stage == "select":
        # 🔴 抽样与执行必须解耦：抽样结果落盘，执行阶段只认这份清单。
        # 否则每次改 --n-facts 都会换一批题（轮转抽样对 n 敏感），
        # 分批续跑时就会漏题或重复题。
        picked_facts, picked_numeric = select_batch(
            rows, flags_list, args.n_facts, args.n_numeric)
        batch = {
            "n_facts_requested": args.n_facts,
            "n_numeric_requested": args.n_numeric,
            "facts": picked_facts,
            "numeric": picked_numeric,
            "meta": {
                "candidates_total": len(rows),
                "docs_covered": sorted({rows[i]["doc_id"]
                                        for i in picked_facts + picked_numeric}),
                "picked_total": len(picked_facts) + len(picked_numeric),
            },
        }
        pathlib.Path(args.batch).write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"抽样：事实题 {len(picked_facts)} + 数值题 {len(picked_numeric)} "
              f"= {len(picked_facts) + len(picked_numeric)}，覆盖文档 "
              f"{len(batch['meta']['docs_covered'])} 份")
        print(f"-> {args.batch}")
        import collections
        for d, n in collections.Counter(
                rows[i]["doc_id"] for i in picked_facts + picked_numeric).most_common():
            print(f"  {d}: {n}")
        return

    if args.stage == "refine":
        batch = json.loads(pathlib.Path(args.batch).read_text(encoding="utf-8"))
        facts = batch["facts"]
        numeric = batch["numeric"]
        # 先做数值题（量少、但保真是重点战场），再补事实题
        todo = [("数值题", i) for i in numeric] + [("事实题", i) for i in facts]
        done_src = {r.get("src_qid") for r in load_jsonl(out_path) if r.get("src_qid")}
        pending = [(label, i) for label, i in todo if rows[i].get("qid") not in done_src]
        print(f"批次清单：共 {len(todo)} 条，已完成 {len(done_src)} 条，"
              f"待处理 {len(pending)} 条，本次最多处理 {args.limit} 条")
        chunk = pending[:args.limit] if args.limit > 0 else pending
        ok = fail = 0
        errors = []
        t0 = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(refine_one, rows[i], flags_list[i]): (label, i)
                        for label, i in chunk}
                for fut in concurrent.futures.as_completed(futs):
                    label, i = futs[fut]
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
        print(f"完成：成功 {ok}，失败 {fail}，耗时 {time.time() - t0:.0f}s")
        if errors:
            err_path = out_path.with_suffix(".errors.jsonl")
            with err_path.open("a", encoding="utf-8") as f:
                for e in errors:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"失败明细 -> {err_path}")
        remaining = len(pending) - len(chunk)
        print(f"剩余待处理 {remaining} 条（重复执行本命令即可续做）")
        return

    if args.stage == "synthesize":
        synthesize(rows, out_path, args)
        return


if __name__ == "__main__":
    main()
