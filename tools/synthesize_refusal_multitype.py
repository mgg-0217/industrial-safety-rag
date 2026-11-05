"""多机制拒答题合成 —— PRD 7.1 的扩展（M2 补充批次）。

## 定位

M2 已有的 42 条拒答题**语义模板只有 1 个**（全是「假位号 X 的检修周期 /
维护项目 / 作业许可」）。42 条 ≈ 1 个样本重复 42 次 → Wilson CI 假设样本独立，
**虚假精度**。本脚本产出**独立于主集**的拒答专项集：

  data/eval/refusal_extended.jsonl

主集 306 条**不动**（已发布的 2×2 对照、生成层数字全部保持可比），
扩展集单独评测、单独报告。

## 🔴 为什么不用 LLM 生成题干

现有 M2 流程用 LLM 写拒答题题干。这里**故意不用**：

* 主体是题目的**全部难点** —— LLM 改写题干时可能把假主体「顺手」写成
  语料里真实存在的设备（"液氩储罐" → "液氨储罐"），题目当场失效；
* 模板化 + **主体确定性注入**保证每条题的主体就是池子里验证过的主体；
* 可复现、可审查：题干与主体的对应关系一眼能查。

多样性由**模板库**保证（每机制 6 套问法结构 × 4 机制 = 24 种句式骨架），
不是靠 LLM 的措辞随机性。

## 统计上的口径（重要）

* 一个主体 = 一个**独立样本**。同一主体出多道题不增信度（CI 会变窄，
  但那是同一个问题的重复）。本集**一个主体一题**。
* 报告时**按机制分层**给出误答率，并给出"同机制 42 vs N"的对照，
  用来证明**扩量 ≠ 增信度，必须扩机制**。

## 机制配额

| 机制 | 条数 | 难度预期 |
|---|---|---|
| `fake_id` | 40 | 低（top-k 变大会被通用条款带偏） |
| `fake_equipment` | 50 | **高**（语料有"锅炉/压力容器"等通用条款可套） |
| `fake_medium` | 30 | 高（容易命中同类通用作业要求） |
| `fake_process` | 30 | 中 |
| **合计** | **150** | |

主体从 **T-9041** 起，与主集已用的 T-9001~T-9040 **完全不相交**。
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from normalize import (build_chunk_text, normalize_digits,  # noqa: E402
                       normalize_equipment_tags)

POOL = ROOT / "data" / "eval" / "refusal_pool_multitype.json"
OUT = ROOT / "data" / "eval" / "refusal_extended.jsonl"
OUT_CONTROL = ROOT / "data" / "eval" / "refusal_extended_control.jsonl"
MD_DIR = ROOT / "data" / "md"

BASE_DATE = "2026-11-02"   # 与主集同一时间口径，保证可比

QUOTA = {
    "fake_id": 40,
    "fake_equipment": 50,
    "fake_medium": 30,
    "fake_process": 30,
}

# ---------------------------------------------------------------------------
# 模板库：每机制 6 套问法骨架，主体用 {s} 占位
# ---------------------------------------------------------------------------

TEMPLATES = {
    "fake_id": [
        "{s}的预防性维护项目、检修周期及作业前准备有哪些要求？",
        "设备 {s} 的检维修作业应履行哪些许可手续和确认程序？",
        "{s}的备件更换与检修记录应如何管理和保存？",
        "{s}的日常维护与定期检修分别有哪些规定？",
        "{s}的设备完好性检查与故障诊断要求是什么？",
        "{s}开展检维修作业时，隔离、置换和作业许可有哪些要求？",
    ],
    "fake_equipment": [
        "{s}的检修隔离与安全停机要求有哪些？",
        "{s}的检维修作业许可、动火与受限空间管理有哪些规定？",
        "{s}的设备维护保养项目与周期应如何确定？",
        "{s}发生故障时的应急处置与抢修要求是什么？",
        "{s}的作业人员资质、特种设备检验与作业审批有哪些要求？",
        "{s}的检维修风险辨识、作业票办理与完工验收有哪些规定？",
    ],
    "fake_medium": [
        "{s}的储存、装卸和搬运作业有哪些安全要求？",
        "{s}的泄漏应急处置与人员防护要求是什么？",
        "{s}的作业岗位职业危害因素与防护措施有哪些规定？",
        "{s}的装卸计量、罐区管理或出入库有哪些具体要求？",
        "{s}的使用、领用和废弃处置应遵守哪些规定？",
        "{s}相关的动火、受限空间和检维修作业有哪些特别要求？",
    ],
    "fake_process": [
        "{s}作业的安全操作规程和作业票有哪些要求？",
        "{s}过程中的人员防护、监测与异常处置要求是什么？",
        "{s}的作业许可、风险辨识与现场监护有哪些规定？",
        "{s}设备或设施的检维修与停工检修有哪些具体要求？",
        "{s}的应急处置、个体防护装备配备有哪些要求？",
        "{s}相关作业的记录、验收与隐患排查有哪些规定？",
    ],
}

# 判分标准：按机制给出**指向主体**的 key_point（比统一的"无相关规定"更精确）
MECH_KP = {
    "fake_id": "明确说明语料中不存在与「{s}」这一设备相关的规定",
    "fake_equipment": "明确说明语料中不存在与「{s}」这一装置/设备相关的规定",
    "fake_medium": "明确说明语料中不存在与「{s}」这一介质/物料相关的规定",
    "fake_process": "明确说明语料中不存在与「{s}」这一工序/作业相关的规定",
}

MECH_LABEL = {
    "fake_id": "假位号",
    "fake_equipment": "假装置/设备",
    "fake_medium": "假介质/物料",
    "fake_process": "假工序/作业",
}


def squashed(text):
    return re.sub(r"[-_]", "", text).upper()


def load_corpus_sides():
    raw_sq, norm_sq = [], []
    for p in sorted(MD_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        _orig, norm = build_chunk_text(raw)
        raw_sq.append(squashed(normalize_equipment_tags(normalize_digits(raw))))
        norm_sq.append(squashed(norm))
    return raw_sq, norm_sq


def make_qid(tag, question):
    h = hashlib.sha1(f"{tag}|{question}".encode("utf-8")).hexdigest()[:12]
    return f"r_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="同时产出「同机制扩量」对照组："
                         "同一批主体 × 多套模板，用来证明"
                         "「扩量 ≠ 扩机制」")
    ap.add_argument("--pool", default="refusal_pool_multitype.json",
                    help="主体池文件名（相对 data/eval/）")
    ap.add_argument("--out", default="refusal_extended.jsonl",
                    help="输出题集文件名（相对 data/eval/）")
    ap.add_argument("--control-out", default="refusal_extended_control.jsonl",
                    help="对照组输出文件名（相对 data/eval/）")
    args = ap.parse_args()

    pool_path = POOL.parent / args.pool
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    raw_sq, norm_sq = load_corpus_sides()

    # 🔴 出厂前再验一次：主体必须在双口径下都不存在于语料。
    #    （池子由 build_refusal_pool_multitype.py 产出，但中间产物可能被手改）
    records = []
    audit = []
    for mech, quota in QUOTA.items():
        items = pool[mech]
        # fake_id 从 T-9041 起，跳过主集已用的 T-9001~T-9040
        if mech == "fake_id":
            items = [it for it in items
                     if int(re.search(r"(\d+)$", it["subject"]).group(1)) > 9040]
        taken = 0
        tpl_i = 0
        for it in items:
            if taken >= quota:
                break
            subj = it["subject"]
            key = squashed(subj)
            raw_hit = any(key in rs for rs in raw_sq)
            norm_hit = any(key in ns for ns in norm_sq)
            if raw_hit or norm_hit:
                audit.append({"mechanism": mech, "subject": subj,
                              "issue": "复查命中语料，已剔除"})
                continue
            tpl = TEMPLATES[mech][tpl_i % len(TEMPLATES[mech])]
            tpl_i += 1
            q = tpl.format(s=subj)
            records.append({
                "qid": make_qid(subj, q),
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
                "key_points": [
                    MECH_KP[mech].format(s=subj),
                    "不给出任何具体条款号或数值",
                ],
                "time_mode": "current",
                "query_time": BASE_DATE,
                "expect_doc_id": None,
                "equipment_id": subj,
                "refusal_mechanism": mech,
                "refusal_mechanism_label": MECH_LABEL[mech],
                "review_status": "pending",
                "origin": "synthesized_refusal_multitype",
                "key_point_source": "refusal_criteria",
            })
            taken += 1

    # 断言：一个主体一题；qid 唯一
    assert len({r["qid"] for r in records}) == len(records), "qid 撞号"
    assert len({r["equipment_id"] for r in records}) == len(records), "主体重复"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT.parent / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"拒答扩展集：**{len(records)} 条** -> {out_path}")
    from collections import Counter
    print("\n=== 按机制 ===")
    for m, c in Counter(r["refusal_mechanism"] for r in records).most_common():
        print(f"  {MECH_LABEL[m]:<14} {c}")
    if audit:
        print(f"\n🔴 复查剔除 {len(audit)} 条：")
        for a in audit[:10]:
            print(f"   {a['subject']} — {a['issue']}")

    # 题干骨架分布（去掉主体后）
    skel = Counter()
    for r in records:
        s = r["question"].replace(r["equipment_id"], "{s}")
        skel[s] += 1
    print(f"\n=== 题干骨架（去主体后）共 {len(skel)} 种 ===")
    for k, v in skel.most_common():
        print(f"  {v:3d} 条 | {k}")

    # ---- 对照组：同机制扩量 --------------------------------------------
    if args.control:
        ids = [r["equipment_id"] for r in records
               if r["refusal_mechanism"] == "fake_id"]
        ctrl = []
        for subj in ids:
            for tpl in TEMPLATES["fake_id"][:4]:
                q = tpl.format(s=subj)
                ctrl.append({
                    "qid": make_qid(subj + "|ctrl", q),
                    "src_qid": None, "doc_id": None, "doc_title": None,
                    "clause_no": None, "page_no": None,
                    "question_type": "拒答题",
                    "question": q,
                    "gold_answer": ("语料中不存在该设备/条款的相关规定，"
                                    "正确行为是明确说明未找到，"
                                    "不得给出具体条款号、条款原文或具体数值。"),
                    "must_include_numbers": [],
                    "key_points": [MECH_KP["fake_id"].format(s=subj),
                                   "不给出任何具体条款号或数值"],
                    "time_mode": "current", "query_time": BASE_DATE,
                    "expect_doc_id": None, "equipment_id": subj,
                    "refusal_mechanism": "fake_id",
                    "refusal_mechanism_label": MECH_LABEL["fake_id"],
                    "control_group": subj,
                    "review_status": "pending",
                    "origin": "synthesized_refusal_multitype_control",
                    "key_point_source": "refusal_criteria",
                })
        assert len({c["qid"] for c in ctrl}) == len(ctrl), "对照集 qid 撞号"
        ctrl_path = OUT_CONTROL.parent / args.control_out
        with ctrl_path.open("w", encoding="utf-8") as f:
            for c in ctrl:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\n对照组（同机制扩量）：**{len(ctrl)} 条** / "
              f"{len(set(c['equipment_id'] for c in ctrl))} 个主体 / 1 种机制 "
              f"-> {ctrl_path.name}")


if __name__ == "__main__":
    main()
