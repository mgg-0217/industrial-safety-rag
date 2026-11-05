"""多机制拒答题主体池 —— PRD 7.1 的扩展。

## 为什么要有这个脚本

M2 产出的 42 条拒答题**语义模板只有 1 个**：全是「假位号 X 的检修周期 /
维护项目 / 作业许可」，只换了位号和措辞。后果：

* 统计上 42 条 ≈ 1 个样本重复 42 次。Wilson CI 假设样本独立 →
  **虚假精度**，CI 会随 n 收窄，让人误以为"结论变显著了"；
* 扩容到 150 条如果只是换位号，CI 会更窄但结论仍是同一个问题的重复
  → **扩量 ≠ 增信度，必须扩机制**。

所以这里不扩位号，而是**扩充"语料里没有"这件事的构造机制**：

| 机制 | 主体形态 | 预期难点 |
|---|---|---|
| `fake_id` | 仿真实位号形状的假位号 | 低（但 top-k 变大时模型会套通用条款） |
| `fake_equipment` | 语料里没有的装置/设备类别 | **高** —— 语料里有"锅炉/压力容器"等通用条款可套 |
| `fake_medium` | 语料里没有的介质/物料 | 高 —— 容易命中同类的通用作业要求 |
| `fake_process` | 语料里没有的工序/作业 | 中 |

## 🔴 验证口径（本脚本最容易写错的地方）

`build_refusal_pool.py` 的教训直接复用：

* `tag_key` 去掉分隔符，而 `norm_text` **保留**分隔符 →
  拿 key 去 count 归一化文本**恒为 0**，等于只查了一半形态；
* 原文侧要**先跑 normalize**（中文破折号 → 半角连字符），否则
  `19—V—4215` 压缩后仍带破折号，查不到 `19V4215`。

本脚本对每个候选同时查两个口径：

* **原文侧**：`squash(normalize_equipment_tags(normalize_digits(raw)))`
* **归一化侧**：`squash(norm)`（`build_chunk_text` 产出的检索用文本）

任一口径命中 → 候选被拒。

## 同域度（nearest_real）

候选必须**与语料同域但不存在** —— 太冷门的词模型随手就拒答，没有区分度；
太常见的词又很可能真实存在。同域度用**字符 bigram 的 Jaccard 相似度**
对语料里的真实标签（manifest 的 equipment_tags / hazard_tags）取最大值，
输出最相近的那个真实词，供人工核对题目难度是否合理。

输出：
  data/eval/refusal_pool_multitype.json
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from normalize import (build_chunk_text, normalize_digits,  # noqa: E402
                       normalize_equipment_tags)

MD_DIR = ROOT / "data" / "md"
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "data" / "eval" / "refusal_pool_multitype.json"

# ---------------------------------------------------------------------------
# 候选清单：与语料同域（化工/危化/特种设备/检维修）但语料里不存在的实体
# ---------------------------------------------------------------------------

FAKE_EQUIPMENT = [
    # 炼油装置
    "加氢反应器", "催化裂化装置", "加氢精制装置", "延迟焦化装置",
    "常压蒸馏塔", "减压蒸馏塔", "催化重整装置", "烷基化装置",
    "溶剂脱沥青装置", "糠醛精馏装置", "乙烯裂解装置", "聚乙烯装置",
    "聚丙烯装置", "聚苯乙烯装置", "聚氯乙烯装置", "合成橡胶装置",
    # 无机化工装置
    "硫酸装置", "硝酸装置", "氯碱装置", "乙炔装置", "合成氨装置",
    "尿素装置", "磷酸装置", "纯碱装置", "硝酸铵装置", "碳酸氢铵装置",
    "聚合反应釜", "缩聚反应釜", "离子交换装置", "吸附分离装置",
    # 公用工程与动力
    "垃圾焚烧炉", "余热锅炉", "汽轮机", "发电机", "主变压器",
    "循环水泵", "空气压缩机", "氨制冷机", "氟制冷机", "螺杆压缩机",
    "汽包", "过热器", "再热器", "省煤器", "给水泵", "送风机",
    "脱硫塔", "脱硝塔", "除尘器", "布袋除尘器", "电除尘器",
    # 冶金与建材
    "转炉", "电炉", "阳极炉", "电解槽", "沸腾炉", "回转窑", "竖炉",
    "焙烧炉", "煅烧炉", "碳化炉", "玻璃熔窑", "水泥回转窑",
    # 轻工与橡胶
    "捏合机", "密炼机", "挤出机", "注塑机", "吹塑机", "纺丝机",
    "织布机", "印染机", "造纸机", "抄纸机", "压延机", "涂布机",
    # 储运
    "输煤皮带", "输油泵", "油品码头", "散装水泥筒仓", "液氨储罐区",
]

FAKE_MEDIUM = [
    # 低温与惰性介质
    "液氩", "液氧", "液氮", "液溴", "液钠", "液钾",
    # 强腐蚀性
    "氢氟酸", "三氯乙酸", "溴素", "氯磺酸", "发烟硫酸", "王水",
    # 常见化工原料
    "甲醇", "乙醇", "丙酮", "丙醇", "异丙醇", "乙二醇", "醋酸",
    "苯酚", "糠醛", "蒽醌", "硝基苯", "二硫化碳", "溴乙烷",
    # 石化中间体
    "氯乙烯", "氯丙烯", "三氯乙烯", "四氯乙烯", "甲基丙烯酸甲酯",
    "对二甲苯", "邻二甲苯", "乙苯", "异构烷烃",
    # 无机盐与农化
    "氰化钠", "硝酸钾", "硫酸锌", "磷酸二氢钾", "尿素",
    # 特殊
    "二甲基汞", "氯化汞", "甲基汞", "氟利昂", "液碱", "溴甲烷",
]

FAKE_PROCESS = [
    # 催化与加氢
    "催化剂再生", "催化剂钝化", "氢气还原", "加氢反应", "催化裂化",
    "催化重整", "催化加氢", "变压吸附", "催化剂还原再生",
    # 脱硫脱硝与环保
    "干法脱硫", "湿法脱硫", "氨法脱硫", "SCR脱硝", "SNCR脱硝",
    "烟气脱硝", "溶剂脱硫", "MDEA脱硫",
    # 分离与反应
    "碱洗", "酸洗", "溶剂萃取", "精馏分离", "闪蒸脱挥", "结晶分离",
    "离心分离", "膜分离", "电渗析", "离子交换再生", "吸附剂置换",
    # 聚合与后处理
    "聚合反应", "缩聚反应", "固相缩聚", "聚合后处理", "熔融纺丝",
    # 其他
    "歧化反应", "磺化反应", "氯化反应", "硝化反应", "重整反应",
    "烷基化反应", "异构化反应", "热裂化", "减粘裂化", "催化裂化再生",
    "湿法冶金浸出", "萃取分离", "堆肥发酵", "垃圾填埋气收集",
]

# 假位号模板（与 build_refusal_pool.py 同构，但扩到 80 个）
FAKE_ID_PATTERNS = [
    "T-{n}", "V-{n}", "MP-{n}", "PV-{n}", "R-{n}", "HE-{n}",
    "T-{n}-A", "P-{n}-B", "HX-{n}", "C-{n}", "E-{n}", "S-{n}",
    "V-{n}-B", "PS-{n}", "PV-{n}-C", "TR-{n}", "HX-{n}-A", "RX-{n}",
]


def squashed(text):
    """压成无分隔符形态（与 normalize.find_tag_occurrences 的口径一致）。"""
    return re.sub(r"[-_]", "", text).upper()


def bigrams(s):
    """字符 bigram 集合（用于同域度）。"""
    s = re.sub(r"\s+", "", s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_corpus():
    """加载全部 md 的原文侧（normalize+压缩）与归一化侧（压缩）。"""
    raw_sq, norm_sq = [], []
    for p in sorted(MD_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        _orig, norm = build_chunk_text(raw)
        raw_sq.append(squashed(normalize_equipment_tags(normalize_digits(raw))))
        norm_sq.append(squashed(norm))
    return raw_sq, norm_sq


def real_tags(manifest):
    """语料里的真实标签（equipment_tags + hazard_tags）——同域度的参照系。"""
    docs = manifest.get("documents", manifest) if isinstance(manifest, dict) else manifest
    eq, hz = set(), set()
    for d in docs:
        for t in (d.get("equipment_tags") or []):
            eq.add(t.strip())
        for t in (d.get("hazard_tags") or []):
            hz.add(t.strip())
    return eq, hz


def verify(subject, raw_sq, norm_sq):
    """双口径验证：原文侧 + 归一化侧。返回 (raw_hit, norm_hit)。"""
    key = squashed(subject)
    raw_hit = any(key in rs for rs in raw_sq)
    norm_hit = any(key in ns for ns in norm_sq)
    return raw_hit, norm_hit


def build_ngram_index(texts, lengths):
    """按长度建 n-gram 索引（每个长度只扫一遍语料）。"""
    idx = {L: set() for L in lengths}
    for t in texts:
        for L in lengths:
            for i in range(len(t) - L + 1):
                idx[L].add(t[i:i + L])
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reject-near-miss", type=float, default=0.6,
                    help="与语料中同长度 n-gram 的字符 bigram 相似度 ≥ 此值 → 剔除"
                         "（同一事物换了说法，精确匹配看不见）。0 = 不剔除")
    ap.add_argument("--out", default="refusal_pool_multitype.json",
                    help="输出文件名（相对 data/eval/）")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_sq, norm_sq = load_corpus()
    total_chars = sum(len(s) for s in raw_sq)
    print(f"语料：{len(raw_sq)} 份 md，双口径压缩后共 {total_chars} 字符")

    eq, hz = real_tags(manifest)
    real_all = sorted(eq | hz)
    real_bg = [(t, bigrams(t)) for t in real_all]
    print(f"真实标签参照系：equipment {len(eq)} 个 + hazard {len(hz)} 个 = {len(real_all)} 个")

    # 真实位号的压缩键 —— 假位号不能与它们撞
    real_keys = set()
    for t in eq:
        real_keys.add(squashed(t))
        real_keys.add(re.sub(r"[^0-9A-Za-z一-鿿]", "", t))

    def nearest(subject):
        """同域度：与真实标签的字符 bigram 最大 Jaccard。"""
        sb = bigrams(subject)
        best, score = "", 0.0
        for t, tb in real_bg:
            s = jaccard(sb, tb)
            if s > score:
                best, score = t, s
        return best, round(score, 3)

    # 🔴 语义近邻索引（精确匹配的必要补充，见下方 near_miss 说明）
    subj_lens = sorted({len(c) for c in FAKE_EQUIPMENT + FAKE_MEDIUM + FAKE_PROCESS})
    ngram_idx = build_ngram_index(raw_sq, subj_lens)
    print(f"近邻索引：长度 {subj_lens} 的 n-gram 种类 "
          f"{ {L: len(ngram_idx[L]) for L in subj_lens} }")

    def near_miss(subject):
        """在语料里找与主体**长度相同**、字符 bigram 最相近的 n-gram。

        🔴 精确子串匹配查不出「同一事物换了说法」：
        「液氨储罐区」查不到，但语料里有「液氨储罐」（相似度 0.75）
        → 主体实质存在 → 题目失效（模型能答出来）。
        """
        L = len(subject)
        sb = bigrams(subject)
        best, score = "", 0.0
        for cand in ngram_idx.get(L, ()):
            s = jaccard(sb, bigrams(cand))
            if s > score:
                best, score = cand, s
        return best, round(score, 3)

    results = {}

    # ---- 机制 A：假位号 -------------------------------------------------
    safe_ids, rejected = [], []
    n = 9000
    for pattern in FAKE_ID_PATTERNS:
        for _ in range(400):
            if len(safe_ids) >= 80:
                break
            n += 1
            cand = pattern.format(n=n)
            if not re.match(r"^[0-9A-Z]+(-[0-9A-Z]+){1,4}$", cand):
                continue
            ck = squashed(cand)
            if ck in real_keys or cand in eq:
                rejected.append({"subject": cand, "reason": "与真实位号形态撞车"})
                continue
            raw_hit, norm_hit = verify(cand, raw_sq, norm_sq)
            if raw_hit or norm_hit:
                rejected.append({"subject": cand,
                                 "reason": "原文命中" if raw_hit else "归一化文本命中"})
                continue
            nb, sc = nearest(cand)
            safe_ids.append({
                "subject": cand, "match_key": ck,
                "raw_hit": False, "norm_hit": False,
                "nearest_real": nb, "domain_score": sc,
            })
    results["fake_id"] = safe_ids

    # ---- 机制 B/C/D：假设备 / 假介质 / 假工序 ---------------------------
    for mech, cands in (("fake_equipment", FAKE_EQUIPMENT),
                        ("fake_medium", FAKE_MEDIUM),
                        ("fake_process", FAKE_PROCESS)):
        kept, rej = [], []
        for c in cands:
            if c in eq or c in hz:
                rej.append({"subject": c, "reason": "就是语料里的真实标签"})
                continue
            raw_hit, norm_hit = verify(c, raw_sq, norm_sq)
            if raw_hit or norm_hit:
                rej.append({"subject": c,
                            "reason": "原文命中" if raw_hit else "归一化文本命中"})
                continue
            # 🔴 语义近邻剔除：精确匹配查不出「同一事物换了说法」
            nm, nsc = near_miss(c)
            if nsc >= args.reject_near_miss:
                rej.append({"subject": c,
                            "reason": f"与语料中的「{nm}」语义极近（{nsc}）"})
                continue
            nb, sc = nearest(c)
            kept.append({
                "subject": c, "raw_hit": False, "norm_hit": False,
                "nearest_real": nb, "domain_score": sc,
                "near_miss": nm, "near_miss_score": nsc,
            })
        results[mech] = kept
        results[mech + "_rejected"] = rej

    # ---- 汇总 -----------------------------------------------------------
    print()
    for mech in ("fake_id", "fake_equipment", "fake_medium", "fake_process"):
        kept = results[mech]
        rej = results.get(mech + "_rejected", [])
        print(f"{mech}: 通过 {len(kept)} / 候选 {len(kept) + len(rej)}")
        if rej:
            print(f"   被拒 {len(rej)} 个：")
            for r in rej[:12]:
                print(f"     ✗ {r['subject']:<16} {r['reason']}")
            if len(rej) > 12:
                print(f"     ... 另 {len(rej) - 12} 个")

    payload = {
        "note": ("多机制拒答题主体池。每个主体都在**原文侧（normalize 后压缩分隔符）**"
                 "与**归一化侧**两个口径上验证过「不存在于语料」。"
                 "nearest_real = 与语料真实标签字符 bigram 最相近的一个，"
                 "用于人工判断题目是否「同域但不存在」。"),
        "corpus": {
            "n_docs": len(raw_sq),
            "total_chars": total_chars,
            "match_rule": "squash(去掉 - 与 _) 后精确子串匹配；原文侧先 normalize",
        },
        "real_equipment_tags": sorted(eq),
        "real_hazard_tags": sorted(hz),
        "mechanisms": {
            "fake_id": "仿真实位号形状的假位号（与 M2 已有池同构，扩到 80 个）",
            "fake_equipment": "语料里没有的装置/设备类别",
            "fake_medium": "语料里没有的介质/物料",
            "fake_process": "语料里没有的工序/作业",
        },
        "fake_id": results["fake_id"],
        "fake_equipment": results["fake_equipment"],
        "fake_medium": results["fake_medium"],
        "fake_process": results["fake_process"],
        "n_by_mechanism": {m: len(results[m]) for m in
                           ("fake_id", "fake_equipment", "fake_medium", "fake_process")},
        "n_rejected": {m: len(results.get(m + "_rejected", [])) for m in
                       ("fake_id", "fake_equipment", "fake_medium", "fake_process")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT.parent / args.out
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print()
    print("=== 各机制通过数 ===")
    for m, c in payload["n_by_mechanism"].items():
        print(f"  {m:16s} {c}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
