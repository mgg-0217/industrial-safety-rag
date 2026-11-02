"""拒答题位号池 —— PRD M2 / 7.1。

拒答题要问「语料里根本没有的设备」。造之前必须**核实它不在全文里出现**，
PRD 7.1 的原话是「必须核实它不在 34 万字符全文里出现，用全文精确匹配验证，
不是靠『看着不像』」。

🔴 匹配口径（本脚本最容易写错的地方，写错过一次）：
  `tag_key` 去掉分隔符（19-V-4215 → 19V4215），而 norm_text **保留**分隔符。
  直接拿 key 去 count 归一化文本，**永远得到 0** —— 等于只查了一半形态，
  看起来"验证通过"，实际漏检。
  正确做法与 `normalize.find_tag_occurrences` 对齐：把待查文本也压成
  无分隔符形态再比。
  原文侧还要先跑一遍 normalize（中文破折号 → 半角连字符），
  否则 `19—V—4215` 压缩后仍带破折号，查不到 `19V4215`。

另外真位号本身也不能用：真位号出现在语料里，问它必然能答出来。

输出：
  data/eval/refusal_pool.json   安全假位号池 + 每个位号的验证证据
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from normalize import (build_chunk_text, find_tag_occurrences, normalize_digits,  # noqa: E402
                       normalize_equipment_tags)

MD_DIR = ROOT / "data" / "md"
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "data" / "eval" / "refusal_pool.json"

# 假位号的构造模板：仿真实位号的形状（字母-数字-数字 / 数字-字母-数字）
FAKE_PATTERNS = [
    "T-{n}", "V-{n}", "MP-{n}", "PV-{n}", "R-{n}", "HE-{n}",
    "T-{n}-A", "P-{n}-B", "HX-{n}", "C-{n}", "E-{n}", "S-{n}",
]


def squashed(text):
    """压成无分隔符形态（与 normalize.find_tag_occurrences 的口径一致）。"""
    return re.sub(r"[-_]", "", text).upper()


def load_corpus():
    """加载全部 md 的原文、归一化文本，以及两者的**压缩形态**。

    压缩形态的原文侧要先跑一遍 normalize（中文破折号 -> 半角连字符），
    否则 `19—V—4215` 这种写法在压缩后仍带破折号，查不到 `19V4215`。
    """
    texts, norm_texts, raw_sq, norm_sq = [], [], [], []
    for p in sorted(MD_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        texts.append(raw)
        _orig, norm = build_chunk_text(raw)
        norm_texts.append(norm)
        raw_sq.append(squashed(normalize_equipment_tags(normalize_digits(raw))))
        norm_sq.append(squashed(norm))
    return texts, norm_texts, raw_sq, norm_sq


def real_tags(manifest):
    """真实位号池（来自 manifest 的 equipment_tags）。"""
    tags = set()
    for d in manifest:
        for t in (d.get("equipment_tags") or []):
            tags.add(t.strip())
    return tags


def key_variants(tag):
    """一个位号在检索侧可能出现的所有形态（原文侧 + 归一化侧）。"""
    out = {tag, squashed(tag)}
    out.add(re.sub(r"[^0-9A-Za-z一-鿿]", "", tag))
    return {v for v in out if v}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest = manifest.get("documents", manifest)
    texts, norm_texts, raw_sq, norm_sq = load_corpus()
    total_chars = sum(len(t) for t in texts)
    print(f"语料：{len(texts)} 份 md，共 {total_chars} 字符（原文侧）")

    real = real_tags(manifest)
    print(f"manifest 里的真实位号：{len(real)} 个")

    # 真位号的压缩键——假位号不能与它们撞
    real_keys = set()
    for t in real:
        real_keys |= key_variants(t)
        real_keys.add(squashed(t))
    print(f"真实位号的所有形态：{len(real_keys)} 个（假位号必须与之不相交）")

    safe, rejected = [], []
    n = 9000
    for pattern in FAKE_PATTERNS:
        for _ in range(400):
            n += 1
            cand = pattern.format(n=n)
            if not re.match(r"^[0-9A-Z]+(-[0-9A-Z]+){1,4}$", cand):
                continue
            cand_key = squashed(cand)
            if cand_key in real_keys or cand in real:
                rejected.append({"candidate": cand, "reason": "与真实位号形态撞车"})
                continue
            # 原文侧（归一化破折号后、压缩分隔符）
            hit_raw = any(cand_key in rs for rs in raw_sq)
            # 归一化侧（检索真正用的形态）
            hit_norm = any(cand_key in ns for ns in norm_sq)
            if hit_raw or hit_norm:
                rejected.append({"candidate": cand, "reason":
                                 "原文命中" if hit_raw else "归一化文本命中"})
                continue
            safe.append({
                "equipment_id": cand,
                "match_key": cand_key,
                "raw_hit": False,
                "norm_hit": False,
            })
            if len(safe) >= 40:
                break
        if len(safe) >= 40:
            break

    # 附带产出：真位号在语料中的实际出现次数
    # 🔴 必须同时给**原文侧**与**归一化侧**两个口径：manifest 存的是规范形式
    #（19-V-4215），而原文写的是 19—V—4215（中文破折号），
    # 只查原文侧会全部得到 0，看起来像"位号根本不在语料里"。
    real_occurrences = {}
    for t in real:
        raw_cnt = sum(r.count(squashed(t)) for r in raw_sq)
        norm_cnt = sum(len(find_tag_occurrences(nt, t)) for nt in norm_texts)
        real_occurrences[t] = {"raw": raw_cnt, "norm": norm_cnt}

    payload = {
        "note": ("拒答题用的假位号池。每个位号都已在**原文与归一化文本**上做过精确匹配验证"
                 "（口径与 normalize.find_tag_occurrences 一致：压缩分隔符后比对），"
                 "确保语料里不存在。用于 PRD 7.1 的拒答题构造。"),
        "corpus": {
            "n_docs": len(texts),
            "total_chars": total_chars,
            "match_rule": "squash(去掉 - 与 _) 后精确子串匹配",
        },
        "real_equipment_tags": sorted(real),
        "real_tag_occurrences": real_occurrences,
        "real_variant_count": len(real_keys),
        "safe_fake_ids": safe,
        "n_safe": len(safe),
        "n_rejected": len(rejected),
        "rejected_sample": rejected[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n安全假位号：**{len(safe)}** 个（候选被拒 {len(rejected)} 个）")
    print(f"样例：{[s['equipment_id'] for s in safe[:8]]}")
    print(f"\n真实位号出现次数（raw = 原文侧, norm = 归一化侧）：")
    for t, c in sorted(real_occurrences.items(), key=lambda kv: -kv[1]["norm"]):
        if c["norm"] or c["raw"]:
            print(f"  {t}: raw={c['raw']} norm={c['norm']}")
    print(f"  （真位号形态设备类名合计 {len(real)} 个，其中位号形态 5 个）")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
