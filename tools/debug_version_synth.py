"""调试：版本陷阱题合成为什么解析失败。抓完整原始输出并逐步测 parse_json。"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import llm_client as L
import refine_eval_set as R

manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
if isinstance(manifest, dict):
    manifest = manifest.get("documents", manifest)
by_id = {d["doc_id"]: d for d in manifest}
old = by_id["yulin_2023_old"]
new = by_id["yulin_donghuo_xianshi"]

prompt = R.VERSION_PROMPT.format(
    old_id=old["doc_id"], old_no=old.get("doc_no") or "-",
    old_from=old.get("effective_from") or "-",
    new_id=new["doc_id"], new_no=new.get("doc_no") or "-",
    new_from=new.get("effective_from") or "-",
    old_text=R._read_md(old["doc_id"])[:6000],
    new_text=R._read_md(new["doc_id"])[:6000],
    n=40, base_date=R.BASE_DATE)

raw = L.call_llm(prompt, system=R.SYSTEM_PROMPT, temperature=0.0, max_tokens=3000)
out = ROOT / "data" / "eval" / "_debug_version_raw.txt"
out.write_text(raw, encoding="utf-8")
print(f"原始长度 {len(raw)} 字符 -> {out}")
print("---- 尾部 200 字 ----")
print(repr(raw[-200:]))
print("---- parse_json 结果 ----")
res = L.parse_json(raw, default=None)
print("类型:", type(res).__name__, "| 长度:", len(res) if hasattr(res, "__len__") else "-")
if res is None:
    # 手动逐步试
    s, e = raw.find("["), raw.rfind("]")
    seg = raw[s:e + 1]
    print("抠出的数组片段长度:", len(seg))
    try:
        json.loads(seg)
        print("直接 json.loads 成功?!")
    except Exception as ex:
        print("json.loads 失败:", ex)
        # 定位出错位置
        try:
            json.loads(seg)
        except json.JSONDecodeError as je:
            pos = je.pos
            print(f"错误位置 {pos}: {seg[max(0,pos-120):pos+120]!r}")
