# -*- coding: utf-8 -*-
"""探测 LLM 端点可用性（**不打印凭据**）。

## 为什么单独做一个探测脚本

换 key / 换端点之后，最省事的做法是直接跑 300 题评测 ——
但那要花十几分钟额度，而**失败原因可能是 key 过期、模型名不对、端点不通**，
三种原因在评测脚本的报错里长得一模一样。

→ 先用一条极短的问句探一下，**输出只含 HTTP 状态码与脱敏后的响应体片段**。

## 🔴 凭据边界（这条是硬要求）

- key **只**通过 `llm_client.load_config()` 读入内存，**从不打印、从不落盘**；
- 打印前一律走 `llm_client._redact()`（HTTP 错误体经常回显 key）；
- 落盘的只有 `eval_runs/<ts>_llm_probe.json`，里面**没有** key 字段。

跑法：
    python tools/probe_llm.py
    python tools/probe_llm.py --base-url https://example.com/anthropic --model SomeModel
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm_client  # noqa: E402

RUN_DIR = ROOT / "eval_runs"

PROMPT = "只回答两个字：可用"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="", help="临时覆盖端点（不写进 .env）")
    ap.add_argument("--model", default="", help="临时覆盖模型名")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    base, key, default_model = llm_client.load_config()
    if args.base_url:
        base = args.base_url.rstrip("/")
    model = args.model or default_model

    # 🔴 只报端点与模型，**不报 key**（连前缀都不给）。
    print(f"[probe] base_url = {base}")
    print(f"[probe] model    = {model}")
    print(f"[probe] key      = <已加载 {len(key)} 字符，不显示>")

    ok, err = False, None
    text = ""
    try:
        text = llm_client.call_llm(PROMPT, model=model, timeout=args.timeout,
                                   retries=args.retries, max_tokens=16)
        ok = bool(text.strip())
    except Exception as e:  # noqa: BLE001
        err = llm_client._redact(f"{type(e).__name__}: {e}", key)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base,
        "model": model,
        "ok": ok,
        "response_preview": llm_client._redact((text or "")[:200], key),
        "error": err,
        "note": "本文件不含任何凭据字段（key 只存在于内存中）",
    }
    RUN_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    p = RUN_DIR / f"{ts}_llm_probe.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[probe] ok = {ok}")
    if err:
        print(f"[probe] error = {err}")
    else:
        print(f"[probe] response = {payload['response_preview']!r}")
    print(f"-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
