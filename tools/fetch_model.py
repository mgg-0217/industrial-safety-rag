#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**预下载句向量模型到本地目录**（绕开 pypi 慢速链路上的 95.8 MB）。

🔴 **为什么需要这个**：
本机 pypi 链路实测只有 **27 kB/s**（transformers 9.7 MB 用了 5 分 17 秒），
而模型 `BAAI/bge-small-zh-v1.5` 的 `model.safetensors` 有 **95.8 MB**
→ 走 `huggingface_hub` 在线下载要 ~1 小时，且**中途失败就得重来**。
→ 先用 `hf-mirror.com` 把文件落到本地目录，再让 `SentenceTransformer` 从本地读。

🔴 **UA 头是必需的**：
镜像会**拒绝 `Python-urllib` 的默认 UA**（实测 403 Forbidden）；
加上常规浏览器 UA 后全部文件可取（`huggingface_hub` 自带正常 UA，不受影响）。

用法：
    python tools/fetch_model.py --out models/bge-small-zh-v1.5
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

#: 句向量模型需要的文件（sentence-transformers 的加载顺序）。
FILES = [
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "1_Pooling/config.json",
    "model.safetensors",
]

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}


def fetch(url: str, dest: pathlib.Path, timeout=120):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        t0 = time.time()
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total and time.time() - t0 > 2:
                    pct = got / total * 100
                    print(f"    {dest.name}: {got/1e6:.1f}/{total/1e6:.1f} MB "
                          f"({pct:.0f}%) {got/1e6/(time.time()-t0):.2f} MB/s",
                          flush=True)
                    t0 = time.time()
                    got_mark = got
    return dest.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--out", default="models/bge-small-zh-v1.5")
    ap.add_argument("--base", default="https://hf-mirror.com")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{args.base}/{args.repo}/resolve/main/"
    print(f"模型 {args.repo} -> {out}", flush=True)
    ok = 0
    for name in FILES:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [已有] {name} ({dest.stat().st_size/1e6:.2f} MB)", flush=True)
            ok += 1
            continue
        try:
            size = fetch(base + name, dest)
            print(f"  [完成] {name} ({size/1e6:.2f} MB)", flush=True)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            if dest.exists():
                dest.unlink()
    print(f"\n{ok}/{len(FILES)} 个文件就绪 -> {out}")
    if ok < len(FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
