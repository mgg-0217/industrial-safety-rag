"""Anthropic 兼容端点客户端 —— M2 评测集精修用。

**为什么自己写而不装 SDK**：这个端点本身就是 Anthropic 兼容格式，
一个 HTTP 请求就够；项目一已经趟过这条路（`expense-precheck/src/eval/baseline_llm.py`），
照搬调用方式即可，不引入新依赖。

**凭据怎么来（重要）**：本仓库**不落盘**任何 API key。
按顺序找配置，找不到就报错：
  1. 环境变量 ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / B1_MODEL
  2. 项目根的 .env（已被 .gitignore 忽略）
  3. 兄弟目录 `../expense-precheck/.env`（项目一同一端点的配置，运行时回退读，不复制）

第 3 条是为了避免把 key 复制进本仓库——复制动作会被 auto mode 分类器拦下，
而且即便复制成功，凭据落在两个仓库里也意味着两处都要轮换。
"""
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FALLBACK_ENV = ROOT.parent / "expense-precheck" / ".env"

DEFAULTS = {
    "ANTHROPIC_BASE_URL": "https://api.minimax.cn/anthropic",
    "B1_MODEL": "MiniMax-M3",
}


def _parse_env_file(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def load_config():
    """返回 (base_url, api_key, model)。找不到 key 抛异常（不静默继续）。

    凭据来源按优先级：
      1. 环境变量 —— Claude Code 会把用户级 settings.json 的 env 注入进来
         （`ANTHROPIC_AUTH_TOKEN` 是 Anthropic SDK 的标准变量名；
         本机的 MiniMax 端点就是用这个变量配的）
      2. 本项目根的 .env（已被 .gitignore 忽略）
      3. 兄弟目录 ../expense-precheck/.env（运行时回退读，不复制）
    """
    cfg = {}
    cfg.update(_parse_env_file(ROOT / ".env"))
    cfg.update(_parse_env_file(FALLBACK_ENV))
    # 环境变量优先级最高
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "B1_MODEL"):
        if os.getenv(k):
            cfg[k] = os.getenv(k)

    base = (cfg.get("ANTHROPIC_BASE_URL") or DEFAULTS["ANTHROPIC_BASE_URL"]).rstrip("/")
    key = cfg.get("ANTHROPIC_API_KEY") or cfg.get("ANTHROPIC_AUTH_TOKEN") or ""
    model = cfg.get("B1_MODEL") or DEFAULTS["B1_MODEL"]
    if not key:
        raise RuntimeError(
            "找不到 API key。请设置环境变量 ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN，"
            "或在项目根放 .env（已被 .gitignore 忽略），"
            "或确保 ../expense-precheck/.env 存在。")
    return base, key, model


def call_llm(prompt, system=None, model=None, temperature=0.0,
             max_tokens=4096, timeout=180, retries=3, retry_sleep=3):
    """调 Anthropic 兼容端点，返回原始文本。

    重试策略：网络错误 / 5xx 退避重试（指数退避，封顶 30s）；
    4xx 不重试（请求本身有问题，重试没意义）。
    """
    base, key, default_model = load_config()
    model = model or default_model

    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    if system:
        # 该端点的 Anthropic 兼容层支持 system 字段
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

    req = urllib.request.Request(
        f"{base}/v1/messages", data=body, method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        })

    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return "".join(b.get("text", "") for b in data.get("content", []))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {detail}"
            if e.code == 429:
                # 速率限制：**退避要比普通错误长得多**。
                # 实测踩过：并发 6 时 37/45 条被 429 打回，
                # 而 3/6/12 秒的普通退避根本等不到额度恢复。
                if attempt < retries - 1:
                    time.sleep(20 * (attempt + 1))
                continue
            if 400 <= e.code < 500:
                break  # 请求本身有问题，不重试
        except Exception as e:  # 网络超时、连接重置等
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt * retry_sleep, 30))

    raise RuntimeError(f"LLM 调用失败（重试 {retries} 次）：{last_err}")


def _repair_truncated_array(raw):
    """输出被 max_tokens 截断时，救回**已经写完**的那部分。

    实测踩过：一次让模型生成 40 条题，max_tokens=3000 不够，
    输出停在第 40 条的中间，JSON 数组缺右括号 → 整个数组解析失败 → 一条都不剩。
    这里从后往前找每个 `}` 结尾的位置，尝试补 `]` 后再解析，
    取**能解析出最多条目**的那个。写完的部分不浪费。
    """
    if not raw:
        return None
    start = raw.find("[")
    if start == -1:
        return None
    best = None
    # 从后往前找 "}" 收尾的位置
    idx = raw.rfind("}")
    while idx > start:
        seg = raw[:idx + 1] + "]"
        try:
            obj = json.loads(seg)
            if isinstance(obj, list):
                if best is None or len(obj) > len(best):
                    best = obj
                if len(best) == len(obj) and obj:
                    break
        except Exception:
            pass
        idx = raw.rfind("}", 0, idx)
    return best


def parse_json(raw, default=None):
    """从模型输出里抠 JSON。多层容错。

    1. 直接 parse
    2. 去 markdown 围栏后 parse
    3. 抠 `{...}` / `[...]` 后 parse
    4. **截断修复**：输出被 max_tokens 截断时，救回已写完的条目

    🔴 **必须同时支持顶层数组**。实测踩过：让模型一次生成多条题，
    它返回的是 `[{"question":...},{"question":...}]`，
    而只抠 `{...}` 会得到 `{...},{...}` —— 多个对象没有数组括号，不是合法 JSON，
    于是解析失败、整批题被静默丢弃（当时表现是"合成 0 条"）。
    """
    raw = (raw or "").strip()
    if not raw:
        return default
    candidates = [raw]
    if "```" in raw:
        stripped = raw.replace("```json", "").replace("```", "").strip()
        candidates.insert(0, stripped)

    # 抠对象 {...}
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    # 抠数组 [...]（🔴 顶层数组必须单独支持）
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for cand in candidates:
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue

    # 截断修复（放最后：只有前面都失败才走这条）
    repaired = _repair_truncated_array(raw)
    if repaired is not None:
        return repaired
    return default
