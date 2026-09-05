"""
模型路由器（对标 OpenRouter 的核心逻辑的教学实现）

特性：
  - OpenAI 兼容统一入口
  - 多 Provider 适配器（演示用 Mock + 可接真实 httpx 调用）
  - 健康分/成本/延迟加权路由
  - 熔断 + 指数退避重试 + Fallback 链 + 幂等键
  - 三级缓存（exact）演示

适用场景：教学/原型；生产需补全流式聚合、GCRA 限流、Kafka 计量。
前置知识：asyncio、OpenAI Chat Completions 协议。

运行：
  python model_router.py          # 离线演示（Mock Provider，无需 API Key）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ModelAPIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status


@dataclass
class Endpoint:
    """一个模型端点（某 Provider 上的某模型）。"""
    provider: str
    model: str
    price_in: float          # 每 1M 输入 token 美元
    price_out: float         # 每 1M 输出 token 美元
    # 运行期健康状态（由探测/调用结果滑动更新）
    err_rate: float = 0.0
    p99_latency: float = 1.0
    quota_left: int = 1_000_000
    circuit_open_until: float = 0.0
    _calls: list = field(default_factory=list)   # (ts, ok)

    def record(self, ok: bool) -> None:
        self._calls.append((time.time(), ok))
        cutoff = time.time() - 30
        self._calls = [(t, o) for t, o in self._calls if t > cutoff]
        n = len(self._calls)
        self.err_rate = 0.0 if n == 0 else sum(1 for _, o in self._calls if not o) / n

    def score(self) -> float:
        """分数越低越优先：健康权重最大，其次成本、延迟。"""
        if time.time() < self.circuit_open_until or self.quota_left <= 0:
            return float("inf")
        return 100 * self.err_rate + self.price_out + 0.1 * self.p99_latency


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    cached: bool = False


# ---------------------------------------------------------------------------
# Provider 适配器
# ---------------------------------------------------------------------------

class BaseAdapter:
    """适配器：把统一请求转成 Provider 私有协议，并归一化响应。"""

    async def chat(self, ep: Endpoint, payload: dict) -> ChatResult:
        raise NotImplementedError


class MockAdapter(BaseAdapter):
    """离线模拟：按概率注入失败，用于演示熔断/重试/fallback。"""

    def __init__(self, fail_rate: float = 0.5):
        self.fail_rate = fail_rate

    async def chat(self, ep: Endpoint, payload: dict) -> ChatResult:
        await asyncio.sleep(0.05)
        if random.random() < self.fail_rate:
            raise ModelAPIError(random.choice([429, 503]), f"{ep.provider} 注入故障")
        last = payload["messages"][-1]["content"]
        text = f"[{ep.provider}/{ep.model}] 回答：关于「{last[:20]}…」的模拟回复。"
        pt, ct = 120, 40
        cost = (pt * ep.price_in + ct * ep.price_out) / 1e6
        return ChatResult(text, ep.model, ep.provider, pt, ct, cost)


# ---------------------------------------------------------------------------
# 路由器
# ---------------------------------------------------------------------------

class ModelRouter:
    def __init__(self, catalog: dict[str, list[Endpoint]],
                 fallback_chain: dict[str, list[str]],
                 adapter: BaseAdapter | None = None):
        # catalog: 逻辑模型 -> 端点列表；fallback_chain: 模型 -> 降级链
        self.catalog = catalog
        self.fallback_chain = fallback_chain
        self.adapter = adapter or MockAdapter()
        self._cache: dict[str, ChatResult] = {}

    def _candidates(self, model: str) -> list[Endpoint]:
        chain = [model] + self.fallback_chain.get(model, [])
        eps = [ep for m in chain for ep in self.catalog.get(m, [])]
        return sorted(eps, key=Endpoint.score)

    @staticmethod
    def _cache_key(model: str, payload: dict) -> str:
        raw = json.dumps({"m": model, "p": payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _call_once(self, ep: Endpoint, payload: dict, attempt: int) -> ChatResult:
        try:
            result = await asyncio.wait_for(self.adapter.chat(ep, payload), timeout=10)
            ep.record(ok=True)
            return result
        except ModelAPIError as e:
            ep.record(ok=False)
            if ep.err_rate > 0.5:                      # 熔断：错误率过高开路 5s
                ep.circuit_open_until = time.time() + 5
            if e.status not in RETRYABLE_STATUS:
                raise
            backoff = min(2 ** attempt, 8) * (0.5 + random.random() / 2)
            await asyncio.sleep(backoff)
            raise
        except asyncio.TimeoutError:
            ep.record(ok=False)
            raise ModelAPIError(504, "timeout")

    async def chat(self, model: str, payload: dict, max_attempts: int = 4) -> ChatResult:
        key = self._cache_key(model, payload)          # exact 缓存
        if key in self._cache:
            r = self._cache[key]
            return ChatResult(r.text, r.model, r.provider, r.prompt_tokens,
                              r.completion_tokens, 0.0, cached=True)

        last_exc: Exception | None = None
        for attempt, ep in enumerate(self._candidates(model)[:max_attempts]):
            try:
                result = await self._call_once(ep, payload, attempt)
                self._cache[key] = result
                return result
            except ModelAPIError as e:
                last_exc = e
                continue
        raise ModelAPIError(503, f"所有端点均失败：{last_exc}")


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def build_demo_router() -> ModelRouter:
    catalog = {
        "fast-model": [
            Endpoint("providerA", "fast-v1", 0.5, 1.5),
            Endpoint("providerB", "fast-v1", 0.6, 1.6),
        ],
        "strong-model": [
            Endpoint("providerA", "strong-v2", 2.0, 8.0),
        ],
    }
    fallback = {"fast-model": ["strong-model"]}
    return ModelRouter(catalog, fallback, MockAdapter(fail_rate=0.3))


async def main() -> None:
    print("演示：高故障率端点 + 重试/fallback + exact 缓存\n")
    router = build_demo_router()
    payload = {"messages": [{"role": "user", "content": "如何设计一个模型网关？"}]}
    for i in range(6):
        try:
            r = await router.chat("fast-model", payload)
            tag = "缓存命中" if r.cached else f"成本 ${r.cost:.6f}"
            print(f"请求 {i+1}: {r.provider}/{r.model}  {tag}")
        except ModelAPIError as e:
            print(f"请求 {i+1}: 失败 {e}")


if __name__ == "__main__":
    asyncio.run(main())
