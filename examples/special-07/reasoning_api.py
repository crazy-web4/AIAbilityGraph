"""
推理模型（Reasoning Model）调用示例

演示：
  - 思考过程（reasoning_content / reasoning_tokens）与最终答案分离
  - 思考预算控制（reasoning_effort / 思考 token 上限）
  - 无 API Key 时用内置 Mock 客户端离线演示返回结构

依赖（真实调用）：
  pip install openai
运行：
  python reasoning_api.py                 # 离线 Mock 演示
  DEEPSEEK_API_KEY=sk-xxx python reasoning_api.py   # 真实调用
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ReasoningResult:
    reasoning: str
    answer: str
    reasoning_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Mock 客户端（离线演示返回结构）
# ---------------------------------------------------------------------------

class MockReasoningClient:
    async def solve(self, question: str, effort: str) -> ReasoningResult:
        reasoning = (
            "让我分析这道题。\n"
            "第一步：明确已知条件……\n"
            "等等，我刚才的假设错了，换一种方法。\n"
            "第二步：代入并验算……结果自洽。\n"
            f"（effort={effort}，这是模拟的长思维链）"
        )
        return ReasoningResult(
            reasoning=reasoning,
            answer=f"（模拟答案）针对「{question[:18]}…」的最终结论。",
            reasoning_tokens=980 if effort == "high" else 320,
            completion_tokens=60,
        )


# ---------------------------------------------------------------------------
# 真实客户端（OpenAI 兼容：DeepSeek-R1 / o 系列 / QwQ 类）
# ---------------------------------------------------------------------------

class OpenAIReasoningClient:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )

    async def solve(self, question: str, effort: str) -> ReasoningResult:
        resp = self.client.chat.completions.create(
            model=os.getenv("REASONING_MODEL", "deepseek-reasoner"),
            messages=[{"role": "user", "content": question}],
            extra_body={"reasoning_effort": effort},   # low/medium/high
        )
        msg = resp.choices[0].message
        details = getattr(resp.usage, "completion_tokens_details", None)
        return ReasoningResult(
            reasoning=getattr(msg, "reasoning_content", "") or "(思考内容未单独返回)",
            answer=msg.content,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) if details else 0,
            completion_tokens=resp.usage.completion_tokens,
        )


async def main() -> None:
    client = (OpenAIReasoningClient() if os.getenv("DEEPSEEK_API_KEY")
              else MockReasoningClient())
    question = "证明：任意 6 个人中，必有 3 人互相认识或 3 人互相不认识。"

    for effort in ("low", "high"):
        r = await client.solve(question, effort)
        print(f"\n=== reasoning_effort = {effort} ===")
        print(f"思考 token: {r.reasoning_tokens} | 答案 token: {r.completion_tokens}")
        print("思维链（前 80 字）:", r.reasoning[:80].replace("\n", " "))
        print("最终答案:", r.answer[:60])
    print("\n要点：思考 token 也计费；高 effort 更贵更慢，按任务难度路由。")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
