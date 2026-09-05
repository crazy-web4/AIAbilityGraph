"""
Durable Agent Loop（教学实现）

演示智能体工程的四个关键机制：
  1. 状态外置：每一步把 RunState 写入检查点（此处用 JSON 文件模拟）
  2. 崩溃恢复：进程中途被杀后重跑，可从最近检查点继续
  3. 幂等工具：用 idempotency_key 保证工具重放不重复执行副作用
  4. 预算护栏：max_steps / max_cost 防止死循环烧钱

LLM 用脚本化 Mock 模拟（离线可跑）；真实环境替换 model_chat 为模型网关调用即可。

运行：
  python durable_agent.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

CHECKPOINT_DIR = "/tmp/durable_agent_demo"


# ---------------------------------------------------------------------------
# 状态与检查点
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    run_id: str
    messages: list[dict] = field(default_factory=list)
    step: int = 0
    cost: float = 0.0


class Checkpointer:
    def __init__(self, root: str = CHECKPOINT_DIR):
        os.makedirs(root, exist_ok=True)
        self.root = root

    def _path(self, run_id: str) -> str:
        return os.path.join(self.root, f"{run_id}.json")

    def save(self, state: RunState) -> None:
        with open(self._path(state.run_id), "w") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)

    def load(self, run_id: str) -> RunState | None:
        path = self._path(run_id)
        if not os.path.exists(path):
            return None
        return RunState(**json.load(open(path)))


# ---------------------------------------------------------------------------
# 幂等工具运行时
# ---------------------------------------------------------------------------

class ToolRuntime:
    """工具执行带幂等键：重放同一 key 直接返回缓存结果，不重复产生副作用。"""

    def __init__(self):
        self._done: dict[str, str] = {}
        self.side_effects = 0          # 统计真实副作用次数（演示用）

    def execute(self, tool: str, args: dict, idempotency_key: str) -> str:
        if idempotency_key in self._done:
            return f"[幂等命中] {self._done[idempotency_key]}"
        # 真实副作用
        self.side_effects += 1
        if tool == "get_weather":
            result = f"{args['city']} 今天晴，25℃"
        elif tool == "send_email":
            result = f"邮件已发送给 {args['to']}"
        else:
            result = f"工具 {tool} 执行完成"
        self._done[idempotency_key] = result
        return f"[真实执行] {result}"


# ---------------------------------------------------------------------------
# Mock LLM：脚本化返回，模拟"先调工具，再给答案"
# ---------------------------------------------------------------------------

def mock_llm(messages: list[dict], step: int) -> dict:
    # 第一步要求查天气，第二步给最终答案
    if step == 0:
        return {"role": "assistant",
                "tool_calls": [{"id": "c1", "name": "get_weather",
                                "arguments": {"city": "北京"}}]}
    return {"role": "assistant", "content": "北京今天晴，25℃，适合出门。"}


# ---------------------------------------------------------------------------
# Agent 运行时
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_steps: int = 10
    max_cost: float = 1.0
    cost_per_step: float = 0.01


class AgentRuntime:
    def __init__(self, tools: ToolRuntime, cp: Checkpointer, budget: Budget):
        self.tools = tools
        self.cp = cp
        self.budget = budget

    async def run(self, run_id: str, user_input: str, simulate_crash: bool = False) -> str:
        state = self.cp.load(run_id)
        if state is None:
            state = RunState(run_id, [{"role": "user", "content": user_input}])
            self.cp.save(state)
            print(f"[新建 Run] {run_id}")
        else:
            print(f"[恢复 Run] {run_id} 从 step {state.step} 继续")

        while True:
            if state.step >= self.budget.max_steps or state.cost >= self.budget.max_cost:
                return "任务终止：超出预算护栏"

            # 恢复点：若上一步 LLM 决定调工具但工具结果缺失（崩溃在执行前），
            # 先幂等地补执行，再继续 —— 这是 Durable Execution 的关键。
            pending = self._pending_tool_calls(state)
            if pending:
                self._run_tools(run_id, state, pending)
                self.cp.save(state)
                continue

            state.step += 1
            state.cost += self.budget.cost_per_step
            msg = mock_llm(state.messages, state.step - 1)
            state.messages.append(msg)
            self.cp.save(state)                       # 每步检查点

            if simulate_crash and state.step == 1:
                print("[模拟崩溃] 进程退出！（检查点已保存）")
                return "__CRASHED__"

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                self.cp.save(state)
                return msg["content"]

            self._run_tools(run_id, state, tool_calls)
            self.cp.save(state)

    @staticmethod
    def _pending_tool_calls(state: RunState) -> list[dict]:
        """找出已有 tool_call 但还没有对应 tool 结果的调用。"""
        answered = {m.get("tool_call_id") for m in state.messages if m.get("role") == "tool"}
        pending = []
        for m in state.messages:
            if m.get("role") == "assistant":
                for c in m.get("tool_calls", []):
                    if c["id"] not in answered:
                        pending.append(c)
        return pending

    def _run_tools(self, run_id: str, state: RunState, calls: list[dict]) -> None:
        for call in calls:
            key = f"{run_id}:{state.step}:{call['id']}"       # 幂等键
            result = self.tools.execute(call["name"], call["arguments"], key)
            state.messages.append({"role": "tool", "tool_call_id": call["id"],
                                   "content": result})


async def main() -> None:
    cp, tools, budget = Checkpointer(), ToolRuntime(), Budget()
    agent = AgentRuntime(tools, cp, budget)
    run_id = "run-demo-001"
    # 清理旧检查点
    if os.path.exists(cp._path(run_id)):
        os.remove(cp._path(run_id))

    print("=== 第一次运行：在第 1 步后模拟崩溃 ===")
    out = await agent.run(run_id, "北京天气怎么样？", simulate_crash=True)
    print("返回:", out, "| 真实副作用次数:", tools.side_effects)

    print("\n=== 第二次运行：从检查点恢复（工具幂等，不重复执行）===")
    out = await agent.run(run_id, "北京天气怎么样？", simulate_crash=False)
    print("最终答案:", out)
    print("真实副作用次数:", tools.side_effects, "（应为 1，证明重放未重复发副作用）")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
