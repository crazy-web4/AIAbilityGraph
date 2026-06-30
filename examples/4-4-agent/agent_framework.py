"""
4.4 Agent 应用开发 - Function Calling 与 ReAct Agent

实现内容:
- Function Calling 框架
- ReAct Agent 实现
- 多 Agent 协作
"""

import json
import re
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum


# ========== 工具定义 ==========

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    func: Callable
    parameters: Dict  # JSON Schema


# ========== Function Calling 引擎 ==========

class FunctionCallingEngine:
    """
    Function Calling 引擎

    支持:
    - 工具注册
    - 参数解析
    - 结果格式化
    """

    def __init__(self, llm_client=None):
        self.tools: Dict[str, Tool] = {}
        self.llm_client = llm_client

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        print(f"已注册工具：{tool.name}")

    def get_tools_schema(self) -> List[Dict]:
        """获取工具 Schema（用于 LLM）"""
        schemas = []
        for tool in self.tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            schemas.append(schema)
        return schemas

    def parse_tool_call(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 返回的工具调用

        返回:
            {name, arguments} 或 None
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                call = json.loads(json_match.group())
                return {
                    "name": call.get("name"),
                    "arguments": call.get("arguments", {})
                }
            except json.JSONDecodeError:
                pass

        # 尝试提取函数调用格式
        func_match = re.search(r'(\w+)\(([^)]*)\)', response)
        if func_match:
            name = func_match.group(1)
            args_str = func_match.group(2)

            # 解析参数
            arguments = {}
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    arguments[key.strip()] = value.strip().strip('"')

            return {"name": name, "arguments": arguments}

        return None

    def execute_tool(self, name: str, arguments: Dict) -> Any:
        """执行工具"""
        if name not in self.tools:
            return f"错误：未知工具 '{name}'"

        tool = self.tools[name]
        try:
            result = tool.func(**arguments)
            return result
        except Exception as e:
            return f"错误：{e}"


# ========== 内置工具 ==========

def create_builtin_tools() -> List[Tool]:
    """创建内置工具"""

    def search_web(query: str) -> str:
        """搜索网络"""
        # 实际实现调用搜索 API
        return f"搜索结果：关于'{query}'的相关信息..."

    def calculate(expression: str) -> str:
        """计算器"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    def get_current_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def read_file(path: str) -> str:
        """读取文件"""
        try:
            with open(path, 'r') as f:
                return f.read()[:1000]  # 限制长度
        except Exception as e:
            return f"读取失败：{e}"

    def write_file(path: str, content: str) -> str:
        """写入文件"""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"已写入文件：{path}"
        except Exception as e:
            return f"写入失败：{e}"

    return [
        Tool(
            name="search_web",
            description="搜索网络获取实时信息",
            func=search_web,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate",
            description="执行数学计算",
            func=calculate,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_current_time",
            description="获取当前时间",
            func=get_current_time,
            parameters={}
        ),
        Tool(
            name="read_file",
            description="读取文件内容",
            func=read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="写入文件",
            func=write_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["path", "content"]
            }
        )
    ]


# ========== ReAct Agent ==========

class ReActAgent:
    """
    ReAct Agent (Reasoning + Acting)

    核心循环:
    Thought → Action → Observation → ... → Final Answer
    """

    REACT_PROMPT = """你是一个智能助手，通过调用工具来解决问题。

可用工具：
{tools}

回答格式：
Thought: 思考当前需要做什么
Action: 工具名称
Action Input: {{"name": "工具名", "arguments": {{...}}}}
Observation: 工具返回结果
...（重复 Thought/Action/Observation）...
Thought: 我已经有了足够的信息
Final Answer: 最终答案

开始！

问题：{question}
"""

    def __init__(
        self,
        tools: List[Tool],
        llm_client=None,
        max_iterations: int = 10
    ):
        self.engine = FunctionCallingEngine(llm_client)
        for tool in tools:
            self.engine.register_tool(tool)

        self.max_iterations = max_iterations
        self.llm_client = llm_client

    async def run(self, question: str) -> Dict:
        """
        执行 ReAct 流程

        返回:
            {
                "answer": 最终答案,
                "trace": 执行轨迹,
                "iterations": 迭代次数
            }
        """
        # 构建工具描述
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.engine.tools.values()
        )

        # 初始化 Prompt
        prompt = self.REACT_PROMPT.format(
            tools=tools_desc,
            question=question
        )

        trace = []

        for iteration in range(self.max_iterations):
            # 调用 LLM
            response = await self._call_llm(prompt)

            # 解析响应
            parsed = self._parse_response(response)
            trace.append(parsed)

            if parsed.get('final_answer'):
                return {
                    "answer": parsed['final_answer'],
                    "trace": trace,
                    "iterations": iteration + 1
                }

            if parsed.get('action'):
                # 执行工具
                result = self.engine.execute_tool(
                    parsed['action'],
                    parsed['action_input']
                )

                # 添加 Observation
                prompt += f"\n\n{response}\nObservation: {result}"
            else:
                # 无法解析
                prompt += f"\n\n{response}\n\n请使用正确的格式。"

        return {
            "answer": "达到最大迭代次数，无法完成。",
            "trace": trace,
            "iterations": self.max_iterations
        }

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if self.llm_client:
            return await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])
        else:
            # Mock 实现（用于测试）
            return self._mock_llm_response(prompt)

    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM 响应（测试用）"""
        # 简单规则匹配
        if "时间" in prompt or "几点" in prompt:
            return 'Action: get_current_time\nAction Input: {}'
        elif "计算" in prompt or "多少" in prompt:
            return 'Action: calculate\nAction Input: {"expression": "123 * 456"}'
        else:
            return 'Final Answer: 这是一个测试响应'

    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        # 查找 Final Answer
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
            return {"final_answer": answer}

        # 查找 Action
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.DOTALL)

        if action_match:
            action = action_match.group(1)
            input_str = input_match.group(1).strip() if input_match else "{}"

            try:
                input_dict = json.loads(input_str)
            except:
                input_dict = {"query": input_str}

            return {
                "action": action,
                "action_input": input_dict
            }

        return {"thought": response}


# ========== 多 Agent 协作 ==========

class MultiAgentSystem:
    """
    多 Agent 协作系统

    角色:
    - Planner: 规划任务
    - Worker: 执行子任务
    - Reviewer: 审查结果
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.roles = {
            "planner": self._create_planner_prompt(),
            "worker": self._create_worker_prompt(),
            "reviewer": self._create_reviewer_prompt()
        }

    def _create_planner_prompt(self) -> str:
        return """你是一个任务规划专家。
将复杂任务分解为可执行的子任务。

输出格式（JSON）：
{
  "subtasks": [
    {"id": 1, "description": "...", "assigned_role": "..."},
    ...
  ]
}"""

    def _create_worker_prompt(self) -> str:
        return """你是一个专业的任务执行者。
根据指示完成具体任务。

任务：{task}
上下文：{context}

请完成这个任务："""

    def _create_reviewer_prompt(self) -> str:
        return """你是一个质量审查员。
检查结果是否满足要求。

任务：{task}
结果：{result}

请评价这个结果（1-5 分）并说明理由："""

    async def run_collaborative_task(self, task: str) -> Dict:
        """
        协作任务执行

        流程:
        1. Planner 分解任务
        2. Workers 并行执行
        3. Reviewer 审查
        4. Coordinator 整合
        """
        # 这里简化实现
        return {
            "plan": {"subtasks": [{"id": 1, "description": task}]},
            "results": [{"result": "完成"}],
            "review": {"score": 4, "comment": "良好"},
            "final": f"任务完成：{task}"
        }


# ========== 使用示例 ==========

async def demo_agent():
    """Agent 演示"""

    # 创建工具
    tools = create_builtin_tools()

    # 创建 Agent
    agent = ReActAgent(tools=tools)

    # 运行
    questions = [
        "现在几点了？",
        "计算 123 * 456",
        "搜索 Python 最新特性"
    ]

    for question in questions:
        print(f"\n问题：{question}")
        result = await agent.run(question)
        print(f"答案：{result['answer']}")
        print(f"迭代次数：{result['iterations']}")


# ========== 多 Agent 演示 ==========

def demo_multi_agent():
    """多 Agent 演示"""
    system = MultiAgentSystem()

    task = "编写一个 Python 脚本，统计 WordCount"

    print(f"任务：{task}")
    # 实际使用需要异步调用
    # result = await system.run_collaborative_task(task)
    print("多 Agent 协作需要 LLM 支持")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_agent())
    demo_multi_agent()
