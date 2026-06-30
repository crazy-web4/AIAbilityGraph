# 4.4 Agent 应用开发

> AI Agent 使大模型能够与外部世界交互，通过工具调用实现复杂任务自动化。

## 学习目标

学完本节后，你将能够：

- [ ] 理解 Function Calling 的工作原理
- [ ] 实现工具调用与多轮对话
- [ ] 构建 ReAct 风格 Agent
- [ ] 设计多 Agent 协作系统

---

## 4.4.1 Function Calling 基础

### 工作原理

```
用户请求 → LLM 分析 → 选择函数 → 提取参数 → 执行 → 返回结果 → 生成回复

┌─────────────────────────────────────────────────────────────────┐
│ 1. 定义工具                                                     │
│    tools = [{"name": "search", "description": "...", ...}]     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. LLM 决定调用                                                 │
│    输入：用户问题 + tools 描述                                   │
│    输出：{"name": "search", "arguments": {"query": "..."}}     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 执行函数                                                     │
│    result = tools[function_name](**arguments)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 生成最终回复                                                 │
│    将 result 作为上下文，LLM 生成自然语言回复                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.4.2 OpenAI Function Calling

```python
# examples/4-4-agent/function_calling.py
"""
OpenAI Function Calling 实现
"""

import json
from typing import List, Dict, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: dict
    func: Callable


class FunctionCaller:
    """
    Function Calling 实现
    """
    
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def _build_tools_schema(self) -> List[dict]:
        """构建工具 Schema"""
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
    
    async def chat_with_tools(self,
                               messages: List[dict],
                               available_tools: List[str] = None) -> dict:
        """
        带工具调用的对话
        
        返回:
            {
                "reply": 最终回复,
                "tool_calls": 工具调用历史,
            }
        """
        
        # 筛选可用工具
        if available_tools:
            tools = [self.tools[name] for name in available_tools]
        else:
            tools = list(self.tools.values())
        
        tool_schemas = self._build_tools_schema()
        
        # 第一轮：LLM 决定是否调用工具
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls
        
        # 如果没有工具调用，直接返回
        if not tool_calls:
            return {
                "reply": assistant_message.content,
                "tool_calls": []
            }
        
        # 执行工具调用
        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            # 查找并执行工具
            if func_name in self.tools:
                result = self.tools[func_name].func(**func_args)
                tool_results.append({
                    "call_id": tool_call.id,
                    "name": func_name,
                    "args": func_args,
                    "result": result
                })
        
        # 将结果返回给 LLM，生成最终回复
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
        ])
        
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["call_id"],
                "content": json.dumps(result["result"], ensure_ascii=False)
            })
        
        final_response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        return {
            "reply": final_response.choices[0].message.content,
            "tool_calls": tool_results
        }


# 工具示例
def search_web(query: str) -> str:
    """搜索网络信息"""
    # 实际实现调用搜索 API
    return f"搜索结果：关于'{query}'的相关信息..."


def calculate(expression: str) -> str:
    """计算器"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def get_weather(city: str) -> str:
    """天气查询"""
    # 实际实现调用天气 API
    return f"{city}今天晴转多云，25-32°C"


# 使用示例
async def demo_function_calling():
    """Function Calling 演示"""
    
    caller = FunctionCaller(api_key="xxx")
    
    # 注册工具
    caller.register_tool(Tool(
        name="search_web",
        description="搜索网络获取实时信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        },
        func=search_web
    ))
    
    caller.register_tool(Tool(
        name="calculate",
        description="执行数学计算",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        },
        func=calculate
    ))
    
    caller.register_tool(Tool(
        name="get_weather",
        description="查询城市天气",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        },
        func=get_weather
    ))
    
    # 对话
    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]
    
    result = await caller.chat_with_tools(messages)
    
    print(f"回复：{result['reply']}")
    print(f"工具调用：{result['tool_calls']}")
```

---

## 4.4.3 LangChain Agent

```python
# examples/4-4-agent/langchain_agent.py
"""
使用 LangChain 构建 Agent
"""

from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool,
    AgentExecutor
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


def create_math_agent(api_key: str):
    """
    创建数学计算 Agent
    """
    
    # 工具定义
    tools = [
        Tool(
            name="Calculator",
            description="用于数学计算",
            func=lambda x: str(eval(x)),
        ),
    ]
    
    # LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key
    )
    
    # Agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent


def create_research_agent(api_key: str):
    """
    创建研究助手 Agent
    
    工具:
    - 搜索
    - 网页读取
    - 笔记整理
    """
    
    from langchain.tools import DuckDuckGoSearchRun
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    
    # 搜索工具
    search = DuckDuckGoSearchRun()
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="搜索网络获取信息"
        ),
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="查询维基百科"
        ),
    ]
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        api_key=api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return agent


# 多轮对话示例
async def demo_conversational_agent():
    """对话 Agent 演示"""
    
    agent = create_research_agent(api_key="xxx")
    
    # 多轮对话
    conversations = [
        "谁是美国第一任总统？",
        "他是什么时候出生的？",
        "那个时代美国有哪些重要事件？"
    ]
    
    for question in conversations:
        print(f"用户：{question}")
        response = agent.run(question)
        print(f"助手：{response}\n")
```

---

## 4.4.4 ReAct Agent 实现

```python
# examples/4-4-agent/react_agent.py
"""
ReAct Agent 从零实现
"""

import json
import re
from typing import Dict, List, Optional


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    核心循环:
    Thought → Action → Observation → ... → Final Answer
    """
    
    REACT_PROMPT = """你是一个智能助手，通过调用工具来解决问题。

可用工具：
{tools}

回答格式：
Thought: 思考当前需要做什么
Action: 工具名称
Action Input: 工具参数（JSON 格式）
Observation: 工具返回结果
...（重复以上步骤）...
Thought: 我已经有了足够的信息
Final Answer: 最终答案

开始！

问题：{question}
"""
    
    def __init__(self, tools: Dict, llm_client):
        """
        参数:
            tools: {name: {"description": str, "func": callable, "params": dict}}
            llm_client: LLM 客户端
        """
        self.tools = tools
        self.llm = llm_client
        self.max_iterations = 10
    
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
            f"- {name}: {info['description']}\n  参数：{info.get('params', {})}"
            for name, info in self.tools.items()
        )
        
        # 初始化 Prompt
        prompt = self.REACT_PROMPT.format(
            tools=tools_desc,
            question=question
        )
        
        trace = []
        
        for iteration in range(self.max_iterations):
            # 调用 LLM
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            
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
                result = self._execute_tool(
                    parsed['action'],
                    parsed['action_input']
                )
                
                # 添加 Observation
                prompt += f"\n\n{response}\nObservation: {result}"
            else:
                # 无法解析，添加提示
                prompt += f"\n\n{response}\n\n请使用正确的格式。"
        
        return {
            "answer": "达到最大迭代次数，无法完成。",
            "trace": trace,
            "iterations": self.max_iterations
        }
    
    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        
        # 查找 Final Answer
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
            return {"final_answer": answer}
        
        # 查找 Action
        action_match = re.search(
            r'Action:\s*(\w+)',
            response
        )
        
        action_input_match = re.search(
            r'Action Input:\s*(.+?)(?=\n|$)',
            response,
            re.DOTALL
        )
        
        if action_match:
            action = action_match.group(1)
            action_input = action_input_match.group(1).strip() if action_input_match else "{}"
            
            try:
                action_input = json.loads(action_input)
            except:
                action_input = {"query": action_input}
            
            return {
                "action": action,
                "action_input": action_input
            }
        
        return {"thought": response}
    
    def _execute_tool(self, name: str, params: dict) -> str:
        """执行工具"""
        if name not in self.tools:
            return f"错误：未知工具 '{name}'"
        
        try:
            result = self.tools[name]['func'](**params)
            return str(result)
        except Exception as e:
            return f"错误：{e}"


# 使用示例
async def demo_react():
    """ReAct Agent 演示"""
    
    # 定义工具
    tools = {
        "search": {
            "description": "搜索网络信息",
            "func": lambda query: f"搜索结果：{query}",
            "params": {"query": "string"}
        },
        "calc": {
            "description": "数学计算",
            "func": lambda expr: str(eval(expr)),
            "params": {"expr": "string"}
        }
    }
    
    # 创建 Agent
    from openai import AsyncOpenAI
    llm = AsyncOpenAI(api_key="xxx")
    
    agent = ReActAgent(tools, llm)
    
    # 运行
    result = await agent.run(
        "计算 123 * 456，然后搜索这个结果是什么意思"
    )
    
    print(f"答案：{result['answer']}")
    print(f"迭代次数：{result['iterations']}")
    print(f"执行轨迹：{result['trace']}")
```

---

## 4.4.5 多 Agent 协作

```python
# examples/4-4-agent/multi_agent.py
"""
多 Agent 协作系统
"""

from typing import List
import asyncio


class MultiAgentSystem:
    """
    多 Agent 协作
    
    角色:
    - Planner: 规划任务
    - Worker: 执行子任务
    - Reviewer: 审查结果
    - Coordinator: 协调整体
    """
    
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
    
    def _create_agent_prompt(self, role: str) -> str:
        """创建角色 Prompt"""
        
        prompts = {
            "planner": """你是一个任务规划专家。
将复杂任务分解为可执行的子任务。

输出格式（JSON）：
{
  "subtasks": [
    {"id": 1, "description": "...", "assigned_role": "..."},
    ...
  ]
}""",
            
            "worker": """你是一个专业的任务执行者。
根据指示完成具体任务。

任务：{task}
上下文：{context}

请完成这个任务：""",
            
            "reviewer": """你是一个质量审查员。
检查结果是否满足要求。

任务：{task}
结果：{result}

请评价这个结果（1-5 分）并说明理由：""",
            
            "coordinator": """你是一个项目协调员。
整合各子任务结果，形成最终输出。

子任务结果：
{results}

请整合以上内容，形成最终答案："""
        }
        
        return prompts.get(role, "")
    
    async def run_collaborative_task(self, 
                                      task: str,
                                      n_workers: int = 2) -> dict:
        """
        协作任务执行
        
        流程:
        1. Planner 分解任务
        2. Workers 并行执行
        3. Reviewer 审查
        4. Coordinator 整合
        """
        
        # 1. 任务规划
        plan = await self._plan_task(task)
        
        # 2. 分配 execution
        results = await asyncio.gather(*[
            self._worker_execute(subtask, task)
            for subtask in plan['subtasks']
        ])
        
        # 3. 审查
        reviewed = await self._review(task, results)
        
        # 4. 整合
        final = await self._coordinate(task, results)
        
        return {
            "plan": plan,
            "results": results,
            "review": reviewed,
            "final": final
        }
    
    async def _plan_task(self, task: str) -> dict:
        """任务规划"""
        prompt = self._create_agent_prompt("planner")
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"任务：{task}"}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _worker_execute(self, 
                               subtask: dict, 
                               original_task: str) -> dict:
        """Worker 执行"""
        prompt = self._create_agent_prompt("worker")
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt.format(
                    task=subtask['description'],
                    context=original_task
                )}
            ]
        )
        
        return {
            "subtask_id": subtask['id'],
            "result": response.choices[0].message.content
        }
    
    async def _review(self, task: str, results: List[dict]) -> dict:
        """审查结果"""
        prompt = self._create_agent_prompt("reviewer")
        
        reviews = []
        for r in results:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": prompt.format(
                        task=task,
                        result=r['result']
                    )
                }]
            )
            reviews.append({
                "subtask_id": r['subtask_id'],
                "review": response.choices[0].message.content
            })
        
        return reviews
    
    async def _coordinate(self, task: str, results: List[dict]) -> str:
        """整合结果"""
        prompt = self._create_agent_prompt("coordinator")
        
        results_text = "\n\n".join(
            f"子任务 {r['subtask_id']}:\n{r['result']}"
            for r in results
        )
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt.format(results=results_text)
            }]
        )
        
        return response.choices[0].message.content
```

---

## 练习题

1. **工具设计**：为一个电商客服场景设计 5 个实用工具。

2. **ReAct 优化**：如何减少 ReAct 的迭代次数？

3. **多 Agent 场景**：设计一个代码评审的多 Agent 系统。

---

[← 上一节：4.3 RAG 应用开发](4-3-rag.md) | [下一节：4.5 前端 AI 应用 →](4-5-frontend-ai.md)
