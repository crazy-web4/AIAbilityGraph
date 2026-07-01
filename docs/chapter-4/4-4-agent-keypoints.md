# 4.4 Agent 应用开发 - 关键知识点详解

> 本节为 4.4 节的补充知识点，包含 Agent 架构详解、Function Calling 实战、多 Agent 协作。

---

## 知识点 1: Agent 架构对比

### 主流 Agent 框架对比

| 框架 | 核心特性 | 适用场景 | 学习曲线 | 生态 |
|------|---------|---------|---------|------|
| **LangChain** | 完整工具链、丰富集成 | 快速原型、生产部署 | 中等 | ⭐⭐⭐⭐⭐ |
| **LlamaIndex** | 文档检索优化 | RAG 应用 | 简单 | ⭐⭐⭐⭐ |
| **AutoGen** | 多 Agent 协作 | 复杂任务分解 | 中等 | ⭐⭐⭐⭐ |
| **CrewAI** | 角色分工、流程编排 | 工作流自动化 | 简单 | ⭐⭐⭐ |
| **Semantic Kernel** | 微软生态、.NET 支持 | 企业应用 | 中等 | ⭐⭐⭐ |

### Agent 核心组件

```
Agent 架构组成

┌─────────────────────────────────────────────────────────┐
│                    Agent Controller                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Planning  │  │   Memory    │  │   Tools     │     │
│  │             │  │             │  │             │     │
│  │ • 任务分解  │  │ • 短期记忆  │  │ • API 调用   │     │
│  │ • 子目标设定│  │ • 长期记忆  │  │ • 代码执行  │     │
│  │ • 反思调整  │  │ • 向量检索  │  │ • 文件操作  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
    ReAct 循环       向量数据库      Function Calling
    ToT 搜索         缓存机制       工具注册表
```

---

## 知识点 2: Function Calling 详解

### Function Calling 原理

```
Function Calling 工作流程

┌──────────────┐
│   用户请求   │ "北京明天的空气质量如何？"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM 判断    │ 分析是否需要调用工具
└──────┬───────┘
       │
   ┌───┴────┐
   │ 需要？ │
   └───┬────┘
       │ Yes
       ▼
┌──────────────┐
│ 生成工具调用 │
│ get_weather( │
│   city="北京",│
│   date="明天" │
│ )            │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 执行函数     │ → 调用 API → 返回结果
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LLM 整合结果 │ "北京明天空气质量：良，AQI=75"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   最终回复   │
└──────────────┘
```

### Function Calling 实战代码

```python
# examples/4-4-agent/function_calling_demo.py
"""
Function Calling 完整实现

包含:
1. 工具定义与注册
2. 函数调用解析
3. 多轮对话支持
4. 错误处理与重试
"""

import json
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ToolParameterType(Enum):
    """参数类型定义"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """
    工具定义
    
    用于注册可被 LLM 调用的函数
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    
    def to_schema(self) -> Dict:
        """转换为 Function Calling Schema"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class FunctionCallingAgent:
    """
    Function Calling Agent
    
    支持:
    1. 多工具注册
    2. 自动参数解析
    3. 错误处理与重试
    4. 工具调用历史
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.tool_history: List[Dict] = []
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        print(f"✓ 已注册工具：{tool.name}")
    
    def get_tools_schema(self) -> List[Dict]:
        """获取所有工具的 Schema (用于 LLM API)"""
        return [tool.to_schema() for tool in self.tools.values()]
    
    def parse_function_call(self, response: str) -> Optional[Dict]:
        """
        从 LLM 响应中解析函数调用
        
        期望格式:
        ```json
        {
            "name": "function_name",
            "arguments": {"arg1": "value1", ...}
        }
        ```
        """
        try:
            # 尝试提取 JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            call_data = json.loads(json_str.strip())
            
            # 验证必要字段
            if "name" not in call_data or "arguments" not in call_data:
                return None
            
            return call_data
        except (json.JSONDecodeError, IndexError, KeyError):
            return None
    
    def execute_tool(self, name: str, arguments: Dict) -> Any:
        """
        执行工具函数
        
        Returns:
            工具执行结果
        """
        if name not in self.tools:
            raise ValueError(f"未知工具：{name}")
        
        tool = self.tools[name]
        
        # 参数验证
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                raise ValueError(f"缺少必需参数：{param.name}")
        
        # 执行函数
        try:
            result = tool.function(**arguments)
            
            # 记录历史
            self.tool_history.append({
                "tool": name,
                "arguments": arguments,
                "result": result,
                "success": True
            })
            
            return result
        except Exception as e:
            self.tool_history.append({
                "tool": name,
                "arguments": arguments,
                "error": str(e),
                "success": False
            })
            raise
    
    def process_request(self, user_input: str, llm_response: str) -> str:
        """
        处理用户请求
        
        流程:
        1. 解析 LLM 响应
        2. 执行工具调用
        3. 返回结果
        
        Args:
            user_input: 用户原始输入
            llm_response: LLM 的响应 (可能包含函数调用)
        
        Returns:
            最终回复
        """
        # 尝试解析函数调用
        call_data = self.parse_function_call(llm_response)
        
        if call_data is None:
            # 没有函数调用，直接返回 LLM 响应
            return llm_response
        
        # 执行工具
        tool_name = call_data["name"]
        arguments = call_data["arguments"]
        
        try:
            result = self.execute_tool(tool_name, arguments)
            
            # 返回格式化结果
            return f"工具 {tool_name} 执行结果：{result}"
        except Exception as e:
            return f"工具调用失败：{e}"


# ==================== 工具函数示例 ====================

def get_weather(city: str, date: str = "今天") -> str:
    """查询天气"""
    # 模拟天气数据
    weather_data = {
        "北京": {"今天": "晴，25°C", "明天": "多云，27°C"},
        "上海": {"今天": "小雨，22°C", "明天": "阴，24°C"},
        "广州": {"今天": "雷阵雨，30°C", "明天": "小雨，29°C"},
    }
    
    city_weather = weather_data.get(city, "未知城市")
    if isinstance(city_weather, dict):
        return city_weather.get(date, "未知日期")
    return city_weather


def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """搜索网络"""
    # 模拟搜索结果
    results = [
        {"title": f"搜索结果 {i}", "url": f"https://example.com/result{i}", "snippet": f"这是搜索结果的摘要..."}
        for i in range(1, num_results + 1)
    ]
    return results


def calculate(expression: str) -> float:
    """计算数学表达式"""
    try:
        # 安全检查：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            return eval(expression)
        else:
            raise ValueError("表达式包含非法字符")
    except Exception as e:
        return f"计算错误：{e}"


def get_news(category: str = "科技", limit: int = 5) -> List[Dict]:
    """获取新闻"""
    # 模拟新闻数据
    news = {
        "科技": ["AI 新突破", "量子计算进展", "电动车销量飙升"],
        "财经": ["股市上涨", "美联储决策", "加密货币波动"],
        "体育": ["世界杯预选赛", "NBA 季后赛", "网球大满贯"],
    }
    return news.get(category, ["无相关新闻"])[:limit]


# ==================== 使用示例 ====================

def create_demo_agent() -> FunctionCallingAgent:
    """创建演示 Agent"""
    
    # 定义工具
    weather_tool = Tool(
        name="get_weather",
        description="查询指定城市指定日期的天气",
        parameters=[
            ToolParameter("city", ToolParameterType.STRING, "城市名称，如'北京'、'上海'"),
            ToolParameter("date", ToolParameterType.STRING, "日期，如'今天'、'明天'", required=False),
        ],
        function=get_weather
    )
    
    search_tool = Tool(
        name="search_web",
        description="搜索网络获取信息",
        parameters=[
            ToolParameter("query", ToolParameterType.STRING, "搜索关键词"),
            ToolParameter("num_results", ToolParameterType.NUMBER, "返回结果数量", required=False),
        ],
        function=search_web
    )
    
    calc_tool = Tool(
        name="calculate",
        description="计算数学表达式",
        parameters=[
            ToolParameter("expression", ToolParameterType.STRING, "数学表达式，如'2+3*4'"),
        ],
        function=calculate
    )
    
    news_tool = Tool(
        name="get_news",
        description="获取指定类别的最新新闻",
        parameters=[
            ToolParameter("category", ToolParameterType.STRING, "新闻类别", 
                         enum=["科技", "财经", "体育", "娱乐"]),
            ToolParameter("limit", ToolParameterType.NUMBER, "返回数量", required=False),
        ],
        function=get_news
    )
    
    # 创建 Agent
    agent = FunctionCallingAgent(tools=[weather_tool, search_tool, calc_tool, news_tool])
    
    return agent


def demo_function_calling():
    """演示 Function Calling"""
    
    print("=" * 70)
    print("Function Calling 演示")
    print("=" * 70)
    
    agent = create_demo_agent()
    
    # 显示可用工具
    print("\n可用工具:")
    for tool in agent.tools.values():
        print(f"  - {tool.name}: {tool.description}")
        for param in tool.parameters:
            req = "必填" if param.required else "可选"
            print(f"    - {param.name} ({param.type.value}): {param.description} [{req}]")
    
    # 模拟 LLM 响应 (实际应由 LLM 生成)
    test_cases = [
        # 天气查询
        {
            "input": "北京明天天气怎么样？",
            "llm_response": '{"name": "get_weather", "arguments": {"city": "北京", "date": "明天"}}'
        },
        # 计算
        {
            "input": "计算 123 + 456 * 7",
            "llm_response": '{"name": "calculate", "arguments": {"expression": "123 + 456 * 7"}}'
        },
        # 新闻查询
        {
            "input": "看看最新的科技新闻",
            "llm_response": '{"name": "get_news", "arguments": {"category": "科技", "limit": 3}}'
        },
    ]
    
    print("\n" + "=" * 70)
    print("测试用例")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['input']}")
        print(f"LLM 响应：{case['llm_response'][:50]}...")
        
        result = agent.process_request(case['input'], case['llm_response'])
        print(f"结果：{result}")
    
    # 显示工具调用历史
    print("\n" + "=" * 70)
    print("工具调用历史")
    print("=" * 70)
    for record in agent.tool_history:
        status = "✓" if record['success'] else "✗"
        print(f"{status} {record['tool']}: {record.get('arguments', record.get('error', ''))}")


if __name__ == "__main__":
    demo_function_calling()
```

---

## 知识点 3: ReAct Agent 实现

### ReAct 完整流程

```
ReAct (Reason + Act) 循环

┌─────────────────────────────────────────────────────────┐
│  Thought: 分析当前情况，决定下一步行动                   │
├─────────────────────────────────────────────────────────┤
│  Action: 选择要执行的工具                               │
│  Action Input: 工具参数                                 │
├─────────────────────────────────────────────────────────┤
│  Observation: 工具执行结果                              │
└─────────────────────────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │      是否有最终答案？        │
         └──────────────┬──────────────┘
               Yes │     │ No
                   │     └──→ 继续循环
                   │
                   ▼
         ┌─────────────────┐
         │  Final Answer   │ 输出最终答案
         └─────────────────┘
```

### ReAct Agent 代码实现

```python
# examples/4-4-agent/react_agent.py
"""
ReAct Agent 完整实现

参考：Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

import re
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum


class ReActStep(Enum):
    """ReAct 步骤类型"""
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    FINAL_ANSWER = "Final Answer"


@dataclass
class ReActRecord:
    """单次 ReAct 记录"""
    step_type: ReActStep
    content: str
    action_input: Optional[Dict] = None
    observation: Optional[str] = None


class ReActAgent:
    """
    ReAct Agent 实现
    
    特性:
    1. 思考 - 行动 - 观察循环
    2. 工具调用历史
    3. 最大迭代限制
    4. 结果验证
    """
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        tool_descriptions: Dict[str, str],
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.tools = tools
        self.tool_descriptions = tool_descriptions
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.history: List[ReActRecord] = []
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        
        tools_info = "\n".join(
            f"- {name}: {desc}"
            for name, desc in self.tool_descriptions.items()
        )
        
        return f"""你是一个智能助手，使用 ReAct 模式解决问题。

你有以下工具可用：
{tools_info}

对于每个问题，按以下格式循环：

Thought: 分析当前情况，决定下一步
Action: 选择要执行的动作 (从上面的工具中选择)
Action Input: {{"param1": "value1", ...}}
Observation: 动作的结果

当有足够信息时：
Thought: 我已经有了所有需要的信息
Final Answer: 给出最终答案

重要规则:
1. 每次只能选择一个工具
2. 确保参数格式正确 (JSON)
3. 如果工具不存在，说明无法完成
4. 最多思考 {self.max_iterations} 次
"""
    
    def _parse_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        解析 LLM 响应
        
        Returns:
            (thought, action, action_input) 或 (thought, None, None) 或 (None, None, final_answer)
        """
        # 查找 Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if final_match:
            return None, None, final_match.group(1).strip()
        
        # 查找 Action
        action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
        action_input_match = re.search(r'Action Input:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        
        thought_match = re.search(r'Thought:\s*(.+?)(?=(Action|$))', response, re.IGNORECASE | re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1) if action_match else None
        action_input = action_input_match.group(1).strip() if action_input_match else None
        
        return thought, action, action_input
    
    def _execute_action(self, action_name: str, action_input: str) -> str:
        """执行动作"""
        if action_name not in self.tools:
            return f"错误：未知工具 '{action_name}'"
        
        try:
            # 解析 JSON 参数
            import json
            try:
                params = json.loads(action_input)
            except json.JSONDecodeError:
                # 尝试简单键值对
                params = {}
                for item in action_input.split(","):
                    if ":" in item:
                        key, value = item.split(":", 1)
                        params[key.strip().strip('"')] = value.strip().strip('"')
            
            # 调用工具
            tool = self.tools[action_name]
            result = tool(**params)
            return str(result)
        
        except Exception as e:
            return f"执行错误：{e}"
    
    def run(self, question: str, llm_callback: Callable[[str], str]) -> Dict:
        """
        执行 ReAct 循环
        
        Args:
            question: 用户问题
            llm_callback: LLM 回调函数，输入.history 文本，返回 LLM 响应
        
        Returns:
            包含 answer, trace, iterations 的字典
        """
        self.history = []
        
        # 初始化对话
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{question}"}
        ]
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n【迭代 {iteration + 1}/{self.max_iterations}】")
            
            # 构建历史文本
            history_text = "\n".join(
                f"{record.step_type.value}: {record.content}"
                + (f"\nAction Input: {record.action_input}" if record.action_input else "")
                + (f"\nObservation: {record.observation}" if record.observation else "")
                for record in self.history
            )
            
            # 调用 LLM
            prompt = f"{history_text}\n"
            llm_response = llm_callback(prompt)
            
            # 解析响应
            thought, action, action_input = self._parse_response(llm_response)
            
            if action and action_input:
                # 记录 Thought
                if thought:
                    self.history.append(ReActRecord(ReActStep.THUGHT, thought))
                
                # 记录 Action
                self.history.append(ReActRecord(ReActStep.ACTION, f"{action}: {action_input}"))
                
                # 执行并记录 Observation
                observation = self._execute_action(action, action_input)
                self.history.append(ReActRecord(ReActStep.OBSERVATION, "", observation=observation))
                
                if self.verbose:
                    print(f"Thought: {thought}")
                    print(f"Action: {action}")
                    print(f"Observation: {observation[:100]}...")
            
            elif action and not action_input:
                # 缺少参数，提示重试
                self.history.append(ReActRecord(
                    ReActStep.THUGHT,
                    f"需要参数才能执行 {action}"
                ))
                if self.verbose:
                    print(f"需要参数")
            
            else:
                # Final Answer
                if thought:
                    self.history.append(ReActRecord(ReActStep.THUGHT, thought))
                final_answer = action_input  # 这里 action_input 实际是 final_answer
                self.history.append(ReActRecord(ReActStep.FINAL_ANSWER, final_answer))
                
                if self.verbose:
                    print(f"\n最终答案：{final_answer}")
                
                return {
                    "answer": final_answer,
                    "trace": self.history,
                    "iterations": iteration + 1,
                    "success": True
                }
        
        # 达到最大迭代次数
        return {
            "answer": "达到最大迭代次数，未能得出结论",
            "trace": self.history,
            "iterations": self.max_iterations,
            "success": False
        }


# ==================== 工具函数 ====================

def search_wiki(query: str) -> str:
    """搜索维基百科"""
    # 模拟维基百科搜索
    wiki_db = {
        "人工智能": "人工智能 (AI) 是模拟人类智能的科学。主要分支包括机器学习、计算机视觉、自然语言处理等。",
        "机器学习": "机器学习是 AI 的子领域，使用算法从数据中学习。分为监督学习、无监督学习和强化学习。",
        "深度学习": "深度学习使用多层神经网络学习数据表示。在图像识别、语音识别等领域取得突破。",
        "北京": "北京是中国的首都，人口约 2100 万。著名景点包括故宫、长城、天坛等。",
        "python": "Python 是一种高级编程语言，以代码可读性著称。广泛用于 Web 开发、数据分析、AI 等领域。",
    }
    return wiki_db.get(query, f"未找到关于'{query}'的词条")


def calculate(expr: str) -> float:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expr):
            return eval(expr)
        return "非法表达式"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def web_search(query: str) -> List[str]:
    """网络搜索"""
    # 模拟搜索结果
    return [
        f"搜索结果 1 关于'{query}'",
        f"搜索结果 2 关于'{query}'",
        f"搜索结果 3 关于'{query}'",
    ]


# ==================== 使用示例 ====================

def demo_react_agent():
    """ReAct Agent 演示"""
    
    print("=" * 70)
    print("ReAct Agent 演示")
    print("=" * 70)
    
    # 定义工具
    tools = {
        "search_wiki": search_wiki,
        "calculate": calculate,
        "get_current_time": get_current_time,
        "web_search": web_search,
    }
    
    tool_descriptions = {
        "search_wiki": "搜索维基百科获取知识",
        "calculate": "计算数学表达式",
        "get_current_time": "获取当前时间",
        "web_search": "搜索网络获取信息",
    }
    
    # 创建 Agent
    agent = ReActAgent(
        tools=tools,
        tool_descriptions=tool_descriptions,
        max_iterations=5,
        verbose=True
    )
    
    # 模拟 LLM (实际应调用真实 LLM API)
    def mock_llm(history: str) -> str:
        """模拟 LLM 响应"""
        # 简单的规则匹配
        if "人工智能" in history:
            return """Thought: 我需要查询人工智能的定义
Action: search_wiki
Action Input: {"query": "人工智能"}"""
        elif "Observation" in history and "人工智能" in history:
            return """Thought: 我已经有了足够的信息
Final Answer: 人工智能 (AI) 是模拟人类智能的科学，主要分支包括机器学习、计算机视觉、自然语言处理等。"""
        elif "计算" in history or "+" in history or "*" in history:
            return """Thought: 我需要计算这个数学表达式
Action: calculate
Action Input: {"expr": "123 + 45 * 6"}"""
        else:
            return """Thought: 我不确定如何回答这个问题
Final Answer: 抱歉，我暂时无法回答这个问题。"""
    
    # 测试问题
    questions = [
        "什么是人工智能？",
        "计算 123 + 45 * 6",
        "现在几点了？",
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print("=" * 60)
        
        result = agent.run(question, mock_llm)
        
        print(f"\n结果：{result['answer']}")
        print(f"迭代次数：{result['iterations']}")
        print(f"成功：{result['success']}")


if __name__ == "__main__":
    demo_react_agent()
```

---

## 知识点 4: 多 Agent 协作

### 多 Agent 架构模式

```
多 Agent 协作模式

1. 主从模式 (Leader-Worker)
┌────────────────────────────────┐
│         Leader Agent           │  ← 任务分解、协调
│   • 接收用户请求               │
│   • 分解为子任务               │
│   • 分配给 Worker              │
│   • 汇总结果                   │
└──────────┬─────────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐
│Worker│ │Worker│ │Worker│
│  A   │ │  B   │ │  C   │
└──────┘ └──────┘ └──────┘

2. 链式模式 (Pipeline)
用户 → Agent1 → Agent2 → Agent3 → 结果
      (规划)  (执行)  (验证)

3. 辩论模式 (Debate)
       ┌──────────┐
       │  Moderator│
       └────┬─────┘
        ┌───┴───┐
        ▼       ▼
   ┌────────┐ ┌────────┐
   │ Agent A│ │ Agent B│
   │ (正方) │ │ (反方) │
   └────────┘ └────────┘
```

### CrewAI 快速入门

```python
# examples/4-4-agent/crewai_demo.py
"""
多 Agent 协作示例

使用 CrewAI 框架实现角色分工的工作流
"""

# 注意：需要先安装 crewai
# pip install crewai


def create_crew_example():
    """
    创建 CrewAI 多 Agent 协作示例
    
    场景：市场研究报告生成
    
    角色:
    1. 研究员 - 收集信息
    2. 分析师 - 分析数据
    3. 作家 - 撰写报告
    """
    
    crew_code = '''
from crewai import Agent, Task, Crew, Process

# 定义 Agent
researcher = Agent(
    role="高级研究员",
    goal="深入调研指定主题",
    backstory="你是一位经验丰富的研究员，善于从海量信息中提取关键洞察。",
    verbose=True,
    allow_delegation=False
)

analyst = Agent(
    role="数据分析师",
    goal="分析研究数据，提取关键趋势",
    backstory="你是一位数据分析师，善于从复杂数据中发现模式和洞察。",
    verbose=True
)

writer = Agent(
    role="技术作家",
    goal="撰写清晰专业的报告",
    backstory="你是一位资深技术作家，善于将复杂信息转化为易读的文档。",
    verbose=True
)

# 定义 Task
research_task = Task(
    description="调研{topic}领域的最新发展趋势",
    expected_output="包含 5-7 个关键趋势的列表",
    agent=researcher
)

analysis_task = Task(
    description="分析研究结果，识别最重要的 3 个趋势",
    expected_output="详细的趋势分析报告",
    agent=analyst
)

writing_task = Task(
    description="基于分析结果，撰写一份专业的市场研究报告",
    expected_output="格式完整、内容专业的报告文档",
    agent=writer
)

# 创建 Crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    verbose=2,
    process=Process.sequential  # 顺序执行
)

# 执行
result = crew.kickoff(inputs={"topic": "生成式 AI"})
print(result)
'''
    
    print("CrewAI 多 Agent 协作示例代码:")
    print(crew_code)


if __name__ == "__main__":
    create_crew_example()
```

---

## 练习题

### 练习 1: 实现自定义工具

为你的 Agent 添加以下工具：
1. 文件读写工具
2. 数据库查询工具
3. HTTP API 调用工具

要求：
- 定义完整的 Tool Schema
- 实现错误处理
- 添加使用示例

### 练习 2: 多轮对话 Agent

实现一个支持多轮对话的 Agent：
1. 维护对话历史
2. 支持上下文引用
3. 实现记忆机制

---

## 延伸阅读

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [LangChain 文档](https://python.langchain.com/docs/modules/agents/)
- [CrewAI 文档](https://docs.crewai.com/)

---

[← 返回 4.4 主文档](4-4-agent.md) | [下一节：前端 AI 应用 →](4-5-frontend-ai.md)
