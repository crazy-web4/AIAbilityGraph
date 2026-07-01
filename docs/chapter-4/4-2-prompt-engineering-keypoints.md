# 4.2 提示词工程 - 关键知识点详解

> 本节为 4.2 节的补充知识点，包含提示词模式库、优化检查清单、实战案例。

---

## 知识点 1: 提示词模式库

### 核心提示模式对比表

| 模式 | 适用场景 | 模板结构 | 效果提升 |
|------|---------|---------|---------|
| **Zero-shot** | 简单任务 | `指令 + 输入` | 基线 |
| **Few-shot** | 复杂任务/少样本学习 | `指令 + 示例 1-N + 输入` | +15-30% |
| **CoT (思维链)** | 推理任务 | `指令 + 让我们一步步思考 + 输入` | +20-40% |
| **ToT (思维树)** | 多路径探索 | `指令 + 生成多个思路 + 评估 + 选择` | +30-50% |
| **Self-Consistency** | 降低幻觉 | `多次采样 + 投票` | +10-20% |
| **ReAct** | 工具调用/Agent | `思考 - 行动 - 观察` 循环 | +25-45% |

---

## 知识点 2: 提示词优化检查清单

### 提示词设计 Checklist

```
□ 角色设定
  □ 是否明确定义了 AI 的角色/身份？
  □ 角色是否与任务匹配？
  □ 是否需要设定语气/风格？

□ 任务描述
  □ 指令是否清晰具体？
  □ 是否包含必要的上下文？
  □ 是否有歧义或模糊表述？

□ 输出格式
  □ 是否指定了输出格式？
  □ 是否需要结构化输出 (JSON/XML)?
  □ 是否有长度限制？

□ 示例 (Few-shot)
  □ 示例数量是否足够 (1-5 个)？
  □ 示例是否覆盖边界情况？
  □ 示例质量是否高？

□ 约束条件
  □ 是否有负面约束 (不要做什么)？
  □ 是否有正面约束 (必须做什么)？
  □ 约束是否可验证？

□ 思维链
  □ 任务是否需要推理？
  □ 是否需要"一步步思考"提示？
  □ 是否需要中间验证步骤？
```

---

## 知识点 3: 高级提示技术详解

### 1. 思维链 (Chain-of-Thought)

**标准 CoT 模板：**

```
问题：{question}

让我们一步步思考：
1. 首先，我们需要理解...
2. 然后，我们计算...
3. 最后，我们得出...

答案：{answer}
```

**代码实现：**

```python
# examples/4-2-prompt/chain_of_thought.py
from typing import List, Dict
from openai import OpenAI

client = OpenAI()


class ChainOfThoughtPrompt:
    """
    思维链提示词生成器
    
    参考：Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = client
    
    def generate(self, question: str, domain: str = "math") -> str:
        """
        生成带思维链的回答
        
        Args:
            question: 用户问题
            domain: 领域类型 (math/logic/code/general)
        
        Returns:
            带推理过程的回答
        """
        
        # 根据领域选择提示模板
        templates = {
            "math": "让我们一步步解决这个问题。首先分析已知条件，然后推导每一步。",
            "logic": "让我们逻辑地分析这个问题。首先明确前提，然后逐步推理。",
            "code": "让我们一步步设计这个程序。先理解需求，再设计架构，最后实现。",
            "general": "让我们一步步思考这个问题。"
        }
        
        cot_instruction = templates.get(domain, templates["general"])
        
        messages = [
            {
                "role": "system",
                "content": "你是一个善于逐步推理的助手。对于每个问题，先展示你的思考过程，然后给出答案。"
            },
            {
                "role": "user",
                "content": f"{question}\n\n{cot_instruction}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def generate_with_verification(self, question: str, num_steps: int = 3) -> Dict:
        """
        生成带验证的思维链
        
        每一步后添加自我检查
        """
        
        messages = [
            {
                "role": "system",
                "content": """你是一个严谨的助手。对于每个问题：
1. 分解问题为多个步骤
2. 逐步解答，每步后进行自我验证
3. 如果发现错误，回溯并修正
4. 最后总结答案"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,  # 较低温度提高一致性
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        # 提取推理过程和最终答案
        return {
            "reasoning": answer,
            "final_answer": self._extract_final_answer(answer),
            "num_steps": num_steps
        }
    
    def _extract_final_answer(self, text: str) -> str:
        """提取最终答案"""
        # 查找"答案"或" Answer:"模式
        import re
        
        patterns = [
            r"(?:答案 | Answer[:：]\s*)(.+?)(?:\n|$)",
            r"(?:综上所述 | Therefore)[:，:\s]*(.+?)(?:\n|$)",
            r"(?:所以 | Thus)[:，:\s]*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return text[-200:]  # 返回最后部分


# 使用示例
if __name__ == "__main__":
    cot = ChainOfThoughtPrompt()
    
    # 数学问题
    math_question = "如果一个小数的小数点向右移动两位后，比原数大 29.7，原数是多少？"
    result = cot.generate(math_question, domain="math")
    print(f"数学问题解答:\n{result}\n")
    
    # 逻辑问题
    logic_question = "A 说 B 在说谎，B 说 C 在说谎，C 说 A 和 B 都在说谎。谁在说真话？"
    result = cot.generate(logic_question, domain="logic")
    print(f"逻辑问题解答:\n{result}")
```

### 2. Few-shot 示例优化

**示例质量评估标准：**

| 标准 | 描述 | 检查方法 |
|------|------|---------|
| **代表性** | 示例是否代表典型输入？ | 覆盖常见用例 |
| **多样性** | 示例是否覆盖不同情况？ | 检查输入分布 |
| **清晰性** | 示例的输入输出是否清晰？ | 人工审查 |
| **一致性** | 示例格式是否一致？ | 格式检查 |

**代码实现：**

```python
# examples/4-2-prompt/few_shot_optimizer.py
from typing import List, Dict, Tuple
import json


class FewShotExampleSelector:
    """
    Few-shot 示例选择器
    
    功能:
    1. 从示例库中选择最相关的示例
    2. 基于相似度或多样性选择
    """
    
    def __init__(self, examples: List[Dict[str, str]], embedding_model: str = "text-embedding-3-small"):
        self.examples = examples
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.example_embeddings = None
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """预计算示例 embeddings"""
        texts = [f"{ex['input']} {ex['output']}" for ex in self.examples]
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        self.example_embeddings = [e.embedding for e in response.data]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def select_examples(
        self,
        query: str,
        k: int = 3,
        strategy: str = "similarity"
    ) -> List[Dict[str, str]]:
        """
        选择示例
        
        Args:
            query: 用户输入
            k: 选择示例数量
            strategy: 选择策略 (similarity/diversity/recency)
        
        Returns:
            选中的示例列表
        """
        
        # 计算查询的 embedding
        query_response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        # 计算相似度
        similarities = [
            (i, self._cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.example_embeddings)
        ]
        
        if strategy == "similarity":
            # 选择最相似的 k 个
            similarities.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [i for i, _ in similarities[:k]]
        
        elif strategy == "diversity":
            # 最大边界相关性 (MMR)
            selected_indices = self._mmr_selection(
                query_embedding, similarities, k
            )
        
        else:  # random
            import random
            selected_indices = random.sample(range(len(self.examples)), min(k, len(self.examples)))
        
        return [self.examples[i] for i in selected_indices]
    
    def _mmr_selection(
        self,
        query_embedding: List[float],
        similarities: List[Tuple[int, float]],
        k: int,
        lambda_param: float = 0.5
    ) -> List[int]:
        """
        最大边际相关性选择 (平衡相似度和多样性)
        
        lambda=1: 只考虑相似度
        lambda=0: 只考虑多样性
        """
        selected = []
        remaining = set(range(len(self.examples)))
        
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # 与查询的相似度
                query_sim = similarities[idx][1]
                
                # 与已选示例的最小相似度 (多样性)
                if selected:
                    min_selected_sim = min(
                        self._cosine_similarity(
                            self.example_embeddings[idx],
                            self.example_embeddings[s]
                        )
                        for s in selected
                    )
                else:
                    min_selected_sim = 0
                
                # MMR 分数
                score = lambda_param * query_sim - (1 - lambda_param) * min_selected_sim
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected
    
    def format_prompt(
        self,
        query: str,
        selected_examples: List[Dict],
        instruction: str = ""
    ) -> str:
        """
        格式化 few-shot 提示
        
        Args:
            query: 用户输入
            selected_examples: 选中的示例
            instruction: 任务指令
        
        Returns:
            完整的 few-shot 提示
        """
        parts = []
        
        if instruction:
            parts.append(f"{instruction}\n")
        
        # 添加示例
        for ex in selected_examples:
            parts.append(f"输入：{ex['input']}")
            parts.append(f"输出：{ex['output']}\n")
        
        # 添加当前查询
        parts.append(f"输入：{query}")
        parts.append("输出：")
        
        return "\n".join(parts)


# 使用示例
if __name__ == "__main__":
    # 示例库
    examples = [
        {"input": "2+2=?", "output": "4"},
        {"input": "10*10=?", "output": "100"},
        {"input": "5-3=?", "output": "2"},
        {"input": "15/3=?", "output": "5"},
        {"input": "2^3=?", "output": "8"},
        {"input": "sqrt(16)=?", "output": "4"},
    ]
    
    selector = FewShotExampleSelector(examples)
    
    # 选择示例
    query = "12 + 8 = ?"
    selected = selector.select_examples(query, k=3, strategy="similarity")
    
    # 生成提示
    prompt = selector.format_prompt(
        query=query,
        selected_examples=selected,
        instruction="请完成以下数学计算："
    )
    
    print(prompt)
```

### 3. ReAct 模式 (Reason + Act)

```python
# examples/4-2-prompt/react_agent.py
"""
ReAct: Reasoning + Acting

参考：Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

from typing import List, Dict, Optional, Callable
import json


class ReActAgent:
    """
    ReAct Agent 实现
    
    循环：思考 -> 行动 -> 观察 -> 思考 -> ... -> 回答
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10
    ):
        self.model = model
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.client = OpenAI()
        
        # 系统提示
        self.system_prompt = """你是一个智能助手，使用 ReAct 模式解决问题。

对于每个问题，按以下格式循环：
Thought: 分析当前情况，决定下一步
Action: 选择要执行的动作 (从 {tools} 中选择)
Observation: 动作的结果

当有足够信息时：
Thought: 我已经有了所有需要的信息
Answer: 给出最终答案

可用工具：{tool_descriptions}
"""
    
    def run(self, question: str) -> Dict:
        """
        执行 ReAct 循环
        
        Returns:
            {
                "trace": [...],      # 思考 - 行动 - 观察 .trace
                "answer": str,       # 最终答案
                "iterations": int    # 迭代次数
            }
        """
        
        trace = []
        
        # 系统提示
        tool_descriptions = "\n".join(
            f"- {name}: {tool.__doc__ or '无描述'}"
            for name, tool in self.tools.items()
        )
        
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    tools=list(self.tools.keys()),
                    tool_descriptions=tool_descriptions
                )
            },
            {
                "role": "user",
                "content": f"问题：{question}"
            }
        ]
        
        for iteration in range(self.max_iterations):
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            
            content = response.choices[0].message.content
            trace.append({"thought": content})
            
            # 解析响应
            action = self._parse_action(content)
            
            if action:
                # 执行动作
                tool_name = action.get("tool")
                tool_input = action.get("input", {})
                
                if tool_name in self.tools:
                    observation = self.tools[tool_name](**tool_input)
                    trace.append({
                        "action": action,
                        "observation": str(observation)
                    })
                    
                    # 添加观察结果到对话
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}"
                    })
                else:
                    trace.append({"error": f"未知工具：{tool_name}"})
            else:
                # 没有行动，提取答案
                answer = self._extract_answer(content)
                return {
                    "trace": trace,
                    "answer": answer,
                    "iterations": iteration + 1
                }
        
        # 达到最大迭代次数
        return {
            "trace": trace,
            "answer": "达到最大迭代次数，未能得出结论",
            "iterations": self.max_iterations
        }
    
    def _parse_action(self, content: str) -> Optional[Dict]:
        """从响应中解析动作"""
        import re
        
        # 查找 "Action: tool_name(input)" 模式
        action_pattern = r"Action:\s*(\w+)\s*\((.*?)\)"
        match = re.search(action_pattern, content, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1)
            input_str = match.group(2)
            
            # 解析输入 (简化版，支持 key=value 格式)
            tool_input = {}
            for pair in input_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    tool_input[key.strip()] = value.strip().strip('"')
            
            return {"tool": tool_name, "input": tool_input}
        
        return None
    
    def _extract_answer(self, content: str) -> str:
        """提取最终答案"""
        import re
        
        answer_pattern = r"Answer:\s*(.+)"
        match = re.search(answer_pattern, content, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return content


# 工具函数示例
def search(query: str) -> str:
    """搜索信息"""
    # 实际使用时调用搜索 API
    return f"搜索结果：关于'{query}'的信息..."


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


# 使用示例
if __name__ == "__main__":
    agent = ReActAgent(
        tools={
            "search": search,
            "calculate": calculate
        }
    )
    
    question = "地球到月球的距离是 384400 公里，如果以 100 公里/小时的速度行驶，需要多少小时？"
    result = agent.run(question)
    
    print(f"迭代次数：{result['iterations']}")
    print(f"答案：{result['answer']}")
    print("\n执行.trace:")
    for step in result['trace']:
        print(step)
```

---

## 练习题

### 练习 1: 设计 CoT 提示

为以下问题设计思维链提示：

```
问题：一个农场有鸡和兔子，总共有 35 个头，94 只脚。问鸡和兔子各有多少只？

要求：
1. 设计引导性的"一步步思考"提示
2. 包含验证步骤
3. 格式清晰，便于检查
```

### 练习 2: Few-shot 示例选择

给定 20 个示例，设计算法选出最相关的 5 个作为 few-shot 示例。

考虑因素：
- 与当前输入的相似度
- 示例之间的多样性
- 示例质量评分

---

## 延伸阅读

- [CoT 论文](https://arxiv.org/abs/2201.11903)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Few-shot Learning 论文](https://arxiv.org/abs/2005.14165)

---

[← 返回 4.2 主文档](4-2-prompt-engineering.md) | [下一节：RAG 应用开发 →](4-3-rag.md)
