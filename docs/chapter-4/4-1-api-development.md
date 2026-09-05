# 4.1 大模型 API 应用开发

> 掌握主流大模型 API 的使用方法，构建稳定高效的 AI 应用。

## 学习目标

学完本节后，你将能够：

- [ ] 调用主流大模型 API（OpenAI、Anthropic、下游模型）
- [ ] 实现流式响应处理
- [ ] 处理错误重试与降级
- [ ] 优化 Token 使用与成本

---

## 4.1.1 主流 API 对比

| 提供商 | 模型 | 输入价格 | 输出价格 | 上下文 |
|--------|------|----------|----------|--------|
| **OpenAI** | GPT-4o | $2.5/1M | $10/1M | 128K |
| **Anthropic** | Claude 3.5 Sonnet | $3/1M | $15/1M | 200K |
| **Google** | Gemini 1.5 Pro | $3.5/1M | $10.5/1M | 2M |
| **Moonshot** | Kimi | ¥2/1M | ¥8/1M | 200K |

---

## 4.1.2 OpenAI API 使用

```python
# examples/4-1-api/openai_client.py
"""
OpenAI API 完整使用示例
"""

from openai import AsyncOpenAI
import asyncio
from typing import List, AsyncIterator


class OpenAIClient:
    """OpenAI API 封装"""
    
    def __init__(self, api_key: str, base_url: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url  # 兼容下游 API
        )
        
        # 默认配置
        self.default_model = "gpt-4o"
        self.default_max_tokens = 2048
        self.default_temperature = 0.7
    
    async def chat(self, 
                   messages: List[dict],
                   model: str = None,
                   temperature: float = None,
                   max_tokens: int = None,
                   **kwargs) -> str:
        """
        单次对话请求
        
        参数:
            messages: 对话历史 [{role, content}, ...]
            model: 模型名称
            temperature: 温度
            max_tokens: 最大输出长度
        """
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    async def chat_stream(self,
                          messages: List[dict],
                          **kwargs) -> AsyncIterator[str]:
        """
        流式对话
        
        产出:
            每次产出一个 token 或 token 片段
        """
        stream = await self.client.chat.completions.create(
            model=self.default_model,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def chat_with_retry(self,
                              messages: List[dict],
                              max_retries: int = 3,
                              backoff: float = 1.0,
                              **kwargs) -> str:
        """
        带重试的对话请求
        
        参数:
            max_retries: 最大重试次数
            backoff: 退避系数 (秒)
        """
        import aiohttp
        
        for attempt in range(max_retries):
            try:
                return await self.chat(messages, **kwargs)
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = backoff * (2 ** attempt)
                print(f"请求失败，{wait_time:.1f}s 后重试：{e}")
                await asyncio.sleep(wait_time)
        
        raise RuntimeError("重试次数耗尽")
    
    def count_tokens(self, text: str, model: str = None) -> int:
        """
        计算 Token 数
        
        使用 tiktoken 库
        """
        import tiktoken
        
        model = model or self.default_model
        encoding = tiktoken.encoding_for_model(model)
        
        return len(encoding.encode(text))
    
    def estimate_cost(self, 
                      input_tokens: int, 
                      output_tokens: int,
                      model: str = None) -> float:
        """
        估算成本（美元）
        """
        # 价格表（每 1K tokens）
        prices = {
            "gpt-4o": {"input": 0.0025, "output": 0.010},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        model = (model or self.default_model).lower()
        price = prices.get(model, prices["gpt-3.5-turbo"])
        
        input_cost = input_tokens / 1000 * price["input"]
        output_cost = output_tokens / 1000 * price["output"]
        
        return input_cost + output_cost


# 使用示例
async def main():
    client = OpenAIClient(api_key="your-api-key")
    
    # 单次请求
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
    
    response = await client.chat(messages)
    print(f"回复：{response}")
    
    # 流式请求
    print("\n流式回复:")
    async for token in client.chat_stream(messages):
        print(token, end="", flush=True)
    
    # Token 计数
    text = "这是一段测试文本。"
    tokens = client.count_tokens(text)
    print(f"\n\nToken 数：{tokens}")
    
    # 成本估算
    cost = client.estimate_cost(1000, 500)
    print(f"估算成本：${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4.1.3 Anthropic Claude API

```python
# examples/4-1-api/anthropic_client.py
"""
Anthropic Claude API 使用示例
"""

from anthropic import AsyncAnthropic
from typing import List, Optional


class AnthropicClient:
    """Anthropic API 封装"""
    
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = "claude-sonnet-4-20250514"
    
    async def message(self,
                      messages: List[dict],
                      system: str = None,
                      max_tokens: int = 4096,
                      temperature: float = 0.7,
                      **kwargs) -> str:
        """
        发送消息请求
        
        注意：Claude 的 messages 格式与 OpenAI 略有不同
        """
        # 转换消息格式
        claude_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "assistant"
            elif role == "user":
                role = "user"
            elif role == "system":
                system = system or msg["content"]
                continue
            else:
                continue
            
            claude_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # 请求
        response = await self.client.messages.create(
            model=self.default_model,
            max_tokens=max_tokens,
            system=system,
            messages=claude_messages,
            temperature=temperature,
            **kwargs
        )
        
        return response.content[0].text
    
    async def message_stream(self, messages: List[dict], **kwargs):
        """流式请求"""
        from anthropic import AsyncStream
        
        stream: AsyncStream = await self.client.messages.create(
            model=self.default_model,
            max_tokens=4096,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for event in stream:
            if event.type == "content_block_delta":
                yield event.delta.text


# 使用示例：长文档分析
async def analyze_long_document(client: AnthropicClient, 
                                 document: str,
                                 questions: List[str]):
    """
    利用 Claude 200K 上下文分析长文档
    """
    
    system_prompt = """你是一个文档分析专家。请仔细阅读文档，然后回答问题。
如果文档中没有相关信息，请如实告知。"""
    
    messages = [
        {"role": "user", "content": f"""
<document>
{document}
</document>

请回答以下问题：
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}
"""},
    ]
    
    response = await client.message(
        messages,
        system=system_prompt,
        max_tokens=4096
    )
    
    return response
```

---

## 4.1.4 批量处理与成本控制

```python

# examples/4-1-api/batch_processing.py
"""
批量处理与成本优化
"""

import asyncio
from dataclasses import dataclass
from typing import List, Callable
import json


@dataclass
class BatchResult:
    """批量处理结果"""
    successful: int
    failed: int
    total_tokens: int
    total_cost: float
    results: List[dict]


class BatchProcessor:
    """
    批量请求处理器
    
    功能:
    - 速率限制
    - 成本追踪
    - 错误处理
    """
    
    def __init__(self, 
                 client,
                 requests_per_minute: int = 60,
                 tokens_per_minute: int = 100000,
                 max_concurrent: int = 10):
        
        self.client = client
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.max_concurrent = max_concurrent
        
        # 追踪
        self.request_count = 0
        self.token_count = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process(self,
                      inputs: List[dict],
                      process_fn: Callable,
                      on_progress: Callable = None) -> BatchResult:
        """
        批量处理
        
        参数:
            inputs: 输入列表
            process_fn: 处理函数
            on_progress: 进度回调
        """
        results = []
        successful = 0
        failed = 0
        
        # 创建任务
        tasks = []
        for i, inp in enumerate(inputs):
            task = self._process_with_retry(inp, process_fn, i)
            tasks.append(task)
        
        # 并发执行
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                results.append(result)
                successful += 1
                self.token_count += result.get('tokens', 0)
            except Exception as e:
                results.append({'error': str(e)})
                failed += 1
            
            # 进度回调
            if on_progress:
                on_progress(i + 1, len(tasks), successful, failed)
        
        return BatchResult(
            successful=successful,
            failed=failed,
            total_tokens=self.token_count,
            total_cost=self._estimate_cost(),
            results=results
        )
    
    async def _process_with_retry(self, 
                                   input_data: dict, 
                                   process_fn: Callable,
                                   index: int,
                                   max_retries: int = 3) -> dict:
        """带重试的单次处理"""
        
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    # 速率限制
                    await self._rate_limit()
                    
                    result = await process_fn(input_data)
                    result['index'] = index
                    return result
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    await asyncio.sleep(2 ** attempt)
    
    async def _rate_limit(self):
        """简单的速率限制"""
        # 实际实现需要记录时间戳
        pass
    
    def _estimate_cost(self) -> float:
        """估算成本"""
        # 简化计算
        return self.token_count / 1000 * 0.002


# 使用示例：批量文本分类
async def batch_classify_texts(texts: List[str]) -> BatchResult:
    """批量文本分类"""
    
    client = OpenAIClient(api_key="xxx")
    processor = BatchProcessor(client, requests_per_minute=30)
    
    async def classify(text: dict) -> dict:
        messages = [
            {"role": "user", 
             "content": f"将以下文本分类为正面/负面/中性：\n\n{text['content']}"}
        ]
        
        response = await client.chat(messages)
        
        return {
            'input': text['content'],
            'classification': response,
            'tokens': client.count_tokens(response)
        }
    
    inputs = [{'content': t} for t in texts]
    
    def on_progress(done, total, success, fail):
        print(f"进度：{done}/{total}, 成功：{success}, 失败：{fail}")
    
    result = await processor.process(inputs, classify, on_progress)
    
    print(f"\n完成：成功={result.successful}, 失败={result.failed}")
    print(f"总 Token: {result.total_tokens}, 估算成本：${result.total_cost:.4f}")
    
    return result
```

---

## 4.1.5 下游模型 API 兼容

```python
# examples/4-1-api/downstream_models.py
"""
下游模型 API 调用（兼容 OpenAI 格式）
"""

from openai import AsyncOpenAI


class DownstreamClient:
    """
    下游模型客户端
    
    支持：Moonshot、智谱、百川等
    """
    
    def __init__(self, provider: str, api_key: str):
        """
        参数:
            provider: 提供商名称
            api_key: API 密钥
        """
        endpoints = {
            "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-128k"),
            "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "glm-4"),
            "baichuan": ("https://api.baichuan-ai.com/v1", "Baichuan4"),
            "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
        }
        
        base_url, default_model = endpoints.get(provider, (None, None))
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.default_model = default_model
        self.provider = provider
    
    async def chat(self, messages: List[dict], **kwargs):
        """通用聊天接口"""
        
        # 特殊处理各厂商差异
        if self.provider == "zhipu":
            # 智谱需要 user 参数
            kwargs.setdefault('user', 'default')
        
        response = await self.client.chat.completions.create(
            model=self.default_model,
            messages=messages,
            **kwargs
        )
        
        return response.choices[0].message.content


# 统一调用接口
async def unified_chat(provider: str, 
                       messages: List[dict],
                       api_key: str) -> str:
    """统一调用接口"""
    
    client = DownstreamClient(provider, api_key)
    return await client.chat(messages)
```

---

## 练习题

1. **成本优化**：如果每天处理 10 万条请求，如何选择合适的模型组合以最小化成本？

2. **错误处理**：设计一个优雅的错误降级策略。

3. **流式处理**：实现一个支持部分结果取消的流式处理器。

---

[← 上一节：章前导引](README.md) | [下一节：4.2 提示词工程 →](4-2-prompt-engineering.md)







作为大模型应用开发者，通过OpenAI API调用模型不仅仅是“发请求收结果”。要构建生产级应用，必须深入理解其背后的**概率采样机制**、**上下文管理策略**和**交互范式**。下面我将从原理到实践，极其详细地拆解你提出的四个核心知识点。

---

### 1. 大模型的工作原理（为何要关心底层？）

在调用API时，模型本身是黑盒，但理解其**自回归（Autoregressive）**本质至关重要：

- **下一个词预测（Next-Token Prediction）**：大模型本质上是一个概率分布函数。输入Prompt，模型会计算词汇表中**所有词**的生成概率，然后根据采样策略选出下一个Token。
- **无状态性（Stateless）**：**HTTP API本身是无状态的**。模型每次推理时，并不“记住”你之前的对话。你之所以感觉它在连续对话，是因为**客户端**每次都将完整的对话历史拼接在 `messages` 数组中重新发送给模型。
- **上下文窗口（Context Window）**：模型能处理的**输入Token数 + 生成Token数**有硬性上限（如128K）。超出窗口，模型会直接截断或报错（Context Overflow）。

---

### 2. 核心 API 参数详解（不只是调参，而是控制概率）

`temperature` 和 `top_p` 不控制“聪明程度”，它们控制**“确定性 vs 随机性”**。

#### (1) `model`（模型选择）
- **策略**：复杂推理（数学、逻辑）选 **`o1` 系列**（擅长思维链但贵且慢）；日常对话/文本生成选 **`gpt-4o`**（平衡速度与智能）；极致成本与速度选 **`gpt-4o-mini`**。
- **关键点**：不同模型的**知识截止日期**和**上下文窗口长度**不同，开发时务必在代码中配置动态 `max_tokens` 上限。

#### (2) `temperature`（温度系数——核心）
- **数学本质**：对模型输出的原始概率分布（Logits）进行**缩放**后，再转为Softmax概率。
  - **公式影响**：`Temperature` 越高（如 1.5），概率分布越**平坦**（高熵），低频词被选中的概率增加，输出越发散、创意。
  - **Temperature = 0**：概率分布被极度锐化（变成One-Hot），模型**永远**取最高概率的词，输出变为确定性（Deterministic）——适合代码生成、数学题、信息抽取。
- **黄金法则**：若需要事实性回答，设为 **0 ~ 0.3**；若需要写小说或头脑风暴，设为 **0.8 ~ 1.2**。

#### (3) `top_p`（核采样——动态截断）
- **数学本质**：从词汇表中选出**累积概率超过 `p` 的最小词集合**（Nucleus），并仅在这个子集中采样。
- **联动机理**：
  - `top_p=0.1`：意味着只从累积概率前10%的“头部词汇”中选，大幅过滤掉生僻词。
  - **`temperature` 与 `top_p` 的关系**：**强烈建议只调其中一个**。两者都会改变输出分布，若同时大幅调整（如 `temp=2, top_p=0.9`），极易产生冲突导致输出崩溃。通常做法：固定 `top_p=1`，仅微调 `temperature`。

#### (4) 其他关键参数
- **`max_completion_tokens`**（注意：GPT-4o系列已弃用 `max_tokens`）：生成回复的最大长度。**陷阱**：这个数值会占用上下文窗口。如果窗口是128K，Prompt占100K，那么 `max_completion_tokens` 最多只能设为28K，否则报错。
- **`frequency_penalty` 与 `presence_penalty`**：作用于Token级别的重复惩罚。
  - `frequency_penalty`：根据某个词**已出现次数**进行降权（次数越多，惩罚越大），扼杀重复循环。
  - `presence_penalty`：只要某个词**出现过一次**，就固定降低其权重（不关心次数），鼓励引入全新话题。

---

### 3. 批量生成（Batch）与流式生成（Streaming）

这是影响用户体验（TTFT，首字延迟）的关键。

#### (1) 批量生成（非流式）
- **机制**：客户端发起请求后，**服务端计算完所有 `max_completion_tokens`**，将完整生成的JSON一次性返回给客户端。
- **优缺点**：实现简单，便于异常重试（失败则重发整个请求）。但**首字延迟极长**（用户盯着空白屏幕），且如果生成中途报错，客户端拿不到任何部分结果。
- **适用场景**：离线数据清洗、批量文件处理、后台任务。

#### (2) 流式生成（`stream=True`）
- **机制**：服务端使用**Server-Sent Events (SSE)** 协议，每生成一个Token（或几个Token），立即以 `data: {"choices":[{"delta": {...}}]}` 格式推送给客户端。
- **关键数据格式**：流式响应中，第一帧通常包含 `role`，后续帧只有 `content` 的增量片段（Delta）。最后一帧是 `data: [DONE]`。
- **工程陷阱**：
  - 流式传输时，**无法通过API直接获取 `usage`（消耗Token数）**。必须在客户端**自行拼接所有Delta文本**，并计算总长度，或者利用官方返回的 `response` 对象事后查询。
  - 必须设置 `read_timeout` 足够长（如60s+），因为流式是长连接，若中间网络抖动，容易丢帧。

---

### 4. 理解 `messages` 与对话历史管理（应用开发的命脉）

既然API无状态，`messages` 数组就是构建记忆的核心载体。

#### (1) 消息结构的三要素（角色体系）
```json
[
  {"role": "system", "content": "你是严谨的Python专家"},
  {"role": "user", "content": "如何捕获异常？"},
  {"role": "assistant", "content": "使用try...except..."},
  {"role": "user", "content": "给个具体例子"}
]
```
- **System（系统提示词）**：设定模型的**人格、规则、输出格式**（如JSON Schema）。**重要**：它占据Prompt前缀位置，影响力极大，越靠前越容易被模型在长上下文中“遗忘”（Lost-in-Middle现象）。
- **User 与 Assistant**：交替构成对话历史。开发者必须**保证消息严格奇偶交替**（User->Assistant->User），否则模型会混乱。

#### (2) 对话历史的致命陷阱：上下文爆炸
随着对话轮次增加，`messages` 数组线性膨胀。**必须自行实现截断策略**，因为API不会帮你筛选记忆：

- **策略A（滑动窗口）**：保留最近 N 轮对话（如最近10轮）。**风险**：一旦超出窗口，模型立即失忆。
- **策略B（核心摘要法）**：维护一个独立的摘要系统。当历史超出阈值，调用一次模型将旧历史压缩为一段话（摘要），替换掉 `messages` 数组开头的一大段User/Assistant轮次。
- **策略C（RAG 检索）**：不保留全量历史，而是将历史向量化存入向量库，每次根据当前提问检索最相关的几条历史片段插入上下文。

#### (3) 指令注入与分隔符
在实际拼接时，**强烈建议**使用XML标签或Markdown分隔符将用户输入与历史区分开，防止用户输入“忽略之前所有指令”来劫持System Prompt（提示词注入攻击）。

---

### 实战总结建议（极简代码骨架思维）

```python
# 伪代码逻辑
def call_llm(messages, is_streaming):
    # 1. 检查当前 messages 总 Token 数（用 tiktoken 库），若超过限制先截断
    # 2. 设置 temperature=0.1 (事实性任务), top_p=1.0
    # 3. 若开启流式:
    #    response = client.chat.completions.create(..., stream=True)
    #    for chunk in response:
    #        if chunk.choices[0].delta.content:
    #            # 实时渲染到前端
    #            yield chunk.choices[0].delta.content
    #    # 注意：此时拿不到 token 用量，需自行累加
    # 4. 若不开流式:
    #    response = client.chat.completions.create(...)
    #    return response.choices[0].message.content, response.usage.total_tokens
```

**最后的忠告**：开发阶段，务必先设 `max_completion_tokens` 为较小值（如500）进行调试，防止因死循环或模型发散产生巨额账单；生产环境，一定要接入**Token计数中间件**，并在超过上下文窗口90%时主动触发截断或告警。这才是工程化落地的核心门槛。
