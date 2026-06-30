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
