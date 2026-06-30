# 4.6 后端 AI 服务开发

> 构建稳定、可扩展的 AI 后端服务，支持高并发、异步处理和任务队列管理。

## 学习目标

学完本节后，你将能够：

- [ ] 使用 FastAPI 构建 AI 服务
- [ ] 实现异步任务处理
- [ ] 集成任务队列（Celery/Redis）
- [ ] 部署和优化 AI 服务

---

## 4.6.1 FastAPI 基础架构

```python
# examples/4-6-backend/app/main.py
"""
AI 服务后端 - FastAPI 实现
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator
import asyncio
import uuid

from .services.llm import LLMService
from .services.embedding import EmbeddingService
from .models import ChatRequest, ChatResponse, Message

# 创建应用
app = FastAPI(
    title="AI Service API",
    description="大模型应用后端服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
llm_service = LLMService()
embedding_service = EmbeddingService()


# ========== 请求/响应模型 ==========

class GenerateRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class GenerateResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict


# ========== API 端点 ==========

@app.post("/api/chat", response_model=GenerateResponse)
async def chat(request: GenerateRequest):
    """
    对话接口
    
    - **messages**: 对话历史
    - **model**: 模型名称
    - **temperature**: 温度 (0-2)
    - **max_tokens**: 最大输出长度
    """
    try:
        response = await llm_service.chat(
            messages=[m.dict() for m in request.messages],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerateResponse(
            id=str(uuid.uuid4()),
            content=response['content'],
            model=request.model,
            usage=response.get('usage', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: GenerateRequest):
    """
    流式对话接口
    
    使用 Server-Sent Events (SSE)
    """
    
    async def generate() -> AsyncGenerator[str, None]:
        async for token in llm_service.chat_stream(
            messages=[m.dict() for m in request.messages],
            model=request.model,
            temperature=request.temperature
        ):
            # SSE 格式
            yield f"data: {token}\n\n"
        
        yield "data: [DONE]\n\n"
    
    from fastapi.responses import StreamingResponse
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/embeddings")
async def create_embeddings(texts: List[str]):
    """
    向量化接口
    """
    embeddings = await embedding_service.encode(texts)
    
    return {
        "data": [
            {"index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "model": embedding_service.model_name,
        "usage": {"total_tokens": sum(len(t) for t in texts)}
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "services": {
            "llm": llm_service.is_available(),
            "embedding": embedding_service.is_available()
        }
    }


@app.get("/models")
async def list_models():
    """获取可用模型列表"""
    return {
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o", "context_window": 128000},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context_window": 16385},
            {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "context_window": 200000},
        ]
    }
```

---

## 4.6.2 LLM 服务封装

```python
# examples/4-6-backend/app/services/llm.py
"""
LLM 服务封装
"""

import asyncio
from typing import List, Dict, AsyncGenerator
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMService:
    """
    LLM 服务
    
    功能:
    - 对话生成
    - 流式输出
    - 自动重试
    - 速率限制
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 速率限制
        self._rate_limiter = asyncio.Semaphore(10)  # 最多 10 并发
        self._request_queue = asyncio.Queue()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def chat(self,
                   messages: List[Dict],
                   model: str = "gpt-4o",
                   temperature: float = 0.7,
                   max_tokens: int = 2048,
                   **kwargs) -> Dict:
        """
        对话请求
        
        内置重试机制（十次失败自动重试）
        """
        async with self._rate_limiter:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
    
    async def chat_stream(self,
                          messages: List[Dict],
                          model: str = "gpt-4o",
                          temperature: float = 0.7,
                          **kwargs) -> AsyncGenerator[str, None]:
        """
        流式对话
        
        产出:
            每次产出一个 token
        """
        async with self._rate_limiter:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    
    def is_available(self) -> bool:
        """检查服务可用性"""
        try:
            asyncio.get_event_loop().run_until_complete(
                self.chat([{"role": "user", "content": "hi"}], max_tokens=1)
            )
            return True
        except:
            return False


class RateLimitedLLMService(LLMService):
    """
    带速率限制的 LLM 服务
    
    支持：
    - RPM (Requests Per Minute)
    - TPM (Tokens Per Minute)
    """
    
    def __init__(self, 
                 rpm_limit: int = 60,
                 tpm_limit: int = 100000,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # 请求追踪
        self._request_times: List[float] = []
        self._token_counts: List[tuple] = []  # (timestamp, tokens)
    
    async def chat(self, messages: List[Dict], **kwargs):
        """带速率限制的对话"""
        
        # 等待直到不超限
        await self._wait_for_rate_limit()
        
        response = await super().chat(messages, **kwargs)
        
        # 记录请求
        self._record_request(response.get('usage', {}))
        
        return response
    
    async def _wait_for_rate_limit(self):
        """等待直到满足速率限制"""
        import time
        
        now = time.time()
        
        # 清理旧记录（只保留 1 分钟内）
        self._request_times = [t for t in self._request_times if now - t < 60]
        self._token_counts = [(t, c) for t, c in self._token_counts if now - t < 60]
        
        # 检查 RPM
        if len(self._request_times) >= self.rpm_limit:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # 检查 TPM
        recent_tokens = sum(c for _, c in self._token_counts)
        if recent_tokens >= self.tpm_limit:
            await asyncio.sleep(1)  # 简单等待 1 秒再检查
            await self._wait_for_rate_limit()  # 递归检查
    
    def _record_request(self, usage: Dict):
        """记录请求用于速率限制"""
        import time
        now = time.time()
        
        self._request_times.append(now)
        self._token_counts.append((now, usage.get('total_tokens', 0)))
```

---

## 4.6.3 任务队列（Celery）

```python
# examples/4-6-backend/app/tasks.py
"""
异步任务队列 - Celery 实现

用于:
- 长时间运行的任务
- 批量处理
- 定时任务
"""

from celery import Celery, Task
from celery.result import AsyncResult
import redis

# Celery 配置
celery_app = Celery(
    'ai_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # 重试配置
    task_acks_late=True,
    task_reject_on_worker_or_death=True,
    task_max_retries=3,
    task_default_retry_delay=60,
)


# ========== 任务定义 ==========

@celery_app.task(bind=True, max_retries=3)
def generate_long_response(self, prompt: str, user_id: str):
    """
    长文本生成任务
    
    适用于需要长时间运行的任务
    """
    try:
        from .services.llm import LLMService
        
        llm = LLMService()
        
        # 调用 LLM
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096  # 长输出
        )
        
        # 保存结果到数据库/缓存
        from .services.cache import TaskCache
        TaskCache.set(user_id, response['content'])
        
        return {
            'status': 'completed',
            'content': response['content'],
            'tokens': response['usage']['total_tokens']
        }
        
    except Exception as exc:
        # 重试
        raise self.retry(exc=exc)


@celery_app.task
def batch_embed_documents(document_ids: List[str], model: str = "bge-large-zh"):
    """
    批量向量化任务
    
    适用于导入大量文档时的异步处理
    """
    from .services.embedding import EmbeddingService
    from .models import Document
    
    embedder = EmbeddingService(model=model)
    
    processed = []
    failed = []
    
    for doc_id in document_ids:
        try:
            doc = Document.get(doc_id)
            embedding = embedder.encode([doc.content])[0]
            
            # 保存到向量数据库
            VectorStore.add(doc_id, embedding, metadata={
                'title': doc.title,
                'created_at': doc.created_at.isoformat()
            })
            
            processed.append(doc_id)
        except Exception as e:
            failed.append({'id': doc_id, 'error': str(e)})
    
    return {
        'processed': processed,
        'failed': failed,
        'total': len(document_ids),
        'success_rate': len(processed) / len(document_ids) if document_ids else 0
    }


@celery_app.task(bind=True)
def process_with_progress(self, items: List[dict], total_steps: int):
    """
    带进度更新的任务
    
    适用于需要向用户展示进度的场景
    """
    results = []
    
    for i, item in enumerate(items):
        # 更新进度
        progress = (i + 1) / len(items) * 100
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': len(items),
                'progress': progress,
                'current_item': item.get('id')
            }
        )
        
        # 处理单项
        result = process_single_item(item)
        results.append(result)
        
        # 可中断检查
        if self.request.called_directly:
            break
    
    return {
        'status': 'completed',
        'results': results,
        'total_processed': len(results)
    }


# ========== 任务管理 ==========

class TaskManager:
    """任务管理器"""
    
    @staticmethod
    def create_task(task_name: str, **kwargs) -> AsyncResult:
        """创建异步任务"""
        task = celery_app.send_task(task_name, kwargs=kwargs)
        return task
    
    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """获取任务状态"""
        result = celery_app.AsyncResult(task_id)
        
        return {
            'task_id': task_id,
            'state': result.state,
            'info': result.info if hasattr(result, 'info') else None,
            'progress': result.info.get('progress', 0) if result.state == 'PROGRESS' else None
        }
    
    @staticmethod
    def revoke_task(task_id: str, terminate: bool = False):
        """撤销任务"""
        celery_app.control.revoke(task_id, terminate=terminate)


# ========== API 端点 ==========

from fastapi import BackgroundTasks

@app.post("/api/tasks/generate")
async def create_generation_task(
    prompt: str,
    background_tasks: BackgroundTasks
):
    """
    创建异步生成任务
    
    适用于长文本生成，不阻塞请求
    """
    user_id = get_current_user_id()  # 假设的认证函数
    
    # 异步任务
    task = generate_long_response.delay(prompt, user_id)
    
    return {
        'task_id': task.id,
        'status': 'queued',
        'estimated_time': '30-60 秒'
    }


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    status = TaskManager.get_task_status(task_id)
    return status
```

---

## 4.6.4 缓存与优化

```python
# examples/4-6-backend/app/services/cache.py
"""
缓存服务

用于:
- 响应缓存
- 嵌入缓存
- 会话状态
"""

import redis
import json
import hashlib
from typing import Optional, Any
from datetime import timedelta


class RedisCache:
    """Redis 缓存服务"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)
    
    def _make_key(self, prefix: str, *args) -> str:
        """生成缓存键"""
        key_data = ":".join(str(a) for a in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        self.redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    
    def delete(self, key: str):
        """删除缓存"""
        self.redis.delete(key)
    
    # 特定功能缓存
    
    def get_response_cache(self, messages_hash: str) -> Optional[str]:
        """获取对话响应缓存"""
        return self.get(f"response:{messages_hash}")
    
    def set_response_cache(self, messages: list, response: str, ttl: int = 86400):
        """缓存对话响应"""
        messages_hash = hashlib.md5(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()
        self.set(f"response:{messages_hash}", response, ttl)
    
    def get_embedding_cache(self, text_hash: str) -> Optional[list]:
        """获取嵌入缓存"""
        return self.get(f"embedding:{text_hash}")
    
    def set_embedding_cache(self, text: str, embedding: list, ttl: int = 604800):
        """缓存嵌入（7 天）"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.set(f"embedding:{text_hash}", embedding, ttl)


# 装饰器方式缓存
def cache_response(ttl: int = 3600):
    """响应缓存装饰器"""
    from functools import wraps
    
    cache = RedisCache()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = hashlib.md5(
                f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()
            ).hexdigest()
            
            # 尝试获取缓存
            cached = cache.get(cache_key)
            if cached:
                return cached
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 设置缓存
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator
```

---

## 4.6.5 部署配置

### Docker 配置

```dockerfile
# examples/4-6-backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# examples/4-6-backend/docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs

  worker:
    build: .
    command: celery -A app.tasks worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

## 练习题

1. **限流实现**：实现基于令牌桶的 API 限流。

2. **负载均衡**：如何部署多实例 AI 服务？

3. **安全加固**：添加 API 认证和请求签名验证。

---

[← 上一节：4.5 前端 AI 应用](4-5-frontend-ai.md) | [第 5 章：AI 产品与业务 →](../chapter-5/README.md)
