# 7.3 GPU/推理资源调度

> 高效的资源调度是降低 AI 服务成本、提升服务质量的关键。

## 学习目标

- [ ] 理解 GPU 虚拟化与隔离技术
- [ ] 掌握推理服务调度策略
- [ ] 学会设计多租户资源管理方案
- [ ] 实施批处理与队列调度

---

## 7.3.1 GPU 虚拟化技术

### GPU 隔离方案对比

| 方案 | 隔离级别 | 性能损耗 | 适用场景 |
|------|----------|----------|----------|
| **物理隔离** | 完整 GPU | 无 | 训练任务 |
| **MIG (A100/H100)** | 硬件级切分 | 无 | 多租户推理 |
| **Time-Slicing** | 时间片共享 | <5% | 开发测试 |
| **MPS** | 进程级共享 | <10% | 多任务推理 |

### MIG 配置示例

```yaml
# examples/7-3-scheduling/mig-config.yaml
# NVIDIA A100 MIG 配置

mig_profiles:
  # 1g.10gb: 1 个计算实例 + 10GB 显存
  - profile: "1g.10gb"
    count: 7  # 单卡最多 7 个
    use_case: "轻量推理"
  
  # 2g.20gb: 2 个计算实例 + 20GB 显存
  - profile: "2g.20gb"
    count: 3
    use_case: "中等负载训练"
  
  # 3g.40gb: 3 个计算实例 + 40GB 显存
  - profile: "3g.40gb"
    count: 2
    use_case: "大模型推理"
  
  # 7g.80gb: 完整 GPU
  - profile: "7g.80gb"
    count: 1
    use_case: "全量训练"

# Kubernetes 配置
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference
spec:
  containers:
  - name: inference
    image: my-llm-service:latest
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1  # 请求 1 个 MIG 实例
```

---

## 7.3.2 推理服务调度

### 多层调度架构

```
                    用户请求
                       │
                       ▼
              ┌─────────────────┐
              │   Global Load   │  ← L1: 全局负载均衡
              │     Balancer    │     (跨区域/可用区)
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Region        │  ← L2: 区域调度
              │   Scheduler     │     (选择最佳集群)
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Cluster       │  ← L3: 集群内调度
              │   Autoscaler    │     (HPA/VPA)
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   GPU           │  ← L4: GPU 级调度
              │   Partition     │     (MIG/时间片)
              └─────────────────┘
```

### 请求队列设计

```python
# examples/7-3-scheduling/request_scheduler.py
"""
推理请求调度器

支持:
- 优先级队列
- 公平调度
- 超时处理
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum
import heapq

class Priority(IntEnum):
    """请求优先级"""
    CRITICAL = 0  # 最高
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass(order=True)
class ScheduledRequest:
    """调度请求"""
    priority: Priority
    timestamp: float
    request_id: str = field(compare=False)
    payload: dict = field(compare=False)
    deadline_ms: int = field(compare=False, default=5000)
    
    def is_expired(self) -> bool:
        """检查是否超时"""
        elapsed_ms = (time.time() - self.timestamp) * 1000
        return elapsed_ms > self.deadline_ms

class MultiLevelScheduler:
    """
    多级反馈队列调度器
    
    特性:
    - 4 个优先级队列
    - 每级有时间片
    - 超时降级机制
    """
    
    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.queues: List[List[ScheduledRequest]] = [
            [] for _ in range(n_levels)
        ]
        self.current_level = 0
        self.time_slices = [1.0, 0.8, 0.5, 0.3]  # 各级时间片 (秒)
    
    async def enqueue(self, request: ScheduledRequest):
        """添加请求到队列"""
        level = request.priority
        self.queues[level].append(request)
        
        # 按优先级排序
        self.queues[level].sort()
    
    async def dequeue(self) -> Optional[ScheduledRequest]:
        """获取下一个请求"""
        for level in range(self.n_levels):
            if self.queues[level]:
                # 清理超时请求
                valid_requests = []
                for req in self.queues[level]:
                    if req.is_expired():
                        # 降级或丢弃
                        if level < self.n_levels - 1:
                            req.priority = Priority(level + 1)
                            self.queues[level + 1].append(req)
                    else:
                        valid_requests.append(req)
                
                self.queues[level] = valid_requests
                
                if valid_requests:
                    return self.queues[level].pop(0)
        
        return None
    
    async def schedule(self, process_fn):
        """运行调度器"""
        while True:
            request = await self.dequeue()
            
            if request:
                # 处理请求
                await process_fn(request)
            else:
                # 空闲等待
                await asyncio.sleep(0.01)
    
    def get_stats(self) -> Dict:
        """获取队列统计"""
        return {
            f"level_{i}_size": len(q)
            for i, q in enumerate(self.queues)
        }


# 公平调度器
class FairScheduler:
    """
    公平调度器 (DRF - Dominant Resource Fairness)
    
    确保不同用户/租户公平共享资源
    """
    
    def __init__(self):
        self.user_queues: Dict[str, List[ScheduledRequest]] = {}
        self.user_shares: Dict[str, float] = {}  # 用户资源份额
    
    def add_user(self, user_id: str, weight: float = 1.0):
        """添加用户"""
        self.user_queues[user_id] = []
        self.user_shares[user_id] = weight
    
    def enqueue(self, user_id: str, request: ScheduledRequest):
        """添加用户请求"""
        if user_id not in self.user_queues:
            self.add_user(user_id)
        self.user_queues[user_id].append(request)
    
    def select_next(self) -> tuple:
        """选择下一个用户和请求 (DRF 算法)"""
        if not self.user_queues:
            return None, None
        
        # 找到份额最小的用户
        min_dominant_share = float('inf')
        selected_user = None
        
        for user_id, queue in self.user_queues.items():
            if queue:
                # 计算主导资源份额 (简化：使用队列长度代理)
                share = len(queue) / (self.user_shares[user_id] + 1e-6)
                if share < min_dominant_share:
                    min_dominant_share = share
                    selected_user = user_id
        
        if selected_user and self.user_queues[selected_user]:
            return selected_user, self.user_queues[selected_user].pop(0)
        
        return None, None
```

---

## 7.3.3 批处理调度

```python
# examples/7-3-scheduling/batch_scheduler.py
"""
批处理调度器

优化目标:
- 最大化 GPU 利用率
- 最小化平均等待时间
"""

import asyncio
from typing import List, Callable, Awaitable
from dataclasses import dataclass

@dataclass
class BatchConfig:
    """批处理配置"""
    max_batch_size: int = 32
    max_wait_ms: float = 100
    min_batch_size: int = 4

class AdaptiveBatchScheduler:
    """
    自适应批处理调度器
    
    根据负载动态调整批处理策略:
    - 低负载：增大 batch，提高吞吐
    - 高负载：减小 batch，降低延迟
    """
    
    def __init__(
        self,
        process_fn: Callable[[List], Awaitable[List]],
        config: BatchConfig = None
    ):
        self.process_fn = process_fn
        self.config = config or BatchConfig()
        self.queue = asyncio.Queue()
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "avg_latency_ms": 0
        }
    
    async def submit(self, item) -> any:
        """提交请求"""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((item, future))
        return await future
    
    async def run(self):
        """运行调度器"""
        while True:
            # 收集一批请求
            batch = []
            start_time = asyncio.get_event_loop().time()
            
            while len(batch) < self.config.max_batch_size:
                remaining_ms = self.config.max_wait_ms - (
                    asyncio.get_event_loop().time() - start_time
                ) * 1000
                
                if remaining_ms <= 0:
                    break
                
                try:
                    item, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining_ms / 1000 / 10  # 短时间检查
                    )
                    batch.append((item, future))
                except asyncio.TimeoutError:
                    continue
            
            if not batch:
                continue
            
            # 处理批次
            batch_start = asyncio.get_event_loop().time()
            
            items = [b[0] for b in batch]
            futures = [b[1] for b in batch]
            
            results = await self.process_fn(items)
            
            # 设置结果
            for future, result in zip(futures, results):
                future.set_result(result)
            
            # 更新统计
            batch_latency = (asyncio.get_event_loop().time() - batch_start) * 1000
            self._update_stats(len(batch), batch_latency)
    
    def _update_stats(self, batch_size: int, latency_ms: float):
        """更新统计信息"""
        n = self.stats["total_batches"]
        self.stats["total_requests"] += batch_size
        self.stats["total_batches"] += 1
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * n + batch_size) / (n + 1)
        )
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * n + latency_ms) / (n + 1)
        )
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()


# 使用示例
async def demo_batch_scheduler():
    """批处理调度器演示"""
    
    # 模拟 GPU 批处理推理
    async def gpu_infer_batch(items: List[str]) -> List[str]:
        # 批处理推理时间 (非完全线性增长)
        await asyncio.sleep(0.05 * len(items) ** 0.7)
        return [f"Result-{item}" for item in items]
    
    scheduler = AdaptiveBatchScheduler(
        process_fn=gpu_infer_batch,
        config=BatchConfig(
            max_batch_size=32,
            max_wait_ms=100,
            min_batch_size=4
        )
    )
    
    # 启动调度器
    asyncio.create_task(scheduler.run())
    
    # 提交请求
    async def submit_requests(n: int):
        tasks = [
            scheduler.submit(f"request-{i}")
            for i in range(n)
        ]
        return await asyncio.gather(*tasks)
    
    # 运行测试
    results = await submit_requests(100)
    stats = scheduler.get_stats()
    
    print("批处理统计:")
    print(f"  总请求数：{stats['total_requests']}")
    print(f"  批次数：{stats['total_batches']}")
    print(f"  平均批次大小：{stats['avg_batch_size']:.1f}")
    print(f"  平均延迟：{stats['avg_latency_ms']:.1f}ms")
```

---

## 7.3.4 多租户管理

```python
# examples/7-3-scheduling/multi_tenant.py
"""
多租户 GPU 资源管理
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class TenantQuota:
    """租户配额"""
    tenant_id: str
    gpu_hours_daily: float  # 每日 GPU 小时配额
    max_gpu_count: int  # 最大 GPU 数
    priority: int  # 优先级 1-10
    budget_limit: float  # 预算上限

@dataclass
class TenantUsage:
    """租户使用量"""
    tenant_id: str
    gpu_hours_used: float = 0
    requests_today: int = 0
    cost_today: float = 0
    last_request: datetime = field(default_factory=datetime.now)

class MultiTenantGPUManager:
    """
    多租户 GPU 管理器
    
    功能:
    - 配额管理
    - 优先级调度
    - 成本追踪
    """
    
    def __init__(self):
        self.quotas: Dict[str, TenantQuota] = {}
        self.usage: Dict[str, TenantUsage] = {}
    
    def register_tenant(
        self,
        tenant_id: str,
        gpu_hours: float,
        max_gpus: int,
        priority: int = 5,
        budget: float = 1000
    ):
        """注册租户"""
        quota = TenantQuota(
            tenant_id=tenant_id,
            gpu_hours_daily=gpu_hours,
            max_gpu_count=max_gpus,
            priority=priority,
            budget_limit=budget
        )
        self.quotas[tenant_id] = quota
        self.usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
    
    def can_schedule(self, tenant_id: str, n_gpus: int, duration_hours: float) -> bool:
        """检查是否可以调度"""
        if tenant_id not in self.quotas:
            return False
        
        quota = self.quotas[tenant_id]
        usage = self.usage[tenant_id]
        
        # 检查 GPU 数量
        if n_gpus > quota.max_gpu_count:
            return False
        
        # 检查配额
        if usage.gpu_hours_used + duration_hours > quota.gpu_hours_daily:
            return False
        
        # 检查预算
        estimated_cost = duration_hours * n_gpus * 25  # ¥25/GPU 小时
        if usage.cost_today + estimated_cost > quota.budget_limit:
            return False
        
        return True
    
    def record_usage(
        self,
        tenant_id: str,
        n_gpus: int,
        duration_hours: float,
        cost: float
    ):
        """记录使用量"""
        if tenant_id in self.usage:
            usage = self.usage[tenant_id]
            usage.gpu_hours_used += n_gpus * duration_hours
            usage.requests_today += 1
            usage.cost_today += cost
            usage.last_request = datetime.now()
    
    def get_priority_score(self, tenant_id: str) -> float:
        """
        计算租户优先级分数
        
        考虑:
        - 基础优先级
        - 配额使用率 (低使用率优先)
        - 公平性 (长时间未调度优先)
        """
        if tenant_id not in self.quotas:
            return 0
        
        quota = self.quotas[tenant_id]
        usage = self.usage[tenant_id]
        
        # 基础分数 (优先级 × 10)
        base_score = quota.priority * 10
        
        # 配额使用率调节 (使用越少优先)
        usage_ratio = usage.gpu_hours_used / quota.gpu_hours_daily if quota.gpu_hours_daily > 0 else 0
        usage_bonus = (1 - usage_ratio) * 5
        
        # 等待时间调节
        wait_hours = (datetime.now() - usage.last_request).total_seconds() / 3600
        wait_bonus = min(wait_hours, 10)  # 最多 10 分
        
        return base_score + usage_bonus + wait_bonus
    
    def get_tenant_stats(self, tenant_id: str) -> Dict:
        """获取租户统计"""
        if tenant_id not in self.usage:
            return {}
        
        quota = self.quotas[tenant_id]
        usage = self.usage[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "gpu_hours_used": usage.gpu_hours_used,
            "gpu_hours_quota": quota.gpu_hours_daily,
            "usage_percent": usage.gpu_hours_used / quota.gpu_hours_daily * 100,
            "cost_today": usage.cost_today,
            "budget_remaining": quota.budget_limit - usage.cost_today,
            "requests_today": usage.requests_today
        }


# 使用示例
if __name__ == "__main__":
    manager = MultiTenantGPUManager()
    
    # 注册租户
    manager.register_tenant("premium", gpu_hours=100, max_gpus=8, priority=10, budget=5000)
    manager.register_tenant("standard", gpu_hours=20, max_gpus=2, priority=5, budget=1000)
    manager.register_tenant("trial", gpu_hours=4, max_gpus=1, priority=1, budget=200)
    
    # 检查调度
    for tenant_id in ["premium", "standard", "trial"]:
        can_schedule = manager.can_schedule(tenant_id, n_gpus=2, duration_hours=1)
        stats = manager.get_tenant_stats(tenant_id)
        priority = manager.get_priority_score(tenant_id)
        
        print(f"\n{tenant_id}:")
        print(f"  可调度：{can_schedule}")
        print(f"  优先级分数：{priority:.1f}")
        print(f"  配额使用：{stats['usage_percent']:.0f}%")
        print(f"  预算剩余：¥{stats['budget_remaining']:.0f}")
```

---

## 练习题

1. **调度策略**: 设计一个支持抢占式调度的方案，允许高优先级任务抢占低优先级任务的 GPU 资源。

2. **配额优化**: 某租户配额使用率长期低于 20%，如何调整配额策略？

3. **公平性分析**: 比较轮转调度、加权公平队列、DRF 三种调度算法的优缺点。

---

[← 上一节：7.2 成本优化](7-2-cost-optimization.md) | [← 返回第 7 章目录](README.md)
