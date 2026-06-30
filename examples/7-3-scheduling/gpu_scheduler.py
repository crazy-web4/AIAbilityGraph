#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 资源调度器

功能:
- 多级反馈队列调度
- 公平调度 (DRF)
- 批处理调度
- 多租户管理
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
from enum import IntEnum
from datetime import datetime


# ========== 基础调度器 ==========

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
    user_id: str = field(compare=False, default="")

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
        self.queues: List[List[ScheduledRequest]] = [[] for _ in range(n_levels)]
        self.current_level = 0
        self.time_slices = [1.0, 0.8, 0.5, 0.3]  # 各级时间片 (秒)
        self.stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "total_expired": 0
        }

    async def enqueue(self, request: ScheduledRequest):
        """添加请求到队列"""
        level = request.priority
        self.queues[level].append(request)
        self.stats["total_enqueued"] += 1

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
                            self.stats["total_expired"] += 1
                    else:
                        valid_requests.append(req)

                self.queues[level] = valid_requests

                if valid_requests:
                    req = self.queues[level].pop(0)
                    self.stats["total_dequeued"] += 1
                    return req

        return None

    def get_stats(self) -> Dict:
        """获取队列统计"""
        return {
            **self.stats,
            "queue_sizes": [len(q) for q in self.queues],
            "total_pending": sum(len(q) for q in self.queues)
        }


# ========== 公平调度器 (DRF) ==========

class FairScheduler:
    """
    公平调度器 (DRF - Dominant Resource Fairness)

    确保不同用户/租户公平共享资源
    """

    def __init__(self):
        self.user_queues: Dict[str, List[ScheduledRequest]] = {}
        self.user_shares: Dict[str, float] = {}  # 用户资源份额
        self.user_weights: Dict[str, float] = {}  # 用户权重

    def add_user(self, user_id: str, weight: float = 1.0):
        """添加用户"""
        self.user_queues[user_id] = []
        self.user_weights[user_id] = weight
        self.user_shares[user_id] = 0

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
                share = len(queue) / (self.user_weights.get(user_id, 1) + 1e-6)
                if share < min_dominant_share:
                    min_dominant_share = share
                    selected_user = user_id

        if selected_user and self.user_queues[selected_user]:
            req = self.user_queues[selected_user].pop(0)
            # 更新份额
            self.user_shares[selected_user] += 1
            return selected_user, req

        return None, None

    def get_fairness_metrics(self) -> Dict:
        """获取公平性指标"""
        if not self.user_shares:
            return {}

        shares = list(self.user_shares.values())
        if not shares:
            return {}

        # Jain's Fairness Index
        sum_shares = sum(shares)
        sum_squares = sum(s * s for s in shares)
        n = len(shares)

        if sum_squares > 0:
            fairness_index = (sum_shares ** 2) / (n * sum_squares)
        else:
            fairness_index = 1.0

        return {
            "fairness_index": fairness_index,  # 1.0 = 完全公平
            "max_share": max(shares) if shares else 0,
            "min_share": min(shares) if shares else 0,
            "user_shares": self.user_shares.copy()
        }


# ========== 批处理调度器 ==========

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
        self.queue: List[tuple] = []  # (item, future)
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "avg_latency_ms": 0
        }

    async def submit(self, item) -> any:
        """提交请求"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.queue.append((item, future))
        return await future

    async def run(self):
        """运行调度器"""
        while True:
            if not self.queue:
                await asyncio.sleep(0.01)
                continue

            # 收集一批请求
            batch = self.queue[:self.config.max_batch_size]
            self.queue = self.queue[self.config.max_batch_size:]

            if not batch:
                continue

            # 处理批次
            items = [b[0] for b in batch]
            futures = [b[1] for b in batch]

            results = await self.process_fn(items)

            # 设置结果
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

            # 更新统计
            self._update_stats(len(batch))

    def _update_stats(self, batch_size: int):
        """更新统计信息"""
        n = self.stats["total_batches"]
        self.stats["total_requests"] += batch_size
        self.stats["total_batches"] += 1
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * n + batch_size) / (n + 1)
        )

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()


# ========== 多租户 GPU 管理器 ==========

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
        - 等待时间调节
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
            "usage_percent": usage.gpu_hours_used / quota.gpu_hours_daily * 100 if quota.gpu_hours_daily > 0 else 0,
            "cost_today": usage.cost_today,
            "budget_remaining": quota.budget_limit - usage.cost_today,
            "requests_today": usage.requests_today
        }


# ========== 使用示例 ==========

async def demo_scheduler():
    """调度器演示"""
    print("=" * 60)
    print("GPU 资源调度器演示")
    print("=" * 60)
    print()

    # ===== 1. 多级队列调度 =====
    print("【1. 多级队列调度】\n")

    scheduler = MultiLevelScheduler()

    # 添加不同优先级的请求
    for i in range(5):
        await scheduler.enqueue(ScheduledRequest(
            priority=Priority(i % 4),
            timestamp=time.time(),
            request_id=f"req-{i}",
            payload={"data": f"payload-{i}"}
        ))

    stats = scheduler.get_stats()
    print(f"队列统计：{stats}")

    # 调度请求
    print("调度请求:")
    for _ in range(3):
        req = await scheduler.dequeue()
        if req:
            print(f"  处理：{req.request_id} (优先级：{req.priority})")

    print()

    # ===== 2. 公平调度 =====
    print("【2. 公平调度 (DRF)】\n")

    fair_scheduler = FairScheduler()

    # 添加用户
    fair_scheduler.add_user("premium", weight=3.0)
    fair_scheduler.add_user("standard", weight=1.0)

    # 添加请求
    for i in range(5):
        fair_scheduler.enqueue("premium", ScheduledRequest(
            priority=Priority.NORMAL,
            timestamp=time.time(),
            request_id=f"prem-{i}",
            payload={}
        ))

    for i in range(3):
        fair_scheduler.enqueue("standard", ScheduledRequest(
            priority=Priority.NORMAL,
            timestamp=time.time(),
            request_id=f"std-{i}",
            payload={}
        ))

    # 选择下一个
    print("调度序列:")
    for _ in range(5):
        user, req = fair_scheduler.select_next()
        if user and req:
            print(f"  用户 {user}: {req.request_id}")

    fairness = fair_scheduler.get_fairness_metrics()
    print(f"\n公平性指数：{fairness.get('fairness_index', 0):.3f}")
    print()

    # ===== 3. 多租户管理 =====
    print("【3. 多租户管理】\n")

    manager = MultiTenantGPUManager()

    # 注册租户
    manager.register_tenant("premium", gpu_hours=100, max_gpus=8, priority=10, budget=5000)
    manager.register_tenant("standard", gpu_hours=20, max_gpus=2, priority=5, budget=1000)
    manager.register_tenant("trial", gpu_hours=4, max_gpus=1, priority=1, budget=200)

    # 租户统计
    print("租户配额状态:")
    for tenant_id in ["premium", "standard", "trial"]:
        can_schedule = manager.can_schedule(tenant_id, n_gpus=2, duration_hours=1)
        stats = manager.get_tenant_stats(tenant_id)
        priority = manager.get_priority_score(tenant_id)

        print(f"\n  {tenant_id}:")
        print(f"    可调度：{can_schedule}")
        print(f"    优先级分数：{priority:.1f}")
        if stats:
            print(f"    配额使用：{stats['usage_percent']:.0f}%")
            print(f"    预算剩余：¥{stats['budget_remaining']:.0f}")


async def demo_batch_scheduler():
    """批处理调度器演示"""
    print("\n\n【4. 批处理调度】\n")

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
        tasks = [scheduler.submit(f"request-{i}") for i in range(n)]
        return await asyncio.gather(*tasks)

    results = await submit_requests(20)
    stats = scheduler.get_stats()

    print(f"批处理统计:")
    print(f"  总请求数：{stats['total_requests']}")
    print(f"  批次数：{stats['total_batches']}")
    print(f"  平均批次大小：{stats['avg_batch_size']:.1f}")


async def main():
    """主程序"""
    await demo_scheduler()
    await demo_batch_scheduler()

    print("\n" + "=" * 60)
    print("调度器演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
