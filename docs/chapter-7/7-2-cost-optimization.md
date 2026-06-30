# 7.2 模型服务容量与成本优化

> 在保障服务质量的前提下，优化 AI 服务的运营成本。

## 学习目标

- [ ] 理解模型服务成本结构
- [ ] 掌握推理优化技术
- [ ] 学会设计自动扩缩容策略
- [ ] 实施成本监控与优化

---

## 7.2.1 成本结构分析

### AI 服务成本构成

```
典型 AI 服务成本结构:

┌─────────────────────────────────────────┐
│  计算成本 (40-60%)                       │
│  ├── GPU 实例费用                         │
│  ├── CPU/内存费用                        │
│  └── 边缘节点费用                        │
├─────────────────────────────────────────┤
│  数据传输 (15-25%)                       │
│  ├── 入站流量 (通常免费)                  │
│  ├── 出站流量 (主要成本)                  │
│  └── 跨区域传输                          │
├─────────────────────────────────────────┤
│  存储成本 (10-15%)                       │
│  ├── 模型存储                            │
│  ├── 向量数据库                          │
│  └── 日志/监控数据存储                   │
├─────────────────────────────────────────┤
│  其他 (5-10%)                            │
│  ├── 监控告警                            │
│  ├── CDN                                 │
│  └── 第三方 API                          │
└─────────────────────────────────────────┘
```

### 单位经济模型

```python
# examples/7-2-cost/unit_economics.py
"""
AI 服务单位经济模型
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class ServiceMetrics:
    """服务指标"""
    requests_per_day: int  # 日请求数
    avg_input_tokens: int  # 平均输入 token
    avg_output_tokens: int  # 平均输出 token
    avg_latency_ms: int  # 平均延迟
    
@dataclass
class CostModel:
    """成本模型"""
    gpu_hourly_rate: float  # GPU 时租
    gpu_count: int  # GPU 数量
    token_price_per_million: float  # Token 单价

def calculate_unit_economics(
    metrics: ServiceMetrics,
    costs: CostModel
) -> Dict:
    """
    计算单位经济指标
    """
    # 日 Token 量
    daily_tokens = metrics.requests_per_day * (
        metrics.avg_input_tokens + metrics.avg_output_tokens
    )
    
    # 日计算成本
    daily_compute_cost = costs.gpu_hourly_rate * costs.gpu_count * 24
    
    # 每 Token 成本
    cost_per_token = daily_compute_cost / daily_tokens
    
    # 每请求成本
    cost_per_request = cost_per_token * (
        metrics.avg_input_tokens + metrics.avg_output_tokens
    )
    
    # 月收入估算
    monthly_revenue = metrics.requests_per_day * 30 * 0.001  # 假设 ¥0.001/请求
    
    # 月度成本
    monthly_cost = daily_compute_cost * 30 * 1.3  # 30% 其他成本
    
    return {
        "daily_tokens": daily_tokens,
        "daily_compute_cost": daily_compute_cost,
        "cost_per_token": cost_per_token,
        "cost_per_request": cost_per_request,
        "monthly_revenue": monthly_revenue,
        "monthly_cost": monthly_cost,
        "monthly_profit": monthly_revenue - monthly_cost,
        "margin": (monthly_revenue - monthly_cost) / monthly_revenue if monthly_revenue > 0 else 0
    }

# 使用示例
if __name__ == "__main__":
    metrics = ServiceMetrics(
        requests_per_day=100000,
        avg_input_tokens=500,
        avg_output_tokens=200,
        avg_latency_ms=100
    )
    
    costs = CostModel(
        gpu_hourly_rate=2.5,  # A10 时租
        gpu_count=4,
        token_price_per_million=0.01
    )
    
    econ = calculate_unit_economics(metrics, costs)
    
    print("单位经济分析:")
    print(f"  日请求量：{metrics.requests_per_day:,}")
    print(f"  日 Token 量：{econ['daily_tokens']:,.0f}")
    print(f"  单次请求成本：¥{econ['cost_per_request']:.6f}")
    print(f"  月收入：¥{econ['monthly_revenue']:,.0f}")
    print(f"  月成本：¥{econ['monthly_cost']:,.0f}")
    print(f"  月利润：¥{econ['monthly_profit']:,.0f}")
    print(f"  利润率：{econ['margin']*100:.1f}%")
```

---

## 7.2.2 推理优化技术

### 1. 批处理优化

```python
# examples/7-2-cost/batching_optimization.py
"""
批处理优化策略
"""

import asyncio
import time
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Request:
    """请求对象"""
    id: str
    prompt: str
    future: asyncio.Future

class DynamicBatcher:
    """
    动态批处理器
    
    策略:
    - 等待更多请求累积 (增加 batch size)
    - 但不超过延迟 SLA
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 100
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: List[Request] = []
        self.lock = asyncio.Lock()
    
    async def enqueue(self, request: Request) -> str:
        """添加请求到队列"""
        future = asyncio.get_event_loop().create_future()
        request.future = future
        
        async with self.lock:
            self.queue.append(request)
            
            # 如果队列满或达到等待时间，立即处理
            if len(self.queue) >= self.max_batch_size:
                asyncio.create_task(self.process_batch())
            elif len(self.queue) == 1:
                # 第一个请求，启动定时器
                asyncio.create_task(self._wait_timer())
        
        return await future
    
    async def _wait_timer(self):
        """等待定时器"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        async with self.lock:
            if self.queue:
                await self.process_batch()
    
    async def process_batch(self):
        """处理一批请求"""
        if not self.queue:
            return
        
        # 取出当前队列
        batch = self.queue[:]
        self.queue = []
        
        # 批量推理 (伪代码)
        # results = await model.infer_batch([r.prompt for r in batch])
        
        # 返回结果
        for i, request in enumerate(batch):
            if not request.future.done():
                request.future.set_result(f"Response-{request.id}")

# 性能对比
async def benchmark_batching():
    """批处理性能对比"""
    
    # 无批处理
    async def no_batching(n_requests):
        start = time.time()
        for i in range(n_requests):
            await asyncio.sleep(0.05)  # 模拟 50ms 推理
        return time.time() - start
    
    # 批处理
    async def with_batching(n_requests, batch_size=8):
        start = time.time()
        n_batches = (n_requests + batch_size - 1) // batch_size
        for _ in range(n_batches):
            await asyncio.sleep(0.05 * batch_size)  # 批处理时间线性增长
        return time.time() - start
    
    n = 100
    t1 = await no_batching(n)
    t2 = await with_batching(n)
    
    print(f"处理 {n} 个请求:")
    print(f"  无批处理：{t1:.2f}s")
    print(f"  批处理：{t2:.2f}s")
    print(f"  加速比：{t1/t2:.1f}x")
```

### 2. 模型量化

| 量化方案 | 模型大小 | 显存占用 | 精度损失 | 适用场景 |
|----------|----------|----------|----------|----------|
| FP16 | 100% | 100% | 无 | 训练/高精度推理 |
| INT8 | 50% | 50% | <1% | 一般推理 |
| INT4 | 25% | 25% | 1-3% | 资源受限 |
| 混合精度 | 60% | 60% | <1% | 推荐 |

### 3. KV Cache 优化

```python
# examples/7-2-cost/kv_cache_optimizer.py
"""
KV Cache 优化策略
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: bytes
    size_mb: float
    last_access: float
    access_count: int = 0

class KVCACHEOptimizer:
    """
    KV Cache 优化器
    
    优化策略:
    1. 共享前缀缓存 (Radix Attention)
    2. LRU 淘汰
    3. 压缩存储
    """
    
    def __init__(self, max_cache_gb: float = 10.0):
        self.max_cache_bytes = max_cache_gb * 1024 * 1024 * 1024
        self.current_cache_bytes = 0
        self.cache: Dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[bytes]:
        """获取缓存"""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access = time.time()
            return entry.value
        return None
    
    def put(self, key: str, value: bytes, size_mb: float):
        """添加缓存"""
        # 检查是否需要淘汰
        while (self.current_cache_bytes + size_mb * 1024 * 1024 > 
               self.max_cache_bytes):
            self._evict_one()
        
        self.cache[key] = CacheEntry(
            key=key,
            value=value,
            size_mb=size_mb,
            last_access=time.time()
        )
        self.current_cache_bytes += size_mb * 1024 * 1024
    
    def _evict_one(self):
        """淘汰一个缓存条目 (LRFU - 最近最少使用+频率)"""
        if not self.cache:
            return
        
        # 计算淘汰分数 (时间越久 + 访问越少 = 分数越高)
        now = time.time()
        for key, entry in self.cache.items():
            entry.score = (now - entry.last_access) / (entry.access_count + 1)
        
        # 淘汰分数最高的
        victim = max(self.cache.keys(), key=lambda k: self.cache[k].score)
        entry = self.cache.pop(victim)
        self.current_cache_bytes -= entry.size_mb * 1024 * 1024
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            "entries": len(self.cache),
            "used_mb": self.current_cache_bytes / 1024 / 1024,
            "max_mb": self.max_cache_bytes / 1024 / 1024,
            "utilization": self.current_cache_bytes / self.max_cache_bytes
        }
```

---

## 7.2.3 自动扩缩容

### HPA 配置示例

```yaml
# examples/7-2-cost/hpa-config.yaml
# Kubernetes HPA 配置

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  
  # 副本数范围
  minReplicas: 2
  maxReplicas: 20
  
  metrics:
  # 基于 CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # 基于自定义指标 (QPS)
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
  
  # 基于延迟
  - type: Pods
    pods:
      metric:
        name: p99_latency_ms
      target:
        type: AverageValue
        averageValue: 200
  
  # 扩缩容策略
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 分钟稳定期
      policies:
      - type: Percent
        value: 50  # 每次最多缩减 50%
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # 立即扩容
      policies:
      - type: Percent
        value: 100  # 每次最多翻倍
        periodSeconds: 30
```

---

## 7.2.4 成本监控

```python
# examples/7-2-cost/cost_monitoring.py
"""
成本监控与告警
"""

import logging
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class CostAlert:
    """成本告警"""
    metric: str
    threshold: float
    current_value: float
    severity: str  # warning/critical
    message: str

class CostMonitor:
    """
    成本监控器
    
    监控指标:
    - 日/周/月累计成本
    - 单位请求成本
    - 资源利用率
    - 预算执行率
    """
    
    def __init__(self, budget_monthly: float):
        self.budget_monthly = budget_monthly
        self.alerts: List[CostAlert] = []
        self.logger = logging.getLogger(__name__)
    
    def check_cost_anomalies(
        self,
        current_cost: float,
        historical_avg: float,
        threshold_percent: float = 50
    ) -> List[CostAlert]:
        """检测成本异常"""
        alerts = []
        
        # 成本突增检测
        if current_cost > historical_avg * (1 + threshold_percent / 100):
            alerts.append(CostAlert(
                metric="daily_cost",
                threshold=historical_avg,
                current_value=current_cost,
                severity="warning",
                message=f"今日成本 ¥{current_cost:.0f} 超出均值 {threshold_percent}%"
            ))
        
        return alerts
    
    def check_budget_progress(
        self,
        spent: float,
        days_elapsed: int,
        total_days: int = 30
    ) -> List[CostAlert]:
        """检查预算执行进度"""
        alerts = []
        
        expected_progress = days_elapsed / total_days
        actual_progress = spent / self.budget_monthly
        
        if actual_progress > expected_progress * 1.5:
            alerts.append(CostAlert(
                metric="budget_pace",
                threshold=expected_progress,
                current_value=actual_progress,
                severity="critical",
                message=f"预算执行过快：{actual_progress:.1%} (预期 {expected_progress:.1%})"
            ))
        
        return alerts
    
    def check_resource_efficiency(
        self,
        gpu_utilization: Dict[str, float],
        min_efficiency: float = 0.5
    ) -> List[CostAlert]:
        """检查资源利用效率"""
        alerts = []
        
        for gpu_id, util in gpu_utilization.items():
            if util < min_efficiency:
                alerts.append(CostAlert(
                    metric="gpu_utilization",
                    threshold=min_efficiency,
                    current_value=util,
                    severity="warning",
                    message=f"GPU {gpu_id} 利用率仅 {util:.1%}"
                ))
        
        return alerts
    
    def generate_report(self) -> Dict:
        """生成成本报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "alerts": [a.message for a in self.alerts],
            "total_alerts": len(self.alerts),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        for alert in self.alerts:
            if alert.metric == "daily_cost":
                recommendations.append(
                    "建议检查是否有异常请求或资源浪费"
                )
            elif alert.metric == "budget_pace":
                recommendations.append(
                    "建议缩减非必要实例或改用 Spot 实例"
                )
            elif alert.metric == "gpu_utilization":
                recommendations.append(
                    "建议整合负载或缩减实例数量"
                )
        
        return recommendations


# 使用示例
if __name__ == "__main__":
    monitor = CostMonitor(budget_monthly=100000)
    
    # 检查预算进度
    alerts = monitor.check_budget_progress(
        spent=60000,
        days_elapsed=15
    )
    
    for alert in alerts:
        print(f"[{alert.severity.upper()}] {alert.message}")
    
    # 资源效率检查
    gpu_util = {"gpu-0": 0.8, "gpu-1": 0.3, "gpu-2": 0.2}
    alerts = monitor.check_resource_efficiency(gpu_util, min_efficiency=0.5)
    
    for alert in alerts:
        print(f"[{alert.severity.upper()}] {alert.message}")
```

---

## 练习题

1. **成本分析**: 某服务日均 10 万请求，每次请求平均 1000 token，使用 4 张 A10，计算单次请求成本。

2. **扩缩容设计**: 为一个有明显峰谷的服务设计 HPA 策略。

3. **优化建议**: 分析某服务成本突增 200% 的可能原因并提出解决方案。

---

[← 上一节：7.1 GPU 资源规划](7-1-gpu-resources.md) | [下一节：7.3 资源调度 →](7-3-gpu-scheduling.md)
