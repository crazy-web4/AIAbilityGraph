#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 服务成本优化与监控

功能:
- 单位经济模型计算
- 成本监控与告警
- 批处理优化
- 自动扩缩容策略
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


# ========== 单位经济模型 ==========

@dataclass
class ServiceMetrics:
    """服务指标"""
    requests_per_day: int           # 日请求数
    avg_input_tokens: int           # 平均输入 token
    avg_output_tokens: int          # 平均输出 token
    avg_latency_ms: int             # 平均延迟


@dataclass
class CostModel:
    """成本模型"""
    gpu_hourly_rate: float          # GPU 时租
    gpu_count: int                  # GPU 数量
    token_price_per_million: float  # Token 单价


@dataclass
class UnitEconomicsResult:
    """单位经济结果"""
    daily_tokens: int
    daily_compute_cost: float
    cost_per_token: float
    cost_per_request: float
    monthly_revenue: float
    monthly_cost: float
    monthly_profit: float
    margin: float

    def generate_report(self) -> str:
        """生成报告"""
        lines = [
            "=" * 60,
            "单位经济模型分析",
            "=" * 60,
            "",
            "【运营指标】",
            f"  日 Token 量：{self.daily_tokens:,}",
            f"  单次请求成本：¥{self.cost_per_request:.6f}",
            "",
            "【财务指标】",
            f"  月收入：¥{self.monthly_revenue:,.0f}",
            f"  月成本：¥{self.monthly_cost:,.0f}",
            f"  月利润：¥{self.monthly_profit:,.0f}",
            f"  利润率：{self.margin*100:.1f}%",
            "",
            "【健康度】",
            f"  {'✓ 盈利' if self.monthly_profit > 0 else '✗ 亏损'}",
            "=" * 60
        ]
        return "\n".join(lines)


def calculate_unit_economics(
    metrics: ServiceMetrics,
    costs: CostModel,
    price_per_request: float = 0.001
) -> UnitEconomicsResult:
    """
    计算单位经济指标

    参数:
        metrics: 服务指标
        costs: 成本模型
        price_per_request: 单次请求价格

    返回:
        单位经济分析结果
    """
    # 日 Token 量
    daily_tokens = metrics.requests_per_day * (
        metrics.avg_input_tokens + metrics.avg_output_tokens
    )

    # 日计算成本
    daily_compute_cost = costs.gpu_hourly_rate * costs.gpu_count * 24

    # 每 Token 成本
    cost_per_token = daily_compute_cost / daily_tokens if daily_tokens > 0 else 0

    # 每请求成本
    cost_per_request = cost_per_token * (
        metrics.avg_input_tokens + metrics.avg_output_tokens
    )

    # 月收入估算
    monthly_revenue = metrics.requests_per_day * 30 * price_per_request

    # 月度成本
    monthly_cost = daily_compute_cost * 30 * 1.3  # 30% 其他成本

    # 月利润
    monthly_profit = monthly_revenue - monthly_cost

    # 利润率
    margin = monthly_profit / monthly_revenue if monthly_revenue > 0 else 0

    return UnitEconomicsResult(
        daily_tokens=daily_tokens,
        daily_compute_cost=daily_compute_cost,
        cost_per_token=cost_per_token,
        cost_per_request=cost_per_request,
        monthly_revenue=monthly_revenue,
        monthly_cost=monthly_cost,
        monthly_profit=monthly_profit,
        margin=margin
    )


# ========== 成本监控 ==========

@dataclass
class CostAlert:
    """成本告警"""
    metric: str
    threshold: float
    current_value: float
    severity: str  # warning/critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


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
        self.daily_costs: Dict[str, float] = {}

    def record_cost(self, date: str, cost: float):
        """记录每日成本"""
        self.daily_costs[date] = cost

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
            alert = CostAlert(
                metric="daily_cost",
                threshold=historical_avg,
                current_value=current_cost,
                severity="warning",
                message=f"今日成本 ¥{current_cost:.0f} 超出均值 {threshold_percent:.0f}%"
            )
            alerts.append(alert)
            self.alerts.append(alert)

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
            alert = CostAlert(
                metric="budget_pace",
                threshold=expected_progress,
                current_value=actual_progress,
                severity="critical",
                message=f"预算执行过快：{actual_progress:.1%} (预期 {expected_progress:.1%})"
            )
            alerts.append(alert)
            self.alerts.append(alert)

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
                alert = CostAlert(
                    metric="gpu_utilization",
                    threshold=min_efficiency,
                    current_value=util,
                    severity="warning",
                    message=f"GPU {gpu_id} 利用率仅 {util:.1%}"
                )
                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def generate_report(self) -> Dict:
        """生成成本报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "alerts": [a.message for a in self.alerts],
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


# ========== 批处理优化 ==========

class BatchOptimizer:
    """
    批处理优化器

    目标:
    - 最大化 GPU 利用率
    - 最小化平均等待时间
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 100,
        min_batch_size: int = 4
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.min_batch_size = min_batch_size
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "avg_latency_ms": 0
        }

    def calculate_optimal_batch_size(
        self,
        request_rate: float,  # 请求/秒
        target_latency_ms: float
    ) -> int:
        """
        计算最优批处理大小

        权衡:
        - 大 batch: 高吞吐，高延迟
        - 小 batch: 低延迟，低吞吐
        """
        # 简单模型：基于请求率和目标延迟
        # 最优 batch = min(最大 batch, 请求率 × 目标延迟)
        optimal = int(request_rate * target_latency_ms / 1000)

        return max(self.min_batch_size, min(optimal, self.max_batch_size))

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


# ========== 自动扩缩容 ==========

@dataclass
class ScalingPolicy:
    """扩缩容策略"""
    min_instances: int = 2
    max_instances: int = 20
    target_cpu_utilization: float = 0.7
    target_latency_ms: float = 200
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300


class AutoScaler:
    """
    自动扩缩容器

    基于:
    - CPU 利用率
    - 请求队列长度
    - 延迟指标
    """

    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.current_instances = policy.min_instances
        self.last_scale_time = datetime.now()
        self.scale_history: List[Dict] = []

    def decide_scaling(
        self,
        current_cpu_util: float,
        current_latency_ms: float,
        queue_length: int
    ) -> Optional[int]:
        """
        决定是否需要扩缩容

        返回:
            目标实例数，或 None 表示不变
        """
        now = datetime.now()

        # 检查冷却时间
        time_since_last_scale = (now - self.last_scale_time).total_seconds()
        if time_since_last_scale < self.policy.scale_up_cooldown_seconds:
            return None

        # 扩容条件
        if (current_cpu_util > self.policy.target_cpu_utilization or
            current_latency_ms > self.policy.target_latency_ms or
            queue_length > self.current_instances * 100):

            # 需要扩容
            scale_factor = max(
                current_cpu_util / self.policy.target_cpu_utilization,
                current_latency_ms / self.policy.target_latency_ms,
                queue_length / (self.current_instances * 100)
            )

            target = min(
                int(self.current_instances * scale_factor) + 1,
                self.policy.max_instances
            )

            if target > self.current_instances:
                self._record_scale_event("up", self.current_instances, target)
                self.current_instances = target
                self.last_scale_time = now
                return target

        # 缩容条件
        elif (current_cpu_util < self.policy.target_cpu_utilization * 0.5 and
              current_latency_ms < self.policy.target_latency_ms * 0.5 and
              queue_length < self.current_instances * 20):

            target = max(
                int(self.current_instances * 0.8),
                self.policy.min_instances
            )

            if target < self.current_instances:
                self._record_scale_event("down", self.current_instances, target)
                self.current_instances = target
                self.last_scale_time = now
                return target

        return None

    def _record_scale_event(self, direction: str, from_count: int, to_count: int):
        """记录扩缩容事件"""
        self.scale_history.append({
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "from": from_count,
            "to": to_count
        })

    def get_history(self) -> List[Dict]:
        """获取扩缩容历史"""
        return self.scale_history.copy()


# ========== 使用示例 ==========

def main():
    """演示成本优化功能"""
    print("=" * 60)
    print("AI 服务成本优化与监控")
    print("=" * 60)
    print()

    # ===== 1. 单位经济模型 =====
    print("【1. 单位经济模型】\n")

    metrics = ServiceMetrics(
        requests_per_day=100000,
        avg_input_tokens=500,
        avg_output_tokens=200,
        avg_latency_ms=100
    )

    costs = CostModel(
        gpu_hourly_rate=2.5,
        gpu_count=4,
        token_price_per_million=0.01
    )

    result = calculate_unit_economics(metrics, costs, price_per_request=0.001)
    print(result.generate_report())
    print()

    # ===== 2. 成本监控 =====
    print("【2. 成本监控】\n")

    monitor = CostMonitor(budget_monthly=100000)

    # 记录成本
    for i in range(15):
        date = f"2024-01-{i+1:02d}"
        cost = 3000 + (i % 5) * 500  # 模拟波动
        monitor.record_cost(date, cost)

    # 检查异常
    alerts = monitor.check_cost_anomalies(
        current_cost=5000,
        historical_avg=3500,
        threshold_percent=30
    )

    # 检查预算进度
    spent = sum(monitor.daily_costs.values())
    alerts = monitor.check_budget_progress(spent=spent, days_elapsed=15)

    # 检查资源效率
    alerts = monitor.check_resource_efficiency(
        gpu_utilization={"gpu-0": 0.8, "gpu-1": 0.3, "gpu-2": 0.2},
        min_efficiency=0.5
    )

    # 生成报告
    report = monitor.generate_report()
    print(f"告警数量：{report['total_alerts']}")
    print(f"优化建议:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    print()

    # ===== 3. 批处理优化 =====
    print("【3. 批处理优化】\n")

    optimizer = BatchOptimizer(
        max_batch_size=32,
        max_wait_ms=100,
        min_batch_size=4
    )

    optimal_batch = optimizer.calculate_optimal_batch_size(
        request_rate=100,  # 100 请求/秒
        target_latency_ms=150
    )
    print(f"最优批处理大小：{optimal_batch}")
    print()

    # ===== 4. 自动扩缩容 =====
    print("【4. 自动扩缩容】\n")

    policy = ScalingPolicy(
        min_instances=2,
        max_instances=20,
        target_cpu_utilization=0.7,
        target_latency_ms=200
    )

    scaler = AutoScaler(policy)

    # 模拟负载场景
    scenarios = [
        {"cpu": 0.9, "latency": 300, "queue": 500},   # 高负载
        {"cpu": 0.5, "latency": 100, "queue": 100},   # 正常
        {"cpu": 0.3, "latency": 80, "queue": 50},     # 低负载
    ]

    current_instances = 2
    for i, scenario in enumerate(scenarios):
        target = scaler.decide_scaling(
            current_cpu_util=scenario["cpu"],
            current_latency_ms=scenario["latency"],
            queue_length=scenario["queue"]
        )
        if target:
            print(f"场景 {i+1}: 扩容到 {target} 实例")
        else:
            print(f"场景 {i+1}: 保持 {scaler.current_instances} 实例")

    print(f"\n扩缩容历史: {len(scaler.scale_history)} 次事件")

    print("\n" + "=" * 60)
    print("成本优化分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
