#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 产品定价与 ROI 分析模型

功能:
- 单位经济模型计算
- ROI 分析
- 定价策略优化
- 盈亏平衡分析
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


# ========== 单位经济模型 ==========

@dataclass
class UnitEconomicsInput:
    """单位经济输入参数"""
    # 收入侧
    price_per_request: float       # 单次请求价格
    avg_requests_per_user_monthly: int  # 单用户月请求量

    # 成本侧
    inference_cost_per_request: float  # 单次推理成本
    support_cost_per_user: float       # 单用户客服成本

    # 获客
    customer_acquisition_cost: float   # 获客成本 (CAC)
    avg_customer_lifetime_months: int  # 平均客户生命周期


@dataclass
class UnitEconomicsResult:
    """单位经济结果"""
    revenue_per_user_monthly: float
    cost_per_user_monthly: float
    gross_margin_per_user: float
    gross_margin_percent: float
    ltv: float                          # 客户生命周期价值
    ltv_cac_ratio: float
    payback_months: float
    is_healthy: bool

    def generate_report(self) -> str:
        """生成分析报告"""
        lines = [
            "=" * 60,
            "单位经济模型分析",
            "=" * 60,
            "",
            "【收入侧】",
            f"  单用户月收入：¥{self.revenue_per_user_monthly:.2f}",
            "",
            "【成本侧】",
            f"  单用户月成本：¥{self.cost_per_user_monthly:.2f}",
            f"  毛利率：{self.gross_margin_percent:.1f}%",
            "",
            "【生命周期的价值】",
            f"  LTV: ¥{self.ltv:.2f}",
            f"  LTV/CAC: {self.ltv_cac_ratio:.2f}",
            f"  回本周期：{self.payback_months:.1f} 月",
            "",
            "【健康度评估】",
            f"  {'✓ 商业模式健康' if self.is_healthy else '⚠ 需优化商业模式'}",
            "=" * 60
        ]
        return "\n".join(lines)


def calculate_unit_economics(inputs: UnitEconomicsInput) -> UnitEconomicsResult:
    """
    计算单位经济指标

    核心公式:
    - 月收入 = 价格 × 请求量
    - 月成本 = 推理成本 + 客服成本
    - 毛利 = 收入 - 成本
    - LTV = 月毛利 × 生命周期
    - LTV/CAC = LTV / 获客成本
    """
    # 月收入
    revenue = inputs.price_per_request * inputs.avg_requests_per_user_monthly

    # 月成本
    variable_cost = inputs.inference_cost_per_request * inputs.avg_requests_per_user_monthly
    total_cost = variable_cost + inputs.support_cost_per_user

    # 毛利
    gross_margin = revenue - total_cost
    gross_margin_percent = (gross_margin / revenue * 100) if revenue > 0 else 0

    # LTV
    ltv = gross_margin * inputs.avg_customer_lifetime_months

    # LTV/CAC
    ltv_cac_ratio = ltv / inputs.customer_acquisition_cost if inputs.customer_acquisition_cost > 0 else float('inf')

    # 回本周期
    monthly_contribution = gross_margin
    payback_months = inputs.customer_acquisition_cost / monthly_contribution if monthly_contribution > 0 else float('inf')

    # 健康度判断 (LTV/CAC >= 3 且回本 < 12 月)
    is_healthy = ltv_cac_ratio >= 3 and payback_months <= 12

    return UnitEconomicsResult(
        revenue_per_user_monthly=revenue,
        cost_per_user_monthly=total_cost,
        gross_margin_per_user=gross_margin,
        gross_margin_percent=gross_margin_percent,
        ltv=ltv,
        ltv_cac_ratio=ltv_cac_ratio,
        payback_months=payback_months,
        is_healthy=is_healthy
    )


# ========== ROI 计算模型 ==========

@dataclass
class ROIAnalysis:
    """ROI 分析结果"""
    total_investment: float           # 总投资
    total_return: float               # 总收益
    net_profit: float                 # 净利润
    roi_percent: float                # ROI 百分比
    payback_months: Optional[int]     # 回本周期
    irr: Optional[float]              # 内部收益率 (简化)

    def generate_report(self) -> str:
        """生成 ROI 分析报告"""
        lines = [
            "=" * 60,
            "ROI 投资回报分析",
            "=" * 60,
            "",
            "【投资】",
            f"  总投资：¥{self.total_investment:,.0f}",
            "",
            "【收益】",
            f"  总收益：¥{self.total_return:,.0f}",
            f"  净利润：¥{self.net_profit:,.0f}",
            "",
            "【回报指标】",
            f"  ROI: {self.roi_percent:.1f}%",
            f"  回本周期：{self.payback_months if self.payback_months else 'N/A'} 月",
            f"  IRR: {self.irr:.1f}%" if self.irr else "  IRR: N/A",
            "",
            "【评估】",
            f"  {'✓ 投资可行' if self.roi_percent > 20 else '⚠ 收益偏低'}",
            "=" * 60
        ]
        return "\n".join(lines)


def calculate_roi(
    initial_investment: float,
    monthly_cash_flows: List[float],
    discount_rate: float = 0.1
) -> ROIAnalysis:
    """
    计算投资回报率

    参数:
        initial_investment: 初始投资
        monthly_cash_flows: 月度现金流列表
        discount_rate: 折现率

    返回:
        ROIAnalysis 结果
    """
    total_investment = initial_investment
    total_return = sum(monthly_cash_flows)
    net_profit = total_return - total_investment

    # ROI
    roi_percent = (net_profit / total_investment * 100) if total_investment > 0 else 0

    # 回本周期
    cumulative = 0
    payback_months = None
    for i, cash_flow in enumerate(monthly_cash_flows):
        cumulative += cash_flow
        if cumulative >= initial_investment:
            payback_months = i + 1
            break

    # 简化 IRR (NPV=0 时的折现率)
    irr = _calculate_irr(initial_investment, monthly_cash_flows)

    return ROIAnalysis(
        total_investment=total_investment,
        total_return=total_return,
        net_profit=net_profit,
        roi_percent=roi_percent,
        payback_months=payback_months,
        irr=irr
    )


def _calculate_irr(initial: float, cash_flows: List[float], max_iterations: int = 100) -> Optional[float]:
    """
    计算内部收益率 (IRR)
    使用牛顿法近似求解
    """
    def npv(rate: float) -> float:
        """计算净现值"""
        npv_value = -initial
        for i, cf in enumerate(cash_flows):
            npv_value += cf / ((1 + rate) ** (i + 1))
        return npv_value

    def npv_derivative(rate: float) -> float:
        """NPV 的导数"""
        deriv = 0.0
        for i, cf in enumerate(cash_flows):
            deriv -= (i + 1) * cf / ((1 + rate) ** (i + 2))
        return deriv

    # 牛顿法迭代
    rate = 0.1  # 初始猜测
    for _ in range(max_iterations):
        npv_value = npv(rate)
        if abs(npv_value) < 0.01:
            return rate
        deriv = npv_derivative(rate)
        if deriv == 0:
            break
        rate = rate - npv_value / deriv

    return rate


# ========== 定价策略分析 ==========

@dataclass
class PricingTier:
    """定价层级"""
    name: str
    price_monthly: float
    included_requests: int
    overage_price_per_1k: float
    features: List[str]


@dataclass
class PricingAnalysis:
    """定价分析结果"""
    tiers: List[PricingTier]
    projected_users_per_tier: Dict[str, int]
    total_mrr: float                    # 月度经常收入
    avg_revenue_per_user: float
    tier_distribution: Dict[str, float]


def analyze_pricing_strategy(
    tiers: List[PricingTier],
    projected_users: Dict[str, int],
    cost_per_request: float
) -> PricingAnalysis:
    """
    分析定价策略

    参数:
        tiers: 定价层级定义
        projected_users: 各层级预估用户数
        cost_per_request: 单次请求成本
    """
    total_mrr = 0
    total_users = sum(projected_users.values())
    tier_dist = {}

    for tier in tiers:
        users = projected_users.get(tier.name, 0)
        tier_revenue = tier.price_monthly * users
        total_mrr += tier_revenue
        tier_dist[tier.name] = users / total_users if total_users > 0 else 0

    avg_revenue = total_mrr / total_users if total_users > 0 else 0

    return PricingAnalysis(
        tiers=tiers,
        projected_users_per_tier=projected_users,
        total_mrr=total_mrr,
        avg_revenue_per_user=avg_revenue,
        tier_distribution=tier_dist
    )


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("AI 产品定价与 ROI 分析")
    print("=" * 60)
    print()

    # ===== 1. 单位经济模型 =====
    print("【1. 单位经济模型】\n")

    inputs = UnitEconomicsInput(
        price_per_request=0.01,              # ¥0.01/请求
        avg_requests_per_user_monthly=1000,   # 1000 请求/月
        inference_cost_per_request=0.003,     # ¥0.003/请求
        support_cost_per_user=5,              # ¥5/用户/月
        customer_acquisition_cost=100,        # ¥100 获客成本
        avg_customer_lifetime_months=12       # 12 个月生命周期
    )

    result = calculate_unit_economics(inputs)
    print(result.generate_report())
    print()

    # ===== 2. ROI 分析 =====
    print("【2. ROI 投资回报分析】\n")

    # 模拟 3 年现金流
    initial_investment = 500000  # 50 万初始投资
    monthly_cash_flows = [
        20000,   # 第 1 月
        40000,   # 第 2 月
        60000,   # 第 3 月
        80000,   # 第 4 月
        100000,  # 第 5 月
        120000,  # 第 6 月
    ] + [150000] * 30  # 第 7-36 月稳定在 15 万/月

    roi_result = calculate_roi(initial_investment, monthly_cash_flows)
    print(roi_result.generate_report())
    print()

    # ===== 3. 定价策略分析 =====
    print("【3. 定价策略分析】\n")

    pricing_tiers = [
        PricingTier(
            name="免费版",
            price_monthly=0,
            included_requests=100,
            overage_price_per_1k=0,
            features=["基础功能", "社区支持"]
        ),
        PricingTier(
            name="专业版",
            price_monthly=99,
            included_requests=10000,
            overage_price_per_1k=5,
            features=["全部功能", "优先支持", "API 访问"]
        ),
        PricingTier(
            name="企业版",
            price_monthly=999,
            included_requests=100000,
            overage_price_per_1k=3,
            features=["定制功能", "专属支持", "SLA 保障"]
        )
    ]

    projected_users = {
        "免费版": 8000,
        "专业版": 1500,
        "企业版": 50
    }

    analysis = analyze_pricing_strategy(pricing_tiers, projected_users, cost_per_request=0.003)

    print("定价层级分布:")
    for tier_name, pct in analysis.tier_distribution.items():
        print(f"  {tier_name}: {pct*100:.1f}%")

    print(f"\n月度经常收入 (MRR): ¥{analysis.total_mrr:,.0f}")
    print(f"平均用户收入 (ARPU): ¥{analysis.avg_revenue_per_user:.2f}")

    print()
    print("=" * 60)
    print("分析完成")
    print("=" * 60)
