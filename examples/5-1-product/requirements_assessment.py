#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 需求评估工具

功能:
- AI 适用性评估清单
- 需求可行性打分
- ROI 估算
"""

from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


# ========== AI 适用性评估清单 ==========

AI_READINESS_CHECKLIST = {
    "problem_fit": [
        "问题是否涉及模式识别？",
        "是否有明确的输入和期望输出？",
        "人类能否完成这个任务（作为上限参考）？"
    ],
    "data_readiness": [
        "是否有足够的训练数据？",
        "数据质量如何（标注准确性、覆盖度）？",
        "数据获取是否合规合法？"
    ],
    "business_value": [
        "预期 ROI 是否为正？",
        "是否有明确的成功指标？",
        "是否有明确的业务 owner？"
    ],
    "technical_feasibility": [
        "当前技术是否能达到要求精度？",
        "是否有可参考的类似案例？",
        "团队是否具备相关技术能力？"
    ]
}


@dataclass
class RequirementAssessment:
    """需求评估结果"""
    project_name: str
    problem_fit_score: int = 0  # 1-5 分
    data_readiness_score: int = 0
    business_value_score: int = 0
    technical_feasibility_score: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def total_score(self) -> int:
        """总分 (满分 20 分)"""
        return (self.problem_fit_score + self.data_readiness_score +
                self.business_value_score + self.technical_feasibility_score)

    @property
    def recommendation(self) -> str:
        """推荐建议"""
        if self.total_score >= 16:
            return "强烈推荐 - 立即启动"
        elif self.total_score >= 12:
            return "推荐 - 建议启动 PoC"
        elif self.total_score >= 8:
            return "谨慎 - 需进一步调研"
        else:
            return "不推荐 - 风险过高"

    def generate_report(self) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 60)
        report.append(f"AI 项目需求评估报告 - {self.project_name}")
        report.append("=" * 60)
        report.append(f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("【维度评分】")
        report.append(f"  问题适配度：    {self.problem_fit_score}/5")
        report.append(f"  数据就绪度：    {self.data_readiness_score}/5")
        report.append(f"  商业价值：      {self.business_value_score}/5")
        report.append(f"  技术可行性：    {self.technical_feasibility_score}/5")
        report.append("")
        report.append(f"【总分】{self.total_score}/20")
        report.append(f"【建议】{self.recommendation}")
        report.append("")

        if self.notes:
            report.append("【备注】")
            for note in self.notes:
                report.append(f"  - {note}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


# ========== ROI 计算模型 ==========

def calculate_roi(
    development_cost: float,
    data_cost: float,
    compute_cost_monthly: float,
    maintenance_cost_monthly: float,
    expected_benefit_monthly: float,
    project_months: int = 36
) -> Dict:
    """
    计算投资回报率

    参数:
        development_cost: 开发成本 (一次性)
        data_cost: 数据成本 (一次性)
        compute_cost_monthly: 月度计算成本
        maintenance_cost_monthly: 月度运维成本
        expected_benefit_monthly: 月度预期收益
        project_months: 项目周期 (月)

    返回:
        {total_cost, total_benefit, roi, payback_months, monthly_avg_profit}
    """
    # 总成本
    one_time_cost = development_cost + data_cost
    monthly_cost = compute_cost_monthly + maintenance_cost_monthly
    total_cost = one_time_cost + monthly_cost * project_months

    # 总收益
    total_benefit = expected_benefit_monthly * project_months

    # ROI
    roi = (total_benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0

    # 回本周期
    monthly_profit = expected_benefit_monthly - monthly_cost
    if monthly_profit > 0:
        payback_months = one_time_cost / monthly_profit
    else:
        payback_months = float('inf')

    return {
        "total_cost": total_cost,
        "total_benefit": total_benefit,
        "roi": roi,
        "payback_months": payback_months,
        "monthly_avg_profit": monthly_profit
    }


def calculate_unit_economics(
    price_per_request: float,
    cost_per_request: float,
    requests_per_month: int,
    customer_acquisition_cost: float = 100,
    avg_customer_lifetime_months: int = 12
) -> Dict:
    """
    计算单位经济模型

    参数:
        price_per_request: 单次请求价格
        cost_per_request: 单次请求成本
        requests_per_month: 月请求量
        customer_acquisition_cost: 获客成本
        avg_customer_lifetime_months: 平均客户生命周期
    """
    # 月度毛利
    monthly_gross_profit = (price_per_request - cost_per_request) * requests_per_month

    # LTV (客户生命周期价值)
    avg_requests_per_customer = requests_per_month / max(1, requests_per_month / 1000)  # 简化估算
    ltv = avg_requests_per_customer * (price_per_request - cost_per_request) * avg_customer_lifetime_months

    # LTV/CAC 比率
    ltv_cac_ratio = ltv / customer_acquisition_cost if customer_acquisition_cost > 0 else float('inf')

    return {
        "monthly_gross_profit": monthly_gross_profit,
        "ltv": ltv,
        "cac": customer_acquisition_cost,
        "ltv_cac_ratio": ltv_cac_ratio,
        "unit_economics_healthy": ltv_cac_ratio >= 3
    }


# ========== 需求评估问卷 ==========

def run_assessment_interview() -> RequirementAssessment:
    """运行需求评估访谈"""
    print("=" * 60)
    print("AI 项目需求评估访谈")
    print("=" * 60)
    print()

    project_name = input("项目名称：")

    print("\n【问题适配度】(1-5 分)")
    print(AI_READINESS_CHECKLIST["problem_fit"])
    problem_fit = int(input("评分："))

    print("\n【数据就绪度】(1-5 分)")
    print(AI_READINESS_CHECKLIST["data_readiness"])
    data_readiness = int(input("评分："))

    print("\n【商业价值】(1-5 分)")
    print(AI_READINESS_CHECKLIST["business_value"])
    business_value = int(input("评分："))

    print("\n【技术可行性】(1-5 分)")
    print(AI_READINESS_CHECKLIST["technical_feasibility"])
    technical_feasibility = int(input("评分："))

    assessment = RequirementAssessment(
        project_name=project_name,
        problem_fit_score=problem_fit,
        data_readiness_score=data_readiness,
        business_value_score=business_value,
        technical_feasibility_score=technical_feasibility
    )

    print("\n" + assessment.generate_report())

    return assessment


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例：评估一个智能客服项目
    print("示例：智能客服 AI 项目评估")
    print()

    assessment = RequirementAssessment(
        project_name="智能客服 AI",
        problem_fit_score=4,  # 客服对话是典型的 NLP 问题
        data_readiness_score=3,  # 有部分历史对话数据
        business_value_score=5,  # 可显著降低人力成本
        technical_feasibility_score=4,  # 技术成熟
        notes=[
            "已有 10 万+ 历史对话数据",
            "预计可替代 60% 人工客服",
            "需 3 个月 PoC 验证"
        ]
    )

    print(assessment.generate_report())

    # ROI 分析
    print("\n【ROI 分析】")
    roi_result = calculate_roi(
        development_cost=500000,  # 开发 50 万
        data_cost=100000,  # 数据标注 10 万
        compute_cost_monthly=20000,  # 月度计算 2 万
        maintenance_cost_monthly=10000,  # 运维 1 万
        expected_benefit_monthly=100000,  # 月度收益 10 万
        project_months=36
    )

    print(f"  总成本：¥{roi_result['total_cost']:,.0f}")
    print(f"  总收益：¥{roi_result['total_benefit']:,.0f}")
    print(f"  ROI: {roi_result['roi']:.1f}%")
    print(f"  回本周期：{roi_result['payback_months']:.1f} 月")
    print(f"  月均利润：¥{roi_result['monthly_avg_profit']:,.0f}")

    # 单位经济分析
    print("\n【单位经济模型】")
    unit econ = calculate_unit_economics(
        price_per_request=0.01,  # ¥0.01/请求
        cost_per_request=0.003,  # ¥0.003/请求
        requests_per_month=1000000,  # 100 万请求/月
        customer_acquisition_cost=500,
        avg_customer_lifetime_months=12
    )

    print(f"  月度毛利：¥{unit_econ['monthly_gross_profit']:,.0f}")
    print(f"  客户生命周期价值：¥{unit_econ['ltv']:,.0f}")
    print(f"  LTV/CAC: {unit_econ['ltv_cac_ratio']:.2f}")
    print(f"  商业健康度：{'✓ 健康' if unit_econ['unit_economics_healthy'] else '⚠ 需优化'}")
