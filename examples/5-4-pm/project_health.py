"""
AI 项目健康度评估工具（配合 docs/chapter-5/5-4-project-management.md）。

把"项目健不健康"从主观感觉变成六维量化评分：
效果进展 / 数据就绪 / 工程就绪 / 业务对齐 / 进度风险 / 成本风险。
每维 0-5 分，输出总分、亮灯（绿/黄/红）与改进建议。

运行方式：
    pip install 无第三方依赖，仅需 Python 3.10+
    python project_health.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 六个健康维度：key -> (中文名, 打分参考)
DIMENSIONS: dict[str, tuple[str, str]] = {
    "effect":  ("效果进展", "核心指标离验收阈值的距离与提升趋势（0=无数据/很差，5=达标且还在涨）"),
    "data":    ("数据就绪", "评测集、训练/知识库数据的齐备与质量（0=没有，5=齐备且持续回流）"),
    "eng":     ("工程就绪", "服务化、集成、稳定性、监控（0=Notebook 原型，5=可灰度上线）"),
    "biz":     ("业务对齐", "业务方参与度、验收标准一致性（0=各说各话，5=共同定义且持续试用）"),
    "schedule":("进度风险", "里程碑推进、阻塞清除（0=严重延期/阻塞，5=按计划推进）"),
    "cost":    ("成本风险", "算力/API 成本相对预算（0=远超预算，5=在预算内且有优化空间）"),
}

# 每维 0-5 分对应的状态描述，帮助打分者对齐口径
RUBRIC = {
    0: "完全没有 / 严重失控",
    1: "很差，基本不可用",
    2: "有雏形，问题明显",
    3: "及格，可用但不稳",
    4: "良好，接近目标",
    5: "优秀，达成且可持续",
}


@dataclass
class HealthReport:
    scores: dict[str, int]
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.scores.values())

    @property
    def max_score(self) -> int:
        return 5 * len(self.scores)

    def light(self, score: int) -> str:
        """单维亮灯：>=4 绿，>=3 黄，否则红。"""
        if score >= 4:
            return "🟢"
        if score >= 3:
            return "🟡"
        return "🔴"

    def overall(self) -> str:
        ratio = self.total / self.max_score
        if ratio >= 0.8:
            return "🟢 健康：可进入下一阶段（通过 Decision Gate）"
        if ratio >= 0.6:
            return "🟡 预警：有明显短板，先补红灯维度再推进"
        return "🔴 危险：不建议加大投入，回到 PoC 重新验证或收窄场景"


def evaluate(report: HealthReport) -> None:
    print("=" * 66)
    print("AI 项目健康度评估")
    print("=" * 66)
    for key, (name, hint) in DIMENSIONS.items():
        score = report.scores.get(key, 0)
        note = report.notes.get(key, "")
        print(f"{report.light(score)} {name:<6} {score}/5  {note}")
        print(f"      口径：{hint}")
    print("-" * 66)
    print(f"总分：{report.total}/{report.max_score}  →  {report.overall()}")

    reds = [DIMENSIONS[k][0] for k, v in report.scores.items() if v < 3]
    if reds:
        print("\n⚠️  红灯维度（优先处理）：" + "、".join(reds))
        print("   建议：AI 项目的 Go/No-Go 由最弱维度决定，先补短板再谈规模化。")


if __name__ == "__main__":
    # 示例：一个进行到试点阶段的客服问答项目
    demo = HealthReport(
        scores={
            "effect":   3,   # 问答命中率 72%，接近 75% 阈值但未稳定
            "data":     4,   # 知识库齐备，badcase 开始回流
            "eng":      2,   # 还是脚本调用，没有服务化和监控
            "biz":      4,   # 客服主管每周参与评测
            "schedule": 3,   # 比里程碑晚 1 周
            "cost":     4,   # API 成本在预算内
        },
        notes={
            "effect": "命中率 72%，目标 75%，幻觉偶发",
            "eng":    "无监控/无灰度，阻塞规模化",
        },
    )
    evaluate(demo)
