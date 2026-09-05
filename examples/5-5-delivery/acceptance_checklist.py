"""
AI 项目 Go-Live 交付检查清单（配合 docs/chapter-5/5-5-integration-delivery.md）。

上线评审会上逐项核对：功能 / 效果 / 性能 / 安全合规 / 运营 五类。
blocker=True 的项不通过则禁止上线；其余为建议项。

运行方式：
    python acceptance_checklist.py            # 跑示例状态
    修改 ITEMS 中各项的 done 状态以反映你的项目
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckItem:
    category: str
    name: str
    blocker: bool          # True = 不通过则阻塞上线
    done: bool = False
    note: str = ""


ITEMS = [
    # —— 功能与集成 ——
    CheckItem("功能集成", "核心场景在真实业务系统中闭环（非独立 Demo）", True),
    CheckItem("功能集成", "与上游/下游系统接口联调通过", True),
    CheckItem("功能集成", "降级预案：模型服务故障时回退到原流程", True),
    CheckItem("功能集成", "灰度开关（feature flag）可一键关闭", True),
    # —— 效果 ——
    CheckItem("效果验收", "双方确认的评测集上核心指标达标", True),
    CheckItem("效果验收", "在线小流量采纳率/接管率达到约定阈值", True),
    CheckItem("效果验收", "badcase 收集与反馈闭环已建立", False),
    # —— 性能与稳定性 ——
    CheckItem("性能稳定", "压测通过：P95 延迟、并发满足业务峰值", True),
    CheckItem("性能稳定", "可用性 SLA 达成（如 99.9%），有监控告警", True),
    CheckItem("性能稳定", "单次/月度成本在预算内，有限流与预算告警", False),
    # —— 数据 ——
    CheckItem("数据治理", "知识库/数据源定时同步，内容非过期", True),
    CheckItem("数据治理", "检索按用户权限过滤，无越权访问", True),
    CheckItem("数据治理", "敏感字段脱敏，数据不出域要求已满足", True),
    # —— 安全合规 ——
    CheckItem("安全合规", "输入输出内容审核通过（有害/敏感）", True),
    CheckItem("安全合规", "Prompt 注入等攻击测试通过", True),
    CheckItem("安全合规", "Agent 写操作有二次确认+权限校验+可回滚", True),
    CheckItem("安全合规", "审计日志完整（谁、何时、问什么、答什么）", False),
    CheckItem("安全合规", "高风险决策保留人工拍板，AI 仅辅助", True),
    # —— 运营交接 ——
    CheckItem("运营交接", "已明确上线后长期 owner（业务+技术）", True),
    CheckItem("运营交接", "用户培训完成，SOP 更新（AI 与人分工）", False),
    CheckItem("运营交接", "运维/回滚/二次开发文档交付", False),
]


def run_checklist(items: list[CheckItem]) -> bool:
    categories: dict[str, list[CheckItem]] = {}
    for item in items:
        categories.setdefault(item.category, []).append(item)

    blockers_failed: list[CheckItem] = []
    print("=" * 70)
    print("AI 项目 Go-Live 交付检查清单")
    print("=" * 70)

    for category, group in categories.items():
        print(f"\n【{category}】")
        for item in group:
            mark = "✅" if item.done else ("⛔" if item.blocker else "⬜")
            tag = " [阻塞项]" if item.blocker else " [建议项]"
            print(f"  {mark} {item.name}{tag}")
            if not item.done and item.blocker:
                blockers_failed.append(item)

    total = len(items)
    done = sum(1 for i in items if i.done)
    print("\n" + "-" * 70)
    print(f"完成度：{done}/{total}")

    if blockers_failed:
        print(f"\n❌ 存在 {len(blockers_failed)} 个阻塞项未通过，不建议上线：")
        for item in blockers_failed:
            print(f"  - [{item.category}] {item.name}")
        return False

    print("\n🟢 所有阻塞项已通过，可以进入灰度/上线（建议项请尽快补齐）。")
    return True


if __name__ == "__main__":
    # 示例：模拟一个"大部分就绪、但安全与降级未完成"的项目
    for item in ITEMS:
        # 默认把非阻塞项设为已完成，阻塞项里故意留几项未做以演示红灯
        item.done = not item.blocker
    # 关键阻塞项已完成
    for name in ["核心场景在真实业务系统中闭环（非独立 Demo）",
                 "双方确认的评测集上核心指标达标",
                 "压测通过：P95 延迟、并发满足业务峰值"]:
        for it in ITEMS:
            if it.name == name:
                it.done = True

    import sys
    sys.exit(0 if run_checklist(ITEMS) else 1)
