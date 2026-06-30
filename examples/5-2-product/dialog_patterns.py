#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 对话设计模式

功能:
- 常见对话模式模板
- 错误状态处理
- 用户体验设计
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import random


# ========== 对话模式定义 ==========

class DialogPatternType(Enum):
    """对话模式类型"""
    CLARIFICATION = "clarification"  # 澄清式
    CONFIRMATION = "confirmation"    # 确认式
    GUIDANCE = "guidance"           # 引导式
    RECOVERY = "recovery"           # 修复式
    ERROR = "error"                 # 错误处理


@dataclass
class DialogPattern:
    """对话模式"""
    name: str
    pattern_type: DialogPatternType
    trigger: str                    # 触发条件
    response_template: str          # 响应模板
    examples: List[Dict] = field(default_factory=list)


# 常见对话模式库
DIALOG_PATTERNS_LIB = {
    "clarification": DialogPattern(
        name="澄清式",
        pattern_type=DialogPatternType.CLARIFICATION,
        trigger="用户问题模糊",
        response_template="我不太确定您的意思，您是想问 {options} 吗？",
        examples=[
            {
                "user": "怎么设置？",
                "bot": "请问您是想设置：\n1. 账号信息\n2. 通知偏好\n3. 隐私设置"
            },
            {
                "user": "我想看数据",
                "bot": "请问您想看：\n1. 今日数据\n2. 本周数据\n3. 自定义时间段"
            }
        ]
    ),

    "confirmation": DialogPattern(
        name="确认式",
        pattern_type=DialogPatternType.CONFIRMATION,
        trigger="执行重要操作",
        response_template="我理解您想要{action}，确认吗？",
        examples=[
            {
                "user": "删除所有数据",
                "bot": "我理解您想要删除所有数据，这个操作无法撤销，确认继续吗？"
            },
            {
                "user": "退出登录",
                "bot": "确认要退出登录吗？"
            }
        ]
    ),

    "guidance": DialogPattern(
        name="引导式",
        pattern_type=DialogPatternType.GUIDANCE,
        trigger="用户不知道能做什么",
        response_template="我可以帮你...{capabilities}",
        examples=[
            {
                "user": "你好",
                "bot": "你好！我可以帮你：\n- 回答问题\n- 创作文档\n- 分析数据\n请问需要什么帮助？"
            },
            {
                "user": "你能做什么",
                "bot": "我是你的 AI 助手，可以帮你：\n1. 查询信息\n2. 生成内容\n3. 数据分析\n请告诉我你的需求"
            }
        ]
    ),

    "recovery": DialogPattern(
        name="修复式",
        pattern_type=DialogPatternType.RECOVERY,
        trigger="AI 理解错误",
        response_template="抱歉我理解错了，您是想要{correction}吗？",
        examples=[
            {
                "user": "不是这个，我要昨天的",
                "bot": "抱歉我理解错了，您是要查看昨天的数据对吗？"
            },
            {
                "user": "不对，我要 export 功能",
                "bot": "抱歉理解错了，您是要使用导出功能对吗？"
            }
        ]
    )
}


# ========== 对话状态机 ==========

class DialogState(Enum):
    """对话状态"""
    IDLE = "idle"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    CLARIFYING = "clarifying"
    CONFIRMING = "confirming"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class DialogContext:
    """对话上下文"""
    conversation_id: str
    state: DialogState = DialogState.IDLE
    user_inputs: List[str] = field(default_factory=list)
    bot_responses: List[str] = field(default_factory=list)
    clarifications_needed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_user_input(self, text: str):
        """添加用户输入"""
        self.user_inputs.append(text)

    def add_bot_response(self, text: str):
        """添加 bot 响应"""
        self.bot_responses.append(text)

    def record_error(self, error: str):
        """记录错误"""
        self.errors.append(error)
        self.state = DialogState.ERROR


class DialogManager:
    """
    对话管理器

    功能:
    - 状态管理
    - 模式匹配
    - 上下文追踪
    """

    def __init__(self):
        self.patterns = DIALOG_PATTERNS_LIB
        self.contexts: Dict[str, DialogContext] = {}

    def create_context(self, conversation_id: str) -> DialogContext:
        """创建新对话上下文"""
        context = DialogContext(conversation_id=conversation_id)
        self.contexts[conversation_id] = context
        return context

    def get_context(self, conversation_id: str) -> Optional[DialogContext]:
        """获取对话上下文"""
        return self.contexts.get(conversation_id)

    def detect_pattern(self, user_input: str, context: DialogContext) -> Optional[str]:
        """
        检测适用的对话模式

        返回:
            模式名称或 None
        """
        # 简单关键词匹配 (实际可用 ML 模型)
        user_input_lower = user_input.lower()

        # 检测确认场景
        confirm_keywords = ["删除", "清空", "重置", "退出", "取消"]
        if any(kw in user_input_lower for kw in confirm_keywords):
            return "confirmation"

        # 检测澄清场景
        vague_keywords = ["怎么", "如何", "哪里", "什么"]
        if any(kw in user_input_lower for kw in vague_keywords):
            # 如果问题太短，可能需要澄清
            if len(user_input) < 10:
                return "clarification"

        # 检测引导场景
        greeting_keywords = ["你好", "hello", "hi", "在吗", "能做什么"]
        if any(kw in user_input_lower for kw in greeting_keywords):
            return "guidance"

        # 检测修复场景
        if len(context.user_inputs) > 0:
            correction_keywords = ["不对", "不是", "错了", "我要"]
            if any(kw in user_input_lower for kw in correction_keywords):
                return "recovery"

        return None

    def generate_response(
        self,
        pattern_name: str,
        user_input: str,
        context: DialogContext
    ) -> str:
        """生成响应"""
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return "抱歉，我还在学习中，请换种方式描述您的问题。"

        if pattern_name == "clarification":
            return self._generate_clarification(user_input, context)
        elif pattern_name == "confirmation":
            return self._generate_confirmation(user_input, context)
        elif pattern_name == "guidance":
            return self._generate_guidance(context)
        elif pattern_name == "recovery":
            return self._generate_recovery(user_input, context)
        else:
            return "我能帮你什么？"

    def _generate_clarification(self, user_input: str, context: DialogContext) -> str:
        """生成澄清式响应"""
        # 提取可能的意图
        if "设置" in user_input:
            return "请问您是想设置：\n1. 账号信息\n2. 通知偏好\n3. 隐私设置\n4. 其他"
        elif "看" in user_input or "查看" in user_input:
            return "请问您想看：\n1. 今日数据\n2. 本周数据\n3. 本月数据\n4. 自定义时间段"
        else:
            return "我不太确定您的意思，能否详细描述一下您的需求？"

    def _generate_confirmation(self, user_input: str, context: DialogContext) -> str:
        """生成确认式响应"""
        if "删除" in user_input:
            return "⚠️ 删除操作无法撤销，确认要删除吗？\n\n请回复"确认"继续，或回复"取消"放弃。"
        elif "退出" in user_input:
            return "确认要退出吗？\n\n请回复"确认"继续。"
        else:
            return f"我理解您想要：{user_input}\n\n确认继续吗？"

    def _generate_guidance(self, context: DialogContext) -> str:
        """生成引导式响应"""
        capabilities = [
            "📝 创作文档、邮件、代码",
            "🔍 查询信息、解答问题",
            "📊 数据分析、总结",
            "🌐 翻译、润色文本",
            "💡 头脑风暴、创意讨论"
        ]
        return "你好！我是你的 AI 助手，可以帮你：\n\n" + "\n".join(capabilities) + "\n\n请问需要什么帮助？"

    def _generate_recovery(self, user_input: str, context: DialogContext) -> str:
        """生成修复式响应"""
        # 尝试理解用户真正的需求
        if "昨天" in user_input or "昨日" in user_input:
            return "抱歉我理解错了，您是要查看昨天的数据对吗？"
        elif "export" in user_input.lower() or "导出" in user_input:
            return "抱歉理解错了，您是要使用导出功能对吗？"
        else:
            return "抱歉我理解错了，能否详细描述一下您的需求？"

    def handle_input(
        self,
        conversation_id: str,
        user_input: str
    ) -> str:
        """
        处理用户输入

        返回:
            bot 响应
        """
        context = self.contexts.get(conversation_id)
        if not context:
            context = self.create_context(conversation_id)

        context.state = DialogState.PROCESSING
        context.add_user_input(user_input)

        # 检测模式
        pattern_name = self.detect_pattern(user_input, context)

        if pattern_name:
            response = self.generate_response(pattern_name, user_input, context)
        else:
            # 默认响应
            response = "收到，我正在处理你的请求：" + user_input

        context.add_bot_response(response)
        context.state = DialogState.WAITING_INPUT

        return response


# ========== 错误状态设计 ==========

@dataclass
class ErrorState:
    """错误状态"""
    error_type: str
    user_message: str
    technical_details: str
    suggested_action: str


ERROR_STATES = {
    "api_timeout": ErrorState(
        error_type="API 超时",
        user_message="响应超时，正在重试...",
        technical_details="API request timeout > 30s",
        suggested_action="请稍后重试或检查网络连接"
    ),

    "content_filtered": ErrorState(
        error_type="内容被过滤",
        user_message="我无法回答这个问题，让我们换个话题吧。",
        technical_details="Content policy violation",
        suggested_action="请修改问题或联系支持"
    ),

    "model_uncertain": ErrorState(
        error_type="模型不确定",
        user_message="这个我不太确定，以下是我的推测...",
        technical_details="Low confidence score",
        suggested_action="建议核实信息准确性"
    ),

    "context_overflow": ErrorState(
        error_type="上下文过长",
        user_message="对话太长，我的记忆有限。我们可以开始新话题吗？",
        technical_details="Context window exceeded",
        suggested_action="建议开始新对话或总结前文"
    ),

    "service_unavailable": ErrorState(
        error_type="服务不可用",
        user_message="服务暂时不可用，请稍后重试。",
        technical_details="Service unavailable",
        suggested_action="请稍后重试"
    )
}


class ErrorHandler:
    """错误处理器"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}

    def handle_error(
        self,
        error_type: str,
        context: DialogContext
    ) -> str:
        """处理错误并返回用户友好的消息"""
        error_state = ERROR_STATES.get(error_type)

        if error_state:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            context.record_error(f"{error_type}: {error_state.technical_details}")
            return error_state.user_message
        else:
            return "抱歉，出现了一个意外错误，请稍后重试。"

    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "by_type": self.error_counts.copy()
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("AI 对话模式演示")
    print("=" * 60)

    # 创建对话管理器
    manager = DialogManager()
    error_handler = ErrorHandler()

    # 模拟对话
    conversation_id = "demo-001"
    manager.create_context(conversation_id)

    test_inputs = [
        "你好",           # 引导式
        "怎么设置？",      # 澄清式
        "删除所有数据",    # 确认式
        "不对，我要导出",  # 修复式
    ]

    print("\n【对话演示】\n")

    for user_input in test_inputs:
        print(f"用户：{user_input}")
        response = manager.handle_input(conversation_id, user_input)
        print(f"Bot: {response}\n")

    # 错误处理演示
    print("\n【错误处理演示】\n")

    context = manager.get_context(conversation_id)
    for error_type in ["api_timeout", "content_filtered"]:
        response = error_handler.handle_error(error_type, context)
        print(f"[{error_type}] → {response}")

    # 统计
    print("\n【错误统计】")
    stats = error_handler.get_error_stats()
    print(f"总错误数：{stats['total_errors']}")
    print(f"分类统计：{stats['by_type']}")

    # 对话上下文
    print("\n【对话上下文】")
    print(f"用户输入数：{len(context.user_inputs)}")
    print(f"Bot 响应数：{len(context.bot_responses)}")
