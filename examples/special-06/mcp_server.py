"""
MCP Server 最小示例（订单查询 + 工单创建）

演示 2025 版 MCP（Model Context Protocol）核心：
  - 用 @mcp.tool() 暴露工具，自动从类型与 docstring 生成 schema
  - Streamable HTTP 传输（单端点，可水平扩展）
  - 工具注解（readOnly / destructive）驱动 HITL 审批策略
  - 结构化返回 + 资源链接（resource link）

依赖：
  pip install "mcp[cli]"
运行（HTTP 模式）：
  python mcp_server.py            # 默认 0.0.0.0:8300，Streamable HTTP
  # 或 stdio 本地模式：把 transport 改成 "stdio"

说明：db 部分用内存数据模拟，真实环境替换为带权限过滤的数据库查询。
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # 允许未安装 mcp 时仍能 import/静态检查
    FastMCP = None

if FastMCP is not None:
    mcp = FastMCP("order-service")
else:  # 降级为普通对象，保证脚本可被导入查看
    class _Stub:
        def tool(self, *a, **k):
            def deco(fn):
                return fn
            return deco
        def run(self, *a, **k):
            print("请先 `pip install mcp[cli]` 再运行 MCP Server。")
    mcp = _Stub()


# 模拟数据库（真实环境应在 SQL 层做行级权限过滤）
_ORDERS = {
    "ORD-1001": {"status": "已发货", "amount": 299.0, "owner": "u123"},
    "ORD-1002": {"status": "待支付", "amount": 1280.0, "owner": "u456"},
}
_TICKETS: dict[str, dict] = {}


@mcp.tool()
def get_order(order_id: str) -> dict:
    """查询订单状态与金额。只读操作，仅返回当前用户有权限查看的订单。

    Args:
        order_id: 订单编号，形如 ORD-xxxx。
    """
    order = _ORDERS.get(order_id)
    if not order:
        return {"ok": False, "error": "订单不存在或无权限"}
    return {
        "ok": True,
        "order_id": order_id,
        "status": order["status"],
        "amount": order["amount"],
        # 资源链接：让 Agent/客户端可进一步取发票等关联资源
        "resource_link": f"mcp://order-service/resources/invoice/{order_id}",
    }


@mcp.tool()
def create_refund_ticket(order_id: str, reason: str) -> dict:
    """为订单创建退款工单。这是写操作（有副作用），生产环境应触发人工审批。

    Args:
        order_id: 订单编号。
        reason: 退款原因，将进入审批流。
    """
    if order_id not in _ORDERS:
        return {"ok": False, "error": "订单不存在"}
    ticket_id = f"TK-{len(_TICKETS) + 1:04d}"
    _TICKETS[ticket_id] = {"order_id": order_id, "reason": reason, "status": "待审批"}
    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "待审批",
        "note": "退款为高危操作，已进入 HITL 审批队列，审批通过后执行。",
    }


if __name__ == "__main__":
    # Streamable HTTP：单端点 /mcp，支持无状态扩展或升级为 SSE 长连接
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8300)
