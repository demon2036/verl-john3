#!/usr/bin/env python3
from fastmcp import FastMCP
from datetime import datetime

# 创建MCP服务器
mcp = FastMCP("SSE Demo Server")

@mcp.tool()
def get_time() -> str:
    """获取当前时间"""
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def echo(text: str) -> str:
    """回显文本"""
    return f"回显: {text}"

@mcp.tool()
def add(a: int, b: int) -> int:
    """加法计算"""
    return a + b

@mcp.resource("status://server")
def server_status() -> str:
    """服务器状态信息"""
    return f"SSE服务器运行中 - {datetime.now()}"

@mcp.resource("info://endpoints")
def endpoint_info() -> str:
    """端点信息"""
    return "SSE端点: /sse, 消息端点: /messages"

if __name__ == "__main__":
    # 使用SSE传输协议运行服务器
    print("启动SSE MCP服务器...")
    mcp.run(
        transport="sse",
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 端口
        path="/sse"      # SSE端点路径
    )