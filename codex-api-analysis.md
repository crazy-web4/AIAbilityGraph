# Codex 桌面 App 后端接口调用分析报告

> 分析时间：2026-08-10
> 分析对象：Codex Desktop App (ChatGPT 2.app)
> 当前模型提供商：火山引擎 (volcengine-coding-plan)
> 当前模型：doubao-seed-2.1-turbo

---

## 一、分析方法

### 方法一：日志数据库分析（推荐，最直接）

Codex 将所有 API 调用日志记录在 SQLite 数据库中：

```
~/.codex/logs_2.sqlite
```

关键表：`logs`
- `target`：日志来源模块（如 `codex_http_client::transport`）
- `feedback_log_body`：日志详细内容（包含请求体、响应头、SSE 事件等）
- `level`：日志级别（TRACE/DEBUG/INFO/WARN/ERROR）

常用查询 SQL：

```sql
-- 查看所有 HTTP 请求
SELECT id, feedback_log_body FROM logs 
WHERE target = 'codex_http_client::transport' 
  AND feedback_log_body LIKE '%POST to%'
ORDER BY id DESC;

-- 查看所有 SSE 事件
SELECT id, feedback_log_body FROM logs 
WHERE target = 'codex_api::sse::responses'
  AND feedback_log_body LIKE 'SSE event: %'
ORDER BY id DESC;

-- 查看响应头
SELECT id, feedback_log_body FROM logs 
WHERE target = 'codex_http_client::client'
  AND feedback_log_body LIKE '%Request completed%'
ORDER BY id DESC;
```

### 方法二：配置文件分析

```
~/.codex/config.toml   # 主配置（模型提供商、API端点、插件等）
~/.codex/auth.json     # 认证信息（API Key 加密存储）
```

### 方法三：网络抓包工具

可用工具：
- **mitmproxy** / **Charles**：HTTPS 抓包（需配置 SSL 证书）
- **Wireshark** / **tcpdump**：底层网络抓包
- Chrome DevTools：Electron 应用可用 `Cmd+Option+I` 打开开发者工具

---

## 二、API 调用基本信息

### 2.1 API 端点

| 项目 | 值 |
|------|-----|
| **HTTP 方法** | `POST` |
| **API 地址** | `https://ark.cn-beijing.volces.com/api/coding/v3/responses` |
| **API 协议** | OpenAI Responses API（兼容格式） |
| **传输方式** | SSE (Server-Sent Events) 流式响应 |
| **Content-Type** | `text/event-stream` |
| **认证方式** | Bearer Token |

> ⚠️ 注意：API 地址取决于你配置的 `model_provider`。以上是当前配置的火山引擎端点。
> 如果你使用 OpenAI 官方，地址会是 `https://api.openai.com/v1/responses`。

### 2.2 认证机制

API Key 存储位置：macOS 钥匙串（Keychain），加密保存。

配置项（`config.toml`）：
```toml
[model_providers.volcengine-coding-plan]
name = "volcengine-coding-plan"
base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
env_key = "ARK_API_KEY"    # 环境变量名
wire_api = "responses"      # API 格式类型
```

认证头：
```
Authorization: Bearer <你的_API_KEY>
```

### 2.3 请求头

标准请求头：
```http
POST /api/coding/v3/responses HTTP/1.1
Host: ark.cn-beijing.volces.com
Authorization: Bearer <API_KEY>
Content-Type: application/json
Accept: text/event-stream
x-client-request-id: <请求唯一ID>
```

Codex 还通过请求体的 `client_metadata` 字段传递额外元数据（不通过 HTTP 头）。

---

## 三、请求体详细结构

请求体遵循 **OpenAI Responses API** 格式。以下是完整字段：

```json
{
  "model": "doubao-seed-2.1-turbo",
  "instructions": "<系统提示词，约20KB>",
  "input": [<消息列表>],
  "tools": [<工具定义列表>],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "reasoning": {
    "effort": "low",
    "summary": "detailed"
  },
  "store": false,
  "stream": true,
  "include": ["reasoning.encrypted_content"],
  "prompt_cache_key": "019fead3-7a15-74e1-8ba6-f88d41b83d57",
  "client_metadata": {
    "x-codex-installation-id": "c211ceff-1584-4aac-bc23-e20a21206530",
    "thread_id": "019fead3-7a15-74e1-8ba6-f88d41b83d57",
    "turn_id": "019fead4-9bb3-7a23-8e49-04ddf7f9dbad",
    "x-codex-window-id": "019fead3-7a15-74e1-8ba6-f88d41b83d57:0",
    "session_id": "019fead3-7a15-74e1-8ba6-f88d41b83d57"
  }
}
```

### 3.1 各字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 模型名称，如 `doubao-seed-2.1-turbo` |
| `instructions` | string | 系统级提示词，定义 AI 的角色和行为（Codex 的系统提示约20KB） |
| `input` | array | 对话消息列表，每条消息包含 `type`, `id`, `role`, `content` |
| `tools` | array | 可用工具定义列表（function 类型） |
| `tool_choice` | string | 工具选择策略，`"auto"` 表示模型自动决定 |
| `parallel_tool_calls` | boolean | 是否允许并行工具调用 |
| `reasoning` | object | 推理配置：`effort` 推理强度(low/medium/high)，`summary` 推理摘要模式 |
| `store` | boolean | 是否存储对话历史 |
| `stream` | boolean | 是否使用流式响应（SSE） |
| `include` | array | 额外包含的响应内容，如 `reasoning.encrypted_content` |
| `prompt_cache_key` | string | 提示缓存键，通常为会话/线程 ID |
| `client_metadata` | object | 客户端元数据（安装ID、会话ID等） |

### 3.2 input 消息格式

```json
{
  "type": "message",
  "id": "msg_019fead4-9f53-7e...",
  "role": "developer",
  "content": [
    {"type": "input_text", "text": "..."},
    {"type": "input_text", "text": "..."}
  ]
}
```

- `role` 取值：`developer`（系统上下文）、`user`（用户消息）、`assistant`（AI回复）
- `content` 是 content part 列表，支持 `input_text`、`input_image` 等类型

### 3.3 tools 工具格式

```json
{
  "type": "function",
  "name": "exec_command",
  "description": "Runs a command in a PTY...",
  "strict": false,
  "parameters": {
    "type": "object",
    "properties": {
      "cmd": {"type": "string", "description": "Shell command to execute."},
      "justification": {"type": "string"}
    },
    "required": ["cmd"],
    "additionalProperties": false
  }
}
```

Codex 默认注册 15 个工具，包括：
- `exec_command` / `write_stdin`：命令执行
- `list_mcp_resources` / `read_mcp_resource`：MCP 资源
- `update_plan` / `request_user_input`：计划与用户输入
- `view_image` / `get_goal` 等

---

## 四、响应格式（SSE 流式）

响应是 **SSE (Server-Sent Events)** 流，每条事件格式：

```
event: message
data: {"type": "<事件类型>", ...}
```

### 4.1 SSE 事件类型完整列表

| 事件类型 | 说明 |
|----------|------|
| `response.created` | 响应创建，返回初始响应对象 |
| `response.in_progress` | 响应进行中（部分字段更新） |
| `response.output_item.added` | 新增输出项（消息/函数调用/推理） |
| `response.content_part.added` | 内容块开始添加 |
| `response.output_text.delta` | 文本输出增量 |
| `response.output_text.done` | 文本输出完成 |
| `response.reasoning_text.delta` | 思考过程文本增量 |
| `response.reasoning_text.done` | 思考过程文本完成 |
| `response.reasoning_summary_text.delta` | 推理摘要文本增量 |
| `response.reasoning_summary_text.done` | 推理摘要文本完成 |
| `response.reasoning_summary_part.added` | 推理摘要部分开始 |
| `response.reasoning_summary_part.done` | 推理摘要部分完成 |
| `response.function_call_arguments.delta` | 函数参数增量 |
| `response.function_call_arguments.done` | 函数参数完成 |
| `response.output_item.done` | 输出项完成 |
| `response.completed` | 整个响应完成 |
| `response.web_search_call.in_progress` | 网页搜索进行中 |
| `response.web_search_call.searching` | 网页搜索中 |
| `response.web_search_call.completed` | 网页搜索完成 |

### 4.2 response.completed 完整结构

```json
{
  "id": "resp_0217863325508161f8649ba3601f67aef8adf8b2a41491046317c",
  "object": "response",
  "model": "doubao-seed-2-1-turbo-260628",
  "created_at": 1786332555,
  "max_output_tokens": 32768,
  "service_tier": "default",
  "status": "completed",
  "output": [
    {
      "id": "rs_021786332555877000000000000...",
      "type": "reasoning",
      "summary": [
        {"type": "summary_text", "text": "我将先核查..."}
      ],
      "status": "completed",
      "encrypted_content": "djF7VsJPrTiW..."
    },
    {
      "id": "msg_02178633255594400000000000...",
      "type": "message",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "..."}
      ]
    }
  ],
  "usage": {
    "input_tokens": 22596,
    "output_tokens": 218,
    "total_tokens": 22814,
    "input_tokens_details": {
      "cached_tokens": 22328
    },
    "output_tokens_details": {
      "reasoning_tokens": 18
    }
  },
  "caching": {...},
  "tools": [...],
  "instructions": "...",
  "store": false,
  "prompt_cache_key": "..."
}
```

### 4.3 响应头示例

```http
HTTP/1.1 200 OK
server: istio-envoy
date: Mon, 10 Aug 2026 08:42:13 GMT
content-type: text/event-stream
cache-control: no-cache
x-client-request-id: 019fead3-7a15-74e1-8ba6-f88d41b83d57
x-request-id: 021786351327816223e95793e6cdc610dbdf3940c5146022deb8e
x-envoy-upstream-service-time: 4978
transfer-encoding: chunked
```

---

## 五、手动调用示例

### 5.1 cURL 调用（非流式）

```bash
curl -X POST "https://ark.cn-beijing.volces.com/api/coding/v3/responses" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-seed-2.1-turbo",
    "input": "你好，请介绍一下你自己",
    "stream": false
  }'
```

### 5.2 cURL 调用（流式 SSE）

```bash
curl -N -X POST "https://ark.cn-beijing.volces.com/api/coding/v3/responses" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "doubao-seed-2.1-turbo",
    "input": "你好",
    "stream": true
  }'
```

### 5.3 Python 调用（流式）

```python
import json
import requests

API_KEY = "your-api-key-here"
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"

def stream_response(prompt, model="doubao-seed-2.1-turbo"):
    url = f"{BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")
                
                if event_type == "response.output_text.delta":
                    print(event.get("delta", ""), end="", flush=True)
                elif event_type == "response.completed":
                    print("\n--- 请求完成 ---")
                    usage = event["response"].get("usage", {})
                    print(f"Token 用量: {usage}")
            except json.JSONDecodeError:
                pass

stream_response("用一句话介绍 Codex")
```

### 5.4 带工具调用的完整示例

```python
import json
import requests

API_KEY = "your-api-key-here"
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

payload = {
    "model": "doubao-seed-2.1-turbo",
    "input": "北京今天天气怎么样？",
    "tools": tools,
    "tool_choice": "auto",
    "stream": False,
}

response = requests.post(
    f"{BASE_URL}/responses",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json=payload
)
result = response.json()

# 检查是否需要调用工具
for output in result.get("output", []):
    if output["type"] == "function_call":
        print(f"调用工具: {output['name']}")
        print(f"参数: {output['arguments']}")
```

---

## 六、关键发现总结

1. **API 标准**：Codex 使用 **OpenAI Responses API** 格式（v2 格式），不是 Chat Completions API。
   - 区别：使用 `input` 而非 `messages`，支持 `instructions` 系统指令，内置工具调用支持更强。

2. **流式传输**：所有对话都通过 **SSE** 流式返回，支持文本、思考过程、工具调用的增量输出。

3. **多模型提供商**：Codex 支持配置多个模型提供商，通过 `wire_api = "responses"` 标识 API 格式类型。
   - 常见提供商：OpenAI 官方、火山引擎、阿里云百炼、Azure OpenAI 等。

4. **提示缓存**：通过 `prompt_cache_key` 实现提示词缓存，大幅减少重复 token 消耗（日志显示缓存命中率可达 98%+）。

5. **加密推理内容**：思考过程（reasoning）内容是加密的（`encrypted_content`），保护模型的思考链条。

6. **客户端元数据**：Codex 通过 `client_metadata` 传递安装ID、会话ID、窗口ID等追踪信息。

7. **认证存储**：API Key 存储在 macOS 钥匙串中，`auth.json` 只存加密后的引用。

---

## 七、获取 API Key 的方法

### 方法 1：从环境变量获取

如果你在配置中设置了环境变量：
```bash
echo $ARK_API_KEY    # 火山引擎
echo $OPENAI_API_KEY # OpenAI 官方
```

### 方法 2：使用 codex login 状态查看

```bash
/Applications/ChatGPT\ 2.app/Contents/Resources/codex login status
```

### 方法 3：直接从模型提供商控制台获取

- **火山引擎**：登录 [火山引擎方舟平台](https://console.volcengine.com/ark) → API Key 管理
- **OpenAI**：登录 [platform.openai.com](https://platform.openai.com/api-keys)
- **阿里云百炼**：登录 [百炼平台](https://bailian.console.aliyun.com) → API Key 管理

---

## 八、参考资料

- [OpenAI Responses API 文档](https://platform.openai.com/docs/api-reference/responses)
- [火山引擎方舟 API 文档](https://www.volcengine.com/docs/82379)
- Codex 配置文件：`~/.codex/config.toml`
- Codex 日志数据库：`~/.codex/logs_2.sqlite`
