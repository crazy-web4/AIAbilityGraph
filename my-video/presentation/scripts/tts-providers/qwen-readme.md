# Qwen3-TTS Provider 使用说明

## 概述

本 Provider 支持使用 **Qwen3-TTS**（通义千问多语言语音合成模型）通过 moma.hq.cmcc 内部 API 服务进行音频合成。

**模型特点**：
- 支持 10 门主流外语、8 类中文方言
- 内置 49 种原生音色
- 支持语速、情绪、停顿控制
- 中英混读自然，长文本朗读流畅
- 开源商用友好

---

## 快速开始

### 1. 环境要求

```bash
# 必需命令行工具
curl --version          # HTTP 请求
jq --version            # JSON 处理
```

### 2. 配置环境变量

```bash
# API Key（JWT Token）
export QWEN_TTS_API_KEY="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoiNjgwNzE4YzcxZDdhMTQzNmY3ZTRkOWQ3IiwiZXhwIjoxODU1NDQzNTk1MTQzMzIsInRpbWVzdGFtcCI6MTc3MjQxMzUzMn0.CKONlwCIWpu1iefavi8yMniOdBsKx5rxWt7R39jfO0o"

# API 基础 URL
export QWEN_TTS_URL="http://moma.hq.cmcc/largemodel/moma/api"
```

### 3. 运行音频合成

```bash
cd presentation

# 使用默认配置合成
PRESENTATION_TTS=qwen npm run synthesize-audio

# 或指定音色（如果模型支持）
PRESENTATION_TTS=qwen PRESENTATION_TTS_VOICE=male npm run synthesize-audio
```

---

## API 规格

### 基础端点

```
Base URL: http://moma.hq.cmcc/largemodel/moma/api/v1
Audio Speech: POST /audio/speech
Models List:  GET  /v1/models
```

### 认证方式

```http
Authorization: Bearer <JWT_TOKEN>
```

JWT Token 格式：
```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoiNjgwNzE4YzcxZDdhMTQzNmY3ZTRkOWQ3IiwiZXhwIjoxODU1NDQzNTk1MTQzMzIsInRpbWVzdGFtcCI6MTc3MjQxMzUzMn0.CKONlwCIWpu1iefavi8yMniOdBsKx5rxWt7R39jfO0o
```

**Token 有效期**：约 10 年（exp: 185544359514332）

### 请求格式

```bash
curl -X POST "http://moma.hq.cmcc/largemodel/moma/api/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $QWEN_TTS_API_KEY" \
  -H "Accept: audio/mpeg" \
  -d '{
    "model": "qwen/qwen3-tts",
    "input": "你好，这是测试文本",
    "response_format": "mp3"
  }'
```

### 响应格式

- **成功**：直接返回 MP3 音频二进制流
  - Content-Type: `audio/mpeg`
  - 格式：MPEG ADTS, layer III, 64 kbps, 24 kHz, Monaural

- **失败**：返回 JSON 错误
  ```json
  {
    "error": {
      "code": 401,
      "message": "请求携带的 API Key 校验未通过"
    }
  }
  ```

---

## Provider 脚本实现

### qwen.sh（完整实现）

```bash
#!/usr/bin/env bash
# Qwen3-TTS provider for presentation audio synthesis

QWEN_BASE_URL="${QWEN_TTS_URL:-http://moma.hq.cmcc/largemodel/moma/api/v1}"
QWEN_API_KEY="${QWEN_TTS_API_KEY:-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoiNjgwNzE4YzcxZDdhMTQzNmY3ZTRkOWQ3IiwiZXhwIjoxODU1NDQzNTk1MTQzMzIsInRpbWVzdGFtcCI6MTc3MjQxMzUzMn0.CKONlwCIWpu1iefavi8yMniOdBsKx5rxWt7R39jfO0o}"

# 校验函数（可选）
tts_check() {
  if ! command -v curl >/dev/null; then
    echo "✗ curl not found." >&2
    return 1
  fi
  if ! command -v jq >/dev/null; then
    echo "✗ jq not found." >&2
    return 1
  fi
  return 0
}

# 安装帮助（可选）
tts_install_help() {
  cat <<'EOF' >&2
Qwen3-TTS provider配置说明：

Required:
  • curl      - HTTP 客户端
  • jq        - JSON 处理器
  • API Key   - JWT Token (Contact admin for new token)

Optional:
  export QWEN_TTS_URL=http://moma.hq.cmcc/largemodel/moma/api/v1
  export QWEN_TTS_API_KEY=your-jwt-token

Run:
  PRESENTATION_TTS=qwen npm run synthesize-audio
EOF
}

# 核心合成函数
tts_synthesize() {
  local text="$1"
  local out="$2"
  local voice="${3:-}"  # 保留参数，虽然当前模型不强制使用

  # 构建 JSON payload
  local payload
  payload=$(jq -n \
    --arg model "qwen/qwen3-tts" \
    --arg input "$text" \
    '{model: $model, input: $input, response_format: "mp3"}')

  # 调用 API
  curl -s -o "$out" -X POST "$QWEN_BASE_URL/audio/speech" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $QWEN_API_KEY" \
    -H "Accept: audio/mpeg" \
    -d "$payload" 2>/dev/null

  # 验证输出（非空且非 JSON 错误）
  if [[ -s "$out" ]] && ! file "$out" | grep -q JSON; then
    return 0
  else
    rm -f "$out" 2>/dev/null
    return 1
  fi
}
```

---

## 集成到 web-video-presentation Skill

### 步骤 1：添加 Provider 脚本

将 `qwen.sh` 保存到：
```
.claude/skills/web-video-presentation/templates/scripts/tts-providers/qwen.sh
```

### 步骤 2：更新技能文档

在 `references/AUDIO.md` 的"内置 provider"表格中添加：

| Provider | 默认 | 何时用 |
|---|---|---|
| `qwen` | ✓ | 内部部署推荐，中文音质优秀，免费额度充足 |

### 步骤 3：更新脚手架

在 `scaffold.sh` 中复制 provider 脚本：
```bash
# Copy Qwen provider
cp "$SKILL_DIR/scripts/tts-providers/qwen.sh" \
   "$OUT_DIR/scripts/tts-providers/"
```

---

## 故障排查

### 问题 1：全部段失败 (FAILED)

**可能原因**：API Key 过期或无效

**解决方案**：
```bash
# 检查 Token 是否有效
curl -X GET "http://moma.hq.cmcc/largemodel/moma/api/v1/models" \
  -H "Authorization: Bearer $QWEN_TTS_API_KEY" | jq '.'

# 如果返回 401，需要联系管理员刷新 Token
```

### 问题 2：部分段失败

**可能原因**：网络波动或 API 限流

**解决方案**：直接重试（自动跳过已有文件）
```bash
PRESENTATION_TTS=qwen npm run synthesize-audio
```

或强制重新合成：
```bash
PRESENTATION_TTS=qwen npm run synthesize-audio -- --force
```

### 问题 3：输出是 JSON 错误而非 MP3

**检查错误响应**：
```bash
# 手动测试单段
curl -X POST "http://moma.hq.cmcc/largemodel/moma/api/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $QWEN_TTS_API_KEY" \
  -d '{"model":"qwen/qwen3-tts","input":"测试","response_format":"mp3"}' | jq '.'
```

常见错误码：
- `401` - API Key 无效
- `400` - 请求格式错误
- `500` - 模型推理错误（重试即可）

---

## 性能基准

**测试环境**：内部网络，moma.hq.cmcc 直连

| 指标 | 数值 |
|------|------|
| 单段平均耗时 | 10-22 秒 |
| 25 段总耗时 | ~6-8 分钟 |
| 输出 bitrate | 64 kbps |
| 输出采样率 | 24 kHz |
| 声道 | Monaural（单声道） |

**成本估算**（按官方定价 ¥0.0008/字）：
- 25 段 × 平均 50 字 ≈ 1250 字
- 总成本 ≈ ¥1.00（1 元人民币）

---

## 可替代部署方案

### 方案 A：本地部署 Qwen3-TTS

```bash
# 使用 vLLM 部署
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-TTS \
  --host 0.0.0.0 \
  --port 8000

# 配置 Provider
export QWEN_TTS_URL="http://localhost:8000/v1"
export QWEN_TTS_API_KEY="not-needed"
PRESENTATION_TTS=qwen npm run synthesize-audio
```

### 方案 B：使用其他 TTS Provider

如果 Qwen3-TTS 不可用，可切换到：

```bash
# MiniMax（中文音质最佳）
PRESENTATION_TTS=minimax PRESENTATION_TTS_VOICE=male-alt npm run synthesize-audio

# Edge TTS（免费，微软语音）
PRESENTATION_TTS=edge-tts PRESENTATION_TTS_VOICE=zh-CN-YunxiNeural npm run synthesize-audio

# macOS say（离线，免费，音质一般）
PRESENTATION_TTS=say npm run synthesize-audio
```

---

## 联系支持

- **API 问题**：联系 moma.hq.cmcc 管理员
- **Token 刷新**：重新生成 JWT Token 并更新环境变量
- **模型问题**：参考 Qwen 官方文档 https://help.aliyun.com/zh/qwen

---

## 更新日志

| 版本 | 日期 | 备注 |
|------|------|------|
| 1.0 | 2026-09-10 | 初始版本，支持 moma.hq.cmcc 部署 |
| | | |

**最后更新**：2026-09-10
