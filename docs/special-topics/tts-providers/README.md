# TTS Providers 文档索引

本目录收录文本转语音（TTS）Provider 相关技术文档，主要用于 web-video-presentation skill 的音频合成。

## 文档列表

| 文档 | 说明 | 阅读时间 |
|------|------|----------|
| [Qwen3-TTS 使用指南](qwen3-tts.md) | 通义千问语音合成完整指南：API 规格、音色列表、语速调节、故障排查 | 20 分钟 |

## TTS Provider 架构

web-video-presentation skill 采用 **Provider-agnostic** 架构：

```
scripts/synthesize-audio.sh    # _provider-agnostic runner_
scripts/tts-providers/
├── README.md                  # 三函数契约 + 内置 provider 说明
├── minimax.sh                 # 默认 provider（中文口播质量稳）
├── openai.sh                  # OpenAI Audio Speech API
└── qwen.sh                    # Qwen3-TTS (moma.hq.cmcc 内部部署)
```

### 三函数契约

每个 Provider 是一个 `.sh` 文件，需实现：

| 函数 | 必需 | 说明 |
|------|------|------|
| `tts_synthesize <text> <out_path> [voice]` | ✓ | 核心合成函数 |
| `tts_check` | ✗ | 启动前校验环境 |
| `tts_install_help` | ✗ | 失败时打印如何修复 |

### 内置 Provider 对比

| Provider | 后端 | 鉴权方式 | 适用场景 |
|----------|------|----------|----------|
| `minimax` | MiniMax mmx CLI | `mmx auth login --api-key` | **默认**；中文口播质量稳 |
| `openai` | OpenAI Audio API | `OPENAI_API_KEY` env var | curl-based；多数已有 key |
| `qwen` | Qwen3-TTS (moma) | JWT Token (Authorization) | **内部部署推荐**；中文音质优秀 |

## 快速开始

```bash
cd presentation

# 使用默认 provider（minimax）
npm run synthesize-audio

# 使用 Qwen3-TTS
PRESENTATION_TTS=qwen npm run synthesize-audio

# 使用 OpenAI
PRESENTATION_TTS=openai npm run synthesize-audio

# 指定音色
PRESENTATION_TTS_VOICE=zh-CN-YunxiNeural npm run synthesize-audio

# 强制重新合成全部
npm run synthesize-audio -- --force
```

## 相关文档

- [AUDIO.md](../../chapter-4/4-4-audio-synthesis.md) - 音频合成完整流程
- [web-video-presentation Skill](../../../../.claude/skills/web-video-presentation/SKILL.md) - 技能主文档

---

[← 返回番外篇目录](../README.md)
