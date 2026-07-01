# Claude Code 配置备份

**备份时间**: 2026-07-01 17:03:59
**备份原因**: 优化上下文响应速度 - 移除低频 Skill 前备份

## 备份内容

| 目录/文件 | 说明 |
|----------|------|
| `skills/` | 所有自定义 Skills |
| `agents/` | Agent 配置文件 |
| `hooks/` | Hook 脚本和配置 |
| `commands/` | 自定义命令 |
| `settings.json` | 主配置文件 |
| `settings.local.json` | 本地配置 |

## 待移除的低频项目

以下 Skill 和 Agent 被识别为使用频率低、可能拖慢响应速度：

### Skills (Local)
- `weather-fetcher/` - 仅用于迪拜天气查询
- `weather-svg-creator/` - 创建 SVG 天气卡片
- `time-skill/` - 显示巴基斯坦时间

### Agents
- `weather-agent.md` - 天气查询 Agent（场景单一）
- `time-agent.md` - 时间显示 Agent（功能可通过 bash 命令替代）

### Hooks
- `oh-my-claudecode` 插件的多个 hooks（建议禁用部分事件）

## 恢复方法

如需恢复备份：

```bash
# 恢复所有配置
cp -r .claude/backups/backup_20260701_170359/* .claude/

# 恢复单个目录
cp -r .claude/backups/backup_20260701_170359/skills .claude/
cp -r .claude/backups/backup_20260701_170359/agents .claude/
```

## 备份版本历史

| 备份时间 | 操作 |
|----------|------|
| 2026-07-01 17:03:59 | 首次完整备份（优化前） |
