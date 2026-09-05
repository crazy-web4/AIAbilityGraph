# 第 5.5 节 AI 项目集成交付代码

AI 项目集成与交付配套示例。

## 文件说明

| 文件 | 说明 |
|------|------|
| `acceptance_checklist.py` | Go-Live 交付检查清单：功能/效果/性能/数据/安全/运营五类核对，阻塞项未过返回退出码 1 |

## 运行方式

```bash
python acceptance_checklist.py    # 有阻塞项时退出码为 1，可用于上线评审/CI
```
