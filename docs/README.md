# AI 能力图谱教程文档

> 系统化 AI 能力体系：从数学基础到大模型应用开发、智能体工程与企业落地。

## 📚 核心资源（先看这里）

| 资源 | 说明 |
|------|------|
| [**术语表（概念释义大全）**](GLOSSARY.md) | 400+ 核心术语，按领域分组，认证/面试/查漏速查 |
| [**认证备考指南**](certification/README.md) | 阿里云 ACP / AWS AI Practitioner 等认证地图 + 备考方法 |
| [**模拟试卷 A（80 题含详解）**](certification/mock-exam-a.md) | 仿真选择题，覆盖全部章节 |
| [**番外篇**](special-topics/README.md) | 评测打榜、面试、选型、创业、智能体平台、推理模型、前沿年报 |

## 目录结构

```
docs/
├── GLOSSARY.md              # 术语表（400+ 概念释义）
├── chapter-1/               # 第 1 章：AI 基础理论（数学/ML/DL/伦理）
├── chapter-2/               # 第 2 章：大模型核心技术（架构/预训练/PEFT/多模态）
├── chapter-3/               # 第 3 章：大模型工程化（分布式/压缩/推理/数据/MLOps/评估）
├── chapter-4/               # 第 4 章：大模型应用开发（API/Prompt/RAG/Agent/前端/后端）
├── chapter-5/               # 第 5 章：AI 产品与业务（需求/设计/落地/项目管理/交付/行业）
├── chapter-6/               # 第 6 章：算法工程能力（Python/工程化/测试/部署）
├── chapter-7/               # 第 7 章：AI 资源与成本（算力/容量成本/GPU 调度）
├── certification/           # 认证备考指南与模拟题库
└── special-topics/          # 番外篇（实战专题）
```

> 每章含 `README.md`（章导引）、主文档与 `*-keypoints.md`（考点/速览补充）。

## 学习路径建议

### 初学者路径
第 1 章（基础理论）→ 第 4 章（应用开发）→ 第 2 章（大模型技术）

### 工程师路径
第 1 章 → 第 2 章 → 第 3 章（工程化）→ 第 4 章 → 第 6 章

### 产品经理路径
第 1 章（概览）→ 第 4 章（应用）→ 第 5 章（产品与业务）→ 第 7 章（成本）

### 认证备考路径（ACP/AWS）
[术语表](GLOSSARY.md) → 按 [认证指南](certification/README.md) 考点分布精读第 2–4 章 → [模拟试卷 A](certification/mock-exam-a.md) → 错题回术语表复盘

### 前沿/架构师路径
[Special-06 智能体平台架构](special-topics/special-06-agent-platform-architecture.md) → [Special-07 推理模型](special-topics/special-07-reasoning-models.md) → [Special-08 前沿年报](special-topics/special-08-frontier-2026.md) → 第 3、7 章

## 番外篇速览

| 编号 | 主题 |
|------|------|
| Special-01 | [算法工程师打榜指南](special-topics/special-01-benchmark-competition.md) |
| Special-02 | [算法工程师面试指南](special-topics/special-02-interview-guide.md) |
| Special-03 | [AI 技术选型决策手册](special-topics/special-03-tech-selection.md) |
| Special-04 | [大模型创业实战指南](special-topics/special-04-llm-startup.md) |
| Special-05 | [Codex Skills 工程实践记录](special-topics/special-05-codex-skills-practice.md) |
| Special-06 | [智能体平台从零到一：系统架构与模型服务平台](special-topics/special-06-agent-platform-architecture.md) |
| Special-07 | [推理模型与测试时计算](special-topics/special-07-reasoning-models.md) |
| Special-08 | [大模型前沿技术年报（2025–2026）](special-topics/special-08-frontier-2026.md) |
| Special-09 | [核心概念图解：20 张图讲透大模型](special-topics/special-09-core-concepts-visualized.md) |
| Special-10 | [TTS Provider 实战：Qwen3-TTS 音频合成](special-topics/tts-providers/qwen3-tts.md) |
| Special-11 | [Qoder 使用指南](special-topics/special-11-qoder-usage-guide.md) |

## 配套资源

- **代码示例**: `examples/` 目录
- **认证参考课程**: `Aliyun-acp/` 目录（阿里云 ACP 官方教程镜像）

---

[← 返回主目录](../README.md)
