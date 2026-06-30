# 第 2 章：大模型核心技术

> 深入解析大模型的核心技术体系，从架构设计到高效微调。

## 章节目标

学完本章后，你将能够：

- 理解 Transformer 架构的演变与大模型设计原则
- 掌握预训练与微调的完整流程
- 运用参数高效微调技术（LoRA、QLoRA）
- 了解多模态大模型的技术原理

## 章节结构

| 小节 | 主题 | 核心内容 | 难度 |
|------|------|----------|------|
| [2.1 大模型架构设计](2-1-architecture.md) | GPT/BERT/Llama 架构对比 | Transformer 演变、注意力机制优化 | ⭐⭐⭐⭐ |
| [2.2 预训练与微调技术](2-2-pretraining.md) | 全流程预训练实践 | 数据准备、分布式训练、检查点管理 | ⭐⭐⭐⭐⭐ |
| [2.3 参数高效微调](2-3-peft.md) | LoRA/QLoRA/P-Tuning | 低秩适配、prompt tuning、Prefix LM | ⭐⭐⭐⭐ |
| [2.4 多模态大模型](2-4-multimodal.md) | CLIP/LLaVA/多模态融合 | 跨模态对齐、视觉语言模型 | ⭐⭐⭐⭐⭐ |

## 前置知识

- 第 1 章：深度学习基础（特别是 Transformer）
- PyTorch 熟练运用
- 了解分布式训练基础概念

## 配套资源

- 📁 代码示例：`examples/2-*/`
- 🤗 HuggingFace Transformers 库
- 📊 预训练模型：HuggingFace Model Hub

---

[← 第 1 章：AI 基础理论](../chapter-1/README.md) | [第 3 章：大模型工程化 →](../chapter-3/README.md)
