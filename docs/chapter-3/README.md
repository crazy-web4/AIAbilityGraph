# 第 3 章：大模型工程化

> 从实验室到生产环境，掌握大模型工程化的核心技术。

## 章节目标

学完本章后，你将能够：

- 配置和优化分布式训练系统
- 实施模型压缩与量化策略
- 部署高性能推理服务
- 构建 MLOps 流水线

## 章节结构

| 小节 | 主题 | 核心内容 | 难度 |
|------|------|----------|------|
| [3.1 分布式训练技术](3-1-distributed-training.md) | DeepSpeed/Megatron-LM | 数据并行、模型并行、ZeRO | ⭐⭐⭐⭐⭐ |
| [3.2 模型压缩与量化](3-2-compression.md) | 剪枝、蒸馏、INT4/INT8 | 后训练量化、量化感知训练 | ⭐⭐⭐⭐ |
| [3.3 推理优化技术](3-3-inference.md) | vLLM、TensorRT-LLM | KV Cache、PagedAttention | ⭐⭐⭐⭐ |
| [3.4 训练数据处理](3-4-data-processing.md) | 数据清洗、去重、质量评估 | Tokenization、数据混合 | ⭐⭐⭐ |
| [3.5 MLOps 与模型交付](3-5-mlops.md) | 模型版本管理、CI/CD、监控 | MLflow、W&B、模型注册表 | ⭐⭐⭐ |
| [3.6 模型评估与可解释性](3-6-evaluation.md) | 基准测试、可解释性工具 | HELM、LangChain Eval | ⭐⭐⭐ |

## 前置知识

- 第 2 章：大模型核心技术
- PyTorch 分布式基础
- Linux/GPU 集群基础

## 🔍 补充知识点

| 文档 | 内容 |
|------|------|
| [3.1 分布式训练详解](3-1-distributed-keypoints.md) | 并行策略决策树、ZeRO 配置详解、显存计算器、NCCL 优化 |

---

[← 第 2 章：大模型核心技术](../chapter-2/README.md) | [第 4 章：大模型应用开发 →](../chapter-4/README.md)
