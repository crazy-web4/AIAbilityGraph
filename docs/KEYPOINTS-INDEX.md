# 关键知识点详解 - 索引目录

> 本文档汇总了所有章节的补充知识点文档，包含代码示例、实践指南、决策工具等。

---

> ℹ️ 本文聚焦各章 `*-keypoints.md` 补充文档。完整文档结构、学习路径与新增资源以 [**文档总目录 README**](README.md) 为准。
> 新增资源：[术语表（400+ 概念）](GLOSSARY.md) ｜ [认证备考与模拟题库](certification/README.md) ｜ 番外 Special-06~09。

## 📚 索引总览

| 章节 | 补充文档 | 核心主题 |
|------|-------------|---------|
| 第 1 章 | ✅ 3 篇 | 数学基础、机器学习、深度学习 |
| 第 2 章 | ✅ 4 篇 | 大模型架构、预训练、PEFT、多模态 |
| 第 3 章 | ✅ 2 篇 | 分布式训练、模型压缩 |
| 第 4 章 | ✅ 3 篇 | 提示词工程、RAG、Agent |
| 第 5 章 | ✅ 1 篇 | 需求分析、项目管理 |
| 第 6 章 | ✅ 1 篇 | 工程化、代码规范 |
| 第 7 章 | ✅ 1 篇 | 资源规划、成本估算 |
| **番外篇** | ✅ 9 篇 | 打榜/面试/选型/创业/Skills/智能体平台/推理模型/前沿年报/概念图解 |
| **术语表** | ✅ [GLOSSARY.md](GLOSSARY.md) | 400+ 核心概念释义 |
| **认证** | ✅ [certification/](certification/README.md) | ACP/AWS 备考指南 + 模拟卷 A/B（120 题） |

---

## 🔍 详细索引

### 第 1 章：AI 基础理论

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [1.1 数学基础知识点](chapter-1/1-1-math-keypoints.md) | • 线性代数应用 (PCA/SVD)<br>• 概率分布速查表<br>• 贝叶斯优化实战 | • 特征值分解示例<br>• 贝叶斯优化代码 |
| [1.2 机器学习知识点](chapter-1/1-2-ml-keypoints.md) | • 监督学习算法对比<br>• XGBoost 实战<br>• K-Means 聚类<br>• 评估指标可视化 | • XGBoost 完整流程<br>• K-Means 聚类分析<br>• 分类评估可视化 |
| [1.3 深度学习知识点](chapter-1/1-3-dl-keypoints.md) | • 激活函数对比<br>• 优化器选择<br>• 训练技巧模板 | • 激活函数可视化<br>• 梯度流动测试<br>• 优化器对比实验 |

---

### 第 2 章：大模型核心技术

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [2.1 架构设计知识点](chapter-2/2-1-architecture-keypoints.md) | • Transformer 架构变体对比<br>• RoPE/RMSNorm/SwiGLU 详解<br>• 架构选择决策树 | • 三种架构实现对比<br>• RoPE 代码实现<br>• SwiGLU vs ReLU 对比 |
| [2.2 预训练知识点](chapter-2/2-2-pretraining-keypoints.md) | • 数据准备清单<br>• 分布式训练配置<br>• 故障排查指南 | • 数据清洗管道<br>• DeepSpeed 训练脚本<br>• Loss 监控脚本 |
| [2.3 PEFT 知识点](chapter-2/2-3-peft-keypoints.md) | • LoRA/QLoRA/P-Tuning详解<br>• PEFT方法选择指南<br>• 4bit 量化微调 | • LoRA 完整实现<br>• QLoRA 微调脚本<br>• P-Tuning v2 实现 |
| [2.4 多模态知识点](chapter-2/2-4-multimodal-keypoints.md) | • CLIP/LLaVA 架构解析<br>• 多模态对齐技术<br>• 零样本分类 | • CLIP 实现<br>• LLaVA 微调代码<br>• 图文检索系统 |

---

### 第 3 章：大模型工程化

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [3.1 分布式训练知识点](chapter-3/3-1-distributed-keypoints.md) | • 并行策略决策树<br>• ZeRO 配置详解<br>• 显存需求计算器<br>• 通信优化技术 | • DeepSpeed ZeRO-3 配置<br>• 显存计算工具<br>• NCCL 优化参数 |
| [3.2 模型压缩知识点](chapter-3/3-2-compression-keypoints.md) | • 剪枝/蒸馏/量化对比<br>• 结构化 vs 非结构化剪枝<br>• 知识蒸馏实战 | • 幅值剪枝实现<br>• 蒸馏训练器<br>• 迭代剪枝流程 |

---

### 第 4 章：大模型应用开发

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [4.2 提示词工程知识点](chapter-4/4-2-prompt-engineering-keypoints.md) | • 提示词模式库<br>• CoT/Few-shot/HyDE详解<br>• ReAct 模式实现 | • Chain-of-Thought 生成器<br>• Few-shot 示例选择器<br>• ReAct Agent 实现 |
| [4.3 RAG 知识点](chapter-4/4-3-rag-keypoints.md) | • RAG 架构选择器<br>• 向量数据库对比<br>• 检索优化技巧<br>• 重排序实现 | • Chroma 快速入门<br>• HyDE 检索实现<br>• Cross-Encoder 重排序<br>• 完整 RAG 系统 |
| [4.4 Agent 知识点](chapter-4/4-4-agent-keypoints.md) | • Function Calling详解<br>• ReAct Agent实现<br>• 多 Agent 协作 | • 工具注册系统<br>• ReAct 循环实现<br>• CrewAI示例 |

---

### 第 5 章：AI 产品与业务

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [5.1 需求分析知识点](chapter-5/5-1-requirements-keypoints.md) | • 可行性评估框架<br>• 需求调研 Checklist<br>• 项目分类器<br>• 资源估算模板 | • 项目估算计算器<br>• 可行性评估表 |

---

### 第 6 章：算法工程能力

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [6.2 工程化知识点](chapter-6/6-2-engineering-keypoints.md) | • 代码规范模板<br>• 性能优化技巧<br>• 重构检查清单 | • 项目目录结构<br>• PyTorch 性能优化<br>• Trainer 类重构示例 |

---

### 第 7 章：AI 资源与成本

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [7.1 资源规划知识点](chapter-7/7-1-resources-keypoints.md) | • GPU 选型决策树<br>• GPU 规格对比表<br>• 显存需求计算器<br>• 云 GPU 价格对比 | • 显存计算器<br>• 成本对比表 |

---

### 番外篇：实战专题

| 文档 | 核心内容 | 代码示例 |
|------|---------|---------|
| [Special-01 打榜指南](special-topics/special-01-benchmark-competition.md) | • 主流评测榜单全景图<br>• 打榜完整流程<br>• 提分 Trick 合集<br>• 避坑指南 | • Prompt 工程提分代码<br>• 数据增强策略<br>• 集成学习方法<br>• 训练技巧模板 |

---

## 🛠️ 实用工具汇总

### 计算器类

| 工具 | 位置 | 功能 |
|------|------|------|
| 显存需求计算器 | [7.1](chapter-7/7-1-resources-keypoints.md) | 计算不同训练模式的显存需求 |
| 项目资源估算器 | [5.1](chapter-5/5-1-requirements-keypoints.md) | 估算时间、人力、GPU 成本 |

### 配置模板类

| 模板 | 位置 | 用途 |
|------|------|------|
| DeepSpeed ZeRO-3 配置 | [3.1](chapter-3/3-1-distributed-keypoints.md) | 大模型训练配置 |
| CLIP 模型实现 | [2.4](chapter-2/2-4-multimodal-keypoints.md) | 多模态学习参考 |
| LLaVA 微调脚本 | [2.4](chapter-2/2-4-multimodal-keypoints.md) | 多模态指令微调 |
| RAG 系统完整实现 | [4.3](chapter-4/4-3-rag-keypoints.md) | 检索增强生成系统 |
| Trainer 类模板 | [6.2](chapter-6/6-2-engineering-keypoints.md) | 训练代码组织参考 |

### 决策树/检查清单

| 名称 | 位置 |
|------|------|
| 架构选择决策树 | [2.1](chapter-2/2-1-architecture-keypoints.md) |
| 并行策略决策树 | [3.1](chapter-3/3-1-distributed-keypoints.md) |
| RAG 架构决策树 | [4.3](chapter-4/4-3-rag-keypoints.md) |
| GPU 选型决策树 | [7.1](chapter-7/7-1-resources-keypoints.md) |
| 提示词模式库 | [4.2](chapter-4/4-2-prompt-engineering-keypoints.md) |
| 需求调研 Checklist | [5.1](chapter-5/5-1-requirements-keypoints.md) |
| 代码重构 Checklist | [6.2](chapter-6/6-2-engineering-keypoints.md) |

---

## 📝 待补充章节

以下章节的关键知识点文档待创建：

- [ ] 第 1 章：AI 基础理论 (数学、ML、DL、伦理)
- [ ] 第 2 章：2.3 参数高效微调 (LoRA/QLoRA 详解)
- [ ] 第 3 章：3.2-3.6 (压缩、推理、数据、MLOps、评估)
- [ ] 第 4 章：4.1/4.4/4.5/4.6 (API、Agent、前后端)
- [ ] 第 5 章：5.2/5.3 (产品设计、业务落地)
- [ ] 第 6 章：6.1/6.3/6.4 (Python 进阶、测试、部署)
- [ ] 第 7 章：7.2/7.3 (成本优化、资源调度)

---

## 🔗 快速链接

### 按主题分类

**模型架构**
- [2.1 架构设计](chapter-2/2-1-architecture-keypoints.md) - Transformer 变体、RoPE、RMSNorm
- [2.4 多模态](chapter-2/2-4-multimodal-keypoints.md) - CLIP、LLaVA

**训练技术**
- [2.2 预训练](chapter-2/2-2-pretraining-keypoints.md) - 数据准备、分布式配置
- [3.1 分布式训练](chapter-3/3-1-distributed-keypoints.md) - ZeRO、并行策略

**应用开发**
- [4.2 提示词工程](chapter-4/4-2-prompt-engineering-keypoints.md) - CoT、Few-shot、ReAct
- [4.3 RAG](chapter-4/4-3-rag-keypoints.md) - 检索、重排序、向量数据库

**工程化**
- [6.2 工程化](chapter-6/6-2-engineering-keypoints.md) - 代码规范、性能优化
- [7.1 资源规划](chapter-7/7-1-resources-keypoints.md) - GPU 选型、成本计算

**产品管理**
- [5.1 需求分析](chapter-5/5-1-requirements-keypoints.md) - 可行性评估、项目估算

**番外篇**
- [Special-01 打榜指南](special-topics/special-01-benchmark-competition.md) - 评测榜单竞赛完整方法论

---

## 📖 使用建议

1. **学习路径**: 先阅读主文档，遇到需要深入理解的概念时查阅对应的知识点文档
2. **代码参考**: 每个知识点文档都包含可运行的代码示例，可直接复用或修改
3. **决策辅助**: 遇到选型问题时，查找对应的决策树文档
4. **实践练习**: 每个文档末尾都有练习题，建议动手实践

---

<div align="center">

**📌 提示**: 本文档持续更新，欢迎贡献更多知识点！

[返回主目录](../README.md) | [项目进度](../PROJECT_STATUS.md)

</div>
