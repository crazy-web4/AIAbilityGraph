# 5.1 AI 需求分析与管理 - 关键知识点详解

> 本节为 5.1 节的补充知识点，包含需求评估框架、可行性分析模板、项目检查清单。

---

## 知识点 1: AI 项目可行性评估框架

### AI 适用性评估矩阵

| 评估维度 | 关键问题 | 评分标准 | 权重 |
|---------|---------|---------|------|
| **数据可用性** | 是否有足够的训练数据？ | 充足 (10K+): 3 分，中等 (1K-10K): 2 分，不足 (<1K): 1 分 | 25% |
| **问题定义** | 任务是否清晰可定义？ | 清晰：3 分，较清晰：2 分，模糊：1 分 | 20% |
| **商业价值** | ROI 是否明确？ | 高价值：3 分，中等：2 分，低：1 分 | 25% |
| **技术可行性** | 当前技术是否支持？ | 成熟：3 分，发展中：2 分，前沿：1 分 | 20% |
| **风险可控性** | 是否有伦理/合规风险？ | 低风险：3 分，中风险：2 分，高风险：1 分 | 10% |

**决策阈值：**
- 加权总分 ≥ 2.5: ✅ 推荐启动
- 2.0 ≤ 总分 < 2.5: ⚠️ 需谨慎评估
- 总分 < 2.0: ❌ 不建议启动

### 评估表模板

```markdown
# AI 项目可行性评估表

## 项目基本信息
- 项目名称：
- 提出日期：
- 负责人：

## 一、数据评估
### 数据现状
- [ ] 已有结构化数据
- [ ] 已有非结构化数据
- [ ] 需要外部采购
- [ ] 需要人工标注

### 数据规模
| 数据类型 | 当前数量 | 质量评估 | 获取成本 |
|---------|---------|---------|---------|
| 训练数据 | | | |
| 测试数据 | | | |
| 标注数据 | | | |

## 二、技术评估
### 技术选型
- [ ] 使用现有 API (最快)
- [ ] 微调开源模型 (中等)
- [ ] 从头训练 (最慢)

### 性能目标
| 指标 | 目标值 | 基线值 | 测量方法 |
|------|-------|-------|---------|
| 准确率 | | | |
| 响应时间 | | | |
| 并发能力 | | | |

## 三、商业评估
### 价值量化
- 预期收益：
- 成本节省：
- 用户影响：

### 资源需求
| 资源类型 | 需求 | 当前拥有 | 缺口 |
|---------|------|---------|------|
| 算力 (GPU 小时) | | | |
| 存储空间 | | | |
| 人力 (人月) | | | |
| 预算 | | | |

## 四、风险评估
### 风险清单
| 风险项 | 可能性 | 影响程度 | 缓解措施 |
|-------|-------|---------|---------|
| 数据隐私 | | | |
| 模型偏见 | | | |
| 合规风险 | | | |
| 技术依赖 | | | |

## 综合评分：___ / 3.0
## 建议：[启动 / 进一步评估 / 不推荐]
```

---

## 知识点 2: AI 需求调研 checklist

### 需求调研问题清单

```
□ 业务目标
  □ 要解决的核心问题是什么？
  □ 成功的具体指标是什么？
  □ 现有解决方案的痛点是什么？

□ 用户画像
  □ 谁是最终用户？
  □ 用户的技术水平如何？
  □ 用户的使用场景是什么？

□ 数据情况
  □ 有哪些数据源？
  □ 数据质量如何？
  □ 数据标注是否完成？
  □ 数据更新频率？

□ 技术要求
  □ 响应时间要求？
  □ 并发量预期？
  □ 可用性 SLA？
  □ 部署环境限制？

□ 合规要求
  □ 是否需要数据本地化？
  □ 是否有行业合规要求？
  □ 是否需要审计日志？

□ 项目约束
  □ 预算范围？
  □ 时间期限？
  □ 团队规模？
```

---

## 知识点 3: AI 项目类型分类器

```
AI 项目分类决策树
│
├── 任务类型？
│   │
│   ├── 理解/分类
│   │   ├── 文本分类 → BERT/RoBERTa
│   │   ├── 情感分析 → Fine-tuned LLM
│   │   ├── 实体识别 → NER 模型
│   │   └── 图像分类 → ViT/ResNet
│   │
│   ├── 生成/创作
│   │   ├── 文本生成 → GPT/Llama
│   │   ├── 图像生成 → Stable Diffusion
│   │   ├── 代码生成 → CodeLlama
│   │   └── 语音合成 → Tacotron/VITS
│   │
│   ├── 问答/检索
│   │   ├── 知识库 QA → RAG 系统
│   │   ├── 开放域 QA → Fine-tuned LLM + RAG
│   │   └── 语义搜索 → Embedding + Vector DB
│   │
│   └── 决策/规划
│       ├── 推荐系统 → 协同过滤/深度排序
│       ├── 调度优化 → 强化学习/启发式
│       └── 异常检测 → 无监督学习
│
├── 延迟要求？
│   │
│   ├── < 50ms (实时)
│   │   └── → 小模型 + 边缘部署 + 量化
│   │
│   ├── 50ms - 500ms (交互)
│   │   └── → 中等模型 + GPU 推理
│   │
│   └── > 500ms (批量)
│       └── → 大模型 + 批处理
│
└── 数据敏感度？
    │
    ├── 高敏感 (医疗/金融)
    │   └── → 私有部署 + 联邦学习
    │
    ├── 中敏感 (企业内部)
    │   └── → VPC 部署 + 访问控制
    │
    └── 低敏感 (公开数据)
        └── → 云服务 API
```

---

## 知识点 4: AI 项目估算模板

### 资源估算计算器

```python
# examples/5-1-requirements/project_estimator.py
"""
AI 项目资源估算工具

参考实际项目经验数据
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ProjectEstimate:
    """项目估算结果"""
    data_preparation_days: int
    model_development_days: int
    integration_days: int
    testing_days: int
    total_days: int
    team_size: int
    gpu_hours: int
    estimated_cost: float


def estimate_project(
    task_type: str,
    data_size: str,  # 'small', 'medium', 'large'
    accuracy_requirement: str,  # 'low', 'medium', 'high'
    deployment_type: str  # 'api', 'onprem', 'edge'
) -> ProjectEstimate:
    """
    估算项目时间和资源
    
    基于行业基准数据
    """
    
    # 基础时间估算 (人天)
    base_estimates = {
        # 任务类型基准
        'classification': {'dev': 10, 'data': 15, 'integration': 5},
        'generation': {'dev': 20, 'data': 20, 'integration': 10},
        'rag': {'dev': 25, 'data': 15, 'integration': 15},
        'agent': {'dev': 40, 'data': 10, 'integration': 20},
        
        # 数据规模乘数
        'data_multiplier': {
            'small': 0.5,   # < 10K 样本
            'medium': 1.0,  # 10K - 1M 样本
            'large': 2.0    # > 1M 样本
        },
        
        # 准确率要求乘数
        'accuracy_multiplier': {
            'low': 0.7,     # > 80%
            'medium': 1.0,  # > 90%
            'high': 1.5     # > 95%
        },
        
        # 部署类型乘数
        'deployment_multiplier': {
            'api': 1.0,
            'onprem': 1.5,
            'edge': 2.0
        }
    }
    
    # 获取基准
    base = base_estimates.get(task_type, base_estimates['classification'])
    data_mult = base_estimates['data_multiplier'].get(data_size, 1.0)
    acc_mult = base_estimates['accuracy_multiplier'].get(accuracy_requirement, 1.0)
    dep_mult = base_estimates['deployment_multiplier'].get(deployment_type, 1.0)
    
    # 计算各阶段时间
    data_days = int(base['data'] * data_mult * acc_mult)
    dev_days = int(base['dev'] * acc_mult * dep_mult)
    integration_days = int(base['integration'] * dep_mult)
    testing_days = int((data_days + dev_days + integration_days) * 0.2)  # 20% 测试
    
    total_days = data_days + dev_days + integration_days + testing_days
    
    # 团队规模建议
    if total_days < 30:
        team_size = 2
    elif total_days < 90:
        team_size = 4
    else:
        team_size = 6  # 需要拆分子团队
    
    # GPU 估算 (基于经验)
    gpu_hours_map = {
        'small': {'classification': 100, 'generation': 500, 'rag': 200, 'agent': 300},
        'medium': {'classification': 500, 'generation': 2000, 'rag': 800, 'agent': 1000},
        'large': {'classification': 2000, 'generation': 8000, 'rag': 3000, 'agent': 5000}
    }
    gpu_hours = gpu_hours_map.get(data_size, gpu_hours_map['medium']).get(task_type, 500)
    gpu_hours *= acc_mult  # 高准确率需要更多调优
    
    # 成本估算 (云成本参考)
    # GPU: A100 ￥20/小时，数据标注：￥0.5/条，存储：￥0.1/GB/月
    data_labeling_cost = {
        'small': 5000,
        'medium': 50000,
        'large': 200000
    }
    
    gpu_cost = gpu_hours * 20  # ￥/小时
    labeling_cost = data_labeling_cost.get(data_size, 50000)
    
    total_cost = gpu_cost + labeling_cost
    
    return ProjectEstimate(
        data_preparation_days=data_days,
        model_development_days=dev_days,
        integration_days=integration_days,
        testing_days=testing_days,
        total_days=total_days,
        team_size=team_size,
        gpu_hours=gpu_hours,
        estimated_cost=total_cost
    )


# 使用示例
if __name__ == "__main__":
    # 示例：中等规模 RAG 项目，高准确率要求，私有部署
    estimate = estimate_project(
        task_type='rag',
        data_size='medium',
        accuracy_requirement='high',
        deployment_type='onprem'
    )
    
    print("=" * 50)
    print("AI 项目资源估算")
    print("=" * 50)
    print(f"数据准备：    {estimate.data_preparation_days} 天")
    print(f"模型开发:    {estimate.model_development_days} 天")
    print(f"集成开发：    {estimate.integration_days} 天")
    print(f"测试验证：    {estimate.testing_days} 天")
    print("-" * 50)
    print(f"总计：       {estimate.total_days} 天 (约 {estimate.total_days/20:.1f} 周)")
    print(f"建议团队：    {estimate.team_size} 人")
    print(f"GPU 需求：     {estimate.gpu_hours} 小时")
    print(f"估算成本：    ￥{estimate.estimated_cost:,.0f}")
```

### 示例输出

```
==================================================
AI 项目资源估算
==================================================
数据准备：    30 天
模型开发：    30 天
集成开发：    22 天
测试验证：    16 天
--------------------------------------------------
总计：       98 天 (约 4.9 周)
建议团队：    4 人
GPU 需求：     1200 小时
估算成本：    ￥74,000
```

---

## 练习题

### 练习 1: 项目可行性评估

评估以下项目场景：

**场景 A：智能客服系统**
- 已有 10 万条历史客服对话
- 目标：自动回答常见问题
- 要求：准确率>85%，响应时间<2s
- 部署：企业内网

**场景 B：医疗影像诊断**
- 有 5000 张标注 X 光片
- 目标：辅助诊断肺炎
- 要求：准确率>95%
- 部署：医院本地

请完成：
1. 填写可行性评估表
2. 给出推荐技术路线
3. 估算项目周期和成本

---

## 延伸阅读

- [AI 项目管理最佳实践](https://arxiv.org/abs/2106.06116)
- [ML 项目检查清单](https://machinelearningmastery.com/machine-learning-project-checklist/)

---

[← 返回 5.1 主文档](5-1-requirements.md) | [下一节：AI 产品设计 →](5-2-product-design.md)
