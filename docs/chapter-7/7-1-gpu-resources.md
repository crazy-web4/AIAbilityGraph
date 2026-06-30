# 7.1 AI 算力与资源规划

> 合理规划和选择 AI 算力资源是项目成功的关键。本章讲解 GPU 选型、采购策略和资源规划方法。

## 学习目标

- [ ] 理解 GPU 硬件参数与性能指标
- [ ] 掌握训练/推理场景的选型策略
- [ ] 学会制定资源规划方案
- [ ] 了解云服务与自建对比

---

## 7.1.1 GPU 硬件基础

### 关键参数解析

| 参数 | 说明 | 对 AI 的影响 |
|------|------|-------------|
| **CUDA 核心数** | 并行计算单元数量 | 影响 FP32/FP16 计算速度 |
| **Tensor Core** | 专用矩阵计算单元 | 影响混合精度训练速度 |
| **显存容量** | GPU 内存大小 | 决定可训练模型规模 |
| **显存带宽** | 显存数据传输速度 | 影响大模型推理吞吐 |
| **互联带宽** | GPU 间通信速度 | 影响多卡扩展效率 |

### 主流 GPU 对比

| GPU | 显存 | FP16算力 | 功耗 | 适用场景 |
|-----|------|----------|------|----------|
| **A100 40G** | 40GB | 312 TFLOPS | 400W | 训练/推理通用 |
| **A100 80G** | 80GB | 312 TFLOPS | 400W | 大模型训练 |
| **H100** | 80GB | 989 TFLOPS | 700W | 超大模型训练 |
| **A10** | 24GB | 125 TFLOPS | 150W | 推理/轻量训练 |
| **A2** | 16GB | 103 TFLOPS | 40W | 推理/转码 |
| **L4** | 24GB | 250 TFLOPS | 72W | 推理/视频 |

---

## 7.1.2 训练场景选型

### 模型规模与 GPU 需求

```
模型参数量 → 推荐配置 → 预估训练时间

7B 模型:
  - 单机 8×A100 40G
  - 全参数微调：3-5 天
  - LoRA 微调：6-12 小时

13B 模型:
  - 2 机 16×A100 80G (DP)
  - 全参数微调：1-2 周
  - LoRA 微调：1-2 天

70B 模型:
  - 8 机 64×A100 80G (3D 并行)
  - 全参数微调：4-8 周
  - LoRA 微调：3-7 天
```

### 训练集群配置计算器

```python
# examples/7-1-resources/gpu_calculator.py
"""
GPU 资源需求计算器
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class GPUConfig:
    """GPU 配置"""
    name: str
    memory_gb: int
    tflops_fp16: int
    power_w: int
    nvlink_bw: int  # GB/s

@dataclass
class ModelConfig:
    """模型配置"""
    params_b: float  # 参数量 (B)
    seq_len: int  # 序列长度
    batch_size: int  # 批次大小

# 主流 GPU
GPUS = {
    "A100-40": GPUConfig("A100-40G", 40, 312, 400, 600),
    "A100-80": GPUConfig("A100-80G", 80, 312, 400, 600),
    "H100": GPUConfig("H100", 80, 989, 700, 900),
    "A10": GPUConfig("A10", 24, 125, 150, 0),
}

def estimate_memory_requirement(model_config: ModelConfig) -> float:
    """
    估算显存需求 (GB)
    
    简化公式:
    模型权重 + 梯度 + 优化器状态 + 激活值 + 其他
    """
    params_b = model_config.params_b
    
    # FP16 权重 (2 bytes/param)
    weights_memory = params_b * 2  # GB
    
    # 梯度 (2 bytes/param)
    gradients_memory = params_b * 2  # GB
    
    # Adam 优化器状态 (8-12 bytes/param)
    optimizer_memory = params_b * 10  # GB
    
    # 激活值 (与 batch_size 和 seq_len 相关)
    activation_memory = params_b * 0.5 * model_config.batch_size / 32  # GB
    
    # 其他 (KV Cache 等)
    other_memory = params_b * 0.5
    
    total = weights_memory + gradients_memory + optimizer_memory + activation_memory + other_memory
    
    return total

def recommend_gpu(model_config: ModelConfig, training_days: float = 7) -> str:
    """
    根据训练时间要求推荐 GPU 配置
    
    返回:
        (GPU 型号，GPU 数量)
    """
    memory_needed = estimate_memory_requirement(model_config)
    
    for gpu_name, gpu_config in GPUS.items():
        if gpu_config.memory_gb > memory_needed:
            return gpu_name, 1  # 单卡可训练
    
    # 需要多卡
    if memory_needed <= 80 * 8:
        return "A100-80G", 8  # 8 卡可训练
    
    # 超大规模需要多机
    n_gpus = int(memory_needed / 80 * 1.5)  # 1.5 倍冗余
    return "A100-80G", n_gpus

# 使用示例
if __name__ == "__main__":
    model = ModelConfig(params_b=7, seq_len=4096, batch_size=32)
    
    memory = estimate_memory_requirement(model)
    print(f"7B 模型显存需求：约{memory:.1f} GB")
    
    gpu, count = recommend_gpu(model)
    print(f"推荐配置：{count}× {gpu}")
```

---

## 7.1.3 采购 vs 租赁

### 成本对比模型

```python
# examples/7-1-resources/cost_comparison.py
"""
采购 vs 租赁 成本分析
"""

def calculate_tco(
    gpu_price: float,
    n_gpus: int,
    usage_hours_per_day: float,
    electricity_cost: float,  # 元/度
    project_months: int
) -> dict:
    """
    计算总拥有成本 (TCO)
    
    参数:
        gpu_price: 单卡价格
        n_gpus: GPU 数量
        usage_hours_per_day: 每天使用小时
        electricity_cost: 电费
        project_months: 项目周期 (月)
    """
    
    # 硬件成本
    hardware_cost = gpu_price * n_gpus
    
    # 服务器/网络/存储 (约硬件 30%)
    infrastructure_cost = hardware_cost * 0.3
    
    # 电费 (GPU + 散热，按 500W/GPU 计)
    power_kwh_per_hour = n_gpus * 0.5
    electricity_per_month = power_kwh_per_hour * usage_hours_per_day * 30
    electricity_cost_total = electricity_per_month * project_months * electricity_cost
    
    # 运维成本 (约硬件 10%)
    maintenance_cost = hardware_cost * 0.1
    
    # 总成本
    total_cost = hardware_cost + infrastructure_cost + electricity_cost_total + maintenance_cost
    
    # 残值 (假设 3 年后残值 30%)
    resale_value = hardware_cost * 0.3
    
    return {
        "hardware": hardware_cost,
        "infrastructure": infrastructure_cost,
        "electricity": electricity_cost_total,
        "maintenance": maintenance_cost,
        "resale": -resale_value,
        "total": total_cost - resale_value,
        "monthly_avg": (total_cost - resale_value) / project_months
    }

def compare_cloud_rental(
    gpu_hourly_rate: float,
    n_gpus: int,
    usage_hours_per_day: float,
    months: int
) -> dict:
    """
    计算云租赁成本
    """
    hours_per_month = usage_hours_per_day * 30
    cloud_cost = gpu_hourly_rate * n_gpus * hours_per_month * months
    
    return {
        "hourly_rate": gpu_hourly_rate,
        "monthly_cost": gpu_hourly_rate * n_gpus * hours_per_month,
        "total": cloud_cost
    }

# 使用示例
if __name__ == "__main__":
    print("=" * 50)
    print("8×A100 配置，使用 12 个月对比")
    print("=" * 50)
    
    # 采购分析
    purchase = calculate_tco(
        gpu_price=100000,  # A100 单价
        n_gpus=8,
        usage_hours_per_day=8,
        electricity_cost=1.0,
        project_months=12
    )
    
    print("\n【采购方案】")
    print(f"  硬件成本：¥{purchase['hardware']:,}")
    print(f"  总拥有成本：¥{purchase['total']:,.0f}")
    print(f"  月均成本：¥{purchase['monthly_avg']:,.0f}")
    
    # 租赁分析
    rental = compare_cloud_rental(
        gpu_hourly_rate=25,  # A100 时租
        n_gpus=8,
        usage_hours_per_day=8,
        months=12
    )
    
    print("\n【租赁方案】")
    print(f"  时租单价：¥{rental['hourly_rate']}")
    print(f"  月均成本：¥{rental['monthly_cost']:,.0f}")
    print(f"  总成本：¥{rental['total']:,.0f}")
    
    # 盈亏平衡点
    print("\n【建议】")
    if purchase['total'] < rental['total']:
        print(f"  ✓ 采购更合算，节省 ¥{rental['total'] - purchase['total']:,.0f}")
    else:
        print(f"  ✓ 租赁更合算，节省 ¥{purchase['total'] - rental['total']:,.0f}")
```

### 决策矩阵

| 使用场景 | 推荐方案 | 理由 |
|----------|----------|------|
| 短期项目 (<3 月) | 云租赁 | 避免资本支出 |
| 中期项目 (3-12 月) | 混合 | 核心用自有，峰值用云 |
| 长期使用 (>12 月) | 自建 | TCO 更低 |
| 训练为主 | 自建/A100 | 独占资源，持续训练 |
| 推理为主 | 按需租赁 | 弹性伸缩 |

---

## 7.1.4 云服务商选型

### 主流云 GPU 对比

| 服务商 | GPU 型号 | 时租 (8 卡) | 特点 |
|--------|----------|-----------|------|
| **AWS** | P4d (A100) | ~$32/小时 | 全球覆盖，生态完善 |
| **Azure** | ND A100 | ~$30/小时 | 企业集成好 |
| **GCP** | A100 | ~$29/小时 | TPU 可选 |
| **阿里云** | GN7 | ¥200/小时 | 本地化好 |
| **腾讯云** | HAI | ¥180/小时 | 性价比高 |

### 成本优化技巧

```markdown
## 云 GPU 省钱指南

### 1. 使用 Spot/竞价实例
- 价格：按需的 30-70%
- 风险：可能被回收
- 策略：Checkpoint 频繁保存，断点续训

### 2. 预留实例
- 1 年期：约 40-50% 折扣
- 3 年期：约 60-70% 折扣

### 3. 自动伸缩
- 闲时降级
- 峰时扩容
- 定时启停

### 4.  spot 实例组合
- 核心训练：按需实例
- 数据处理：Spot 实例
- 推理服务：混合
```

---

## 练习题

1. **资源配置**: 为 13B 模型全参数微调设计 GPU 配置方案。

2. **成本分析**: 对比自建 8 卡 A100 集群与阿里云 3 年期的成本差异。

3. **混合策略**: 设计一个训练 + 推理混合场景的资源调度方案。

---

[← 章前导引](README.md) | [下一节：7.2 成本优化 →](7-2-cost-optimization.md)
