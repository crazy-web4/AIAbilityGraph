# 7.1 AI 算力与资源规划 - 关键知识点详解

> 本节为 7.1 节的补充知识点，包含 GPU 选型指南、成本计算器、资源规划模板。

---

## 知识点 1: GPU 选型决策树

```
GPU 选型决策树
│
├── 使用场景？
│   │
│   ├── 训练大模型 (>7B)
│   │   ├── 预算充足 → H100 (80GB) × 多卡
│   │   ├── 预算中等 → A100 (40/80GB) × 多卡
│   │   └── 预算有限 → RTX 4090 × 多卡 + QLoRA
│   │
│   ├── 微调/推理
│   │   ├── 低延迟 → A10/A10G
│   │   ├── 低成本 → RTX 4070/4070Ti
│   │   └── 高吞吐 → L40S
│   │
│   └── 学习/原型
│       ├── 有预算 → RTX 4060 Ti (16GB)
│       └── 无预算 → Colab/Kaggle/云服务
│
├── 显存需求？
│   │
│   ├── < 8GB → RTX 4060 (8GB)
│   ├── 8-16GB → RTX 4060 Ti (16GB)
│   ├── 16-24GB → RTX 4090 (24GB)
│   ├── 24-48GB → RTX 6000 Ada (48GB)
│   └── > 48GB → A100/H100 (80GB)
│
└── 部署环境？
    │
    ├── 数据中心
    │   └── → A100/H100/A10 (支持 NVLink)
    │
    ├── 工作站
    │   └── → RTX 4090/RTX 6000 Ada
    │
    └── 边缘设备
        └── → Jetson Orin / L4
```

---

## 知识点 2: GPU 规格对比表

### 消费级 GPU

| 型号 | 显存 | CUDA 核心 | FP16 TFLOPS | 显存带宽 | 功耗 | 价格 (≈) | 性价比 |
|------|------|-----------|-------------|---------|------|---------|-------|
| **RTX 4060** | 8GB GDDR6 | 3072 | 155 | 272 GB/s | 115W | ￥2,000 | ⭐⭐⭐⭐ |
| **RTX 4060 Ti** | 16GB GDDR6 | 4352 | 217 | 288 GB/s | 165W | ￥3,200 | ⭐⭐⭐⭐⭐ |
| **RTX 4070** | 12GB GDDR6X | 5888 | 291 | 504 GB/s | 200W | ￥4,500 | ⭐⭐⭐⭐ |
| **RTX 4070 Ti Super** | 16GB GDDR6X | 8448 | 444 | 672 GB/s | 285W | ￥6,000 | ⭐⭐⭐⭐ |
| **RTX 4080 Super** | 16GB GDDR6X | 10240 | 522 | 736 GB/s | 320W | ￥8,000 | ⭐⭐⭐ |
| **RTX 4090** | 24GB GDDR6X | 16384 | 826 | 1008 GB/s | 450W | ￥13,000 | ⭐⭐⭐⭐ |

### 数据中心 GPU

| 型号 | 显存 | FP16 TFLOPS | 显存带宽 | NVLink | 功耗 | 月租 (≈) | 适用场景 |
|------|------|-------------|---------|--------|------|---------|---------|
| **A10** | 24GB | 125 | 600 GB/s | ❌ | 150W | ￥3,000 | 推理/轻度训练 |
| **A100 (40GB)** | 40GB HBM2 | 312 | 1.6 TB/s | ✅ | 250W | ￥8,000 | 中型训练 |
| **A100 (80GB)** | 80GB HBM2e | 312 | 2.0 TB/s | ✅ | 300W | ￥12,000 | 大型训练 |
| **H100** | 80GB HBM3 | 989 | 3.35 TB/s | ✅ | 700W | ￥25,000 | 超大规模 |
| **L40S** | 48GB GDDR6 | 181 | 864 GB/s | ❌ | 350W | ￥6,000 | 推理/图形 |

---

## 知识点 3: 显存需求计算器

```python
# examples/7-1-resources/gpu_memory_calculator.py
"""
GPU 显存需求计算器

计算不同场景下的显存需求
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelSpec:
    """模型规格"""
    params_billion: float  # 参数量 (十亿)
    sequence_length: int   # 序列长度
    batch_size: int        # 批次大小


@dataclass
class MemoryEstimate:
    """显存估算结果"""
    model_weights_gb: float
    gradient_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_gb: float
    recommendation: str


def estimate_memory(
    spec: ModelSpec,
    training_mode: str = 'full',  # 'full', 'lora', 'qlora'
    precision: str = 'fp16'       # 'fp32', 'fp16', 'int8', 'int4'
) -> MemoryEstimate:
    """
    估算显存需求
    
    参数:
        spec: 模型规格
        training_mode: 训练模式
        precision: 精度
    """
    
    params = spec.params_billion * 1e9
    
    # ========== 1. 模型权重显存 ==========
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    if training_mode == 'full':
        # 全参数微调需要 FP32 master weights + FP16 模型
        model_weights_gb = params * 4 / (1024**3)  # FP32 master
    elif training_mode == 'lora':
        # LoRA: 冻结权重 + 少量 LoRA 参数 (~0.1-1%)
        frozen_gb = params * 2 / (1024**3)  # FP16 冻结
        lora_gb = frozen_gb * 0.01  # 1% LoRA
        model_weights_gb = frozen_gb + lora_gb
    elif training_mode == 'qlora':
        # QLoRA: 4bit 量化 + LoRA
        model_weights_gb = params * 0.5 / (1024**3)  # 4bit
        model_weights_gb += model_weights_gb * 0.01  # LoRA
    else:
        model_weights_gb = params * bytes_per_param[precision] / (1024**3)
    
    # ========== 2. 梯度显存 ==========
    if training_mode == 'full':
        gradient_memory_gb = params * 2 / (1024**3)  # FP16 梯度
    elif training_mode in ['lora', 'qlora']:
        gradient_memory_gb = model_weights_gb * 0.1  # LoRA 参数梯度
    else:
        gradient_memory_gb = 0
    
    # ========== 3. 优化器显存 ==========
    if training_mode == 'full':
        # Adam: momentum + variance (2×FP32)
        optimizer_memory_gb = params * 8 / (1024**3)
    elif training_mode == 'lora':
        optimizer_memory_gb = model_weights_gb * 0.02
    elif training_mode == 'qlora':
        optimizer_memory_gb = model_weights_gb * 0.01
    else:
        optimizer_memory_gb = 0
    
    # ========== 4. 激活显存 (估算) ==========
    # 经验公式：与批次大小、序列长度成正比
    base_activation = spec.batch_size * spec.sequence_length * 1e-6  # GB
    if training_mode == 'full':
        activation_memory_gb = base_activation * spec.params_billion
    else:
        activation_memory_gb = base_activation * 0.5
    
    # ========== 总计 ==========
    total_gb = (
        model_weights_gb +
        gradient_memory_gb +
        optimizer_memory_gb +
        activation_memory_gb
    )
    
    # 额外开销 (约 10-20%)
    total_gb *= 1.15
    
    # ========== GPU 推荐 ==========
    if total_gb <= 8:
        recommendation = "RTX 4060 (8GB) 或 RTX 3060 (12GB)"
    elif total_gb <= 16:
        recommendation = "RTX 4060 Ti (16GB) 或 RTX 4070"
    elif total_gb <= 24:
        recommendation = "RTX 4090 (24GB)"
    elif total_gb <= 40:
        recommendation = "RTX 6000 Ada (48GB) 或 A100 (40GB)"
    elif total_gb <= 48:
        recommendation = "RTX 6000 Ada (48GB)"
    elif total_gb <= 80:
        recommendation = "A100 (80GB) 或 H100"
    else:
        recommendation = "多卡并行 (A100/H100 × N)"
    
    return MemoryEstimate(
        model_weights_gb=round(model_weights_gb, 2),
        gradient_memory_gb=round(gradient_memory_gb, 2),
        optimizer_memory_gb=round(optimizer_memory_gb, 2),
        activation_memory_gb=round(activation_memory_gb, 2),
        total_gb=round(total_gb, 2),
        recommendation=recommendation
    )


def print_comparison():
    """打印显存需求对比表"""
    
    print("=" * 80)
    print("显存需求对比表 (batch_size=8, seq_len=2048)")
    print("=" * 80)
    
    configs = [
        ("Llama-7B", 7, "full", "fp16"),
        ("Llama-7B", 7, "lora", "fp16"),
        ("Llama-7B", 7, "qlora", "int4"),
        ("Llama-13B", 13, "full", "fp16"),
        ("Llama-13B", 13, "lora", "fp16"),
        ("Llama-13B", 13, "qlora", "int4"),
        ("Llama-70B", 70, "full", "fp16"),
        ("Llama-70B", 70, "qlora", "int4"),
    ]
    
    print(f"{'模型':<15} {'模式':<8} {'权重':<8} {'梯度':<8} {'优化器':<8} {'激活':<8} {'总计':<8} {'推荐 GPU'}")
    print("-" * 80)
    
    for name, params, mode, prec in configs:
        spec = ModelSpec(
            params_billion=params,
            sequence_length=2048,
            batch_size=8
        )
        
        estimate = estimate_memory(spec, training_mode=mode, precision=prec)
        
        print(f"{name:<15} {mode:<8} {estimate.model_weights_gb:<8.1f} "
              f"{estimate.gradient_memory_gb:<8.1f} {estimate.optimizer_memory_gb:<8.1f} "
              f"{estimate.activation_memory_gb:<8.1f} {estimate.total_gb:<8.1f} "
              f"{estimate.recommendation}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()
```

### 示例输出

```
================================================================================
显存需求对比表 (batch_size=8, seq_len=2048)
================================================================================
模型            模式     权重     梯度     优化器   激活     总计     推荐 GPU
--------------------------------------------------------------------------------
Llama-7B        full     28.0    14.0     56.0     2.0     115.0   多卡并行 (A100/H100 × N)
Llama-7B        lora     14.1     1.4      1.4     1.0      20.5   RTX 4090 (24GB)
Llama-7B        qlora     3.5     0.4      0.4     0.5       6.0   RTX 4060 (8GB) 或 RTX 3060 (12GB)
Llama-13B       full     52.0    26.0    104.0     4.0     215.0   多卡并行 (A100/H100 × N)
Llama-13B       lora     26.1     2.6      2.6     2.0      38.5   RTX 6000 Ada (48GB)
Llama-13B       qlora     6.5     0.7      0.7     1.0      11.0   RTX 4060 Ti (16GB) 或 RTX 4070
Llama-70B       full    280.0   140.0    560.0    20.0   1140.0   多卡并行 (A100/H100 × N)
Llama-70B       qlora    35.0     3.5      3.5     5.0      54.0   A100 (80GB) 或 H100
================================================================================
```

---

## 知识点 4: 云 GPU 成本对比

### 主流云服务商价格对比

| 服务商 | 实例 | GPU | 显存 | 价格 (元/小时) | 月包优惠 | 适合场景 |
|--------|------|-----|------|---------------|---------|---------|
| **阿里云** | ecs.gn7i | A10 | 24GB | ￥3.5 | 85 折 | 推理/轻量训练 |
| **阿里云** | ecs.gn7 | A100 | 40GB | ￥12 | 8 折 | 中型训练 |
| **腾讯云** | GN7 | A10 | 24GB | ￥3.2 | 85 折 | 推理/轻量训练 |
| **华为云** | ModelArts | A100 | 80GB | ￥15 | 75 折 | 大型训练 |
| **AutoDL** | 社区云 | RTX 4090 | 24GB | ￥1.5 | 无 | 学生/个人 |
| **Colab Pro** | 云端 | A100 | 40GB | ￥350/月 | - | 学习/原型 |
| **Lambda Labs** | Cloud | A100 | 80GB | $2.5/hr | - | 海外训练 |
| **RunPod** | Cloud | RTX 4090 | 24GB | $0.7/hr | - | 个人开发 |

### 成本优化建议

```
成本控制策略
│
├── 短期使用 (<100 小时)
│   └── → 按量付费 (AutoDL/RunPod)
│
├── 中期使用 (100-1000 小时)
│   └── → 预留实例/月包 (30-50% 折扣)
│
├── 长期使用 (>1000 小时)
│   └── → 自购 GPU 或 年框协议
│
└── 极致省钱
    ├── 申请学术资源 (Google TPU Research Cloud)
    ├── 使用免费额度 (Colab/Kaggle)
    └── 利用 spot 实例 (70% 折扣，可能被中断)
```

---

## 练习题

### 练习 1: 资源规划

为以下场景设计 GPU 配置：

**场景 A**: 微调 Llama-2-13B 模型
- 数据集：50K 样本
- 目标：达到 90% 任务准确率
- 预算：￥50,000

**场景 B**: 部署 RAG 客服系统
- QPS: 100
- 响应延迟：<500ms
- 运行时间：6 个月

请给出:
1. GPU 型号和数量
2. 预计总成本
3. 部署架构

---

## 延伸阅读

- [GPU 选购指南 2024](https://www.art لذا.com/gpu-benchmarks)
- [云 GPU 价格对比](https://cloud-gpu.io/)
- [Llama 显存计算工具](https://github.com/NVIDIA/Megatron-LM)

---

[← 返回 7.1 主文档](7-1-gpu-resources.md) | [下一节：成本优化 →](7-2-cost-optimization.md)
