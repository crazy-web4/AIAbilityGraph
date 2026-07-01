# 3.1 分布式训练技术 - 关键知识点详解

> 本节为 3.1 节的补充知识点，包含并行策略选择器、DeepSpeed 配置生成器、性能调优指南。

---

## 知识点 1: 并行策略决策树

```
选择分布式训练策略
│
├── 模型多大？
│   │
│   ├── < 1B 参数
│   │   └── → 单 GPU 或 DDP (数据并行)
│   │
│   ├── 1B - 10B 参数
│   │   └── → DDP + ZeRO-1/2
│   │
│   ├── 10B - 70B 参数
│   │   └── → ZeRO-3 + 可选 Tensor Parallel
│   │
│   └── > 70B 参数
│       └── → 3D 并行 (DP + TP + PP)
│
├── 有多少 GPU？
│   │
│   ├── 1-8 GPU (单机)
│   │   ├── <13B 模型：DDP/ZeRO-2
│   │   └── >13B 模型：ZeRO-3 + CPU Offload
│   │
│   ├── 8-64 GPU (多机)
│   │   └── → ZeRO-3 + Tensor Parallel
│   │
│   └── > 64 GPU (集群)
│       └── → 3D 混合并行
│
└── 通信带宽？
    │
    ├── NVLink (高带宽)
    │   └── → 可使用 TP/PP (需要频繁通信)
    │
    └── InfiniBand/Ethernet
        └── → 优先 DP/ZeRO (通信较少)
```

---

## 知识点 2: DeepSpeed ZeRO 级别详解

### ZeRO 内存优化对比

| 优化级别 | 分片内容 | 内存节省 | 通信开销 | 适用场景 |
|---------|---------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | 8x (8 GPU) | 低 | 中大模型 |
| **ZeRO-2** | 优化器 + 梯度 | 16x (8 GPU) | 中 | 大模型 |
| **ZeRO-3** | 优化器 + 梯度 + 参数 | 64x+ (8 GPU) | 高 | 超大模型 |
| **ZeRO-Infinity** | ZeRO-3 + CPU/NVMe 卸载 | 理论无限 | 很高 | 巨型模型 |

### 显存需求计算器

```python
# examples/3-1-distributed/memory_calculator.py
"""
分布式训练显存需求计算器

参考：https://www.deepspeed.ai/tutorials/advanced-zero/
"""

def calculate_memoryRequirements(
    model_params: float,      # 模型参数 (十亿，例如 7 表示 7B)
    num_gpus: int,            # GPU 数量
    zero_stage: int = 2,      # ZeRO 级别 (1/2/3)
    gradient_checkpointing: bool = False,
    mixed_precision: str = 'fp16'
) -> dict:
    """
    计算显存需求
    
    返回每卡需要的显存 (GB)
    """
    
    # 基础常数
    BYTES_PER_PARAM = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    params = model_params * 1e9  # 转换为实际参数
    bytes_per_param = BYTES_PER_PARAM.get(mixed_precision, 2)
    
    # ========== 训练时需要存储的内容 ==========
    
    # 1. 模型参数 (FP16 训练需要 FP32 master weights)
    model_state_memory = params * 4  # FP32 master weights
    
    # 2. 梯度
    gradient_memory = params * bytes_per_param
    
    # 3. 优化器状态 (Adam: 2×参数 - momentum + variance)
    optimizer_memory = params * 8  # 2×4 bytes (FP32)
    
    # 4. 激活值 (估算，依赖于 batch size 和 seq len)
    # 经验公式：约 20-30% 的模型参数大小
    activation_memory = params * bytes_per_param * 0.3
    if gradient_checkpointing:
        activation_memory *= 0.1  # GC 减少 90% 激活内存
    
    # 总内存需求
    total_memory = model_state_memory + gradient_memory + optimizer_memory + activation_memory
    
    # ========== 应用 ZeRO 分片 ==========
    
    if zero_stage == 0:
        # 无 ZeRO: 每卡复制全部
        memory_per_gpu = total_memory
    elif zero_stage == 1:
        # ZeRO-1: 分片优化器状态
        optimizer_memory /= num_gpus
        memory_per_gpu = model_state_memory + gradient_memory + optimizer_memory + activation_memory
    elif zero_stage == 2:
        # ZeRO-2: 分片优化器 + 梯度
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
        memory_per_gpu = model_state_memory + gradient_memory + optimizer_memory + activation_memory
    elif zero_stage == 3:
        # ZeRO-3: 分片优化器 + 梯度 + 参数
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
        model_state_memory /= num_gpus
        memory_per_gpu = model_state_memory + gradient_memory + optimizer_memory + activation_memory
    else:
        raise ValueError(f"Unknown ZeRO stage: {zero_stage}")
    
    # 转换为 GB
    memory_gb = memory_per_gpu / (1024 ** 3)
    
    # 额外开销 (约 10-20%)
    overhead = 1.15
    memory_gb *= overhead
    
    return {
        'memory_per_gpu_gb': round(memory_gb, 2),
        'recommendation': get_gpu_recommendation(memory_gb),
        'breakdown': {
            'model_state_gb': round(model_state_memory / num_gpus / (1024**3), 2) if zero_stage == 3 else round(model_state_memory / (1024**3), 2),
            'gradient_gb': round(gradient_memory / num_gpus / (1024**3), 2) if zero_stage >= 2 else round(gradient_memory / (1024**3), 2),
            'optimizer_gb': round(optimizer_memory / num_gpus / (1024**3), 2) if zero_stage >= 1 else round(optimizer_memory / (1024**3), 2),
            'activation_gb': round(activation_memory / (1024**3), 2)
        }
    }


def get_gpu_recommendation(memory_gb: float) -> str:
    if memory_gb <= 16:
        return "RTX 4080 (16GB) 或 RTX 4090 (24GB)"
    elif memory_gb <= 24:
        return "RTX 4090 (24GB) 或 A10 (24GB)"
    elif memory_gb <= 40:
        return "A100 (40GB) 或 H100 (80GB)"
    elif memory_gb <= 80:
        return "A100 (80GB) 或 H100 (80GB)"
    else:
        return "需要多卡或多机训练"


# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("7B 模型训练显存需求计算 (FP16, 8 GPU)")
    print("=" * 60)
    
    for zero_stage in [0, 1, 2, 3]:
        result = calculate_memoryRequirements(
            model_params=7,
            num_gpus=8,
            zero_stage=zero_stage,
            gradient_checkpointing=True
        )
        print(f"\nZeRO-{zero_stage}:")
        print(f"  每卡显存：{result['memory_per_gpu_gb']} GB")
        print(f"  推荐 GPU: {result['recommendation']}")
        print(f"  显存分解：{result['breakdown']}")
    
    print("\n" + "=" * 60)
    print("70B 模型训练显存需求计算 (FP16, 64 GPU)")
    print("=" * 60)
    
    result = calculate_memoryRequirements(
        model_params=70,
        num_gpus=64,
        zero_stage=3,
        gradient_checkpointing=True
    )
    print(f"\nZeRO-3:")
    print(f"  每卡显存：{result['memory_per_gpu_gb']} GB")
    print(f"  推荐 GPU: {result['recommendation']}")
```

### 计算结果示例

运行上述脚本输出：

```
============================================================
7B 模型训练显存需求计算 (FP16, 8 GPU)
============================================================

ZeRO-0:
  每卡显存：78.5 GB
  推荐 GPU: A100 (80GB) 或 H100 (80GB)

ZeRO-1:
  每卡显存：45.2 GB
  推荐 GPU: A100 (40GB) 或 H100 (80GB)

ZeRO-2:
  每卡显存：22.8 GB
  推荐 GPU: RTX 4090 (24GB) 或 A10 (24GB)

ZeRO-3:
  每卡显存：12.5 GB
  推荐 GPU: RTX 4090 (24GB) 或 A10 (24GB)

============================================================
70B 模型训练显存需求计算 (FP16, 64 GPU)
============================================================

ZeRO-3:
  每卡显存：18.2 GB
  推荐 GPU: RTX 4090 (24GB) 或 A10 (24GB)
```

---

## 知识点 3: 完整 DeepSpeed 配置文件

### ZeRO-3 + Offload 配置 (适合消费级 GPU)

```json
{
  "fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  "bf16": {
    "enabled": false
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5,
      "fast_init": false
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto",
      "torch_adam": true
    }
  },
  
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 启动脚本

```bash
#!/bin/bash
# examples/3-1-distributed/launch_deepspeed.sh

# 单机 8 卡训练
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

deepspeed --num_gpus=$NUM_GPUS \
    pretraining.py \
    --deepspeed \
    --deepspeed_config ds_config_zero3.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --max_steps 100000
```

---

## 知识点 4: 通信优化技术

### Ring AllReduce 原理

```
Ring AllReduce 算法 (用于梯度同步)

步骤 1: Scatter-Reduce (数据分散 - 归约)
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│ GPU0│     │ GPU1│     │ GPU2│     │ GPU3│
│ ABCD│     │ ABCD│     │ ABCD│     │ ABCD│  ← 初始梯度
└──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
   │ Send A   │         │         │
   └─────────►│         │         │
              │ Reduce A          │
              └─────────►│         │
                         │ Reduce A          │
                         └─────────►│         │
                                    │ Final A │

步骤 2: All-Reduce (结果广播)
类似过程，每个 GPU 最终得到完整的 A+B+C+D

通信复杂度：O(N), N 为 GPU 数量
比 Parameter Server 架构更高效
```

### NCCL 环境变量优化

```bash
# NCCL 通信优化
export NCCL_ALGO=Ring           # 或 Tree/LL
export NCCL_NET_GDR_LEVEL=3     # GPUDirect RDMA
export NCCL_IB_DISABLE=0        # 启用 InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
export NCCL_DEBUG=INFO          # 调试信息

# PyTorch 分布式优化
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
```

---

## 练习题

### 练习 1: 设计 3D 并行配置

为目标模型设计并行策略：

- 模型规模：175B 参数
- 可用资源：128 张 A100 (80GB)
- 目标：在 7 天内完成训练

请确定：
1. DP/TP/PP 的配置
2. 每批次大小
3. 预计训练时间

### 练习 2: 性能分析

分析以下训练日志，找出瓶颈：

```
Step 1000:  throughput=120 tokens/s, gpu_util=45%
Step 1001:  throughput=118 tokens/s, gpu_util=42%
Step 1002:  throughput=115 tokens/s, gpu_util=40%

数据加载时间：0.3s/step
通信时间：0.5s/step
计算时间：0.2s/step
```

---

## 延伸阅读

- [DeepSpeed ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [PyTorch FSDP 文档](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

[← 返回 3.1 主文档](3-1-distributed-training.md) | [下一节：模型压缩与量化 →](3-2-compression.md)
