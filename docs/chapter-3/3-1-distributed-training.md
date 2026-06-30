# 3.1 分布式训练技术

> 当模型大到无法放入单张 GPU 时，需要分布式训练技术。本章深入讲解数据并行、模型并行及混合并行策略。

## 学习目标

学完本节后，你将能够：

- [ ] 理解数据并行、模型并行、流水线并行的区别
- [ ] 配置 DeepSpeed ZeRO 优化器
- [ ] 实现 tensor parallelism 和 pipeline parallelism
- [ ] 选择适合模型规模的并行策略

---

## 3.1.1 分布式训练基础

### 并行策略对比

| 策略 | 核心思想 | 通信开销 | 适用场景 |
|------|----------|----------|----------|
| **数据并行 (DP)** | 模型复制，数据分片 | 每步梯度同步 | 小模型 (<1B) |
| **分布式数据并行 (DDP)** | DP 的优化版，每卡一个进程 | 梯度 Ring-AllReduce | 中小模型 |
| ** ZeRO 数据并行** | 优化器状态分片 | 减少内存冗余 | 中大模型 |
| **张量并行 (TP)** | 矩阵运算切分到多卡 | 每层前向/后向通信 | 超大模型 |
| **流水线并行 (PP)** | 按层切分到不同设备 | 流水线气泡 | 超大模型 |

### 混合并行策略

```
典型大模型训练配置 (70B 参数)

┌─────────────────────────────────────────────────┐
│  Data Parallel (DP=8)                           │
│  ┌─────────────┬─────────────┬─────────────┐   │
│  │ TP=4, PP=2  │ TP=4, PP=2  │    ...      │   │
│  │ ┌───┬───┐   │ ┌───┬───┐   │             │   │
│  │ │L0 │L1 │   │ │L0 │L1 │   │             │   │
│  │ ├───┼───┤   │ ├───┼───┤   │             │   │
│  │ │L2 │L3 │   │ │L2 │L3 │   │             │   │
│  │ └───┴───┘   │ └───┴───┘   │             │   │
│  └─────────────┴─────────────┴─────────────┘   │
└─────────────────────────────────────────────────┘

总 GPU 数 = DP × TP × PP = 8 × 4 × 2 = 64 GPU
```

---

## 3.1.2 DeepSpeed ZeRO 优化器

### ZeRO 三个级别

```python
# examples/3-1-distributed/zero_config.py
"""
DeepSpeed ZeRO 配置详解
"""

ZERO_CONFIG = {
    # ========== ZeRO-1: 优化器状态分片 ==========
    "zero_1": {
        "stage": 1,
        # 优化器状态 (Adam: 2×参数) 分片到各 GPU
        # 内存收益：减少 8x (假设 8 GPU)
    },
    
    # ========== ZeRO-2: 优化器 + 梯度分片 ==========
    "zero_2": {
        "stage": 2,
        # 梯度也分片，进一步减少内存
        # 内存收益：减少 8x (优化器) + 8x (梯度)
    },
    
    # ========== ZeRO-3: 优化器 + 梯度 + 参数分片 ==========
    "zero_3": {
        "stage": 3,
        # 参数也分片，训练超大模型
        # 需要额外的通信收集参数
        "offload_optimizer": {
            "device": "cpu",  # 优化器状态卸载到 CPU
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",  # 参数卸载到 CPU
            "pin_memory": True
        },
    },
    
    # ========== ZeRO-Infinity: ZeRO-3 + CPU/NVMe 卸载 ==========
    "zero_infinity": {
        "stage": 3,
        "offload_optimizer": {"device": "nvme", "nvme_path": "/mnt/nvme"},
        "offload_param": {"device": "nvme", "nvme_path": "/mnt/nvme"},
        "contiguous_gradients": True,
        "overlap_comm": True,  # 通信计算重叠
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
    }
}
```

### 完整 DeepSpeed 训练配置

```json
// examples/3-1-distributed/ds_config_zero3.json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  
  "fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  "bf16": {
    "enabled": true
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_parition_grads": true,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 训练脚本

```python
# examples/3-1-distributed/deepspeed_training.py
"""
使用 DeepSpeed ZeRO-3 训练大模型
"""

import argparse
import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class DummyDataset(Dataset):
    def __init__(self, seq_len=2048, num_samples=10000):
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 32000, (self.seq_len,)),
            'labels': torch.randint(0, 32000, (self.seq_len,))
        }


def train():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # 初始化分布式环境
    deepspeed.init_distributed()
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16
    )
    
    # 启用梯度检查点（节省显存）
    model.gradient_checkpointing_enable()
    
    # 加载数据
    train_dataset = DummyDataset()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # DeepSpeed 初始化
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=train_dataset,
    )
    
    # 训练循环
    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            loss = model_engine(**batch).loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
        
        # 保存 checkpoint
        if args.local_rank == 0:
            model_engine.save_checkpoint(
                "./checkpoints",
                tag=f"epoch-{epoch}"
            )


if __name__ == "__main__":
    train()
```

---

## 3.1.3 张量并行 (Tensor Parallelism)

### Megatron-LM 风格 TP

```python
# examples/3-1-distributed/tensor_parallel.py
"""
张量并行实现原理

核心思想：将矩阵乘法切分到多个 GPU
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层
    
    Y = XA 被切分为：
    Y1 = XA1, Y2 = XA2, ...
    
    每个 GPU 计算一部分输出列
    """
    
    def __init__(self, in_features, out_features, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        
        # 每个 GPU 只负责一部分输出
        self.out_features_per_gpu = out_features // world_size
        
        self.weight = nn.Parameter(torch.randn(
            in_features, self.out_features_per_gpu
        ))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_gpu))
    
    def forward(self, x):
        # 本地计算
        y = x @ self.weight + self.bias
        
        # All-Gather 收集所有 GPU 的输出
        output_list = [torch.zeros_like(y) for _ in range(self.world_size)]
        dist.all_gather(output_list, y)
        
        # 拼接结果
        return torch.cat(output_list, dim=-1)


class RowParallelLinear(nn.Module):
    """
    行并行线性层
    
    Y = XA 被切分为：
    Y = X1A1 + X2A2 + ... (输入分片，输出累加)
    """
    
    def __init__(self, in_features, out_features, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        
        self.in_features_per_gpu = in_features // world_size
        
        self.weight = nn.Parameter(torch.randn(
            self.in_features_per_gpu, out_features
        ))
    
    def forward(self, x):
        # 输入分片
        x_local = x[:, self.rank * self.in_features_per_gpu:
                       (self.rank + 1) * self.in_features_per_gpu]
        
        # 本地计算
        y = x_local @ self.weight
        
        # All-Reduce 累加结果
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        
        return y


class TensorParallelAttention(nn.Module):
    """
    Megatron 风格的张量并行注意力
    """
    
    def __init__(self, hidden_size, num_heads, rank, world_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # QKV 投影使用列并行
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, rank, world_size
        )
        
        # 输出投影使用行并行
        self.o_proj = RowParallelLinear(
            hidden_size, hidden_size, rank, world_size
        )
    
    def forward(self, x):
        # QKV 投影 (列并行)
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # 注意力计算
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v
        
        # 输出投影 (行并行)
        return self.o_proj(attn_output)
```

---

## 3.1.4 流水线并行 (Pipeline Parallelism)

### GPipe 风格流水线

```python
# examples/3-1-distributed/pipeline_parallel.py
"""
流水线并行实现

将模型按层切分到不同 GPU，像流水线一样处理 micro-batch
"""

import torch
import torch.nn as nn
from torch.autograd.profiler import record_function


class PipelineStage(nn.Module):
    """
    流水线的一个阶段（一部分层）
    """
    
    def __init__(self, layers, is_first=False, is_last=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.is_first = is_first
        self.is_last = is_last
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineParallelModel(nn.Module):
    """
    GPipe 风格流水线并行
    
    流程:
    1. 将模型分成 N 个阶段
    2. 将 batch 分成 M 个 micro-batch
    3. 流水线执行，filling pipeline bubble
    """
    
    def __init__(self, model, num_stages, stage_idx, rank, world_size):
        super().__init__()
        self.stage_idx = stage_idx
        self.rank = rank
        self.world_size = world_size
        
        # 将模型切分到不同阶段
        layers = list(model.children())
        layers_per_stage = len(layers) // num_stages
        
        start_idx = stage_idx * layers_per_stage
        end_idx = start_idx + layers_per_stage if stage_idx < num_stages - 1 else len(layers)
        
        self.stage = PipelineStage(
            layers[start_idx:end_idx],
            is_first=(stage_idx == 0),
            is_last=(stage_idx == num_stages - 1)
        )
        
        # 进程组
        self.group = dist.new_group(list(range(world_size)))
    
    def forward(self, x):
        # 第一阶段从 CPU 接收输入
        if not self.is_first:
            src_rank = self.rank - 1
            x = self._recv_tensor(src_rank)
        
        # 执行当前阶段
        x = self.stage(x)
        
        # 最后阶段输出
        if not self.is_last:
            dst_rank = self.rank + 1
            self._send_tensor(x, dst_rank)
            return None
        
        return x
    
    def _send_tensor(self, tensor, dst_rank):
        dist.send(tensor, dst_rank, group=self.group)
    
    def _recv_tensor(self, src_rank, shape=None):
        if shape is None:
            shape = [1]  # 占位
            dist.recv(shape, src_rank, group=self.group)
        
        tensor = torch.zeros(shape, device=torch.cuda.current_device())
        dist.recv(tensor, src_rank, group=self.group)
        return tensor


def bubble_schedule(num_stages, num_micro_batches):
    """
    计算流水线气泡比例
    
    气泡时间占比 = (num_stages - 1) / (num_stages × num_micro_batches)
    """
    total_steps = num_stages + num_micro_batches - 1
    bubble_steps = num_stages - 1
    
    bubble_ratio = bubble_steps / total_steps
    
    print(f"流水线气泡：{bubble_ratio:.1%}")
    print(f"有效时间占比：{1 - bubble_ratio:.1%}")
    
    return bubble_ratio


# 示例：8 层模型，4 个阶段，8 个 micro-batch
# 气泡 = (4-1) / (4 + 8 - 1) = 3/11 ≈ 27%
# 通过增加 micro-batch 数可以降低气泡
```

---

## 3.1.5 3D 并行最佳实践

### 配置计算器

```python
# examples/3-1-distributed/parallel_config.py
"""
3D 并行配置计算器

根据模型规模和 GPU 资源自动推荐并行策略
"""

def recommend_parallel_config(
    model_params_b: float,  # 模型参数量 (B)
    num_gpus: int,
    gpu_memory_gb: int = 80,  # A100 80GB
    model_type: str = "llama"
):
    """
    推荐并行配置
    
    返回:
        dict: {dp, tp, pp, zero_stage}
    """
    
    # 估算模型内存需求 (16-bit)
    model_memory_gb = model_params_b * 2  # 2 bytes per param
    
    # 可用的总显存
    total_memory_gb = num_gpus * gpu_memory_gb
    
    # 简单启发式策略
    if model_params_b <= 7:
        # 小模型：纯数据并行
        return {
            "dp": num_gpus,
            "tp": 1,
            "pp": 1,
            "zero_stage": 1,
            "description": "数据并行 + ZeRO-1"
        }
    
    elif model_params_b <= 30:
        # 中等模型：TP + DP
        tp = min(4, num_gpus)
        dp = num_gpus // tp
        return {
            "dp": dp,
            "tp": tp,
            "pp": 1,
            "zero_stage": 2,
            "description": "张量并行 + 数据并行 + ZeRO-2"
        }
    
    else:
        # 大模型：3D 并行
        # 优先满足 PP 以减少通信
        pp = min(8, num_gpus // 2)
        remaining = num_gpus // pp
        
        tp = min(4, remaining)
        dp = remaining // tp
        
        return {
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "zero_stage": 3,
            "description": "3D 并行 (DP+TP+PP) + ZeRO-3"
        }


# 使用示例
if __name__ == "__main__":
    configs = [
        (7, 8),      # 7B 模型，8 GPU
        (13, 16),    # 13B 模型，16 GPU
        (70, 64),    # 70B 模型，64 GPU
        (70, 128),   # 70B 模型，128 GPU
    ]
    
    for params, gpus in configs:
        config = recommend_parallel_config(params, gpus)
        print(f"\n{params}B 模型，{gpus} GPU:")
        print(f"  DP={config['dp']}, TP={config['tp']}, PP={config['pp']}")
        print(f"  策略：{config['description']}")
```

---

## 练习题

### 基础题

1. **内存计算**：计算训练 70B 模型所需的显存（全精度 vs 16-bit vs ZeRO-3）。

2. **通信分析**：比较 All-Reduce、All-Gather、Reduce-Scatter 的通信量。

3. **气泡计算**：给定 PP=8, micro-batch=32，计算流水线气泡比例。

### 编程题

4. 实现一个支持 TP+PP 混合并行的 Transformer 层。

5. 使用 DeepSpeed 训练一个 7B 模型，对比 ZeRO-1/2/3 的显存占用和速度。

---

## 延伸阅读

- 🌐 [DeepSpeed Documentation](https://www.deepspeed.ai/)
- 🌐 [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- 🌐 [ColossalAI](https://www.colossalai.org/)
- 📖 《Distributed Deep Learning》- O'Reilly

---

[← 上一节：章前导引](README.md) | [下一节：3.2 模型压缩与量化 →](3-2-compression.md)
