# 3.3 推理优化技术

> 大模型推理优化是降低部署成本、提升用户体验的关键。本章深入讲解 vLLM、TensorRT-LLM 等推理框架的核心技术。

## 学习目标

学完本节后，你将能够：

- [ ] 理解 KV Cache 机制与优化
- [ ] 掌握 PagedAttention 原理
- [ ] 配置 vLLM 推理服务
- [ ] 使用 TensorRT-LLM 优化推理

---

## 3.3.1 KV Cache 优化

### 为什么需要 KV Cache？

在自回归生成中，每一步都要重新计算之前所有 token 的 K/V：

```
Step 1: 计算 token_1 的 K1, V1
Step 2: 重新计算 K1, V1 + 计算 K2, V2
Step 3: 重新计算 K1, V1, K2, V2 + 计算 K3, V3
...

浪费！之前计算的 KV 可以复用
```

### KV Cache 显存计算

```python
# examples/3-3-inference/kv_cache_memory.py

def calculate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16
) -> float:
    """
    计算 KV Cache 显存占用
    
    公式：2 × layers × kv_heads × head_dim × seq_len × batch × dtype
    """
    kv_elements = (
        2  # K 和 V
        * num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * batch_size
    )
    
    memory_bytes = kv_elements * dtype_bytes
    memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_mb / 1024
    
    return memory_gb


# 示例：Llama-2-7B 生成 2048 token
if __name__ == "__main__":
    # Llama-2-7B 配置
    config = {
        "num_layers": 32,
        "num_kv_heads": 32,  # GQA 时可能更少
        "head_dim": 128,
    }
    
    # 显存需求
    mem = calculate_kv_cache_memory(
        batch_size=1,
        seq_len=2048,
        **config
    )
    
    print(f"Llama-2-7B, seq_len=2048, batch=1")
    print(f"KV Cache 显存：{mem:.2f} GB")
    
    # 输出：约 2.0 GB
```

### KV Cache 实现

```python
# examples/3-3-inference/kv_cache_impl.py
import torch


class KVCache:
    """
    KV Cache 实现
    
    支持:
    - 增量更新
    - 位置索引管理
    """
    
    def __init__(self, batch_size, max_seq_len, num_kv_heads, 
                 head_dim, device='cuda', dtype=torch.float16):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        # 预分配 KV Cache (避免动态分配)
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        
        # 当前序列长度
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    def update(self, k: torch.Tensor, v: torch.Tensor, 
               layer_idx: int) -> tuple:
        """
        更新 KV Cache
        
        参数:
            k: 当前 step 的 K (batch, heads, 1, head_dim)
            v: 当前 step 的 V (batch, heads, 1, head_dim)
            layer_idx: 层索引
        
        返回:
            k_full, v_full: 完整的 KV (包含历史)
        """
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # 将新的 KV 存入 cache
        self.k_cache[batch_indices, :, self.seq_lens, :] = k.squeeze(2)
        self.v_cache[batch_indices, :, self.seq_lens, :] = v.squeeze(2)
        
        # 返回完整的 KV（用于注意力计算）
        k_full = self.k_cache[batch_indices, :, :self.seq_lens.max()+1, :]
        v_full = self.v_cache[batch_indices, :, :self.seq_lens.max()+1, :]
        
        # 更新长度
        self.seq_lens += 1
        
        return k_full, v_full
    
    def clear(self):
        """清空 Cache"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_lens.zero_()


class PagedKVCache:
    """
    Paged KV Cache - vLLM 核心创新
    
    将 KV Cache 分块存储，类似操作系统的虚拟内存
    """
    
    def __init__(self, block_size=16, num_blocks=1000, 
                 num_kv_heads=32, head_dim=128, dtype=torch.float16):
        self.block_size = block_size  # 每块 token 数
        
        # KV Cache 块
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self.k_blocks = torch.zeros(shape, dtype=dtype, device='cuda')
        self.v_blocks = torch.zeros(shape, dtype=dtype, device='cuda')
        
        # 块表：sequence -> [block_id, block_id, ...]
        self.block_tables = {}
        self.sequence_lengths = {}
    
    def allocate_sequence(self, seq_id, max_tokens):
        """为序列分配块"""
        num_blocks_needed = (max_tokens + self.block_size - 1) // self.block_size
        self.block_tables[seq_id] = list(range(num_blocks_needed))
        self.sequence_lengths[seq_id] = 0
    
    def get_kv(self, seq_id, q_len):
        """获取指定序列的 KV"""
        blocks = self.block_tables[seq_id]
        seq_len = self.sequence_lengths[seq_id]
        
        # 收集所有块的 KV
        k_list = []
        v_list = []
        
        for block_id in blocks:
            k_list.append(self.k_blocks[block_id])
            v_list.append(self.v_blocks[block_id])
        
        k = torch.cat(k_list, dim=1)[:, :seq_len, :]
        v = torch.cat(v_list, dim=1)[:, :seq_len, :]
        
        return k, v
```

---

## 3.3.2 vLLM 推理服务

### 配置与使用

```python
# examples/3-3-inference/vllm_serving.py
"""
使用 vLLM 部署推理服务
"""

from vllm import LLM, SamplingParams


def setup_vllm(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    配置 vLLM 推理引擎
    
    关键参数:
    - tensor_parallel_size: TP 并行 GPU 数
    - gpu_memory_utilization: GPU 显存利用率
    - max_num_seqs: 最大并发序列数
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,      # 单卡
        gpu_memory_utilization=0.9,  # 90% 显存用于 KV Cache
        max_num_seqs=256,            # 最大并发
        max_model_len=4096,          # 最大序列长度
        trust_remote_code=True,
    )
    
    return llm


def generate_with_vllm(llm, prompts):
    """使用 vLLM 生成"""
    
    # 采样配置
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["</s>", "\n\n"],
        repetition_penalty=1.1,
    )
    
    # 批量生成（vLLM 自动使用 Continuous Batching）
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        results.append({
            "prompt": output.prompt,
            "generated_text": output.outputs[0].text,
            "tokens": len(output.outputs[0].token_ids),
        })
    
    return results


# API 服务
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
llm = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.on_event("startup")
async def startup():
    global llm
    llm = setup_vllm()

@app.post("/generate")
async def generate(req: GenerateRequest):
    params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    outputs = llm.generate([req.prompt], params)
    return {"text": outputs[0].outputs[0].text}
```

### 性能对比

| 框架 | Token/s (A100) | 显存效率 | 并发支持 |
|------|----------------|----------|----------|
| Transformers | ~50 | 低 | 无 |
| vLLM | ~300 | 高 (PagedAttention) | Continuous Batching |
| TensorRT-LLM | ~400 | 最高 | 支持 |

---

## 3.3.3 TensorRT-LLM 优化

```python
# examples/3-3-inference/tensorrt_llm.py
"""
TensorRT-LLM 推理优化
"""

import tensorrt_llm
from tensorrt_llm import LLM


def build_trt_llm(model_path, engine_dir="./engines"):
    """
    构建 TensorRT-LLM 引擎
    
    优化步骤:
    1. 模型导出到 ONNX
    2. TensorRT 优化
    3. 生成推理引擎
    """
    
    # 构建配置
    build_config = {
        "max_batch_size": 8,
        "max_input_len": 2048,
        "max_output_len": 512,
        "max_beam_width": 1,
        "fp16_enable": True,
        "int8_enable": False,
    }
    
    # 构建引擎
    llm = LLM(model=model_path, engine_dir=engine_dir)
    
    return llm


def run_inference(llm, prompts):
    """运行推理"""
    from tensorrt_llm.runtime import generate
    
    outputs = generate(
        llm,
        prompts=prompts,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    
    return outputs
```

---

## 3.3.4 推理优化技巧

### 1. Continuous Batching

```python
"""
Continuous Batching vs Traditional Batching

Traditional:
[Req1] ████████████████████ (等待所有完成)
[Req2] ████████░░░░░░░░░░░░
[Req3] ░░░░░░░░████████████

Continuous:
[Req1] ████████████████░░░░
[Req2] ████████░░░░░░░░░░░░
[Req3] ░░░░░░░░████████████░░

新请求可以立即插入，无需等待整批完成
"""
```

### 2. Speculative Decoding

```python
# examples/3-3-inference/speculative_decoding.py
"""
投机解码：小模型草稿 + 大模型验证

加速比：2-4x
"""

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model):
        self.draft = draft_model  # 小模型
        self.target = target_model  # 大模型
    
    def decode(self, prompt, k=5):
        """
        每步生成 k 个候选 token，大模型并行验证
        
        流程:
        1. 小模型自回归生成 k 个 token
        2. 大模型并行验证所有 token
        3. 接受前缀，拒绝后缀
        """
        tokens = []
        
        for _ in range(k):
            # 小模型生成
            draft_token = self.draft.generate_one(tokens)
            tokens.append(draft_token)
        
        # 大模型验证
        accepted = self.target.verify(prompt, tokens)
        
        return accepted
```

---

## 练习题

1. **KV Cache 计算**：计算 Llama-2-70B (GQA: 64Q/8KV) 生成 4096 token 的 KV Cache 大小。

2. **吞吐量分析**：为什么 Continuous Batching 能提高吞吐量？

3. **Speculative Decoding**：什么情况下投机解码可能减速？

---

[← 上一节：3.2 模型压缩](3-2-compression.md) | [下一节：3.4 训练数据处理 →](3-4-data-processing.md)
