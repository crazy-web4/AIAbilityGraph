# 2.1 大模型架构设计 - 关键知识点详解

> 本节为 2.1 节的补充知识点，包含架构对比表、核心组件详解、实践练习。

---

## 知识点 1: Transformer 架构变体对比

### 完整架构对比表

| 特性 | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|---------------------|-------------------|---------------------|
| **注意力类型** | 双向自注意力 | 因果掩码自注意力 | 双向 + 交叉注意力 |
| **训练目标** | MLM (掩码语言建模) | 自回归语言建模 | 序列到序列 |
| **典型应用** | 分类、NER、问答 | 文本生成、对话 | 翻译、摘要 |
| **代表模型** | BERT, RoBERTa, ALBERT | GPT 系列，Llama | T5, BART, mT5 |
| **上下文长度** | 512 (BERT) | 可扩展至 32K+ | 通常 512-1024 |
| **推理速度** | 快 (并行) | 慢 (自回归) | 中等 |
| **微调数据需求** | 较少 | 中等 | 较少 |

### 代码实践：三种架构的实现差异

```python
# examples/2-1-architecture/architecture_comparison.py
import torch
import torch.nn as nn
import math


class EncoderOnlyBlock(nn.Module):
    """BERT 风格的 Encoder 块 - 双向注意力"""
    
    def __init__(self, dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        """
        BERT: attention_mask 为双向，无因果掩码
        [CLS] token 可以 attend 到所有位置
        """
        # 自注意力 (双向)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈网络
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderOnlyBlock(nn.Module):
    """GPT 风格的 Decoder 块 - 因果注意力"""
    
    def __init__(self, dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码缓冲区 (不占参数)
        self.register_buffer("causal_mask", None)
    
    def forward(self, x, attention_mask=None):
        """
        GPT: 因果掩码防止看未来位置
        """
        seq_len = x.size(1)
        
        # 生成因果掩码
        if self.causal_mask is None or self.causal_mask.size(1) != seq_len:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            self.causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        
        # 自注意力 (带因果掩码)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self.causal_mask[:seq_len, :seq_len])
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


# 测试对比
def compare_architectures():
    batch_size, seq_len, dim = 4, 32, 512
    
    # Encoder-only (BERT)
    encoder = EncoderOnlyBlock(dim, 8, dim * 4)
    x = torch.randn(batch_size, seq_len, dim)
    enc_out = encoder(x)
    print(f"Encoder output shape: {enc_out.shape}")
    # 输出：(4, 32, 512) - 所有位置并行处理
    
    # Decoder-only (GPT)
    decoder = DecoderOnlyBlock(dim, 8, dim * 4)
    dec_out = decoder(x)
    print(f"Decoder output shape: {dec_out.shape}")
    # 输出：(4, 32, 512) - 但位置 i 只能看到 0..i


if __name__ == "__main__":
    compare_architectures()
```

---

## 知识点 2: 现代大模型关键优化技术

### RoPE (Rotary Position Embedding)

RoPE 通过旋转变换将位置信息编码到注意力中，相比绝对位置编码具有更好的外推性。

**数学原理：**

对于查询向量 $q_m$ 和键向量 $k_n$ 在位置 $m$ 和 $n$：

$$Q_m = R_\Theta^m q_m, \quad K_n = R_\Theta^n k_n$$

其中旋转矩阵 $R_\Theta$ 定义为：

$$R_\Theta = \begin{pmatrix} 
\cos\theta_0 & -\sin\theta_0 & 0 & 0 \\
\sin\theta_0 & \cos\theta_0 & 0 & 0 \\
0 & 0 & \cos\theta_1 & -\sin\theta_1 \\
0 & 0 & \sin\theta_1 & \cos\theta_1
\end{pmatrix}$$

**代码实现：**

```python
class RotaryEmbedding(nn.Module):
    """
    RoPE 旋转位置编码
    
    参考：RoFormer: Enhanced Transformer with Rotary Position Embedding
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算余弦和正弦表
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len):
        """预计算旋转缓存"""
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # 扩展到完整维度
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, q, k, seq_len=None):
        """
        应用 RoPE 到 Q 和 K
        
        Args:
            q: (batch, heads, seq_len, dim)
            k: (batch, heads, seq_len, dim)
        """
        seq_len = seq_len or q.size(2)
        
        # 取出对应长度的旋转参数
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # 应用旋转 (使用 einsum 高效计算)
        q_rotated = self._rotate(q, cos, sin)
        k_rotated = self._rotate(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        应用旋转变换
        
        x: (batch, heads, seq_len, dim)
        cos/sin: (seq_len, dim)
        """
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # x * cos + rotate_half(x) * sin
        return x * cos + rotate_half(x) * sin


# 使用示例
def test_rope():
    batch, heads, seq_len, dim = 2, 8, 128, 64
    
    rope = RotaryEmbedding(dim=dim)
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    
    q_rot, k_rot = rope(q, k)
    
    # 验证 RoPE 性质：注意力分数只依赖相对位置
    # <R(q,i), R(k,j)> = f(i-j)
    print(f"RoPE 后 Q 形状：{q_rot.shape}")
    print(f"RoPE 保持模长：{torch.allclose(q.norm(), q_rot.norm(), atol=1e-5)}")


if __name__ == "__main__":
    test_rope()
```

### RMSNorm (Root Mean Square Layer Normalization)

相比 LayerNorm，RMSNorm 去掉均值中心化，计算更高效。

**公式对比：**

| 归一化类型 | 公式 | 计算复杂度 |
|-----------|------|-----------|
| LayerNorm | $y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | O(n) |
| RMSNorm | $y = \gamma \odot \frac{x}{\sqrt{\frac{1}{n}\sum x^2 + \epsilon}}$ | O(n) (实际快 7-10%) |

**代码实现：**

```python
class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization
    
    Llama, Llama2, Mistral 等现代 LLM 使用该归一化
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 只有 scale 参数
    
    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        return self.weight * x / rms


class LlamaRMSNorm(nn.Module):
    """
    Llama 官方实现风格 (带权重初始化为 1)
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        # Llama 使用 f32 计算方差保证精度
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)


# 性能对比实验
def benchmark_normalization():
    import time
    
    dim = 4096
    seq_len = 512
    batch = 32
    
    x = torch.randn(batch, seq_len, dim, device="cuda")
    
    layer_norm = nn.LayerNorm(dim).cuda()
    rms_norm = RMSNorm(dim).cuda()
    
    # 预热
    for _ in range(100):
        _ = layer_norm(x)
        _ = rms_norm(x)
    
    # 计时
    iterations = 1000
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    ln_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_time = time.time() - start
    
    print(f"LayerNorm: {ln_time*1000:.2f} ms")
    print(f"RMSNorm: {rms_time*1000:.2f} ms")
    print(f"RMSNorm 加速：{(ln_time/rms_time - 1)*100:.1f}%")


if __name__ == "__main__":
    benchmark_normalization()
```

### SwiGLU 激活函数

SwiGLU 是 Swish 和 GLU 的组合，相比 ReLU 有更好的表现。

**公式：**

$$\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV) = (\sigma(xW) \odot xW) \odot (xV)$$

其中 $\sigma$ 是 Sigmoid 函数。

**代码实现：**

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数
    
    Llama2, Mistral 等使用的激活函数
    比 ReLU 有更好的表达能力和训练稳定性
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # SiLU(x) = x * sigmoid(x)
        return F.silu(x)


class SwiGLUFFN(nn.Module):
    """
    使用 SwiGLU 的前馈网络
    
    注意：SwiGLU FFN 有三组权重而不是两组
    """
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)  # W 门控
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)    # U 上投影
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)  # V 下投影
    
    def forward(self, x):
        # SwiGLU(Wx, Ux) = Swish(Wx) * Ux
        gate = self.w_gate(x)
        up = self.w_up(x)
        
        # SwiGLU 激活
        hidden = F.silu(gate) * up
        
        # 输出投影
        return self.w_down(hidden)


class ReLUFFN(nn.Module):
    """标准 ReLU FFN 对比"""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# 对比实验
def compare_ffn():
    dim = 512
    hidden_dim = 2048
    batch = 32
    seq_len = 128
    
    swiglu_ffn = SwiGLUFFN(dim, hidden_dim)
    relu_ffn = ReLUFFN(dim, hidden_dim)
    
    x = torch.randn(batch, seq_len, dim)
    
    swiglu_out = swiglu_ffn(x)
    relu_out = relu_ffn(x)
    
    print(f"SwiGLU 输出形状：{swiglu_out.shape}")
    print(f"ReLU 输出形状：{relu_out.shape}")
    
    # 参数量对比
    swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())
    relu_params = sum(p.numel() for p in relu_ffn.parameters())
    
    print(f"SwiGLU 参数量：{swiglu_params:,}")  # 3×dim×hidden
    print(f"ReLU 参数量：{relu_params:,}")      # 2×dim×hidden
    # 注意：SwiGLU 多一重投影，但效果显著更好


if __name__ == "__main__":
    compare_ffn()
```

---

## 知识点 3: 架构选择决策树

```
选择大模型架构
│
├── 任务类型是什么？
│   │
│   ├── 文本分类/情感分析/NER
│   │   └── → Encoder-only (BERT, RoBERTa)
│   │       原因：需要双向上下文理解
│   │
│   ├── 文本生成/续写/对话
│   │   └── → Decoder-only (GPT, Llama)
│   │       原因：自回归生成，因果掩码
│   │
│   └── 翻译/摘要/Text2Text
│       └── → Encoder-Decoder (T5, BART)
│           原因：编码 - 解码架构最适合 seq2seq
│
├── 资源约束？
│   │
│   ├── 单卡/消费级 GPU
│   │   └── → 使用 7B 以下模型 + QLoRA 微调
│   │
│   ├── 多卡 (2-8 GPU)
│   │   └── → 7B-13B 模型 + LoRA/全微调
│   │
│   └── 集群 (64+ GPU)
│       └── → 70B+ 模型 + 3D 并行训练
│
└── 部署场景？
    │
    ├── 低延迟在线服务
    │   └── → GPTQ/AWQ 4bit 量化 + vLLM
    │
    ├── 边缘设备
    │   └── → TinyLlama/Phi 系列 + INT8
    │
    └── 云端批量推理
        └── → FP16/BF16 + DeepSpeed-Inference
```

---

## 练习题

### 练习 1: 实现多查询注意力 (MQA)

多头注意力 (MHA) 的 KV 缓存占用大量内存。MQA 通过共享 KV 头减少缓存大小。

```python
# TODO: 实现 MQA 注意力
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: 多查询共享 KV 头
    
    与 MHA 对比:
    - MHA: Q, K, V 都有 num_heads
    - MQA: Q 有 num_heads, K 和 V 只有 1 个头
    
    优点：减少 KV 缓存 8-16x
    缺点：可能损失一些质量
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        # 实现提示：
        # 1. Q 投影：dim -> dim
        # 2. K, V 投影：dim -> dim // num_heads (只投影一次)
        # 3. 在计算注意力时，对 K, V 使用 expand 复制 num_heads 次
        pass
    
    def forward(self, q, k, v, mask=None):
        pass
```

### 练习 2: 分析 Llama2 架构

下载 Llama2-7B 的配置文件，分析以下问题：

1. RoPE 的 base 频率是多少？
2. 使用了多少个 Transformer 层？
3. 注意力头数是多少？是否使用 GQA？
4. 中间层维度 (MLP hidden dim) 是多少？

---

## 延伸阅读

- [RoFormer 论文](https://arxiv.org/abs/2104.09864) - RoPE 原始论文
- [Llama2 论文](https://arxiv.org/abs/2307.09288) - 现代架构详解
- [HuggingFace Transformers Llama 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

---

[← 返回 2.1 主文档](2-1-architecture.md) | [下一节：预训练与微调 →](2-2-pretraining.md)
