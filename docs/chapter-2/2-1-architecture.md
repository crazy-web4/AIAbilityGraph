# 2.1 大模型架构设计

> 深入理解从 Transformer 到现代大语言模型的架构演进，掌握 GPT、BERT、Llama 等核心模型的设计原理。

## 学习目标

学完本节后，你将能够：

- [ ] 理解 Transformer 架构的核心组件与注意力机制
- [ ] 对比 Encoder-only、Decoder-only、Encoder-Decoder 架构
- [ ] 掌握现代大模型的关键优化技术（RoPE、RMSNorm、SwiGLU）
- [ ] 根据任务需求选择合适的模型架构

---

## 2.1.1 Transformer 架构回顾

### 原始 Transformer 架构

```
                    Transformer
                         │
        ┌────────────────┴───────────────┐
        │                                │
    Encoder                           Decoder
        │                                │
   ┌────┴────┐                      ┌────┴────┐
   │         │                      │         │
Multi-Head   Feed               Multi-Head  Feed
Self-Attention Forward          Self-Attention Forward
   │         │                      │         │
   └────┬────┘                      └────┬────┘
        │                                │
     Add & Norm                       Add & Norm
        │                                │
     Output  ──────────────────────→  Cross-Attention
```

### 核心组件代码实现

```python
# examples/2-1-architecture/transformer_components.py
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0  # 在 forward 中计算
    
    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: (batch, heads, seq_len, d_k)
            key: (batch, heads, seq_len, d_k)
            value: (batch, heads, seq_len, d_k)
            mask: (batch, 1, 1, seq_len) 或 None
        """
        d_k = query.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用 mask（用于 decoder 防止看未来）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights
```

---



## 2.1.2 架构范式对比

### 三种 Transformer 架构

| 架构类型 | 代表模型 | 注意力类型 | 适用场景 |
|----------|----------|------------|----------|
| **Encoder-only** | BERT, RoBERTa | 双向自注意力 | 理解任务（分类、NER） |
| **Decoder-only** | GPT, Llama | 因果/单向注意力 | 生成任务（文本生成） |
| **Encoder-Decoder** | T5, BART | 双向 + 交叉注意力 | 序列到序列（翻译、摘要） |

### 1. Encoder-only (BERT 风格)

```python
class BERTStyleEncoder(nn.Module):
    """
    BERT 风格编码器
    
    特点:
    - 双向自注意力（可以看到所有位置）
    - [CLS] token 用于分类
    - [MASK] token 用于预训练
    """
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, max_seq_len=512):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size + 2, d_model, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 特殊 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embeddings(input_ids) + self.position_embeddings[:, :seq_len, :]
        x = self.layer_norm(x)
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 编码
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # 返回 CLS token 表示（用于分类）
        return x[:, 0, :]  # CLS token
```

### 2. Decoder-only (GPT 风格)

```python
class GPTStyleDecoder(nn.Module):
    """
    GPT 风格解码器
    
    特点:
    - 因果掩码（只能看到过去和当前位置）
    - 自回归生成
    """
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, max_seq_len=2048):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 因果掩码
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 嵌入
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        # 创建因果掩码
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # 解码
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        
        x = self.layer_norm(x)
        logits = self.output_head(x)
        
        return logits
```

### 3. Encoder-Decoder (T5/BART 风格)

```python
class T5StyleModel(nn.Module):
    """
    T5 风格的 Encoder-Decoder 模型
    
    特点:
    - Encoder 双向注意力
    - Decoder 单向 + 交叉注意力
    - 统一文本到文本框架
    """
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=6):
        super().__init__()
        
        self.encoder = BERTStyleEncoder(vocab_size, d_model, n_heads, n_layers)
        self.decoder = GPTStyleDecoder(vocab_size, d_model, n_heads, n_layers)
        
        # 交叉注意力的投影
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads)
    
    def forward(self, input_ids, decoder_input_ids, 
                encoder_attention_mask=None, decoder_attention_mask=None):
        # 编码
        encoder_output = self.encoder(input_ids, encoder_attention_mask)
        
        # 解码（带交叉注意力）
        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask
        )
        
        return decoder_output
```

---



## 2.1.3 现代大模型关键技术

### 1. RoPE 旋转位置编码 (Rotary Position Embedding)

**问题**：传统位置编码外推性差，长序列效果下降

**RoPE 解决方案**：通过旋转矩阵编码相对位置

```python
class RotaryEmbedding(nn.Module):
    """
    RoPE 旋转位置编码
    
    核心思想：用二维旋转表示位置信息
    q ⨂ RoPE(m) · k ⨂ RoPE(n) = f(q, k, m-n)  只依赖相对位置
    """
    
    def __init__(self, dim, max_seq_len=10000, base=10000):
        super().__init__()
        
        # 频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self.max_seq_len = max_seq_len
        self.dim = dim
    
    def forward(self, x, seq_len=None):
        """
        参数:
            x: (batch, heads, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # 位置
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # 相位
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)
        
        # 计算 cos 和 sin
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用 RoPE"""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

### 2. RMSNorm (Root Mean Square Layer Normalization)

**优势**：比 LayerNorm 更稳定、更快

```python
class RMSNorm(nn.Module):
    """
    RMSNorm - 更高效的归一化
    
    与 LayerNorm 对比:
    - LayerNorm: x / std(x) * gamma + beta
    - RMSNorm: x / rms(x) * gamma (无 beta，更简洁)
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化
        x = x / rms
        
        return self.weight * x
```

### 3. SwiGLU 激活函数

**优势**：比 ReLU、GELU 更好的性能

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数 (Swish Gated Linear Unit)
    
    SwiGLU(x) = Swish(xW) ⊗ (xV)
    其中 Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(dim, hidden_dim)
        self.V = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        # Swish 门控
        return F.silu(self.W(x)) * self.V(x)


# 在 FFN 中使用
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.output(self.swiglu(x))
```

---



## 2.1.4 Llama 架构详解

### Llama 2 配置

| 模型 | 层数 | 注意力头数 | KV 头数 | 隐藏维度 |
|------|------|-----------|--------|----------|
| Llama 2 7B | 32 | 32 | 32 | 4096 |
| Llama 2 13B | 40 | 40 | 40 | 5120 |
| Llama 2 70B | 80 | 64 | 8 | 8192 |

### 完整 Llama 风格模型实现

```python
# examples/2-1-architecture/llama_style_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaRMSNorm(nn.Module):
    """Llama 使用的 RMSNorm"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class LlamaRotaryEmbedding(nn.Module):
    """Llama 实现的 RoPE"""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算 cos/sin 缓存
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        )


def rotate_half(x):
    """旋转一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """应用 RoPE"""
    cos = cos[0, 0, position_ids].unsqueeze(1)
    sin = sin[0, 0, position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Llama 风格的 Grouped Query Attention"""
    
    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position_embeddings):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=max_position_embeddings
        )
        
        self.attention_dropout = nn.Dropout(0.0)
    
    def _repeat_kv(self, x, n_rep):
        """GQA: 当 num_kv_heads < num_heads 时重复 KV"""
        if n_rep == 1:
            return x
        
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # GQA：重复 KV 头
        n_rep = self.num_heads // self.num_kv_heads
        k = self._repeat_kv(k, n_rep)
        v = self._repeat_kv(v, n_rep)
        
        # 注意力
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return self.o_proj(output)


class LlamaMLP(nn.Module):
    """Llama 风格的 SwiGLU MLP"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Llama Decoder 层"""
    
    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, max_position_embeddings):
        super().__init__()
        
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.attention = LlamaAttention(
            hidden_size, num_heads, num_kv_heads, max_position_embeddings
        )
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
    
    def forward(self, x, attention_mask=None, position_ids=None):
        # Pre-norm 架构
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, attention_mask, position_ids)
        x = residual + x
        
        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class LlamaStyleModel(nn.Module):
    """
    Llama 风格模型完整实现
    
    配置示例:
        Llama 2 7B: hidden_size=4096, num_heads=32, num_layers=32, intermediate_size=11008
    """
    
    def __init__(self, 
                 vocab_size=32000,
                 hidden_size=4096,
                 intermediate_size=11008,
                 num_hidden_layers=32,
                 num_attention_heads=32,
                 num_kv_heads=32,
                 max_position_embeddings=2048):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size, num_attention_heads, num_kv_heads,
                intermediate_size, max_position_embeddings
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.norm = LlamaRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
```

---



## 架构对比总结

| 特性 | BERT | GPT | Llama |
|------|------|-----|-------|
| **架构类型** | Encoder-only | Decoder-only | Decoder-only |
| **注意力** | 双向 | 因果单向 | 因果单向 + GQA |
| **位置编码** | 学习型/绝对 | 学习型 | RoPE |
| **归一化** | LayerNorm (Post) | LayerNorm (Pre) | RMSNorm (Pre) |
| **激活函数** | GELU | GELU | SwiGLU |
| **典型应用** | 分类、理解 | 生成 | 对话、生成 |

---











## 练习题

### 基础题

1. **注意力计算**：给定 Q, K, V 维度，计算自注意力的输出维度和计算复杂度。

2. **RoPE 推导**：证明 RoPE 只依赖相对位置 m-n。

3. **GQA 分析**：当 num_heads=32, num_kv_heads=8 时，相比 MHA 减少多少 KV Cache 内存？

### 编程题

4. 实现一个支持 Flash Attention 的 Llama 模型（使用 `torch.nn.functional.scaled_dot_product_attention`）

5. 添加 LoRA 支持到 Llama 模型的注意力层

---

## 延伸阅读

- 🌐 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 🌐 [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- 🌐 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- 🌐 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

[← 上一节：章前导引](README.md) | [下一节：2.2 预训练与微调技术 →](2-2-pretraining.md)
