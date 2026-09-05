# 1.3 深度学习基础

> 深度学习是当代 AI 的核心驱动力，本章从 MLP 到 Transformer 系统讲解神经网络的核心架构。

## 学习目标

学完本节后，你将能够：

- [ ] 理解 MLP 的结构与反向传播原理
- [ ] 掌握 CNN 的核心组件（卷积、池化）及经典架构
- [ ] 理解 RNN/LSTM/GRU 处理序列数据的机制
- [ ] 掌握 Self-Attention 与 Transformer 架构
- [ ] 使用 PyTorch 实现基础深度学习模型

---

## 1.3.1 多层感知机 (MLP)

### 结构原理

**多层感知机**是最基础的神经网络，由输入层、隐藏层、输出层组成。

```
输入层 → [线性层 + 激活函数] → [线性层 + 激活函数] → 输出层
         ↓ 隐藏层 1            ↓ 隐藏层 2
```

**前向传播公式**：
$$\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$$

### PyTorch 实现

```python
# examples/1-3-dl/mlp_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    多层感知机示例
    
    架构：Input -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)
    """
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(MLP, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout 防止过拟合
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # 展平输入 (batch, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)

# 创建模型
model = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10)
print(model)

# 测试前向传播
dummy_input = torch.randn(32, 28, 28)  # batch_size=32
output = model(dummy_input)
print(f"输出形状：{output.shape}")  # torch.Size([32, 10])
```

### 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 | 适用场景 |
|----------|------|------|------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 输出 (0,1)，概率解释 | 梯度消失 | 二分类输出层 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 零中心化 | 梯度消失 | RNN |
| ReLU | $\max(0, x)$ | 计算快、缓解梯度消失 | Dead ReLU | 默认选择 |
| Leaky ReLU | $\max(\alpha x, x)$ | 解决 Dead ReLU | 需要调α | 深层网络 |
| GELU | $x \cdot \Phi(x)$ | 平滑、Transformer 标配 | 计算稍慢 | Transformer |

```python
# 激活函数可视化
def plot_activation_functions():
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5, 5, 100)
    
    functions = {
        'Sigmoid': 1 / (1 + np.exp(-x)),
        'Tanh': np.tanh(x),
        'ReLU': np.maximum(0, x),
        'Leaky ReLU': np.where(x > 0, x, 0.01 * x),
        'GELU': x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, y) in enumerate(functions.items()):
        axes[idx].plot(x, y, linewidth=2)
        axes[idx].set_title(name, fontsize=14)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(0, color='black', linewidth=0.5)
        axes[idx].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
```

---

## 1.3.2 卷积神经网络 (CNN)

### 为什么 CNN 适合图像？

| 全连接网络 | 卷积网络 |
|------------|----------|
| 忽略空间结构 | 保留空间关系 |
| 参数量巨大 | 参数共享，效率高 |
| 平移不敏感 | 平移不变性 |

### 核心组件

#### 1. 卷积层 (Conv2d)

```python
# 卷积操作演示
import torch
import torch.nn as nn

# Conv2d 参数详解
conv = nn.Conv2d(
    in_channels=3,      # 输入通道数 (RGB 图像)
    out_channels=64,    # 输出通道数 (滤波器数量)
    kernel_size=3,      # 卷积核大小 3×3
    stride=1,           # 步长
    padding=1,          # 填充 (保持尺寸)
    bias=True           # 是否加偏置
)

# 输入：(batch, channels, height, width)
input_tensor = torch.randn(32, 3, 224, 224)  # 32 张 224×224 RGB 图像
output = conv(input_tensor)
print(f"输出形状：{output.shape}")  # (32, 64, 224, 224)
```

**输出尺寸计算公式**：
$$\text{Output} = \left\lfloor\frac{\text{Input} - \text{Kernel} + 2 \times \text{Padding}}{\text{Stride}}\right\rfloor + 1$$

#### 2. 池化层 (Pooling)

```python
# 最大池化 vs 平均池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 全局平均池化 (常用于分类前)
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 1×1
```

### 经典 CNN 架构

#### LeNet-5 (1998) - 手写数字识别

```python
class LeNet5(nn.Module):
    """
    LeNet-5 架构
    输入：32×32 灰度图像
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 32×32 -> 32×32
            nn.AvgPool2d(2),                             # -> 16×16
            nn.Tanh(),
            nn.Conv2d(6, 16, kernel_size=5),             # -> 12×12
            nn.AvgPool2d(2),                             # -> 6×6
            nn.Tanh()
        )
        
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
```

#### VGG-16 (2014) - 深度 CNN 基准

```python
# examples/1-3-dl/cnn_architectures.py
class VGG16(nn.Module):
    """
    VGG-16 架构
    特点：统一使用 3×3 卷积，深度增加
    """
    def __init__(self, num_classes=1000):
        super(VGG16, self). __init__()
        
        # VGG 配置：每段卷积层的通道数
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512, 'M']
        
        self.features = self._make_layers(cfg)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 现代 CNN 改进技巧

```python
class ModernCNN(nn.Module):
    """
    现代 CNN 技巧集合：
    - BatchNorm 加速收敛
    - Depthwise Separable Conv 减少参数
    - Global Average Pooling 替代全连接
    - Skip Connection 缓解梯度消失
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Conv Block with BatchNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32->16
        )
        
        # 残差连接示例
        self.residual_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.skip = nn.Conv2d(64, 128, 1)  # 1×1 卷积调整通道
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # 残差连接
        identity = self.skip(x)
        x = self.residual_block(x) + identity
        x = F.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## 1.3.3 循环神经网络 (RNN)

### RNN 基础

**适用场景**：序列数据（文本、语音、时间序列）

**核心思想**：隐藏状态传递历史信息

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

```python
# PyTorch RNN 实现
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_classes=10):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN 层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入格式 (batch, seq, feature)
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # RNN 前向传播
        out, hn = self.rnn(x, h0)  # out: (batch, seq_len, hidden)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 使用示例
rnn = SimpleRNN(input_size=128, hidden_size=256, num_classes=10)
x = torch.randn(32, 50, 128)  # batch=32, seq_len=50, features=128
output = rnn(x)
```

### LSTM - 长短期记忆网络

解决传统 RNN 的**长期依赖问题**：

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # 双向 LSTM
        )
        
        # 双向 LSTM 输出维度 ×2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) - 词索引序列
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # LSTM 输出
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # 拼接双向最后状态
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        return self.fc(hidden)
```

### GRU - 简化版 LSTM

```python
# GRU vs LSTM 选择
"""
LSTM:
- 3 个门控（遗忘门、输入门、输出门）
- 参数更多，训练慢
- 长序列依赖更强

GRU:
- 2 个门控（重置门、更新门）
- 参数少 25%，训练快
- 大多数任务效果相当
"""

gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)
```

---

## 1.3.4 Transformer 架构

> 2017 年 Google 论文《Attention Is All You Need》提出 Transformer：完全基于注意力机制、抛弃循环结构，是 GPT、BERT、Llama 等所有现代大模型的共同基石。

### 为什么需要 Transformer？

RNN/LSTM 有两个根本瓶颈：

| 瓶颈 | 说明 |
|------|------|
| 无法并行 | 第 t 步依赖第 t-1 步的隐藏状态，必须沿时间轴串行计算，GPU 利用率低 |
| 长距离依赖弱 | 相距 n 步的两个 token，信息传递路径长度为 O(n)，梯度仍会衰减 |

Transformer 用 **Self-Attention（自注意力）** 替代循环结构，带来三个核心优势：

1. **完全并行**：序列所有位置同时计算，训练吞吐数量级提升
2. **全局视野**：任意两个位置直接交互，最大信息路径长度为 O(1)
3. **易于扩展**：参数、数据、算力可持续放大模型能力（Scaling Law 的架构基础）

### 整体架构总览

原始 Transformer 是为机器翻译设计的 **Encoder-Decoder（编码器-解码器）** 结构：

```
源序列 src                                   目标序列 tgt（右移一位，首 token 为 <sos>）
   │                                              │
词嵌入 + 位置编码                              词嵌入 + 位置编码
   │                                              │
┌──────────────────┐                      ┌─────────────────────────┐
│ Encoder × N      │                      │ Decoder × N             │
│                  │                      │                         │
│ Multi-Head       │                      │ Masked Self-Attention   │ ← 因果 mask
│ Self-Attention   │                      │  + Add & Norm          │
│  + Add & Norm    │                      │                         │
│                  │                      │ Cross-Attention         │
│ Feed-Forward     │  memory (K, V) ─────►│   Q 来自 Decoder        │
│  + Add & Norm    │                      │   K、V 来自 Encoder     │
│                  │                      │  + Add & Norm           │
│                  │                      │                         │
│                  │                      │ Feed-Forward + Norm     │
└────────┬─────────┘                      └───────────┬─────────────┘
         │                                            │
         └──────────── memory 贯穿每一层 ─────────────┤
                                                      │
                                              Linear → Softmax
                                                      │
                                              下一个 token 概率分布
```

数据流：

- **Encoder**：读入源序列，每层 = Self-Attention + 前馈网络，输出每个位置融合了全局上下文的表示
- **Decoder**：自回归地生成目标序列，每层做三件事——Masked Self-Attention（只看已生成内容）、Cross-Attention（看 Encoder 输出）、前馈网络
- **输出头**：Linear 投影到词表维度 + Softmax，预测下一个 token

> 💡 现代大模型大多只取其中一半：Encoder-only（BERT，擅长理解/分类）、Decoder-only（GPT/Llama，擅长生成）、Encoder-Decoder（T5/BART，翻译/摘要）。三种范式对比与现代改进（RoPE、RMSNorm、SwiGLU、Llama 架构）见 [2.1 大模型架构设计](../chapter-2/2-1-architecture.md)。

### 输入表示：词嵌入 + 位置编码

Self-Attention 本身**对输入顺序无感**：把输入序列打乱，输出只是相应地置换位置（置换等变）。因此必须显式注入位置信息：

$$\text{input}_i = \text{Embedding}(w_i) + \text{PE}(i)$$

原始论文使用**正弦位置编码**：

$$PE_{(pos,\ 2i)} = \sin\left(pos / 10000^{2i/d_{model}}\right)$$

$$PE_{(pos,\ 2i+1)} = \cos\left(pos / 10000^{2i/d_{model}}\right)$$

直觉：每个维度对应一个不同波长的正弦波（波长从 $2\pi$ 到 $10000 \times 2\pi$ 几何级数分布），组合后每个位置获得唯一的"时间戳"编码；且相对位置可以表示为位置编码的线性函数，便于模型学习 token 间距离。

常见位置编码方案：

| 方案 | 代表模型 | 特点 |
|------|----------|------|
| 正弦位置编码 | 原版 Transformer | 无需学习、可外推到更长序列 |
| 可学习位置嵌入 | BERT、GPT-2 | 每个位置一个可训练向量 |
| 相对位置编码 | T5、DeBERTa | 直接编码 token 间相对距离 |
| RoPE 旋转位置编码 | Llama、Qwen | 通过旋转矩阵把相对位置注入 Q/K，现代大模型主流，见 [2.1.3](../chapter-2/2-1-architecture.md) |

### Self-Attention：Q、K、V 到底是什么？

**检索类比**：把注意力想象成"在序列里查资料"：

- **Query (Q)**：当前位置发出的"查询"——我需要什么信息？
- **Key (K)**：每个位置的"标题/标签"——我这里有什么信息？
- **Value (V)**：每个位置的"正文内容"——实际被聚合的信息

当前位置用自己的 Q 与所有位置的 K 计算相似度，softmax 归一化为权重，再对 V 加权求和：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

四步拆解：

1. **打分**：`scores = Q @ K.T`，得到每个位置对所有位置的相关性矩阵（n×n）
2. **缩放**：除以 $\sqrt{d_k}$。点积方差随维度 $d_k$ 线性增大，不缩放时 softmax 会趋近 one-hot、梯度几乎为零；缩放后方差回到 1 左右
3. **掩码 + 归一化**：`softmax(scores)` 得到和为 1 的注意力权重（可先加 mask）
4. **聚合**：`weights @ V`，按权重汇总所有位置的信息

> 🔑 **Self-Attention** 中 Q、K、V 来自同一个序列；**Cross-Attention（交叉注意力）** 中 Q 来自 Decoder，K、V 来自 Encoder 输出——这正是 Decoder "查阅" 源序列信息的方式。

### Multi-Head Attention（多头注意力）

单头注意力只能学一种"关注模式"。多头机制把 $d_{model}$ 切成 $h$ 个子空间，并行做 $h$ 次注意力再拼接：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

不同头会自发学到不同的关系模式：有的关注相邻词、有的关注句法搭配、有的做指代消解。总计算量与单头全维度注意力基本相同。

```python
# examples/1-3-dl/transformer_from_scratch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q、K、V 与输出的线性变换
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, Lq, d_model)，key/value: (batch, Lk, d_model)
        mask:  (batch, 1, Lq, Lk) 布尔张量，True = 允许注意该位置
        """
        B = query.size(0)

        # 线性变换后分头：(B, L, D) -> (B, H, L, head_dim)
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)  # 屏蔽位置 softmax 后权重趋近 0
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # 加权求和后合并多头：(B, H, Lq, head_dim) -> (B, Lq, D)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(context)
```

### Mask 机制

Transformer 有两种 mask，缺一不可：

| Mask | 用在哪里 | 作用 |
|------|----------|------|
| Padding Mask | Encoder Self-Attn、Cross-Attn | 屏蔽 batch 中补齐的 `<pad>` 位置 |
| Causal Mask（因果掩码） | Decoder Self-Attn | 保证位置 t 只能看到 ≤ t 的位置，训练/推理行为一致 |

因果掩码是一个上三角被屏蔽的矩阵：

```
              我   喜欢   学   Transformer
我            ✓    ✗     ✗    ✗
喜欢          ✓    ✓     ✗    ✗
学            ✓    ✓     ✓    ✗
Transformer   ✓    ✓     ✓    ✓
```

```python
def make_causal_mask(length, device):
    """下三角为 True：位置 t 只能注意 0..t"""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device)) \
                .unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

def make_pad_mask(tokens, pad_idx=0):
    """(B, L) -> (B, 1, 1, L)，True 表示非 pad 的有效位置"""
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(1)

# Decoder Self-Attention 需要两种 mask 取交集
# tgt_mask = make_pad_mask(tgt_in) & make_causal_mask(L, device)
```

### 残差连接与 LayerNorm（Add & Norm）

每个子层（Attention / FFN）外面都包了残差连接和层归一化：

- **残差连接**：`x + Sublayer(x)`，梯度有直通路径，深层网络可训练（同 ResNet 思想）
- **LayerNorm**：对每个样本的特征维归一化，稳定每层输入分布。NLP 中不用 BatchNorm——序列长度可变、padding 会污染 batch 统计量

归一化放在残差的里面还是外面，形成两种架构：

| 方式 | 公式 | 特点 |
|------|------|------|
| Post-Norm（原版论文） | `LN(x + Sublayer(x))` | 先残差后归一化；深层训练对 warmup 敏感、易不稳定 |
| Pre-Norm（现代主流） | `x + Sublayer(LN(x))` | 先归一化再进子层；残差路径完全直通，易训练深层模型 |

> ⚠️ 注意：本节代码与配套示例均采用 **Pre-Norm**，并在整个栈结束后补一次 LayerNorm；现代大模型进一步用 RMSNorm 替代 LayerNorm（见 [2.1.3](../chapter-2/2-1-architecture.md)）。

### Position-wise Feed-Forward Network

注意力层之后，每个位置**独立**过同一个两层 MLP（中间升维 4 倍，激活后降回）：

$$\text{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2$$

注意力层负责"位置之间混合信息"，FFN 负责"单个位置内的特征变换"，两者交替堆叠。原版用 ReLU，现代模型多用 GELU/SwiGLU。

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # 升维（通常 4×d_model）
        self.linear2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
```

### Encoder 层与 Decoder 层

```python
class EncoderLayer(nn.Module):
    """Encoder 层：Self-Attention + FFN（Pre-Norm 结构）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.ffn(normed))
        return x


class DecoderLayer(nn.Module):
    """Decoder 层：Masked Self-Attention + Cross-Attention + FFN"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)   # 带因果 mask
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Q 来自 Decoder
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        # 1) Masked Self-Attention：只能看已生成的目标位置
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))
        # 2) Cross-Attention：Q 来自 Decoder，K、V 来自 Encoder 输出 memory
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, src_mask))
        # 3) FFN
        normed = self.norm3(x)
        x = x + self.dropout(self.ffn(normed))
        return x
```

完整模型（词嵌入、正弦位置编码、N 层堆叠、输出投影、权重绑定）与训练/推理代码见配套示例 `examples/1-3-dl/transformer_from_scratch.py`。

### 训练与推理的差异

| | 训练 | 推理 |
|---|------|------|
| Decoder 输入 | 完整目标序列右移一位（Teacher Forcing） | 从 `<sos>` 开始，逐 token 拼接 |
| 并行性 | 因果 mask 保证不偷看答案，所有位置并行预测 | 自回归逐 token 生成 |
| 优化手段 | 交叉熵损失 + 梯度裁剪 + lr warmup | KV Cache 缓存历史 K/V，避免重复计算（见 [3.3 推理优化](../chapter-3/3-3-inference.md)） |

两个常用技巧：

- **Teacher Forcing**：训练时直接喂真实目标序列（而非模型上一步的预测），配合因果 mask 实现一次前向计算所有位置的损失，收敛快且稳定
- **权重绑定（Weight Tying）**：输出投影层与目标词嵌入共享同一个权重矩阵，减少参数量且通常提升效果

### 架构复杂度对比

设序列长度 $n$、隐藏维度 $d$、卷积核大小 $k$：

| 指标 | RNN | CNN | Transformer |
|------|-----|-----|-------------|
| 每层复杂度 | $O(n \cdot d^2)$ | $O(n \cdot k \cdot d^2)$ | $O(n^2 \cdot d + n \cdot d^2)$ |
| 并行度 | O(n) 串行 | O(1) 并行 | O(1) 并行 |
| 最大信息路径 | O(n) | O(n/k) | **O(1)** |
| 适合场景 | 短序列、流式数据 | 图像、局部模式 | 长序列、大模型 |

注意力的 $O(n^2)$ 项是长文本的主要瓶颈，催生了 FlashAttention、稀疏/线性注意力等优化（见 [3.3 推理优化](../chapter-3/3-3-inference.md)）。

### 实战案例：从零实现并训练 Transformer

配套示例 `examples/1-3-dl/transformer_from_scratch.py` 用 PyTorch 从零实现完整的 Encoder-Decoder Transformer（约 300 行），并在"序列反转"玩具任务上端到端训练：

```bash
pip install torch
python examples/1-3-dl/transformer_from_scratch.py
```

- 任务：输入随机数字序列 `[9, 7, 6, 12, 12, 7, 7]`，目标输出反转序列 `[7, 7, 12, 12, 6, 7, 9]`
- 涵盖：正弦位置编码、多头注意力、Padding/Causal 两种 mask、Cross-Attention、Pre-Norm、权重绑定、Teacher Forcing 训练、贪心解码
- 预期结果：CPU 上 1-2 分钟，loss 从 ~2.56（随机猜测水平 $\ln 13$）降到接近 0，解码准确率 100%

---

## 1.3.5 训练技巧与最佳实践

### 1. 权重初始化

```python
def init_weights(model):
    """
    权重初始化策略
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # Xavier 初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
```

### 2. 学习率调度器

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Cosine Annealing - Transformer 训练常用
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# ReduceLROnPlateau - 验证集不下降时降低
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

### 3. 梯度裁剪与混合精度训练

```python
from torch.cuda.amp import GradScaler, autocast

# 梯度裁剪（防止梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 混合精度训练
scaler = GradScaler()

with autocast():
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| "网络越深越好" | 过深导致梯度消失/过拟合，ResNet 用残差连接解决 |
| "ReLU 一定比 Sigmoid 好" | 输出层用 Sigmoid/Softmax，RNN 用 Tanh |
| "Transformer 总是优于 CNN" | 小数据集 CNN 可能更好，且推理更快 |
| "位置编码可有可无，模型自己能学会顺序" | Self-Attention 对词序置换等变，没有位置编码时就是词袋模型，位置信息是必需输入 |
| "Pre-Norm 和 Post-Norm 只是写法区别" | 数值行为差异大：Post-Norm 深层训练依赖 warmup，Pre-Norm 残差路径直通、更稳定，现代大模型基本用 Pre-Norm/RMSNorm |
| "Decoder 训练时也是逐个 token 生成" | 训练用 Teacher Forcing 一次喂入整句 + 因果 mask 并行算损失；逐个自回归生成只是推理方式 |
| "Batch Size 越大越好" | 太大导致泛化差，常用 32-256 |

---

## 练习题

### 基础题

1. **反向传播推导**：推导两层 MLP 的梯度计算公式。

2. **CNN 参数计算**：输入 224×224×3，经过 Conv(64, 3×3, padding=1) 后输出尺寸？参数量？

3. **Attention 计算**：解释为什么需要除以 $\sqrt{d_k}$？

4. **位置编码**：为什么说 Self-Attention 本身"看不到"词序？正弦位置编码如何让模型感知相对位置？

5. **Cross-Attention 方向**：Decoder 的 Cross-Attention 中，为什么 Q 来自 Decoder、K/V 来自 Encoder？反过来会怎样？

### 编程题

6. 用 PyTorch 实现完整的 MNIST 分类训练流程（数据加载→模型→训练→评估）

7. 实现一个简单的 Transformer Encoder 并进行文本分类

8. **改造配套示例**：运行 `examples/1-3-dl/transformer_from_scratch.py` 后，分别尝试（a）去掉位置编码、（b）把 Pre-Norm 改成 Post-Norm，观察 loss 与准确率变化并解释原因

---

## 延伸阅读

- 📘 《Deep Learning》- Ian Goodfellow (第 6-10 章)
- 📘 《Dive into Deep Learning》- 交互式深度学习教材
- 🌐 [PyTorch 官方教程](https://pytorch.org/tutorials/)
- 🌐 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

[← 上一节：1.2 机器学习算法](1-2-ml-algorithms.md) | [下一节：1.4 AI 安全与伦理 →](1-4-ai-ethics.md)
