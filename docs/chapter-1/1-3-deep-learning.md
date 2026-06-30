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

### Self-Attention 机制

**注意力公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
# examples/1-3-dl/transformer_from_scratch.py
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"
        
        # Q, K, V 线性变换
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        
        # 输出线性变换
        self.out_linear = nn.Linear(embed_size, embed_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention 分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Mask（用于解码器防止看未来）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax + V
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        
        return self.out_linear(context)
```

### Position-wise Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, ff_hidden)
        self.linear2 = nn.Linear(ff_hidden, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### 完整的 Transformer Encoder

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden, dropout)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm 架构（更稳定）
        # Attention + 残差
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN + 残差
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 位置编码
        self.pos_encoding = self._generate_position_encoding(max_seq_len, embed_size)
        
        # Encoder 层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, ff_hidden, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def _generate_position_encoding(self, max_len, embed_size):
        """生成正弦位置编码"""
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, embed_size)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len) - 词索引
        seq_len = x.size(1)
        
        # 嵌入 + 位置编码
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过所有 Encoder 层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
```

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
| "Batch Size 越大越好" | 太大导致泛化差，常用 32-256 |

---

## 练习题

### 基础题

1. **反向传播推导**：推导两层 MLP 的梯度计算公式。

2. **CNN 参数计算**：输入 224×224×3，经过 Conv(64, 3×3, padding=1) 后输出尺寸？参数量？

3. **Attention 计算**：解释为什么需要除以 $\sqrt{d_k}$？

### 编程题

4. 用 PyTorch 实现完整的 MNIST 分类训练流程（数据加载→模型→训练→评估）

5. 实现一个简单的 Transformer Encoder 并进行文本分类

---

## 延伸阅读

- 📘 《Deep Learning》- Ian Goodfellow (第 6-10 章)
- 📘 《Dive into Deep Learning》- 交互式深度学习教材
- 🌐 [PyTorch 官方教程](https://pytorch.org/tutorials/)
- 🌐 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

[← 上一节：1.2 机器学习算法](1-2-ml-algorithms.md) | [下一节：1.4 AI 安全与伦理 →](1-4-ai-ethics.md)
