"""
2.1 大模型架构 - Transformer 组件实现

实现 Transformer 核心组件：
- Multi-Head Attention
- Feed-Forward Network
- Positional Encoding
- 完整的 Transformer 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码

    使用正弦和余弦函数生成位置信息
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 使用指数衰减生成不同频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数列用 sin，奇数列用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        # 注册为 buffer（不更新）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch_size, seq_len, d_model)
        返回:
            添加位置编码的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    支持:
    - Encoder 自注意力
    - Decoder 掩码自注意力
    - Encoder-Decoder 交叉注意力
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """分割多头"""
        batch_size, seq_len, _ = x.shape
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并多头"""
        batch_size, _, seq_len, _ = x.shape
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        参数:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: (batch, 1, 1, seq_len_k) 或 None

        返回:
            attention_output: (batch, seq_len_q, d_model)
        """
        batch_size = query.shape[0]

        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割多头
        Q = self._split_heads(Q)  # (batch, heads, seq_len_q, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq_len_q, seq_len_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq_len_q, d_k)

        # 合并多头 + 输出变换
        attn_output = self._combine_heads(attn_output)
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    """
    前馈神经网络

    两层线性变换 + ReLU 激活
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 多头注意力 + 残差连接 + LayerNorm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder 层

    包含:
    - 掩码自注意力
    - 交叉注意力
    - FFN
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 掩码自注意力
        attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力
        cross_output = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # FFN
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    """
    完整 Transformer 模型

    可用于:
    - 机器翻译
    - 文本生成
    - 序列标注
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        self.d_model = d_model

        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成 Decoder 掩码（防止看未来）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        参数:
            src: 源序列 (batch, src_len)
            tgt: 目标序列 (batch, tgt_len)
            src_mask: 源序列 padding 掩码
            tgt_mask: 目标序列掩码

        返回:
            output: (batch, tgt_len, tgt_vocab_size)
        """
        device = src.device

        # Encoder 输入
        src_embed = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))

        # Decoder 输入
        tgt_embed = self.dropout(self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 生成目标序列掩码
        if tgt_mask is None:
            tgt_mask = self._generate_mask(tgt.shape[1], device)

        # Encoder
        encoder_output = src_embed
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        # Decoder
        decoder_output = tgt_embed
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)

        # 输出
        output = self.fc_out(decoder_output)

        return output


# ========== 使用示例 ==========

def demo_transformer():
    """Transformer 使用演示"""

    # 模型参数
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params / 1e6:.2f}M")

    # 模拟输入
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 前向传播
    output = model(src, tgt)
    print(f"输出形状：{output.shape}")  # (batch, tgt_seq_len, tgt_vocab_size)

    # 测试注意力可视化
    print("\n多头注意力测试:")
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadAttention(d_model=512, num_heads=8)
    attn_output = attention(x, x, x)
    print(f"注意力输出形状：{attn_output.shape}")


if __name__ == "__main__":
    demo_transformer()
