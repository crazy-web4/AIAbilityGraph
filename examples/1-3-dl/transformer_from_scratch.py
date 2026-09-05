"""
从零实现 Transformer（Encoder-Decoder）并在"序列反转"玩具任务上训练。

适用场景：配合 docs/chapter-1/1-3-deep-learning.md 的 1.3.4 节，
         把 Self-Attention / Cross-Attention / 位置编码 / Mask /
         残差连接与 LayerNorm 每个组件对应到可运行代码。
前置知识：PyTorch 基本用法（nn.Module、autograd、torch.optim）。

运行方式：
    pip install torch
    python transformer_from_scratch.py

预期结果：CPU 上训练约 1-2 分钟，loss 从 ~2.56（随机猜测水平 ln(13)）
         降到接近 0，贪心解码能够正确反转随机数字序列。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 特殊 token
# ----------------------------------------------------------------------
PAD_IDX = 0
SOS_IDX = 1   # 目标序列起始符 (Start Of Sequence)
EOS_IDX = 2   # 目标序列结束符 (End Of Sequence)
SYMBOL_START = 3
NUM_SYMBOLS = 10          # 普通 token: 3 ~ 12
VOCAB_SIZE = SYMBOL_START + NUM_SYMBOLS

MIN_LEN = 5               # 序列最短长度
MAX_LEN = 7               # 序列最长长度（目标序列含 EOS，最长为 MAX_LEN + 1）


# ----------------------------------------------------------------------
# 1. Multi-Head Attention（Self-Attention / Cross-Attention 通用）
# ----------------------------------------------------------------------
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
        query: (batch, Lq, d_model)
        key/value: (batch, Lk, d_model)
        mask: (batch, 1, Lq, Lk) 布尔张量，True 表示该位置允许被注意
        """
        batch_size = query.size(0)

        # 线性变换后分头：(B, L, D) -> (B, H, L, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力：scores = QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask 为 False 的位置填 -1e9，softmax 后权重趋近 0
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和并合并多头：(B, H, Lq, head_dim) -> (B, Lq, D)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)


# ----------------------------------------------------------------------
# 2. Position-wise Feed-Forward Network
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 原版论文用 ReLU，现代大模型多用 GELU / SwiGLU
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ----------------------------------------------------------------------
# 3. Encoder 层（Self-Attention + FFN，Pre-Norm 结构）
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-Norm：先 LayerNorm 再进子层，残差连接在最外层
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.ffn(normed))
        return x


# ----------------------------------------------------------------------
# 4. Decoder 层（Masked Self-Attention + Cross-Attention + FFN）
# ----------------------------------------------------------------------
class DecoderLayer(nn.Module):
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
        # 2) Cross-Attention：Q 来自 Decoder，K、V 来自 Encoder 输出
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, src_mask))
        # 3) FFN
        normed = self.norm3(x)
        x = x + self.dropout(self.ffn(normed))
        return x


# ----------------------------------------------------------------------
# 5. 完整 Transformer
# ----------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 d_ff=512, max_len=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.register_buffer("pos_enc", self._make_pos_enc(max_len, d_model), persistent=False)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # Pre-Norm 架构在最后一层输出后还要补一次 LayerNorm
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        # 输出投影到词表（权重在 __init__ 末尾与 tgt_embed 绑定）
        self.generator = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # GPT/BERT 风格的小方差初始化，必须在权重绑定之前 apply
        self.apply(self._init_weights)
        # 权重绑定：输出投影与目标词嵌入共享矩阵（减少参数、增强泛化）
        self.generator.weight = self.tgt_embed.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.weight[PAD_IDX])

    @staticmethod
    def _make_pos_enc(max_len, d_model):
        """正弦位置编码：PE(pos, 2i)=sin(pos/10000^(2i/d))，2i+1 维取 cos"""
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _embed(self, embedding, tokens):
        # 词嵌入 + 位置编码。
        # 原版论文会把嵌入乘以 sqrt(d_model)；GPT-2/Llama 等现代模型配合
        # 小方差初始化（见 _init_weights）直接相加，训练更稳定。
        seq_len = tokens.size(1)
        x = embedding(tokens)
        return self.dropout(x + self.pos_enc[:, :seq_len, :])

    def encode(self, src, src_mask):
        x = self._embed(self.src_embed, src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self._embed(self.tgt_embed, tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.dec_norm(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(dec_out)  # (B, L_tgt, vocab_size)


# ----------------------------------------------------------------------
# 6. Mask 工具函数
# ----------------------------------------------------------------------
def make_pad_mask(tokens):
    """Padding mask：(B, L) -> (B, 1, 1, L)，True 表示非 PAD 的有效位置"""
    return (tokens != PAD_IDX).unsqueeze(1).unsqueeze(1)


def make_causal_mask(length, device):
    """因果 mask：(1, 1, L, L) 下三角为 True，保证位置 t 只能看到 <= t 的位置"""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


# ----------------------------------------------------------------------
# 7. 玩具任务数据：把随机数字序列反转
#    src: [3, 7, 2(token 5), ...]      tgt: SOS + 反转序列 + EOS
# ----------------------------------------------------------------------
def make_batch(batch_size, device):
    lengths = torch.randint(MIN_LEN, MAX_LEN + 1, (batch_size,))
    src = torch.full((batch_size, MAX_LEN), PAD_IDX, dtype=torch.long)
    tgt_out = torch.full((batch_size, MAX_LEN + 1), PAD_IDX, dtype=torch.long)

    for i, length in enumerate(lengths):
        seq = torch.randint(SYMBOL_START, SYMBOL_START + NUM_SYMBOLS, (length,))
        src[i, :length] = seq
        tgt_out[i, :length] = seq.flip(0)   # 反转
        tgt_out[i, length] = EOS_IDX

    # Teacher forcing：解码器输入 = SOS + 目标序列（去掉最后一个 token）
    tgt_in = torch.full_like(tgt_out, PAD_IDX)
    tgt_in[:, 0] = SOS_IDX
    tgt_in[:, 1:] = tgt_out[:, :-1]

    return src.to(device), tgt_in.to(device), tgt_out.to(device)


# ----------------------------------------------------------------------
# 8. 贪心解码（自回归推理）
# ----------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN + 1):
    model.eval()
    src_mask = make_pad_mask(src)
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), SOS_IDX, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        tgt_mask = make_pad_mask(ys) & make_causal_mask(ys.size(1), ys.device)
        logits = model.generator(model.decode(ys, memory, tgt_mask, src_mask))
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
    return ys


# ----------------------------------------------------------------------
# 9. 训练与验证
# ----------------------------------------------------------------------
def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=128, num_heads=4, num_layers=3, d_ff=512,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 前向传播形状自检
    src_demo, tgt_in_demo, tgt_out_demo = make_batch(4, device)
    src_mask = make_pad_mask(src_demo)
    tgt_mask = make_pad_mask(tgt_in_demo) & make_causal_mask(tgt_in_demo.size(1), device)
    logits = model(src_demo, tgt_in_demo, src_mask, tgt_mask)
    assert logits.shape == (4, MAX_LEN + 1, VOCAB_SIZE), logits.shape
    print(f"前向传播形状自检通过: {tuple(logits.shape)}\n")

    model.train()
    for step in range(1, 1001):
        src, tgt_in, tgt_out = make_batch(128, device)
        src_mask = make_pad_mask(src)
        # Decoder Self-Attention：padding mask 与因果 mask 取交集
        tgt_mask = make_pad_mask(tgt_in) & make_causal_mask(tgt_in.size(1), device)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:>3d} | loss {loss.item():.4f}")

    # 推理验证
    model.eval()
    src, _, tgt_out = make_batch(5, device)
    pred = greedy_decode(model, src)

    print("\n===== 贪心解码结果（数字为 token id）=====")
    correct = total = 0
    for i in range(src.size(0)):
        src_seq = [t for t in src[i].tolist() if t != PAD_IDX]
        gold = [t for t in tgt_out[i].tolist() if t not in (PAD_IDX, EOS_IDX)]
        hyp = [t for t in pred[i].tolist() if t not in (PAD_IDX, SOS_IDX, EOS_IDX)][:len(gold)]
        ok = hyp == gold
        correct += sum(a == b for a, b in zip(hyp, gold))
        total += len(gold)
        print(f"输入: {src_seq} | 期望反转: {gold} | 模型输出: {hyp} | {'✓' if ok else '✗'}")
    print(f"\ntoken 准确率: {correct / total:.1%}")


if __name__ == "__main__":
    train()
