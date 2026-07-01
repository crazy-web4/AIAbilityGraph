# 2.2 预训练与微调技术 - 关键知识点详解

> 本节为 2.2 节的补充知识点，包含数据准备清单、训练配置模板、故障排查指南。

---

## 知识点 1: 预训练数据准备清单

### 数据质量检查清单

在开始预训练前，确保完成以下检查：

```
□ 数据收集
  □ 原始语料来源明确 (CommonCrawl, C4, Wikipedia, Books, Code 等)
  □ 数据版权和许可证已确认
  □ 数据量达到目标 (例如：7B 模型至少需要 1T tokens)

□ 数据清洗
  □ 去除 HTML 标签和特殊字符
  □ 标准化 Unicode 编码
  □ 去除重复内容 (文档级去重)
  □ 过滤低质量内容 (基于启发式规则)
  □ 毒性内容过滤

□ 数据混合
  □ 各数据源比例确定 (见下表推荐配置)
  □ 混合策略文档化
  □ 验证混合后数据分布

□ Tokenization
  □ Tokenizer 词汇表大小确定 (32K-100K)
  □ 特殊 token 定义 (< BOS >, < EOS >, < PAD >等)
  □ Token 效率检查 (chars/token 比率)
```

### 推荐数据源混合比例

| 数据类型 | Llama2 配置 | OPT 配置 | 适用场景 |
|---------|------------|---------|---------|
| **网页爬取** | 67% | 70% | 通用知识 |
| **百科 (Wikipedia)** | 3% | 5% | 事实性知识 |
| **书籍** | 4.5% | 5% | 长上下文理解 |
| **代码** | 4.5% | 5% | 逻辑推理能力 |
| **GitHub** | 4.5% | 5% | 代码能力 |
| **学术论文** | 3% | 5% | 专业领域知识 |
| **新闻** | 3% | 3% | 时效性内容 |
| **论坛/对话** | 6% | 2% | 对话能力 |

### 数据清洗代码示例

```python
# examples/2-2-pretraining/data_cleaning.py
import re
import os
from typing import List, Dict
from bs4 import BeautifulSoup
from langdetect import detect
import hashlib


class TextCleaner:
    """
    预训练数据清洗管道
    
    功能：
    1. HTML 标签移除
    2. 特殊字符清理
    3. 语言过滤
    4. 重复内容检测
    5. 质量评分
    """
    
    def __init__(self, min_words=50, max_words=10000, target_lang='en'):
        self.min_words = min_words
        self.max_words = max_words
        self.target_lang = target_lang
        self.seen_hashes = set()
    
    def clean_html(self, text: str) -> str:
        """移除 HTML 标签"""
        soup = BeautifulSoup(text, 'html.parser')
        
        # 移除 script 和 style 标签
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()
        
        return soup.get_text(separator=' ')
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode 标准化"""
        import unicodedata
        # NFKC 规范化：兼容字符 -> 标准形式
        return unicodedata.normalize('NFKC', text)
    
    def remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        # 保留基本的标点符号
        text = re.sub(r'[^\w\s.,!?;:()"\']', ' ', text)
        # 多个空格合并为一个
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def filter_by_language(self, text: str) -> bool:
        """语言过滤"""
        try:
            detected = detect(text[:1000])  # 只检测前 1000 字符
            return detected == self.target_lang
        except:
            return False
    
    def filter_by_length(self, text: str) -> bool:
        """长度过滤"""
        words = text.split()
        return self.min_words <= len(words) <= self.max_words
    
    def dedup_hash(self, text: str) -> bool:
        """文档级去重 (MinHash 简化版)"""
        # 使用文本哈希进行简单去重
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return False  # 重复
        
        self.seen_hashes.add(text_hash)
        return True  # 新文档
    
    def quality_score(self, text: str) -> float:
        """
        简单质量评分
        
        评分因子:
        - 标点符号密度
        - 大写字母比例 (英文)
        - 平均词长
        """
        words = text.split()
        if not words:
            return 0.0
        
        # 标点符号密度
        punct_count = sum(1 for c in text if c in '.,!?;:')
        punct_density = punct_count / len(text) if text else 0
        
        # 理想标点密度约 5-15%
        punct_score = 1.0 if 0.05 <= punct_density <= 0.15 else 0.5
        
        # 平均词长 (英文单词平均 4-6 字符)
        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_score = 1.0 if 4 <= avg_word_len <= 8 else 0.5
        
        return (punct_score + word_len_score) / 2
    
    def clean_document(self, text: str) -> Dict:
        """
        完整清洗流程
        
        返回：
        {
            'text': str,           # 清洗后的文本
            'passed': bool,        # 是否通过所有过滤器
            'quality_score': float,# 质量评分
            'reason': str          # 拒绝原因 (如果 failed)
        }
        """
        # Step 1: HTML 清理
        text = self.clean_html(text)
        
        # Step 2: Unicode 标准化
        text = self.normalize_unicode(text)
        
        # Step 3: 特殊字符清理
        text = self.remove_special_chars(text)
        
        # Step 4: 长度过滤
        if not self.filter_by_length(text):
            return {'text': '', 'passed': False, 'quality_score': 0, 
                    'reason': f'长度不符合要求，当前词数：{len(text.split())}'}
        
        # Step 5: 语言过滤
        if not self.filter_by_language(text):
            return {'text': '', 'passed': False, 'quality_score': 0,
                    'reason': '语言不符合'}
        
        # Step 6: 去重检查
        if not self.dedup_hash(text):
            return {'text': '', 'passed': False, 'quality_score': 0,
                    'reason': '重复内容'}
        
        # Step 7: 质量评分
        score = self.quality_score(text)
        if score < 0.3:  # 质量阈值
            return {'text': '', 'passed': False, 'quality_score': score,
                    'reason': f'质量评分过低：{score:.2f}'}
        
        return {'text': text, 'passed': True, 'quality_score': score, 'reason': ''}


# 使用示例
def process_dataset(input_files: List[str], output_file: str):
    """批量处理数据集"""
    cleaner = TextCleaner()
    
    passed_count = 0
    rejected_counts = {'length': 0, 'language': 0, 'duplicate': 0, 'quality': 0}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = cleaner.clean_document(line.strip())
                    
                    if result['passed']:
                        passed_count += 1
                        out_f.write(result['text'] + '\n')
                    else:
                        # 统计拒绝原因
                        if '长度' in result['reason']:
                            rejected_counts['length'] += 1
                        elif '语言' in result['reason']:
                            rejected_counts['language'] += 1
                        elif '重复' in result['reason']:
                            rejected_counts['duplicate'] += 1
                        elif '质量' in result['reason']:
                            rejected_counts['quality'] += 1
    
    print(f"通过：{passed_count}")
    print(f"拒绝统计: {rejected_counts}")


if __name__ == "__main__":
    # 示例用法
    files = ["data/raw_part_*.txt"]
    process_dataset(files, "data/cleaned_output.txt")
```

---

## 知识点 2: 分布式训练配置模板

### DeepSpeed + PyTorch 完整训练脚本

```python
# examples/2-2-pretraining/distributed_pretraining.py
"""
大模型预训练完整示例

使用 DeepSpeed ZeRO-3 + FlashAttention 2 + Gradient Checkpointing

运行命令:
    deepspeed --num_gpus=8 pretraining.py \
        --deepspeed_config ds_config_zero3.json
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig
import deepspeed
import argparse
from typing import Dict, Any
from tqdm import tqdm


class SimpleLMModel(nn.Module):
    """
    简化语言模型 (用于演示)
    
    实际训练应使用 Llama/Mistral 等完整架构
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dim_ff, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_ff,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN 架构
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        causal_mask = self.causal_mask[:x.size(1), :x.size(1)]
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        for layer in self.layers:
            x = layer(x, is_causal=True)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


class PretrainingDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, file_path: str, tokenizer, max_seq_len=2048):
        import json
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 加载数据
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data['text'])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)  # 自回归任务 labels=input_ids
        }


def train_model():
    """分布式训练主循环"""
    
    # ========== 参数配置 ==========
    config = {
        # 模型配置
        'vocab_size': 32000,
        'embed_dim': 1024,
        'num_heads': 16,
        'num_layers': 24,
        'dim_ff': 4096,
        'max_seq_len': 2048,
        
        # 训练配置
        'batch_size': 4,  # per GPU
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'warmup_steps': 1000,
        'max_steps': 100000,
        'gradient_accumulation_steps': 8,
        'gradient_clip': 1.0,
        
        # DeepSpeed 配置
        'fp16': True,
        'gradient_checkpointing': True,
    }
    
    # ========== 初始化 ==========
    deepspeed.init_distributed(dist_backend='nccl')
    
    # 创建模型
    model = SimpleLMModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_ff=config['dim_ff'],
        max_seq_len=config['max_seq_len']
    )
    
    # 梯度检查点 (节省显存)
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    
    # ========== DeepSpeed 初始化 ==========
    ds_config = {
        "train_batch_size": config['batch_size'] * 8 * config['gradient_accumulation_steps'],
        "train_micro_batch_size_per_gpu": config['batch_size'],
        "gradient_accumulation_steps": config['gradient_accumulation_steps'],
        "gradient_clipping": config['gradient_clip'],
        "steps_per_print": 100,
        
        # FP16 训练
        "fp16": {
            "enabled": config['fp16'],
            "auto_cast": False,
            "initial_scale_power": 16,
            "loss_scale_window": 1000
        },
        
        # ZeRO-2 优化 (平衡速度和显存)
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        
        # 优化器
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config['learning_rate'],
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": config['weight_decay']
            }
        },
        
        # 学习率调度器
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config['learning_rate'],
                "warmup_num_steps": config['warmup_steps'],
                "total_num_steps": config['max_steps']
            }
        }
    }
    
    # 创建 DeepSpeed 模型
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # ========== 数据加载 ==========
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = PretrainingDataset('data/pretrain_data.jsonl', tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # ========== 训练循环 ==========
    model_engine.train()
    
    for epoch in range(10):  # 10 epochs
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for step, batch in pbar:
            # 数据移到 GPU
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)
            
            # 前向传播
            outputs = model_engine(input_ids)
            
            # 计算损失 (忽略 padding)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            # 日志
            if step % 100 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}, LR: {scheduler.get_lr()[0]:.2e}")
                
                # 保存检查点
                if step > 0 and step % 5000 == 0:
                    model_engine.save_checkpoint(f'checkpoints/epoch{epoch}_step{step}')
    
    # 保存最终模型
    model_engine.save_checkpoint('checkpoints/final')


if __name__ == "__main__":
    train_model()
```

---

## 知识点 3: 训练故障排查指南

### 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| **OOM (显存不足)** | 批次太大/梯度累积太高 | 减少 batch_size, 启用 ZeRO-3, 使用 Gradient Checkpointing |
| **损失爆炸 (NaN)** | 学习率太高/梯度累积不稳 | 降低学习率，增加 warmup, 减小梯度裁剪阈值 |
| **训练不收敛** | 数据质量问题 | 检查数据清洗，验证 tokenizer |
| **训练速度慢** | I/O 瓶颈/通信开销 | 增加数据加载 worker，使用更快的存储 (NVMe)，优化通信 |
| **Loss 震荡** | 批次太小/学习率太高 | 增加有效批次大小，降低学习率，延长 warmup |

### Loss 监控脚本

```python
# examples/2-2-pretraining/monitor_training.py
import json
from datetime import datetime


def analyze_training_logs(log_file: str):
    """分析训练日志"""
    
    losses = []
    lrs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'loss' in line:
                data = json.loads(line)
                losses.append(data['loss'])
                lrs.append(data['learning_rate'])
    
    # 计算移动平均
    import numpy as np
    
    window = 100
    ma_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
    
    # 分析趋势
    if len(ma_losses) > 100:
        recent_trend = ma_losses[-100] - ma_losses[-200]
        
        if recent_trend > 0.1:
            print("⚠️ 警告：Loss 呈上升趋势，考虑降低学习率")
        elif recent_trend < -0.5:
            print("✅ Loss 正在良好下降")
        else:
            print("⚠️ Loss 下降缓慢，考虑调整学习率或数据混合")
    
    print(f"平均 Loss: {np.mean(losses):.4f}")
    print(f"最低 Loss: {np.min(losses):.4f}")
    print(f"最高 Loss: {np.max(losses):.4f}")


if __name__ == "__main__":
    analyze_training_logs("training.log")
```

---

## 练习题

### 练习 1: 设计数据混合策略

给定以下场景，设计数据混合比例：

1. **客服对话模型** - 目标是提升客服回复质量
2. **代码助手模型** - 目标是提升代码生成能力
3. **学术研究模型** - 目标是理解专业文献

### 练习 2: 显存计算

计算以下配置的显存需求：

- 模型：7B 参数，FP16 训练
- 序列长度：4096
- 批次大小：8 per GPU
- 使用 ZeRO-2 优化

---

## 延伸阅读

- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [Llama2 训练细节](https://arxiv.org/abs/2307.09288)

---

[← 返回 2.2 主文档](2-2-pretraining.md) | [下一节：参数高效微调 →](2-3-peft.md)
