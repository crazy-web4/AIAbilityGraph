# 2.2 预训练与微调技术

> 掌握大模型预训练的全流程，从数据准备到分布式训练，再到下游任务微调。

## 学习目标

学完本节后，你将能够：

- [ ] 准备和清洗大规模预训练数据
- [ ] 理解和实现预训练目标函数（MLM、CLM）
- [ ] 配置分布式训练环境（DeepSpeed、FSDP）
- [ ] 执行全参数微调和高效微调

---

## 2.2.1 预训练数据准备

### 数据来源与配比

| 数据类型 | 来源 | 占比 | 处理要点 |
|----------|------|------|----------|
| 网页文本 | CommonCrawl | 60-80% | 去重、质量过滤 |
| 书籍 | Project Gutenberg | 5-10% | OCR 校正、格式统一 |
| 代码 | GitHub | 5-10% | 语言过滤、许可证检查 |
| 百科 | Wikipedia | 3-5% | 多语言、结构化 |
| 问答 | StackExchange | 2-5% | 质量评分 |

### 数据清洗流程

```python
# examples/2-2-pretraining/data_preprocessing.py
import re
import hashlib
from typing import List, Set
from datasketch import MinHash, MinHashLSH


def clean_text(text: str) -> str:
    """
    文本清洗基础流程
    """
    # 1. 移除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. 规范化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 移除过短行（可能是噪声）
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 10]
    text = '\n'.join(lines)
    
    # 4. Unicode 规范化
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text.strip()


def detect_language(text: str, target_lang: str = 'en') -> bool:
    """
    语言检测（使用 langdetect 或 fasttext）
    """
    from langdetect import detect
    
    try:
        return detect(text) == target_lang
    except:
        return False


def compute_quality_score(text: str) -> float:
    """
    文本质量评分
    
    考量因素:
    - 字符/词比例（检测关键词重复）
    - 标点符号密度
    - 停词比例
    - 链接密度
    """
    words = text.split()
    if len(words) < 10:
        return 0.0
    
    # 词重复检测
    word_counts = {}
    for w in words:
        word_counts[w.lower()] = word_counts.get(w.lower(), 0) + 1
    
    max_word_freq = max(word_counts.values()) if word_counts else 1
    repetition_penalty = 1.0 - (max_word_freq / len(words))
    
    # 标点密度
    punctuations = set('.,!?;:')
    punct_ratio = sum(1 for c in text if c in punctuations) / len(text)
    punct_score = min(1.0, punct_ratio * 10)
    
    # 综合评分
    score = 0.5 * repetition_penalty + 0.3 * punct_score + 0.2 * min(1.0, len(words) / 1000)
    
    return score


class DocumentDeduplicator:
    """
    文档级去重（使用 MinHash + LSH）
    """
    
    def __init__(self, num_perm=128, threshold=0.8):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_hashes: Set[str] = set()
    
    def compute_minhash(self, text: str) -> MinHash:
        """计算文本的 MinHash"""
        m = MinHash(num_perm=self.num_perm)
        
        # 按字符 n-gram 计算
        n = 5
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n].encode('utf-8')
            m.update(ngram)
        
        return m
    
    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """检查文档是否重复"""
        minhash = self.compute_minhash(text)
        
        # 查询 LSH
        candidates = self.lsh.query(minhash)
        
        if candidates:
            return True
        
        # 插入新的 hash
        self.lsh.insert(doc_id, minhash)
        return False


def preprocess_dataset(input_files: List[str], output_file: str):
    """
    完整数据预处理流程
    
    步骤:
    1. 读取原始文件
    2. 文本清洗
    3. 语言过滤
    4. 质量打分与过滤
    5. 去重
    6. 分词与保存
    """
    from tqdm import tqdm
    import json
    
    deduplicator = DocumentDeduplicator()
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in tqdm(input_files, desc="Processing files"):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get('text', '')
                    doc_id = data.get('id', hashlib.md5(text.encode()).hexdigest())
                    
                    # 1. 清洗
                    text = clean_text(text)
                    
                    # 2. 质量过滤
                    if len(text) < 200:
                        continue
                    
                    quality_score = compute_quality_score(text)
                    if quality_score < 0.3:
                        continue
                    
                    # 3. 去重
                    if deduplicator.is_duplicate(text, doc_id):
                        continue
                    
                    # 4. 保存
                    f_out.write(f"{text}\n")
    
    print(f"预处理完成，输出：{output_file}")
```

### 分词器训练

```python
# examples/2-2-pretraining/train_tokenizer.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import NFKC, Sequence


def train_bpe_tokenizer(data_files: list, vocab_size=32000):
    """
    训练 BPE 分词器
    
    配置类似 Llama 的分词器：
    - ByteFallback: 处理未知字符
    - Normalization: NFKC 规范化
    """
    # 初始化分词器
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 规范化
    tokenizer.normalizer = Sequence([NFKC()])
    
    # 预分词
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 特殊 token
    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<bos>", "<eos>", "<mask>"
    ]
    
    # 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 训练
    tokenizer.train(files=data_files, trainer=trainer)
    
    # 后处理（添加特殊 token）
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    
    # 保存
    tokenizer.save("tokenizer.json")
    
    return tokenizer


# 使用 SentencePiece 训练（Llama 使用）
def train_sentencepiece_tokenizer(data_file: str, vocab_size=32000):
    """
    使用 SentencePiece 训练分词器
    """
    import sentencepiece as spm
    
    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix='llama',
        vocab_size=vocab_size,
        model_type='bpe',
        max_sentence_length=16384,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_digits=True,  # 数字切分
    )
    
    print("SentencePiece 模型已保存：llama.model")
```

---

## 2.2.2 预训练目标函数

### 1. 因果语言建模 (Causal Language Modeling)

**适用**：GPT、Llama 等 Decoder-only 模型

**损失函数**:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

```python
# examples/2-2-pretraining/pretrain_clm.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """
    因果语言建模数据集
    
    将长文本切分为固定长度的序列
    """
    
    def __init__(self, text_file: str, tokenizer, max_length=2048):
        with open(text_file, 'r') as f:
            self.text = f.read()
        
        # 分词
        self.tokenized = tokenizer.encode(self.text)
        
        self.max_length = max_length
        self.stride = max_length  # 无重叠
    
    def __len__(self):
        return (len(self.tokenized) - 1) // self.stride
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length + 1
        
        # 输入和不重叠的标签（shift 1 position）
        input_ids = torch.tensor(self.tokenized[start:end-1])
        labels = torch.tensor(self.tokenized[start+1:end])
        
        return {'input_ids': input_ids, 'labels': labels}


class CausalLMTrainer:
    """
    因果语言模型训练器
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        self.scaler = torch.cuda.amp.GradScaler()  # 混合精度
    
    def train_epoch(self, dataloader, gradient_accumulation_steps=4):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)
```

### 2. 掩码语言建模 (Masked Language Modeling)

**适用**：BERT、RoBERTa 等 Encoder-only 模型

**损失函数**:
$$\mathcal{L} = -\sum_{(i, m) \in M} \log P(x_m | x_{\setminus M})$$

```python
# examples/2-2-pretraining/mlm_pretraining.py
import torch
import torch.nn as nn


class MLMDataset(Dataset):
    """
    掩码语言建模数据集（类似 BERT）
    """
    
    def __init__(self, text_file, tokenizer, max_length=512, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        
        with open(text_file, 'r') as f:
            self.lines = f.readlines()
    
    def _mask_tokens(self, input_ids):
        """
        BERT 风格 masking:
        - 80% 替换为 [MASK]
        - 10% 替换为随机 token
        - 10% 保持不变
        """
        labels = input_ids.clone()
        
        # 找到非特殊 token 的位置
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids)
        mask_candidates = torch.where(special_tokens_mask == 0)[0]
        
        # 随机选择要 mask 的位置
        n_mask = int(len(mask_candidates) * self.mlm_prob)
        mask_indices = mask_candidates[torch.randperm(len(mask_candidates))[:n_mask]]
        
        # 应用 masking 策略
        for idx in mask_indices:
            rand = torch.rand(1).item()
            if rand < 0.8:
                input_ids[idx] = self.tokenizer.mask_token_id
            elif rand < 0.9:
                input_ids[idx] = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()
            # else: 保持不变
        
        # 非 mask 位置的 label 设为 -100（忽略）
            labels[~torch.isin(torch.arange(len(input_ids)), mask_indices)] = -100
        
        return input_ids, labels
```

---

## 2.2.3 分布式训练配置

### DeepSpeed 配置

```json
// examples/2-2-pretraining/deepspeed_config.json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 8,
  
  "optimizer": {
    "type": "AdamW",
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
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000
  },
  
  "zero_optimization": {
    "stage": 2,  // ZeRO-2: 优化器 + 梯度分片
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null
  }
}
```

### 训练脚本

```python
# examples/2-2-pretraining/run_pretraining.py
"""
使用 DeepSpeed 进行分布式预训练
"""

import argparse
import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # 加载数据
    train_dataset = TextDataset(args.train_data, tokenizer, max_length=2048)
    
    # DeepSpeed 初始化
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        train_data=train_dataset,
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            loss = model_engine(**batch).loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
        
        # 保存 checkpoint
        if args.local_rank == 0:
            model_engine.save_checkpoint(args.output_dir, tag=f"epoch-{epoch}")


if __name__ == "__main__":
    main()
```

---

## 2.2.4 全参数微调

### 指令微调数据格式

```json
// examples/2-2-pretraining/data/instruction_data.json
[
  {
    "id": "001",
    "conversations": [
      {
        "role": "system",
        "content": "你是一个有帮助的 AI 助手。"
      },
      {
        "role": "user",
        "content": "什么是机器学习？"
      },
      {
        "role": "assistant",
        "content": "机器学习是人工智能的一个分支..."
      }
    ]
  }
]
```

### 微调代码

```python
# examples/2-2-pretraining/finetuning.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch


def prepare_instruction_dataset(data_path, tokenizer, max_length=2048):
    """
    准备指令微调数据集
    """
    dataset = load_dataset('json', data_files=data_path)
    
    def format_example(example):
        """格式化为对话模板"""
        messages = example['conversations']
        
        # 构建对话文本（类似 ChatML 格式）
        text = ""
        for msg in messages:
            if msg['role'] == 'system':
                text += f"<|system|>\n{msg['content']}</s>\n"
            elif msg['role'] == 'user':
                text += f"<|user|>\n{msg['content']}</s>\n"
            elif msg['role'] == 'assistant':
                text += f"<|assistant|>\n{msg['content']}</s>\n"
        
        return {'text': text}
    
    # 处理数据
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # 分词
    def tokenize(example):
        tokens = tokenizer(example['text'], truncation=True, max_length=max_length)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens
    
    dataset = dataset.map(tokenize, batched=True)
    
    return dataset


def run_finetuning():
    model_name = "meta-llama/Llama-2-7b-hf"
    data_path = "data/instruction_data.json"
    output_dir = "output/finetuned-model"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据
    train_dataset = prepare_instruction_dataset(data_path, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
```

---

## 练习题

### 基础题

1. **数据配比**：如果你的目标是多语言模型，各语言数据应如何配比？

2. **去重策略**：解释 MinHash 和局部敏感哈希的工作原理。

3. **损失计算**：推导 Causal LM 和 MLM 的梯度差异。

### 编程题

4. 实现一个支持多种预训练目标（CLM + MLM）的多任务模型。

5. 使用 DeepSpeed ZeRO-3 配置训练一个 7B 参数模型。

---

## 延伸阅读

- 🌐 [DeepSpeed Documentation](https://www.deepspeed.ai/)
- 🌐 [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- 🌐 [The Pile Dataset](https://arxiv.org/abs/2101.00027)
- 📖 《Natural Language Processing with Transformers》

---

[← 上一节：2.1 大模型架构](2-1-architecture.md) | [下一节：2.3 参数高效微调 →](2-3-peft.md)
