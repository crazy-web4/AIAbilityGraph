# 2.3 参数高效微调 (PEFT)

> 在大模型时代，全参数微调变得昂贵。本节讲解如何用**极少量可训练参数**实现接近全微调的效果。

## 学习目标

学完本节后，你将能够：

- [ ] 理解参数高效微调的必要性与核心思想
- [ ] 掌握 LoRA、QLoRA 的原理与实现
- [ ] 运用 Prompt Tuning、Prefix Tuning 等技术
- [ ] 根据资源约束选择合适的微调方案

---

## 2.3.1 为什么需要 PEFT？

### 全参数微调的问题

| 模型规模 | 参数量 | 全微调显存 (16-bit) | 消费级显卡 |
|----------|--------|---------------------|------------|
| 7B | 7B | ~28GB | ❌ |
| 13B | 13B | ~52GB | ❌ |
| 70B | 70B | ~280GB | ❌ |

### PEFT 优势

```
全参数微调 vs PEFT 对比 (Llama-2-7B 为例)

                参数量    可训练参数   显存需求    训练时间
全微调           7B       7B (100%)    80GB+      100%
LoRA             7B       8M (0.1%)    16GB       20%
QLoRA            7B       8M (0.1%)    12GB       25%
```

---

## 2.3.2 LoRA (Low-Rank Adaptation)

### 核心思想

冻结预训练权重 $W$，通过低秩分解学习增量 $\Delta W$：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, 且 $r \ll d, k$

```python
# examples/2-3-peft/lora_from_scratch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层
    
    前向传播：y = xW + xBA·scaling
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA 秩 (通常 4-64)
        alpha: 缩放系数 (通常 2*rank 或 1.0)
    """
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # LoRA 参数
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放系数
        
        # A 用高斯初始化，B 用零初始化（保证初始 ΔW=0）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 原始前向
        original = F.linear(x, self.weight)
        
        # LoRA 分支
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return original + lora
    
    def get_lora_params(self):
        """只返回 LoRA 参数用于优化"""
        return [self.lora_A, self.lora_B]


class LoRAEmbedding(nn.Module):
    """
    LoRA 应用于 Embedding 层
    
    对于词汇表大的模型，Embedding 参数量巨大
    LoRA 可以显著减少可训练参数
    """
    
    def __init__(self, vocab_size, embed_dim, rank=8):
        super().__init__()
        
        # 冻结的预训练 embedding
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        
        # LoRA 参数 (只学习低秩分解)
        self.lora_A = nn.Parameter(torch.randn(rank, embed_dim))
        self.lora_B = nn.Parameter(torch.randn(vocab_size, rank))
        
        self.rank = rank
    
    def forward(self, input_ids):
        # 原始 embedding
        orig_embed = F.embedding(input_ids, self.weight)
        
        # LoRA 增量
        lora_embed = F.embedding(input_ids, self.lora_B) @ self.lora_A
        
        return orig_embed + lora_embed


def convert_linear_to_lora(module, rank=8, alpha=16):
    """
    递归将 nn.Linear 转换为 LoRALinear
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 创建 LoRALinear
            lora_layer = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                rank=rank,
                alpha=alpha
            )
            
            # 复制原始权重
            lora_layer.weight.data = child.weight.data
            lora_layer.to(child.weight.device)
            
            # 替换
            setattr(module, name, lora_layer)
        else:
            # 递归处理子模块
            convert_linear_to_lora(child, rank, alpha)
    
    return module
```

### LoRA 配置最佳实践

| 应用场景 | rank | alpha | dropout | 适用模块 |
|----------|------|-------|---------|----------|
| 简单任务 | 4-8 | 8-16 | 0.1 | Q, V |
| 复杂推理 | 16-32 | 32-64 | 0.05 | Q, K, V, O |
| 领域适配 | 8-16 | 16-32 | 0.1 | 全部注意力 + MLP |

```python
# 示例：配置 LoRA 目标模块
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA 秩
    lora_alpha=32,           # 缩放系数
    lora_dropout=0.1,
    target_modules=[         # 应用 LoRA 的模块
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    modules_to_save=None,    # 额外可训练模块
)
```

---

## 2.3.3 QLoRA (Quantized LoRA)

### 核心创新

1. **4-bit 量化**：将预训练权重压缩到 4-bit
2. **分页优化器**：防止梯度峰值 OOM
3. **Double Quantization**：量化常数也量化

```python
# examples/2-3-peft/qlora_training.py
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


def setup_qlora_model(model_name, lora_r=16, lora_alpha=32):
    """
    配置 QLoRA 模型
    
    显存需求 (Llama-2-7B):
    - 4-bit 基础模型：~4GB
    - 梯度 + 优化器状态：~6GB
    - 总计：~12GB (单卡可训练)
    """
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # 正态 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # 双重量化
    )
    
    # 加载模型
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    # 应用 PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 输出示例:
    # trainable params: 8,388,608 || all params: 7,000,000,000
    # trainable%: 0.1198%
    
    return model, tokenizer


def train_qlora(model, tokenizer, train_dataset, output_dir):
    """
    QLoRA 训练循环
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    # 训练参数（QLoRA 需要更小的 batch）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,     # QLoRA batch size 较小
        gradient_accumulation_steps=8,
        learning_rate=2e-4,                # QLoRA 可以用更大学习率
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        fp16=False,                        # QLoRA 自身处理精度
        gradient_checkpointing=True,       # 节省显存
        optim="paged_adamw_8bit",          # 分页优化器
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    # 保存 LoRA 权重（仅几 MB）
    trainer.save_model()
```

---

## 2.3.4 Prompt Tuning & Prefix Tuning

### Prompt Tuning

**思想**：只学习软提示（soft prompts），冻结模型主体

```python
# examples/2-3-peft/prompt_tuning.py
import torch
import torch.nn as nn


class PromptTuning(nn.Module):
    """
    Prompt Tuning 实现
    
    可训练参数：prompt embeddings
    冻结参数：整个预训练模型
    """
    
    def __init__(self, model, num_prompts=20, prompt_dim=768):
        super().__init__()
        self.model = model
        self.num_prompts = num_prompts
        
        # 可学习的 prompt embeddings
        self.prompt_embeddings = nn.Embedding(num_prompts, prompt_dim)
        
        # 初始化（可选：用词向量初始化）
        nn.init.uniform_(self.prompt_embeddings.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 获取输入嵌入
        input_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 扩展 prompt 到 batch
        prompts = self.prompt_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接到输入前面
        combined_embeds = torch.cat([prompts, input_embeds], dim=1)
        combined_attention = torch.cat(
            [torch.ones(batch_size, self.num_prompts, device=input_ids.device), 
             attention_mask], 
            dim=1
        ) if attention_mask is not None else None
        
        # 前向传播（需要修改模型接受 embed 输入）
        outputs = self.model(inputs_embeds=combined_embeds, 
                            attention_mask=combined_attention,
                            labels=labels)
        
        return outputs
```

### Prefix Tuning

**与 Prompt Tuning 区别**：
- Prompt Tuning：只加到输入层
- Prefix Tuning：加到每一层的 key/value

```python
class PrefixTuning(nn.Module):
    """
    Prefix Tuning 实现
    
    在每一层的 K/V 前添加可学习前缀
    """
    
    def __init__(self, model, num_layers=12, num_heads=12, prefix_len=20, hidden_dim=768):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        
        # 每层的前缀（K 和 V 分别学习）
        self.prefix_keys = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, num_heads, prefix_len, hidden_dim // num_heads))
            for _ in range(num_layers)
        ])
        self.prefix_values = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, num_heads, prefix_len, hidden_dim // num_heads))
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, *args, **kwargs):
        # 原始前向
        outputs = self.model(input_ids, *args, output_attentions=True, **kwargs)
        
        # 在注意力计算中注入前缀（需要修改模型内部）
        # 这里简化示意
        
        return outputs
```

---

## 2.3.5 PEFT 方法对比

| 方法 | 可训练参数 | 显存 (7B) | 效果 | 适用场景 |
|------|-----------|-----------|------|----------|
| **全微调** | 100% | 80GB | 100% | 资源充足、领域差异大 |
| **LoRA** | 0.1-1% | 16GB | 95-99% | 通用场景 |
| **QLoRA** | 0.1% | 12GB | 90-95% | 资源受限 |
| **Prompt Tuning** | 0.01% | 10GB | 80-90% | 简单任务、少样本 |
| **Adapter** | 1-5% | 24GB | 90-95% | 多任务学习 |

---

## 2.3.6 实战：多场景 PEFT 对比

```python
# examples/2-3-peft/peft_comparison.py
"""
对比不同 PEFT 方法在相同任务上的效果
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import (
    LoraConfig, 
    PrefixTuningConfig, 
    PromptTuningConfig, 
    get_peft_model, 
    TaskType
)
from datasets import load_dataset


def compare_peft_methods(model_name, dataset_name, output_base="./peft_results"):
    """
    对比 LoRA、Prefix Tuning、Prompt Tuning
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载数据
    dataset = load_dataset(dataset_name, split="train")
    
    methods = {
        "lora": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
        ),
        "prefix": PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
        ),
        "prompt": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
        ),
    }
    
    results = {}
    
    for method_name, peft_config in methods.items():
        print(f"\n{'='*50}")
        print(f"训练 {method_name.upper()}...")
        print(f"{'='*50}")
        
        # 重置模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 应用 PEFT
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 训练配置
        training_args = TrainingArguments(
            output_dir=f"{output_base}/{method_name}",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            logging_steps=50,
            save_strategy="no",
        )
        
        # 训练
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        trainer.train()
        
        # 记录结果
        results[method_name] = {
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "final_loss": trainer.state.log_history[-1]['loss'],
        }
    
    # 输出对比
    print("\n" + "="*50)
    print("PEFT 方法对比结果")
    print("="*50)
    
    for name, metrics in results.items():
        print(f"{name:15s}: 可训练参数={metrics['trainable_params']:,}, 最终损失={metrics['final_loss']:.4f}")
    
    return results
```

---

## 练习题

### 基础题

1. **秩的选择**：为什么 LoRA 通常使用较小的秩（4-64）？秩过大或过小有什么问题？

2. **初始化分析**：为什么 LoRA 的 B 矩阵用零初始化？如果不用零初始化会怎样？

3. **量化误差**：4-bit 量化相比 16-bit 引入多少误差？QLoRA 为什么还能保持较好效果？

### 编程题

4. 实现 LoRA + QLoRA 的可切换训练脚本，支持运行时选择方法。

5. 在 Alpaca 数据集上对比 LoRA、QLoRA、全微调的效果和 resource 消耗。

---

## 延伸阅读

- 🌐 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 🌐 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- 🌐 [PEFT Library Documentation](https://huggingface.co/docs/peft)
- 🌐 [bitsandbytes 量化](https://github.com/TimDettmers/bitsandbytes)

---

[← 上一节：2.2 预训练与微调](2-2-pretraining.md) | [下一节：2.4 多模态大模型 →](2-4-multimodal.md)
