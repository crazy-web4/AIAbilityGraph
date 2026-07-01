# 2.3 参数高效微调 (PEFT) - 关键知识点详解

> 本节为 2.3 节的补充知识点，包含 LoRA/QLoRA/P-Tuning 详解、实战代码、调参指南。

---

## 知识点 1: PEFT 方法对比

### 主流 PEFT 技术对比表

| 方法 | 可训练参数 | 显存需求 | 训练速度 | 效果 | 适用场景 |
|------|-----------|---------|---------|------|---------|
| **全参数微调** | 100% | 100% | 100% | 基准 | 小模型/算力充足 |
| **LoRA** | 0.1-1% | ~30% | 快 | ≈全量 | 主流选择 |
| **QLoRA** | 0.1-1% | ~20% | 稍慢 | ≈LoRA | 消费级 GPU |
| **Adapter** | 1-5% | ~50% | 中等 | 稍逊 | 特定任务 |
| **Prefix Tuning** | 0.1-1% | ~25% | 快 | 中等 | 生成任务 |
| **P-Tuning v2** | 0.1-1% | ~25% | 快 | 较好 | 理解任务 |

### LoRA 原理详解

```
LoRA (Low-Rank Adaptation) 核心思想

原始层：h = Wx + b  (W: d×k)

LoRA 改造:
     W = W₀ + ΔW = W₀ + BA
     其中 B: d×r, A: r×k, r ≪ d,k

前向传播:
h = W₀x + BAx + b

关键设计:
1. W₀ 冻结 (不更新)
2. A 随机高斯初始化，B 零初始化
3. 初始状态 ΔW=0，保证训练起点与预训练一致
4. 只在 A 和 B 上反向传播
```

### LoRA 完整实现

```python
# examples/2-3-peft/lora_implementation.py
"""
LoRA (Low-Rank Adaptation) 完整实现

包含:
1. LoRALinear 层实现
2. 转换为 LoRA 模型
3. 与全量微调对比
4. QLoRA 量化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class LoRALinear(nn.Module):
    """
    LoRA 线性层
    
    前向公式：y = xW₀ + xBA·scale
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA 秩 (通常 4-64，越大表达能力越强)
        alpha: 缩放系数 (通常设为 rank 或 2*rank)
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
        use_rslora: bool = False  # 是否使用 Rank-Stabilized LoRA
    ):
        super().__init__()
        
        # 原始权重 (冻结)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False  # 冻结
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features),
            requires_grad=False
        )
        
        # LoRA 参数
        self.rank = rank
        self.alpha = alpha
        
        # RSLoRA 缩放 (更稳定)
        if use_rslora:
            self.scale = alpha / (rank ** 0.5)
        else:
            self.scale = alpha / rank
        
        # LoRA 矩阵 A 和 B
        # A: 高斯初始化，B: 零初始化 (保证初始 ΔW=0)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) / torch.sqrt(torch.tensor(rank, dtype=torch.float))
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性 (无梯度)
        with torch.no_grad():
            base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 分支
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scale
        
        return base_output + lora_output
    
    def get_lora_params(self) -> List[torch.Parameter]:
        """获取 LoRA 参数 (用于优化器)"""
        return [self.lora_A, self.lora_B]
    
    def merge_weights(self) -> torch.Tensor:
        """合并 LoRA 权重到原始权重 (推理时使用)"""
        delta_w = (self.lora_B @ self.lora_A) * self.scale
        return self.weight + delta_w


class LoRAEmbedding(nn.Module):
    """
    LoRA Embedding
    
    对于大词表模型，Embedding 参数量巨大
    LoRA 可以显著减少可训练参数
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int = 8):
        super().__init__()
        
        # 冻结的预训练 embedding
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim),
            requires_grad=False
        )
        
        # LoRA 参数
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(rank, embedding_dim))
        self.lora_B = nn.Parameter(torch.zeros(num_embeddings, rank))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 原始 embedding
        orig_embed = F.embedding(input_ids, self.weight)
        
        # LoRA 增量
        lora_embed = F.embedding(input_ids, self.lora_B) @ self.lora_A
        
        return orig_embed + lora_embed


def convert_linear_to_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    递归将 nn.Linear 转换为 LoRALinear
    
    Args:
        module: 原始模型
        rank: LoRA 秩
        alpha: 缩放系数
        target_modules: 目标模块名称列表 (None 表示全部)
    
    Returns:
        转换后的模型
    """
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if target_modules is None or name in target_modules:
                # 创建 LoRA 层
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=rank,
                    alpha=alpha
                )
                
                # 复制原始权重
                lora_layer.weight.data = child.weight.data
                lora_layer.bias.data = child.bias.data if child.bias is not None else torch.zeros(lora_layer.out_features)
                lora_layer.to(child.weight.device)
                
                # 替换
                setattr(module, name, lora_layer)
        
        else:
            # 递归处理子模块
            convert_linear_to_lora(child, rank, alpha, target_modules)
    
    return module


# ==================== 使用示例 ====================

def lora_finetune_demo():
    """
    LoRA 微调示例
    
    对比:
    1. 全量微调参数
    2. LoRA 参数
    3. 显存占用
    """
    
    print("=" * 70)
    print("LoRA 微调参数对比")
    print("=" * 70)
    
    # 创建一个简单的 Transformer
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        ),
        num_layers=6
    )
    
    # 计算参数量
    def count_params(m):
        return sum(p.numel() for p in m.parameters())
    
    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n原始模型:")
    print(f"  总参数：{total_params:,}")
    print(f"  可训练：{trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 转换为 LoRA 模型
    lora_model = convert_linear_to_lora(model, rank=16, alpha=32)
    
    # 冻结原始参数
    for param in lora_model.parameters():
        param.requires_grad = False
    
    # 只开放 LoRA 参数
    for module in lora_model.modules():
        if isinstance(module, LoRALinear):
            for param in module.get_lora_params():
                param.requires_grad = True
    
    lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    
    print(f"\nLoRA 模型:")
    print(f"  总参数：{count_params(lora_model):,} (不变)")
    print(f"  可训练：{lora_trainable:,} ({lora_trainable/total_params*100:.2f}%)")
    print(f"  参数减少：{(total_params - lora_trainable) / total_params * 100:.1f}%")
    
    # 显存估算
    # Adam 优化器状态：2×参数量×4 bytes (FP32)
    # 梯度：1×参数量×2 bytes (FP16)
    full_mem = total_params * 4 * 4 / (1024**3)  # GB
    lora_mem = lora_trainable * 4 * 4 / (1024**3)  # GB
    
    print(f"\n显存估算 (Adam FP32):")
    print(f"  全量微调：{full_mem:.2f} GB")
    print(f"  LoRA 微调：{lora_mem:.2f} GB")
    print(f"  显存节省：{(1 - lora_mem/full_mem) * 100:.1f}%")


if __name__ == "__main__":
    lora_finetune_demo()
```

---

## 知识点 2: QLoRA 量化微调

### QLoRA 原理

```
QLoRA = LoRA + 4bit 量化

核心技巧:
1. 4bit NormalFloat (NF4) 量化预训练权重
2. 双重量化：量化器的常量也量化
3. 分页优化器：防止梯度检查点内存峰值
4. LoRA 适配器：在量化权重上添加
```

### QLoRA 实战代码

```python
# examples/2-3-peft/qlora_finetune.py
"""
QLoRA 量化微调实战

基于 bitsandbytes 的 4bit 量化 + LoRA 适配器

环境要求:
pip install bitsandbytes>=0.39.0
pip install peft>=0.4.0
pip install transformers>=4.31.0
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset


def create_4bit_config():
    """
    创建 4bit 量化配置
    
    NF4 (Normal Float 4) 专为正态分布权重设计
    比标准 4bit 量化效果更好
    """
    
    return BitsAndBytesConfig(
        load_in_4bit=True,              # 启用 4bit 量化
        bnb_4bit_quant_type="nf4",      # NF4 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
        bnb_4bit_use_double_quant=True,  # 双重量化 (额外节省显存)
        llm_int8_threshold=6.0,          # INT8 异常值阈值
        llm_int8_has_fp16_weight=False,
    )


def create_lora_config(
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: list = None
) -> LoraConfig:
    """
    创建 LoRA 配置
    
    Args:
        rank: LoRA 秩 (越大表达能力越强)
        alpha: 缩放系数 (通常是 rank 的 2 倍)
        dropout: LoRA dropout
        target_modules: 目标模块 (None 表示自动推断)
    """
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",  # 不训练 bias
        inference_mode=False,
    )


def print_trainable_parameters(model):
    """
    打印可训练参数统计
    """
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"可训练参数：{trainable_params:,} / {all_param:,} "
          f"({trainable_params / all_param * 100:.2f}%)")


def qlora_finetune(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    dataset_name: str = "timdettmers/openassistant-guanaco",
    output_dir: str = "./qlora-output",
    rank: int = 64,
    batch_size: int = 4,
    max_steps: int = 1000
):
    """
    QLoRA 微调完整流程
    
    Args:
        model_name: 预训练模型名称
        dataset_name: 数据集名称
        output_dir: 输出目录
        rank: LoRA 秩
        batch_size: 批次大小
        max_steps: 最大训练步数
    """
    
    print("=" * 70)
    print("QLoRA 4bit 量化微调")
    print("=" * 70)
    
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 避免警告
    
    # 2. 加载 4bit 量化模型
    print("\n1. 加载 4bit 量化模型...")
    
    quant_config = create_4bit_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",  # 自动分配到 GPU
        trust_remote_code=True
    )
    
    # 3. 准备模型 (添加 gradient checkpointing 等)
    model = prepare_model_for_kbit_training(model)
    
    # 4. 配置 LoRA
    # 自动推断目标模块
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "gate" not in name:
            if module.out_features % 64 == 0:  # 确保能被整除
                target_modules.append(name.split(".")[-1])
    
    target_modules = list(set(target_modules))
    print(f"\n目标模块：{target_modules}")
    
    lora_config = create_lora_config(
        rank=rank,
        alpha=rank * 2,
        dropout=0.05,
        target_modules=target_modules
    )
    
    # 5. 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    print_trainable_parameters(model)
    
    # 6. 加载数据集
    print("\n2. 加载数据集...")
    dataset = load_dataset(dataset_name, split="train")
    
    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=max_steps,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # 分页优化器
        report_to="none",
    )
    
    # 8. 创建 Trainer
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # 9. 开始训练
    print("\n3. 开始训练...")
    print(f"   Steps: {max_steps}")
    print(f"   Learning Rate: 2e-4")
    print(f"   Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # 10. 保存模型
    print("\n4. 保存模型...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ 模型已保存至：{output_dir}")
    print(f"  使用：peft_model = PeftModel.from_pretrained(base_model, output_dir)")
    
    return model, tokenizer


# ==================== 推理示例 ====================

def qlora_inference(
    base_model_name: str = "meta-llama/Llama-2-7b-hf",
    adapter_path: str = "./qlora-output",
    prompt: str = "什么是人工智能？"
):
    """
    加载 QLoRA 适配器进行推理
    """
    from peft import PeftModel
    
    # 加载基础模型 (4bit)
    quant_config = create_4bit_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # 推理
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n输入：{prompt}")
    print(f"\n输出：{response}")
    
    return response


if __name__ == "__main__":
    # 微调示例 (取消注释运行)
    # qlora_finetune(
    #     model_name="mistralai/Mistral-7B-v0.1",
    #     rank=64,
    #     batch_size=2,
    #     max_steps=500
    # )
    
    # 推理示例
    print("QLoRA 推理示例")
    print("运行 qlora_inference() 加载适配器进行测试")
```

---

## 知识点 3: P-Tuning v2 详解

### P-Tuning v2 vs Prefix Tuning

| 特性 | Prefix Tuning | P-Tuning v2 | P-Tuning v2 改进 |
|------|--------------|-------------|-----------------|
| **Prompt 添加位置** | 仅输入层 | 每层都添加 | 更深层优化 |
| **Prompt 维度** | 序列维度 | 序列 + 层维度 | 更多可训练参数 |
| **效果** | 中等 | 更好 | 接近全量微调 |
| **适用任务** | 生成任务 | 理解+生成 | 通用 |

### P-Tuning v2 实现

```python
# examples/2-3-peft/p_tuning_v2.py
"""
P-Tuning v2 实现

连续 Prompt Tuning 的改进版本
在每一层的输入都添加可学习的 prompt vectors
"""

import torch
import torch.nn as nn
from typing import Optional


class PromptEncoder(nn.Module):
    """
    Prompt Encoder for P-Tuning v2
    
    为每一层生成 continuous prompt embeddings
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_prompts: int = 10,
        prompt_dim: int = 512
    ):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # 可学习的 prompt 嵌入 (每层独立)
        # shape: (num_layers, 2, num_prompts, hidden_size)
        # 2 表示 key 和 value 的 prompt
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, 2, num_prompts, hidden_size)
        )
        
        # 初始化
        nn.init.normal_(self.prompt_embeddings, std=0.02)
    
    def forward(
        self,
        batch_size: int,
        device: torch.device
    ) -> tuple:
        """
        获取 prompt embeddings
        
        Returns:
            past_key_values: tuple of (key_prompt, value_prompt) for each layer
        """
        
        # 扩展到 batch
        key_prompts = self.prompt_embeddings[:, 0].unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # (batch, num_layers, num_prompts, hidden)
        
        value_prompts = self.prompt_embeddings[:, 1].unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        
        # 转换为每个 layer 的格式
        past_key_values = []
        for layer_idx in range(self.num_layers):
            layer_key = key_prompts[:, layer_idx]  # (batch, num_prompts, hidden)
            layer_value = value_prompts[:, layer_idx]
            
            # Transformer 需要 (batch, heads, seq, head_dim)
            past_key_values.append((layer_key, layer_value))
        
        return tuple(past_key_values)


class P_TuningV2Model(nn.Module):
    """
    集成 P-Tuning v2 的 Transformer 模型
    """
    
    def __init__(self, base_model, num_prompts: int = 10):
        super().__init__()
        
        self.base_model = base_model
        self.config = base_model.config
        
        # 创建 prompt encoder
        self.prompt_encoder = PromptEncoder(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_hidden_layers,
            num_prompts=num_prompts,
            prompt_dim=self.config.hidden_size
        )
        
        # 冻结 base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 只训练 prompt embeddings
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
    
    def get_prompt_params(self):
        """获取 prompt 参数 (用于优化器)"""
        return self.prompt_encoder.parameters()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """前向传播"""
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 获取 prompt
        prompt_past_key_values = self.prompt_encoder(batch_size, device)
        
        # 调用 base model，传入 prompt
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=prompt_past_key_values,
            labels=labels,
            **kwargs
        )
        
        return outputs


# ==================== 使用示例 ====================

def p_tuning_v2_demo():
    """P-Tuning v2 使用示例"""
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("P-Tuning v2 示例")
    print("=" * 70)
    
    # 加载基础模型
    model_name = "gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 添加 P-Tuning v2
    peft_model = P_TuningV2Model(base_model, num_prompts=10)
    
    # 参数统计
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    print(f"\n总参数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 优化器 (只更新 prompt 参数)
    optimizer = torch.optim.AdamW(
        peft_model.get_prompt_params(),
        lr=1e-3
    )
    
    print(f"\n优化器配置:")
    print(f"  - 仅训练 Prompt Embeddings")
    print(f"  - Learning Rate: 1e-3")
    print(f"  - Base Model: 冻结")
    
    # 简单训练示例
    inputs = tokenizer("Hello, how are", return_tensors="pt")
    
    # 单次前向
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    print(f"\n初始 Loss: {loss.item():.4f}")
    
    # 一次梯度更新
    loss.backward()
    optimizer.step()
    
    print("✓ 一次训练步骤完成")


if __name__ == "__main__":
    p_tuning_v2_demo()
```

---

## 知识点 4: PEFT 方法选择指南

### 决策树

```
选择 PEFT 方法
│
├── 显存限制？
│   │
│   ├── 极度受限 (<12GB)
│   │   └── → QLoRA (4bit 量化 + LoRA)
│   │
│   ├── 中度受限 (12-24GB)
│   │   └── → LoRA (秩 16-32)
│   │
│   └── 充足 (>24GB)
│       └── → 全量微调 或 LoRA (秩 64+)
│
├── 任务类型？
│   │
│   ├── 生成任务 (文本续写、翻译)
│   │   └── → LoRA 或 Prefix Tuning
│   │
│   ├── 理解任务 (分类、NER)
│   │   └── → P-Tuning v2
│   │
│   └── 多任务/通用
│       └── → LoRA (适应性强)
│
└── 微调目标？
    │
    ├── 快速原型/实验
    │   └── → LoRA (容易实现)
    │
    ├── 最佳效果
    │   └── → 全量微调 或 LoRA (高秩)
    │
    └── 多适配器切换
        └── → LoRA (可热插拔)
```

### 各方法推荐配置

| 方法 | 推荐秩 | Alpha | Dropout | 学习率 |
|------|-------|-------|---------|-------|
| **LoRA** | 8-64 | 16-128 | 0.05-0.1 | 1e-4 - 2e-4 |
| **QLoRA** | 64 | 128 | 0.05 | 2e-4 |
| **P-Tuning v2** | n/a | n/a | n/a | 1e-3 |
| **Adapter** | n/a | n/a | 0.1 | 1e-4 |

---

## 练习题

### 练习 1: LoRA 适配不同模型

为目标模型实现 LoRA 微调：
1. Llama-2-7B
2. ChatGLM3-6B
3. Baichuan2-7B

要求:
- 自动适配目标模块
- 支持多 GPU 训练
- 保存和加载适配器

### 练习 2: PEFT 方法对比实验

在同一任务上对比以下方法：
1. 全量微调
2. LoRA (r=8, 16, 64)
3. QLoRA
4. P-Tuning v2

评估指标：
- 最终性能
- 训练时间
- 显存占用
- 收敛速度

---

## 延伸阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [P-Tuning v2 论文](https://arxiv.org/abs/2110.07602)
- [PEFT 库文档](https://huggingface.co/docs/peft)

---

[← 返回 2.3 主文档](2-3-peft.md) | [下一节：多模态大模型 →](2-4-multimodal.md)
