# 3.2 模型压缩与量化

> 模型压缩是将大模型部署到资源受限环境的关键技术。本章涵盖剪枝、蒸馏、量化三大核心技术。

## 学习目标

学完本节后，你将能够：

- [ ] 理解并应用模型剪枝技术
- [ ] 实现知识蒸馏训练
- [ ] 掌握 PTQ 和 QAT 量化方法
- [ ] 选择合适的压缩策略

---

## 3.2.1 模型剪枝 (Pruning)

### 剪枝类型对比

| 类型 |  granularity | 优点 | 缺点 | 加速比 |
|------|-------------|------|------|--------|
| **非结构化剪枝** | 单个权重 | 剪枝率高 | 需要稀疏硬件支持 | 1-2x |
| **结构化剪枝** | 神经元/通道 | 直接加速 | 精度损失较大 | 2-4x |
| **半结构化剪枝** | 2:4, 4:8 模式 | NVIDIA Ampere 支持 | 灵活性受限 | 2x |

### 非结构化剪枝实现

```python
# examples/3-2-compression/pruning.py
import torch
import torch.nn as nn
from torch import nn


class MagnitudePruner:
    """
    基于权重大小的剪枝
    
    核心思想：移除绝对值最小的权重
    """
    
    def __init__(self, model, prune_percent=0.5):
        self.model = model
        self.prune_percent = prune_percent
        self.masks = {}
    
    def compute_masks(self):
        """计算剪枝掩码"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.abs()
                threshold = torch.kthvalue(
                    weight.flatten(), 
                    int(weight.numel() * self.prune_percent)
                ).values
                
                mask = (weight > threshold).float()
                self.masks[name] = mask
    
    def apply_masks(self):
        """应用掩码"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.masks:
                module.weight.data *= self.masks[name]
    
    def prune(self):
        self.compute_masks()
        self.apply_masks()


class IterativePruner:
    """
    迭代剪枝 - 更温和的剪枝策略
    
    流程：
    1. 训练模型
    2. 剪枝 20%
    3. 微调恢复
    4. 重复直到目标剪枝率
    """
    
    def __init__(self, model, target_sparsity=0.7, prune_per_iter=0.2):
        self.model = model
        self.target_sparsity = target_sparsity
        self.prune_per_iter = prune_per_iter
        self.current_sparsity = 0
    
    def get_current_sparsity(self):
        """计算当前稀疏度"""
        total_params = 0
        zero_params = 0
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def prune_step(self):
        """执行一步剪枝"""
        pruner = MagnitudePruner(self.model, self.prune_per_iter)
        pruner.prune()
        self.current_sparsity = self.get_current_sparsity()
        
        print(f"当前稀疏度：{self.current_sparsity:.2%}")
        
        return self.current_sparsity < self.target_sparsity
```

### 结构化剪枝

```python
class StructuredPruner:
    """
    L1 范数结构化剪枝
    
    按神经元的 L1 范数重要性进行剪枝
    """
    
    def __init__(self, model):
        self.model = model
    
    def get_neuron_importance(self, module):
        """计算神经元重要性 (L1 范数)"""
        weight = module.weight.data
        # 输出维度求和
        importance = weight.abs().sum(dim=1)
        return importance
    
    def prune_neurons(self, prune_ratio=0.1):
        """剪枝神经元"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                importance = self.get_neuron_importance(module)
                
                # 找到重要性最低的神经元
                k = int(module.out_features * prune_ratio)
                _, indices = torch.topk(importance, k, largest=False)
                
                # 将对应神经元权重置零
                module.weight.data[indices] = 0
                
                if module.bias is not None:
                    module.bias.data[indices] = 0
        
        return self.model
```

---

## 3.2.2 知识蒸馏 (Knowledge Distillation)

### 蒸馏损失函数

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(S, y) + (1-\alpha) \cdot \mathcal{L}_{KL}(\text{softmax}(S/T), \text{softmax}(T/T))$$

```python
# examples/3-2-compression/distillation.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失
    
    三温度蒸馏：
    - 学生温度 Ts > 1
    - 教师温度 Tt = Ts
    - 硬标签温度 T = 1
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha  # 硬标签权重
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, hard_labels):
        """
        参数:
            student_logits: 学生输出 (不除以 T)
            teacher_logits: 教师输出 (不除以 T)
            hard_labels: 真实标签
        """
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, hard_labels)
        
        # 软标签损失 (KL 散度)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        # 组合损失
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return loss


class DistillationTrainer:
    """
    知识蒸馏训练器
    """
    
    def __init__(self, student_model, teacher_model, 
                 device='cuda', temperature=4.0, alpha=0.3):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device).eval()
        self.device = device
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            student.parameters(), lr=2e-4, weight_decay=0.01
        )
        self.criterion = DistillationLoss(
            temperature=temperature, alpha=alpha
        )
    
    def train_epoch(self, dataloader):
        self.student.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 教师前向（无梯度）
            with torch.no_grad():
                teacher_logits = self.teacher(inputs).logits
            
            # 学生前向
            student_logits = self.student(inputs).logits
            
            # 计算蒸馏损失
            loss = self.criterion(
                student_logits, teacher_logits, labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)


class TinyLLaMA(nn.Module):
    """
    小型学生模型示例
    """
    
    def __init__(self, vocab_size=32000, hidden=512, layers=4, heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=heads)
            for _ in range(layers)
        ])
        self.lm_head = nn.Linear(hidden, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def distill_llama_to_tiny():
    """
    将 LLaMA 蒸馏到 TinyLLaMA
    """
    from transformers import AutoModelForCausalLM
    
    # 加载教师模型（冻结）
    teacher = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16
    )
    
    # 创建学生模型
    student = TinyLLaMA(
        vocab_size=32000, hidden=512, layers=6, heads=8
    )
    
    # 蒸馏训练器
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        temperature=4.0,
        alpha=0.3
    )
    
    # 训练...
    return trainer
```

---

## 3.2.3 量化技术

### 量化基础

| 量化类型 | 精度 | 内存节省 | 适用场景 |
|----------|------|----------|----------|
| FP16/BF16 | 16-bit | 2x | 训练/推理 |
| INT8 | 8-bit | 4x | 推理 |
| INT4 | 4-bit | 8x | 边缘部署 |
| NF4 | 4-bit normal | 8x | QLoRA |

### 训练后量化 (PTQ)

```python
# examples/3-2-compression/quantization.py
"""
训练后量化 (Post-Training Quantization)
"""

import torch
from torch.ao.quantization import (
    QuantWrapper, QuantStub, DeQuantStub,
    QConfig, default_qconfig, float_qparams_weight_only_qconfig
)


class QuantizedModel:
    """
    PTQ 量化模型
    """
    
    def __init__(self, model, quant_type='int8'):
        self.model = model
        self.quant_type = quant_type
        
        # 添加量化/反量化节点
        self.model.quant = QuantStub()
        self.model.dequant = DeQuantStub()
    
    def prepare(self):
        """准备量化（插入 fake quant 节点）"""
        self.model = torch.ao.quantization.prepare(self.model)
    
    def calibrate(self, calibration_loader):
        """
        校准：收集激活值的 min/max
        
        参数:
            calibration_loader: 校准数据（不需要标签）
        """
        self.model.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                inputs = batch['input_ids'].cuda()
                _ = self.model(inputs)
    
    def convert(self):
        """转换为量化模型"""
        self.model = torch.ao.quantization.convert(self.model)
        
        print(f"量化完成：{self.quant_type}")
        return self.model


def quantize_llm_ptq(model, calibration_texts, tokenizer):
    """
    LLM 的 PTQ 量化
    """
    from transformers import AutoModelForCausalLM
    
    # 准备校准数据
    model.eval()
    model.cuda()
    
    # 激活统计
    def activation_hook(module, input, output):
        if hasattr(module, 'act_min'):
            module.act_min = min(module.act_min, output.min().item())
            module.act_max = max(module.act_max, output.max().item())
        else:
            module.act_min = output.min().item()
            module.act_max = output.max().item()
    
    # 注册 hook
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(activation_hook))
    
    # 校准
    with torch.no_grad():
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            _ = model(**inputs)
    
    # 移除 hook
    for h in handles:
        h.remove()
    
    # 应用量化
    qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(model, qconfig, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    
    return model


# ========== 使用 bitsandbytes 量化 ==========
def quantize_with_bnb(model_path, bits=4):
    """
    使用 bitsandbytes 进行量化加载
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model
```

### 量化感知训练 (QAT)

```python
class QATTrainer:
    """
    量化感知训练
    """
    
    def __init__(self, model, quant_bits=8):
        self.model = model
        
        # QAT 配置
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(
            'fbgemm' if quant_bits == 8 else 'qnnpack'
        )
        
        # 准备 QAT
        self.model = torch.ao.quantization.prepare_qat(self.model)
    
    def train(self, dataloader, epochs=5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            self.model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss {loss.item():.4f}")
        
        # 转换为量化模型
        self.model = torch.ao.quantization.convert(self.model)
        
        return self.model
```

---

## 3.2.4 模型大小对比

| 方法 | 7B 模型大小 | 显存需求 | 精度损失 |
|------|------------|----------|----------|
| FP16 | 14 GB | 28 GB | 无 |
| INT8 (PTQ) | 7 GB | 16 GB | <1% |
| INT4 (NF4) | 3.5 GB | 12 GB | 1-2% |
| 2:4 半结构化 | 7 GB | 20 GB | <1% |

---

## 练习题

1. **剪枝率计算**：对 7B 模型应用 50% 非结构化剪枝，实际能节省多少内存？

2. **蒸馏温度调优**：为什么蒸馏要用高温？温度过高有什么问题？

3. **量化误差分析**：比较 per-tensor 和 per-channel 量化的误差差异。

---

[← 上一节：3.1 分布式训练](3-1-distributed-training.md) | [下一节：3.3 推理优化技术 →](3-3-inference.md)
