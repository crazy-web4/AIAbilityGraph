"""
2.3 参数高效微调 - LoRA/QLoRA 实战

实现内容:
- LoRA 核心实现
- QLoRA 配置
- 完整微调流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ========== LoRA 核心实现 ==========

class LoRALinear(nn.Module):
    """
    LoRA 线性层

    W' = W + BA
    其中 B 和 A 是可训练的低秩矩阵
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False
    ):
        """
        参数:
            r: LoRA 秩
            alpha: 缩放系数
            dropout: Dropout 率
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        # 原始权重（冻结）
        self.original_weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 可选的 bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向
        original = F.linear(x, self.original_weight, self.bias)

        # LoRA 分支
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return original + lora

    def merge_weights(self) -> torch.Tensor:
        """合并 LoRA 权重到原始权重"""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.original_weight + delta


class LoRAEmbedding(nn.Module):
    """
    LoRA Embedding 层

    用于词汇表嵌入的高效微调
    """

    def __init__(self, vocab_size: int, embed_dim: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 原始嵌入（冻结）
        self.original_weight = nn.Parameter(
            torch.randn(vocab_size, embed_dim),
            requires_grad=False
        )

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.randn(r, embed_dim))
        self.lora_B = nn.Parameter(torch.randn(vocab_size, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 原始嵌入
        original = F.embedding(input_ids, self.original_weight)

        # LoRA 增量
        lora_A_embed = F.linear(self.lora_A, self.lora_B[input_ids])

        return original + lora_A_embed * self.scaling


# ========== QLoRA 配置 ==========

@dataclass
class QLoRAConfig:
    """QLoRA 配置"""
    r: int = 16
    alpha: float = 32
    dropout: float = 0.05
    quantization_bits: int = 4  # 4-bit 量化
    double_quant: bool = True
    quant_type: str = "nf4"


def create_qlora_config(qlora_config: QLoRAConfig) -> BitsAndBytesConfig:
    """
    创建 QLoRA 量化配置
    """
    return BitsAndBytesConfig(
        load_in_4bit=(qlora_config.quantization_bits == 4),
        bnb_4bit_quant_type=qlora_config.quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=qlora_config.double_quant,
    )


# ========== LoRA 应用到 Llama ==========

class LoRALlamaForCausalLM(nn.Module):
    """
    使用 LoRA 微调 Llama 模型
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_config: QLoRAConfig = None,
        target_modules: List[str] = None
    ):
        super().__init__()
        self.lora_config = lora_config or QLoRAConfig()
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        # 量化配置
        quant_config = create_qlora_config(self.lora_config)

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 注入 LoRA
        self._inject_lora()

    def _inject_lora(self):
        """注入 LoRA 到目标模块"""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels, **kwargs)

    def save_pretrained(self, save_dir: str):
        """保存 LoRA 权重"""
        self.model.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, model_name: str, lora_path: str, **kwargs):
        """加载预训练模型和 LoRA 权重"""
        instance = cls(model_name=model_name, **kwargs)
        instance.model = PeftModel.from_pretrained(instance.model, lora_path)
        return instance


# ========== 训练器 ==========

class LoRATrainer:
    """
    LoRA 微调训练器
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000
    ):
        self.model = model
        self.tokenizer = tokenizer

        # 优化器 - 只优化 LoRA 参数
        lora_params = [
            p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=0
        )

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """单步训练"""
        self.model.train()

        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss

        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in self.model.named_parameters() if "lora_" in n],
            max_norm=1.0
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # 学习率调度
        if self.warmup_steps > 0:
            progress = self.step_count / self.warmup_steps
            lr_factor = min(progress, 1.0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

        self.scheduler.step()
        self.step_count += 1

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]['lr']}

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader = None,
        save_dir: str = None
    ):
        """完整训练循环"""
        self.step_count = 0

        for epoch in range(3):  # 默认 3 个 epoch
            print(f"\nEpoch {epoch + 1}/3")

            for batch in train_dataloader:
                train_metrics = self.train_step(
                    input_ids=batch["input_ids"].cuda(),
                    labels=batch["labels"].cuda(),
                    attention_mask=batch["attention_mask"].cuda()
                )

                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: loss={train_metrics['loss']:.4f}, lr={train_metrics['lr']:.6f}")

                if eval_dataloader and self.step_count % 500 == 0:
                    self.evaluate(eval_dataloader)

            # 保存检查点
            if save_dir:
                self.model.save_pretrained(f"{save_dir}/epoch_{epoch + 1}")

    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"].cuda(),
                    labels=batch["labels"].cuda()
                )
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(eval_dataloader)
        perplexity = math.exp(avg_loss)

        print(f"Evaluation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        return {"loss": avg_loss, "perplexity": perplexity}


# ========== 使用示例 ==========

def demo_lora_finetuning():
    """LoRA 微调演示"""

    # 配置
    model_name = "meta-llama/Llama-2-7b-hf"
    lora_config = QLoRAConfig(r=16, alpha=32, dropout=0.05)

    print("加载模型并配置 LoRA...")

    # 创建模型（简化版，实际使用需要 peft 库）
    # model = LoRALlamaForCausalLM(
    #     model_name=model_name,
    #     lora_config=lora_config
    # )

    # 打印 LoRA 参数
    print(f"\nLoRA 配置:")
    print(f"  秩 (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.alpha}")
    print(f"  缩放: {lora_config.alpha / lora_config.r:.2f}")
    print(f"  参数量: 约 {(lora_config.r * 4096 * 2 * 32):,} (仅 LoRA)")
    print(f"  保存率：约 0.1% (相比全参数)")

    # 演示 LoRA 层
    print("\nLoRALinear 测试:")
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)

    lora_layer = LoRALinear(
        in_features=d_model,
        out_features=d_model,
        r=8,
        alpha=16
    )

    output = lora_layer(x)
    print(f"  输入：{x.shape}")
    print(f"  输出：{output.shape}")

    # 合并权重
    merged = lora_layer.merge_weights()
    print(f"  合并后权重形状：{merged.shape}")


if __name__ == "__main__":
    import math
    demo_lora_finetuning()
