# 3.2 模型压缩与量化 - 关键知识点详解

> 本节为 3.2 节的补充知识点，包含剪枝、蒸馏、量化技术详解与实战代码。

---

## 知识点 1: 模型压缩方法对比

### 压缩技术对比表

| 方法 | 原理 | 压缩率 | 精度损失 | 推理加速 | 适用场景 |
|------|------|--------|---------|---------|---------|
| **剪枝** | 移除不连接/权重 | 50-90% | 小 | 2-4x | 大模型压缩 |
| **蒸馏** | 小模型学习大模型 | 70-95% | 中 | 3-10x | 部署到边缘 |
| **量化** | 降低精度 (FP32→INT8/4) | 75-87% | 小 | 2-4x | 通用 |
| **低秩分解** | 矩阵分解 | 50-80% | 中 | 2-3x | 特定层 |
| **动态推理** | 早退/条件计算 | - | 无 | 1.5-3x | 可变难度输入 |

### 量化方法对比

| 量化类型 | 精度 | 代表方案 | 优点 | 缺点 |
|---------|------|---------|------|------|
| **PTQ** (训练后量化) | INT8/INT4 | GPTQ、AWQ、SmoothQuant | 无需训练、快速 | 大模型精度损失 |
| **QAT** (量化感知训练) | INT8/INT4 | QAT、LSQ | 精度高 | 需要微调 |
| **KV Cache 量化** | INT8/FP8 | vLLM、TensorRT | 显存节省 50% | 仅推理 |

---

## 知识点 2: 剪枝技术详解

### 结构化 vs 非结构化剪枝

```
剪枝类型对比

非结构化剪枝 (Unstructured Pruning)
┌─────────────────────────────────┐
│  ●  ○  ●  ○  ●  ○  ●  ○       │  ← 随机稀疏
│  ○  ●  ○  ●  ○  ●  ○  ●       │
│  ●  ○  ●  ○  ●  ○  ●  ○       │
│  ○  ●  ○  ●  ○  ●  ○  ●       │
└─────────────────────────────────┘
优点：压缩率高 (90%+)
缺点：需要稀疏计算库支持

结构化剪枝 (Structured Pruning)
┌─────────────────────────────────┐
│  ●  ●  ○  ○  ●  ●  ○  ○       │  ← 整列/整行移除
│  ●  ●  ○  ○  ●  ●  ○  ○       │
│  ●  ●  ○  ○  ●  ●  ○  ○       │
│  ●  ●  ○  ○  ●  ●  ○  ○       │
└─────────────────────────────────┘
优点：直接加速、无需特殊库
缺点：压缩率较低 (50-70%)
```

### 剪枝实战代码

```python
# examples/3-2-compression/pruning_demo.py
"""
模型剪枝完整示例

包含:
1. 基于幅值的非结构化剪枝
2. 结构化剪枝 (移除神经元)
3. 迭代剪枝流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleMLP(nn.Module):
    """简单的 MLP 用于剪枝演示"""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MagnitudePruner:
    """
    基于幅值的剪枝器
    
    原理：绝对值小的权重对输出影响小，可以剪除
    """
    
    def __init__(self, model: nn.Module, sparsity_target: float = 0.5):
        self.model = model
        self.sparsity_target = sparsity_target
        self.orig_masks = {}  # 保存原始 mask
        self._save_original_masks()
    
    def _save_original_masks(self):
        """保存原始权重 mask（用于恢复）"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.orig_masks[name] = {
                    'weight': module.weight.abs() > 0
                }
                if module.bias is not None:
                    self.orig_masks[name]['bias'] = module.bias.abs() > 0
    
    def compute_importance(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        计算权重重要性
        
        这里使用简单的绝对值，也可以用:
        - 梯度幅值
        - Hessian 矩阵
        - 泰勒展开近似
        """
        return tensor.abs()
    
    def prune_to_sparsity(self, sparsity: float):
        """
        剪枝到指定稀疏度
        
        Args:
            sparsity: 目标稀疏度 (0.9 表示 90% 的权重为 0)
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.orig_masks:
                weight = module.weight.data
                mask = self.orig_masks[name]['weight']
                
                # 计算重要性
                importance = self.compute_importance(weight) * mask.float()
                
                # 计算阈值
                flat_importance = importance.flatten()
                k = int((1 - sparsity) * flat_importance.numel())
                if k == 0:
                    threshold = 0
                else:
                    threshold = torch.topk(flat_importance, k, largest=True)[0][-1]
                
                # 生成新 mask
                new_mask = (importance >= threshold).float()
                
                # 应用 mask
                module.weight.data = module.weight.data * new_mask
                
                print(f"  {name}: 稀疏度={1 - new_mask.sum() / new_mask.numel():.2%}")
    
    def iterative_prune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_iterations: int = 10,
        prune_amount: float = 0.2,
        epochs_per_iter: int = 5
    ):
        """
        迭代剪枝流程
        
        1. 训练模型
        2. 剪枝一定比例
        3. 微调恢复精度
        4. 重复直到达到目标稀疏度
        """
        current_sparsity = 0
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # 1. 训练 epoch
            self._train_for_epochs(train_loader, epochs_per_iter)
            
            # 2. 计算当前稀疏度
            current_sparsity = self.get_current_sparsity()
            print(f"当前稀疏度：{current_sparsity:.2%}")
            
            if current_sparsity >= self.sparsity_target:
                print("已达到目标稀疏度！")
                break
            
            # 3. 剪枝
            remaining = 1 - current_sparsity
            prune_ratio = min(prune_amount / remaining, 0.5)  # 每次最多剪 50%
            new_sparsity = current_sparsity + prune_ratio * (1 - current_sparsity)
            
            print(f"剪枝到 {new_sparsity:.2%}...")
            self.prune_to_sparsity(new_sparsity)
            
            # 4. 验证
            val_acc = self._evaluate(val_loader)
            print(f"验证准确率：{val_acc:.4f}")
        
        return self.model
    
    def _train_for_epochs(self, loader, epochs):
        """训练指定 epoch"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            for X, y in loader:
                optimizer.zero_grad()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
    
    def _evaluate(self, loader) -> float:
        """评估准确率"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in loader:
                output = self.model(X)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return correct / total
    
    def get_current_sparsity(self) -> float:
        """计算当前稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                total_params += w.numel()
                zero_params += (w == 0).sum().item()
        
        return zero_params / total_params


# ==================== 结构化剪枝 ====================

class StructuredPruner:
    """
    结构化剪枝器
    
    移除整个神经元/通道，直接获得加速
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compute_neuron_importance(
        self,
        layer: nn.Linear,
        data_loader: DataLoader
    ) -> torch.Tensor:
        """
        计算神经元重要性
        
        方法：L1 范数 (简单有效)
        """
        weight = layer.weight.data  # (out_features, in_features)
        
        # L1 范数
        importance = weight.abs().sum(dim=1)  # 每个输出神经元的重要性
        
        return importance
    
    def prune_neurons(
        self,
        layer: nn.Linear,
        prune_ratio: float = 0.5
    ) -> nn.Linear:
        """
        移除不重要的神经元
        
        Args:
            layer: 要剪枝的线性层
            prune_ratio: 剪枝比例
        """
        # 计算重要性
        importance = self.compute_neuron_importance(layer, None)
        
        # 选择保留的神经元
        num_keep = int((1 - prune_ratio) * len(importance))
        _, indices = torch.topk(importance, num_keep)
        indices = indices.sort()[0]
        
        # 创建新层
        old_out_features = layer.out_features
        new_out_features = num_keep
        
        new_layer = nn.Linear(layer.in_features, new_out_features, bias=layer.bias is not None)
        
        # 复制权重
        with torch.no_grad():
            new_layer.weight.data = layer.weight.data[indices]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[indices]
        
        # 复制 mask (如果存在)
        if hasattr(layer, 'weight_mask'):
            new_layer.weight_mask = layer.weight_mask[indices]
        
        return new_layer, indices.tolist()


# ==================== 使用示例 ====================

def pruning_demo():
    """剪枝演示"""
    
    print("=" * 60)
    print("模型剪枝演示")
    print("=" * 60)
    
    # 创建模型
    model = SimpleMLP()
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"原始模型参数：{total_params:,}")
    
    # 创建模拟数据
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 非结构化剪枝
    print("\n1. 非结构化剪枝 (幅值剪枝)")
    print("-" * 60)
    
    pruner = MagnitudePruner(model, sparsity_target=0.8)
    pruner.prune_to_sparsity(0.5)  # 剪枝 50%
    
    sparsity = pruner.get_current_sparsity()
    print(f"剪枝后稀疏度：{sparsity:.2%}")
    
    # 结构化剪枝
    print("\n2. 结构化剪枝 (移除神经元)")
    print("-" * 60)
    
    model2 = SimpleMLP()
    struct_pruner = StructuredPruner(model2)
    
    # 剪枝第一个线性层
    new_layer, kept_indices = struct_pruner.prune_neurons(
        model2.network[0],
        prune_ratio=0.3
    )
    
    print(f"移除 {len(model2.network[0].weight) - len(new_layer.weight)} 个神经元")
    print(f"保留 {len(kept_indices)} 个神经元")
    
    print("\n✓ 剪枝演示完成!")
    print("  非结构化剪枝：高压缩率，需要稀疏库支持")
    print("  结构化剪枝：直接加速，压缩率较低")


if __name__ == "__main__":
    pruning_demo()
```

---

## 知识点 3: 蒸馏技术详解

### 蒸馏方法对比

| 方法 | 教师模型 | 学生模型 | 损失函数 | 适用场景 |
|------|---------|---------|---------|---------|
| **标准蒸馏** | 大模型 | 小模型 | KL 散度 + CE | 通用 |
| **自蒸馏** | 自身 (深层) | 自身 (浅层) | MSE | 单模型提升 |
| **数据无关蒸馏** | 预训练模型 | 小模型 | 合成数据 + KL | 无原数据场景 |
| **在线蒸馏** | 动态教师 | 学生 | 多模型平均 | 分布式训练 |

### 蒸馏实战代码

```python
# examples/3-2-compression/knowledge_distillation.py
"""
知识蒸馏完整实现

包含:
1. 标准 KD (Hinton)
2. 自蒸馏
3. 特征图蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失
    
    L = α * T² * KL(student_soft, teacher_soft) + (1-α) * CE(student, label)
    
    参数:
        temperature: 温度参数 T (软化概率分布)
        alpha: 蒸馏损失权重
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        计算蒸馏损失
        
        Returns:
            包含各部分损失的字典
        """
        # 软化概率分布
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL 散度 (蒸馏损失)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        kd_loss *= (self.temperature ** 2)  # 平衡量级
        
        # 交叉熵 (学生与真实标签)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total_loss': total_loss,
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item()
        }


class FeatureDistillation(nn.Module):
    """
    特征图蒸馏 (FitNets)
    
    不仅蒸馏输出，还蒸馏中间层特征
    """
    
    def __init__(
        self,
        student_features_dim: list,
        teacher_features_dim: list,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 投影层 (将学生特征映射到教师维度)
        self.projection_layers = nn.ModuleList()
        for s_dim, t_dim in zip(student_features_dim, teacher_features_dim):
            self.projection_layers.append(
                nn.Sequential(
                    nn.Linear(s_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, t_dim)
                )
            )
        
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        student_features: list,
        teacher_features: list
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生模型各层特征
            teacher_features: 教师模型各层特征
        """
        loss = 0
        
        for proj, s_feat, t_feat in zip(
            self.projection_layers,
            student_features,
            teacher_features
        ):
            # 投影学生特征
            s_proj = proj(s_feat)
            
            # MSE 损失
            loss += self.mse_loss(s_proj, t_feat)
        
        return loss / len(self.projection_layers)


# ==================== 完整蒸馏训练器 ====================

class DistillationTrainer:
    """
    蒸馏训练器
    
    支持:
    1. 输出蒸馏 (logits)
    2. 特征蒸馏 (中间层)
    3. 多教师蒸馏
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        temperature: float = 4.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-3,
        feature_distillation: bool = False,
        feature_weight: float = 0.1
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50
        )
        
        # 蒸馏损失
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature, alpha)
        
        # 特征蒸馏 (可选)
        self.feature_distillation = feature_distillation
        if feature_distillation:
            self.feature_loss_fn = FeatureDistillation(
                student_features_dim=[256, 512],  # 学生特征维度
                teacher_features_dim=[512, 1024]  # 教师特征维度
            )
            self.feature_weight = feature_weight
    
    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch"""
        self.student.train()
        
        total_loss = 0
        kd_loss_total = 0
        ce_loss_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 教师前向 (无梯度)
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
            
            # 学生前向
            student_outputs = self.student(inputs)
            
            # 计算损失
            losses = self.kd_loss_fn(student_outputs, teacher_outputs, targets)
            loss = losses['total_loss']
            
            # 特征蒸馏损失 (可选)
            if self.feature_distillation:
                # 这里需要获取中间层特征
                # 简化示例，实际需要在模型中注册 hook
                pass
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            kd_loss_total += losses['kd_loss']
            ce_loss_total += losses['ce_loss']
        
        self.scheduler.step()
        
        return {
            'avg_loss': total_loss / len(self.train_loader),
            'avg_kd_loss': kd_loss_total / len(self.train_loader),
            'avg_ce_loss': ce_loss_total / len(self.train_loader),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """验证"""
        self.student.eval()
        
        correct = 0
        total = 0
        total_loss = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            student_outputs = self.student(inputs)
            loss = self.kd_loss_fn(
                student_outputs,
                self.teacher(inputs),
                targets
            )['total_loss']
            
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def train(self, num_epochs: int, save_path: str = None):
        """完整训练流程"""
        best_acc = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.evaluate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['avg_loss']:.4f} "
                  f"(KD: {train_metrics['avg_kd_loss']:.4f}, "
                  f"CE: {train_metrics['avg_ce_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                if save_path:
                    torch.save(
                        self.student.state_dict(),
                        f"{save_path}/best_student.pth"
                    )
                    print(f"  ✓ 已保存最佳模型 (Acc: {best_acc:.2f}%)")
        
        return best_acc


# ==================== 使用示例 ====================

def distillation_demo():
    """蒸馏演示"""
    
    print("=" * 60)
    print("知识蒸馏演示")
    print("=" * 60)
    
    # 定义教师模型 (大)
    teacher = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 10)
    )
    
    # 定义学生模型 (小)
    student = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 参数量对比
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n教师模型参数：{teacher_params:,}")
    print(f"学生模型参数：{student_params:,}")
    print(f"压缩比：{teacher_params / student_params:.1f}x")
    print(f"参数量减少：{(1 - student_params/teacher_params)*100:.1f}%")
    
    # 创建模拟数据
    X_train = torch.randn(5000, 784)
    y_train = torch.randint(0, 10, (5000,))
    X_val = torch.randn(1000, 784)
    y_val = torch.randint(0, 10, (1000,))
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128
    )
    
    # 蒸馏训练
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        temperature=4.0,
        alpha=0.7
    )
    
    print(f"\n开始蒸馏训练...")
    print(f"  温度 T = {trainer.temperature}")
    print(f"  蒸馏权重 α = {trainer.alpha}")
    
    # 实际使用时调用:
    # trainer.train(num_epochs=50, save_path='./checkpoints')
    
    print("\n✓ 蒸馏演示完成!")
    print("  关键要点:")
    print("  1. 高温软化概率分布，传递'暗知识'")
    print("  2. α 控制蒸馏与真实标签的平衡")
    print("  3. 特征蒸馏进一步提升效果")


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    distillation_demo()
```

---

## 练习题

### 练习 1: 量化感知训练

实现一个量化感知训练 (QAT) 流程：
1. 定义伪量化模块 (FakeQuantize)
2. 在训练中模拟量化效果
3. 导出 INT8 模型

### 练习 2: 蒸馏对比实验

对比以下蒸馏方案：
1. 标准 KD (Hinton)
2. FitNets (特征蒸馏)
3. AT (Attention Transfer)

评估指标：准确率、参数量、推理速度

---

## 延伸阅读

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton KD 论文
- [Awesome Model Compression](https://github.com/GoodGoodHaoFeng/Awesome-Model-Compression) - 压缩技术汇总
- [pytorch-model-zoo](https://github.com/pytorch/vision) - 预训练模型

---

[← 返回 3.2 主文档](3-2-compression.md) | [下一节：推理优化技术 →](3-3-inference.md)
