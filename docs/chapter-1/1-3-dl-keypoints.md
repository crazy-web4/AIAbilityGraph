# 1.3 深度学习基础 - 关键知识点详解

> 本节为 1.3 节的补充知识点，包含 CNN/RNN/Transformer 架构详解、训练技巧、调试指南。

---

## 知识点 1: 神经网络核心组件对比

### 激活函数对比表

| 激活函数 | 公式 | 导数 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|------|---------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $f(x)(1-f(x))$ | 输出 (0,1) 可解释为概率 | 梯度消失、非零中心 | 二分类输出层 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-f(x)^2$ | 零中心、比 Sigmoid 好 | 梯度消失 | RNN 常用 |
| **ReLU** | $\max(0, x)$ | 1 if x>0 else 0 | 计算快、缓解梯度消失 | Dead ReLU 问题 | **默认选择** |
| **Leaky ReLU** | $\max(\alpha x, x)$ | 1 if x>0 else α | 解决 Dead ReLU | 需要调 α 参数 | ReLU 的替代 |
| **GELU** | $x\Phi(x)$ | 复杂 | 平滑、性能好 | 计算稍慢 | Transformer 默认 |
| **Swish** | $x \cdot \sigma(x)$ | 复杂 | 性能优于 ReLU | 计算稍慢 | 替代 ReLU |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | 复杂 | 多分类概率输出 | 需配合交叉熵 | 多分类输出层 |

### 激活函数可视化与测试

```python
# examples/1-3-dl/activation_functions.py
"""
激活函数详解与对比

包含:
1. 各激活函数可视化
2. 梯度流动测试
3. Dead ReLU 演示
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ========== 激活函数实现 ==========

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.clamp(x, min=0)


def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)


def gelu(x):
    """GELU: Gaussian Error Linear Unit"""
    return x * 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def swish(x):
    """Swish: x * sigmoid(x)"""
    return x * sigmoid(x)


# ========== 可视化 ==========

def plot_activation_functions():
    """绘制各激活函数曲线"""
    
    x = torch.linspace(-5, 5, 1000)
    
    functions = {
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'ReLU': relu,
        'Leaky ReLU': leaky_relu,
        'GELU': gelu,
        'Swish': swish
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, fn) in enumerate(functions.items()):
        y = fn(x)
        
        # 计算导数 (数值微分)
        eps = 1e-6
        y_prime = (fn(x + eps) - fn(x - eps)) / (2 * eps)
        
        axes[idx].plot(x.numpy(), y.numpy(), 'b-', linewidth=2, label='f(x)')
        axes[idx].plot(x.numpy(), y_prime.numpy(), 'r--', linewidth=2, label="f'(x)")
        axes[idx].set_title(name, fontsize=14)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(0, color='black', linewidth=0.5)
        axes[idx].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150)
    print("激活函数图已保存：activation_functions.png")
    plt.show()


# ========== 梯度流动测试 ==========

def test_gradient_flow():
    """
    测试不同激活函数的梯度流动
    
    深层网络中，梯度是否会消失/爆炸
    """
    
    print("=" * 60)
    print("梯度流动测试 (100 层网络)")
    print("=" * 60)
    
    class DeepNet(nn.Module):
        def __init__(self, n_layers, activation):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(nn.Linear(100, 100))
            self.activation = activation
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = self.activation(x)
            return x
    
    activations = {
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.01),
        'GELU': nn.GELU()
    }
    
    n_layers = 50
    batch_size = 32
    input_dim = 100
    
    results = {}
    
    for name, act in activations.items():
        model = DeepNet(n_layers, act)
        
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        
        # 反向传播
        loss = output.sum()
        loss.backward()
        
        # 统计各层梯度范数
        grad_norms = []
        for layer in model.layers:
            if layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                grad_norms.append(grad_norm)
        
        results[name] = {
            'mean_grad': np.mean(grad_norms),
            'std_grad': np.std(grad_norms),
            'min_grad': np.min(grad_norms),
            'max_grad': np.max(grad_norms)
        }
    
    # 打印结果
    print(f"\n{'激活函数':<12} {'平均梯度':<12} {'最小梯度':<12} {'最大梯度':<12}")
    print("-" * 50)
    for name, stats in results.items():
        print(f"{name:<12} {stats['mean_grad']:.6f}      "
              f"{stats['min_grad']:.6f}      {stats['max_grad']:.6f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    for name, stats in results.items():
        plt.bar(name, stats['mean_grad'], yerr=stats['std_grad'], 
                capsize=5, alpha=0.7, label=name)
    
    plt.ylabel('平均梯度范数')
    plt.title(f'{n_layers}层网络的梯度流动对比')
    plt.yscale('log')  # 对数刻度更好观察
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150)
    print("\n梯度流动图已保存：gradient_flow.png")
    plt.show()


# ========== Dead ReLU 演示 ==========

def demonstrate_dead_relu():
    """
    演示 Dead ReLU 问题
    
    当神经元输出始终<=0 时，梯度为 0，永远无法更新
    """
    
    print("\n" + "=" * 60)
    print("Dead ReLU 演示")
    print("=" * 60)
    
    # 创建一个简单网络
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    # 用很大的负偏置初始化 (模拟 Dead ReLU)
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                layer.bias.fill_(-10)  # 很大的负偏置
    
    # 训练数据
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 统计每层 active 的神经元比例
    def count_dead_neurons(model):
        dead_ratios = []
        for layer in model:
            if isinstance(layer, nn.ReLU):
                # 检查输出是否全为 0
                pass
        
        return dead_ratios
    
    # 训练几轮观察
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # 检查每层 ReLU 的 active 比例
        active_ratio = []
        for layer in model:
            if isinstance(layer, nn.ReLU):
                # 这里简化处理，实际应该检查训练数据的输出
                pass
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("\nDead ReLU 问题：使用 Leaky ReLU 或适当的学习率可缓解")


if __name__ == "__main__":
    # 1. 可视化激活函数
    plot_activation_functions()
    
    # 2. 梯度流动测试
    test_gradient_flow()
    
    # 3. Dead ReLU 演示
    demonstrate_dead_relu()
```

---

## 知识点 2: 优化器对比与选择

### 优化器对比表

| 优化器 | 核心思想 | 超参数 | 适用场景 |
|-------|---------|-------|---------|
| **SGD** | 基础梯度下降 | lr, momentum | 凸优化、小网络 |
| **SGD+Momentum** | 添加动量加速 | lr, momentum | 一般 CNN |
| **AdaGrad** | 自适应学习率 | lr | 稀疏数据、NLP |
| **RMSProp** | 解决 AdaGrad 学习率衰减 | lr, decay | RNN/LSTM |
| **Adam** | Momentum + RMSProp | lr, β1, β2, ε | **默认选择** |
| **AdamW** | Adam + 解耦权重衰减 | lr, β1, β2, ε, weight_decay | **Transformer 默认** |
| **LAMB** | 层自适应大批次 | lr, β1, β2, ε | 大批次训练 |

### 优化器对比实验

```python
# examples/1-3-dl/optimizer_comparison.py
"""
优化器对比实验

测试不同优化器在同一任务上的收敛速度和最终性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_classification_task(n_samples=1000, n_features=100):
    """创建二分类任务"""
    
    # 生成数据
    X = torch.randn(n_samples, n_features)
    y = (X[:, :10].sum(dim=1) > 0).float()  # 前 10 个特征决定标签
    
    # 添加噪声
    X += 0.1 * torch.randn_like(X)
    
    return X, y


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))


def compare_optimizers():
    """比较不同优化器"""
    
    # 准备数据
    X_train, y_train = create_classification_task()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 优化器配置
    optimizers_config = {
        'SGD': lambda params: optim.SGD(params, lr=0.01),
        'SGD+Momentum': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        'AdaGrad': lambda params: optim.Adagrad(params, lr=0.01),
        'RMSProp': lambda params: optim.RMSprop(params, lr=0.001),
        'Adam': lambda params: optim.Adam(params, lr=0.001),
        'AdamW': lambda params: optim.AdamW(params, lr=0.001, weight_decay=0.01),
    }
    
    results = {}
    
    print("=" * 70)
    print("优化器对比实验")
    print("=" * 70)
    
    for opt_name, opt_fn in optimizers_config.items():
        print(f"\n训练 {opt_name}...")
        
        # 重置随机种子保证公平比较
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建模型
        model = MLP(input_dim=100)
        optimizer = opt_fn(model.parameters())
        criterion = nn.BCELoss()
        
        losses = []
        accuracies = []
        
        # 训练 50 轮
        for epoch in range(50):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                predicted = (output > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)
            
            avg_loss = epoch_loss / len(train_loader)
            acc = correct / (len(train_dataset))
            
            losses.append(avg_loss)
            accuracies.append(acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/50: Loss = {avg_loss:.4f}, Acc = {acc:.4f}")
        
        results[opt_name] = {
            'losses': losses,
            'accuracies': accuracies
        }
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    for name, data in results.items():
        ax1.plot(data['losses'], label=name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss 曲线对比')
    ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    for name, data in results.items():
        ax2.plot(data['accuracies'], label=name, linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy 曲线对比')
    ax2.legend(ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    print("\n对比图已保存：optimizer_comparison.png")
    plt.show()
    
    # 打印最终结果
    print("\n" + "=" * 70)
    print("最终结果 (Epoch 50)")
    print("=" * 70)
    print(f"{'优化器':<15} {'最终 Loss':<15} {'最终 Accuracy':<15}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<15} {data['losses'][-1]:.4f}          {data['accuracies'][-1]:.4f}")


if __name__ == "__main__":
    compare_optimizers()
```

---

## 知识点 3: 深度学习训练技巧

### 训练技巧清单

| 技巧 | 作用 | 实现方式 | 适用场景 |
|------|------|---------|---------|
| **BatchNorm** | 加速收敛、减少过拟合 | `nn.BatchNorm2d` | CNN 标配 |
| **LayerNorm** | 稳定训练 | `nn.LayerNorm` | Transformer 标配 |
| **Dropout** | 减少过拟合 | `nn.Dropout(p)` | 全连接层常用 |
| **残差连接** | 解决梯度消失 | `x + F(x)` | 深层网络必备 |
| **学习率预热** | 稳定初期训练 | 前 N 轮线性升温 | 大模型训练 |
| **学习率衰减** | 精细收敛 | Cosine/Step/CosineAnnealing | 所有场景 |
| **梯度裁剪** | 防止梯度爆炸 | `torch.nn.utils.clip_grad_norm_` | RNN/Transformer |
| **早停法** | 防止过拟合 | 验证集 loss 不改善则停止 | 小数据集 |
| **标签平滑** | 减少过拟合 | `LabelSmoothingLoss` | 分类任务 |
| **混合精度** | 加速训练、省显存 | `torch.cuda.amp` | GPU 训练 |

### 完整训练模板

```python
# examples/1-3-dl/training_template.py
"""
深度学习训练完整模板

包含最佳实践:
1. 学习率调度 (预热 + 衰减)
2. 梯度裁剪
3. 早停法
4. 检查点保存
5. TensorBoard 日志
6. 混合精度训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp
        
        # AdamW 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器 (在 fit 中设置)
        self.scheduler = None
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == 'cuda')
        
        # 保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _get_lr_schedule(self, epoch: int, step: int, total_steps: int) -> float:
        """
        学习率调度: 预热 + 余弦退火
        """
        # 预热阶段
        if epoch < self.warmup_epochs:
            warmup_progress = (epoch * total_steps + step) / (self.warmup_epochs * total_steps)
            return 0.1 + 0.9 * warmup_progress
        
        # 余弦退火
        progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    def _update_learning_rate(self, epoch: int, step: int, total_steps: int):
        """更新学习率"""
        lr_ratio = self._get_lr_schedule(epoch, step, total_steps)
        base_lr = self.optimizer.param_groups[0]['lr']
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr * lr_ratio
    
    def train_epoch(self, epoch: int) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        total_steps = len(self.train_loader)
        
        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 更新学习率
            self._update_learning_rate(epoch, step, total_steps)
            
            # 混合精度前向
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device == 'cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return avg_loss, accuracy, current_lr
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新
        torch.save(checkpoint, self.save_dir / 'checkpoint_latest.pth')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pth')
            print(f"  ✓ 已保存最佳模型 (Epoch {epoch})")
    
    def fit(self) -> dict:
        """完整训练流程"""
        
        print("=" * 70)
        print("开始训练")
        print("=" * 70)
        print(f"设备：{self.device}")
        print(f"训练集大小：{len(self.train_loader.dataset)}")
        print(f"验证集大小：{len(self.val_loader.dataset)}")
        print(f"最大 Epoch: {self.max_epochs}")
        print(f"早停 Patience: {self.patience}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # 训练
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 打印日志
            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n触发早停条件 (连续{self.patience}个 epoch 未改善)")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时：{total_time/60:.1f} 分钟")
        
        return self.history
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"已加载检查点：{path}")


# ==================== 使用示例 ====================

def create_sample_model():
    """创建示例模型"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )


if __name__ == "__main__":
    # 这里需要实际的数据加载器
    # 仅作为模板参考
    print("训练模板已就绪，请替换为您的数据和模型")
```

---

## 练习题

### 练习 1: CNN 实现

实现一个 CNN 分类 MNIST 数字，要求：
1. 至少 3 个卷积层
2. 使用 BatchNorm 和 Dropout
3. 达到>99% 测试准确率

### 练习 2: 学习率调度器对比

比较以下学习率调度器的效果：
- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau
- OneCycleLR

---

## 延伸阅读

- [Deep Learning Book](https://www.deeplearningbook.org/) - Ian Goodfellow
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Papers With Code](https://paperswithcode.com/) - 最新论文与代码

---

[← 返回 1.3 主文档](1-3-deep-learning.md) | [下一节：AI 安全与伦理 →](1-4-ai-ethics.md)
