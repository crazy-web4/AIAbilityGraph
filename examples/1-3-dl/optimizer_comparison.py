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
