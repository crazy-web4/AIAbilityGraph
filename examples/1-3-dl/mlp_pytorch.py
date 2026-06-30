"""
1.3 深度学习基础 - PyTorch MLP 实战

从零搭建多层感知机并训练 MNIST 分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    """
    多层感知机 (MLP) for MNIST

    架构：Input(784) -> Linear(256) -> ReLU -> Dropout -> Linear(128) -> ReLU -> Dropout -> Output(10)
    """

    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, dropout=0.2):
        super(MLP, self).__init__()

        # 构建网络层
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # 展平输入 (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)


class ModernMLP(nn.Module):
    """
    现代 MLP 改进版

    特性:
    - BatchNorm 加速收敛
    - LeakyReLU 避免 Dead Neurons
    - 残差连接（如果维度匹配）
    """

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super(ModernMLP, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm
            layers.append(nn.LeakyReLU(0.01))  # LeakyReLU
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def train_mlp(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    训练 MLP 模型

    参数:
        model: PyTorch 模型
        train_loader: 训练数据 DataLoader
        test_loader: 测试数据 DataLoader
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
    """
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"开始训练 {'ModernMLP' if isinstance(model, ModernMLP) else 'MLP'}...")
    print("-" * 60)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 更新学习率
        scheduler.step()

        train_loss /= len(train_loader)
        train_acc = correct / total

        # 评估阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(test_loader)
        val_acc = val_correct / val_total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("-" * 60)
    print(f"训练完成! 最终测试准确率：{val_acc:.4f}")

    return history


def plot_training_history(history_list, model_names):
    """绘制训练历史曲线"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax = axes[0]
    for history, name in zip(history_list, model_names):
        ax.plot(history['train_loss'], label=f'{name} Train')
        ax.plot(history['val_loss'], label=f'{name} Val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率曲线
    ax = axes[1]
    for history, name in zip(history_list, model_names):
        ax.plot(history['train_acc'], label=f'{name} Train')
        ax.plot(history['val_acc'], label=f'{name} Val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('训练准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("PyTorch MLP 实战 - MNIST 手写数字分类")
    print("=" * 60)

    # 超参数
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n设备：{DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"学习率：{LR}")
    print(f"训练轮数：{EPOCHS}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])

    # 加载数据
    print("\n加载 MNIST 数据集...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小：{len(train_dataset)}")
    print(f"测试集大小：{len(test_dataset)}")

    # 创建模型
    mlp = MLP(input_size=784, hidden_sizes=[256, 128], num_classes=10)
    modern_mlp = ModernMLP(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10)

    print("\n" + "=" * 60)
    print("MLP 模型结构:")
    print("=" * 60)
    print(mlp)

    print("\n" + "=" * 60)
    print("ModernMLP 模型结构:")
    print("=" * 60)
    print(modern_mlp)

    # 训练模型
    print("\n" + "=" * 60)
    print("训练 MLP...")
    print("=" * 60)
    mlp_history = train_mlp(mlp, train_loader, test_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    print("\n" + "=" * 60)
    print("训练 ModernMLP...")
    print("=" * 60)
    modern_mlp_history = train_mlp(modern_mlp, train_loader, test_loader,
                                   epochs=EPOCHS, lr=LR, device=DEVICE)

    # 可视化训练历史
    plot_training_history(
        [mlp_history, modern_mlp_history],
        ['MLP', 'ModernMLP']
    )

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
