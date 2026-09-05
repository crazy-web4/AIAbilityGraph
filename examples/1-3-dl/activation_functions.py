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
