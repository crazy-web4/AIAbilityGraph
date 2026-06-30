"""
梯度下降法可视化 - 动画版本

展示不同学习率、动量对优化过程的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def rosenbrock(x, y):
    """
    Rosenbrock 函数 - 非凸优化测试函数
    f(x, y) = (a - x)² + b(y - x²)²
    全局最小值在 (a, a²)，通常 a=1, b=100
    """
    a, b = 1, 100
    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_gradient(x, y):
    """Rosenbrock 函数的梯度"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


def gradient_descenttrajectory(start, lr, momentum=0, num_steps=100):
    """
    梯度下降轨迹（支持动量）

    参数:
        start: 起始点 [x, y]
        lr: 学习率
        momentum: 动量系数 (0 = 无动量)
        num_steps: 迭代步数
    """
    path = [start.copy()]
    velocity = np.array([0.0, 0.0])

    x, y = start

    for _ in range(num_steps):
        grad = rosenbrock_gradient(x, y)

        if momentum > 0:
            # 带动量的梯度下降
            velocity = momentum * velocity - lr * grad
            x = x + velocity[0]
            y = y + velocity[1]
        else:
            # 标准梯度下降
            x = x - lr * grad[0]
            y = y - lr * grad[1]

        path.append([x, y])

    return np.array(path)


def plot_optimization_comparison():
    """
    比较不同优化设置的效果
    """
    print("正在计算优化轨迹...")

    # 起始点
    start = np.array([-2.0, 2.0])

    # 不同学习率
    trajectories = {
        'LR=0.0001': gradient_descenttrajectory(start, lr=0.0001, num_steps=200),
        'LR=0.0005': gradient_descenttrajectory(start, lr=0.0005, num_steps=200),
        'LR=0.001': gradient_descenttrajectory(start, lr=0.001, num_steps=200),
        'LR=0.002 (发散)': gradient_descenttrajectory(start, lr=0.002, num_steps=50),
    }

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 创建网格
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-1.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # 绘制等高线
    for ax in axes:
        ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 3.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Rosenbrock 函数优化轨迹')
        ax.grid(True, alpha=0.3)

    # 绘制不同学习率的轨迹
    colors = ['red', 'orange', 'green', 'purple']
    for (name, path), color in zip(trajectories.items(), colors):
        axes[0].plot(path[:, 0], path[:, 1], 'o-', color=color,
                     linewidth=1, markersize=3, alpha=0.7, label=name)

    axes[0].legend(loc='upper right')
    axes[0].plot([1], [1], 'w*', markersize=20, label='全局最优 (1, 1)')

    # 动量对比
    momentum_paths = {
        '无动量': gradient_descenttrajectory(start, lr=0.0005, momentum=0, num_steps=200),
        '动量=0.5': gradient_descenttrajectory(start, lr=0.0005, momentum=0.5, num_steps=200),
        '动量=0.9': gradient_descenttrajectory(start, lr=0.0005, momentum=0.9, num_steps=200),
    }

    colors = ['blue', 'cyan', 'magenta']
    for (name, path), color in zip(momentum_paths.items(), colors):
        axes[1].plot(path[:, 0], path[:, 1], 'o-', color=color,
                     linewidth=1, markersize=3, alpha=0.6, label=name)

    axes[1].legend(loc='upper right')
    axes[1].plot([1], [1], 'w*', markersize=20, label='全局最优 (1, 1)')

    plt.tight_layout()
    plt.show()

    # 打印收敛结果
    print("\n最终位置对比:")
    print("-" * 40)
    for name, path in trajectories.items():
        final_loss = rosenbrock(path[-1, 0], path[-1, 1])
        print(f"{name:15s}: f({path[-1, 0]:.3f}, {path[-1, 1]:.3f}) = {final_loss:.4f}")

    print("\n动量效果对比:")
    print("-" * 40)
    for name, path in momentum_paths.items():
        final_loss = rosenbrock(path[-1, 0], path[-1, 1])
        print(f"{name:15s}: f({path[-1, 0]:.3f}, {path[-1, 1]:.3f}) = {final_loss:.4f}")


def plot_learning_rate_schedules():
    """
    不同学习率调度策略对比
    """
    print("\n" + "=" * 50)
    print("学习率调度策略对比")
    print("=" * 50)

    # 总步数
    total_steps = 100

    # 不同调度策略
    lr_const = np.ones(total_steps) * 0.01
    lr_step = np.ones(total_steps) * 0.01
    lr_step[50:] = 0.001  # 50 步后衰减

    lr_exp = 0.01 * (0.95 ** np.arange(total_steps))  # 指数衰减
    lr_cosine = 0.01 * (1 + np.cos(np.pi * np.arange(total_steps) / total_steps)) / 2  # 余弦退火

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.plot(lr_const, label='Constant LR=0.01', linewidth=2)
    plt.plot(lr_step, label='Step Decay (step=50)', linewidth=2)
    plt.plot(lr_exp, label='Exponential Decay (γ=0.95)', linewidth=2)
    plt.plot(lr_cosine, label='Cosine Annealing', linewidth=2)

    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("梯度下降可视化 - 学习率与动量影响")
    print("=" * 60)

    # 学习率比较
    plot_optimization_comparison()

    # 学习率调度
    plot_learning_rate_schedules()

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
