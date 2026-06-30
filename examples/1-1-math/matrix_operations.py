"""
1.1 数学基础 - 矩阵运算可视化

本示例演示线性代数核心概念及其在 AI 中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def demonstrate_matrix_multiplication():
    """
    演示矩阵乘法的几何意义
    矩阵乘法本质是线性变换
    """
    print("=" * 50)
    print("矩阵乘法演示")
    print("=" * 50)

    # 原始向量
    v = np.array([2, 1])

    # 变换矩阵
    scale_matrix = np.array([[2, 0], [0, 0.5]])  # 缩放变换
    rotate_matrix = np.array([[0, -1], [1, 0]])  # 旋转 90 度
    shear_matrix = np.array([[1, 0.5], [0, 1]])  # 剪切变换

    print(f"\n原始向量：{v}")
    print(f"缩放后：{scale_matrix @ v}")
    print(f"旋转后：{rotate_matrix @ v}")
    print(f"剪切后：{shear_matrix @ v}")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制原始向量
    ax.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, label='Original')

    # 绘制变换后的向量
    ax.arrow(0, 0, (scale_matrix @ v)[0], (scale_matrix @ v)[1],
             head_width=0.2, head_length=0.3, fc='red', ec='red',
             linewidth=2, label='Scaled')

    ax.arrow(0, 0, (rotate_matrix @ v)[0], (rotate_matrix @ v)[1],
             head_width=0.2, head_length=0.3, fc='green', ec='green',
             linewidth=2, label='Rotated')

    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('矩阵变换可视化')
    plt.show()


def demonstrate_eigenvalues():
    """
    特征值与特征向量可视化
    特征向量是在变换中保持方向的向量
    """
    print("\n" + "=" * 50)
    print("特征值与特征向量演示")
    print("=" * 50)

    # 创建一个矩阵
    A = np.array([[3, 1], [1, 2]])

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\n矩阵 A:\n{A}")
    print(f"\n特征值：{eigenvalues}")
    print(f"\n特征向量:\n{eigenvectors}")

    # 验证：Av = λv
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_v = eigenvalues[i] * v
        Av = A @ v
        print(f"\n验证特征值 {eigenvalues[i]:.4f}:")
        print(f"  Av = {Av}")
        print(f"  λv = {lambda_v}")
        print(f"  误差 = {np.linalg.norm(Av - lambda_v):.10f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制特征向量
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        ax.arrow(0, 0, v[0] * 2, v[1] * 2,
                 head_width=0.15, head_length=0.2,
                 fc='red' if eigenvalues[i] > 1 else 'blue',
                 ec='red' if eigenvalues[i] > 1 else 'blue',
                 linewidth=3, label=f'λ={eigenvalues[i]:.2f}')

    # 绘制单位圆变换后的椭圆
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    ax.plot(ellipse[0], ellipse[1], 'g--', alpha=0.5, label='单位圆变换后')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('特征向量方向（变换后的不变方向）')
    plt.show()


def demonstrate_svd():
    """
    奇异值分解 (SVD) 演示
    任何矩阵都可以分解为旋转 - 缩放 - 旋转
    """
    print("\n" + "=" * 50)
    print("奇异值分解 (SVD) 演示")
    print("=" * 50)

    # 创建一个矩阵
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(f"\n原始矩阵 A ({A.shape[0]}×{A.shape[1]}):")
    print(A)

    # SVD 分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"\nU 矩阵 ({U.shape}):")
    print(U[:, :3])  # 显示前 3 列

    print(f"\n奇异值 (S): {S}")

    print(f"\nV^T 矩阵:")
    print(Vt)

    # 验证重建
    S_matrix = np.diag(S)
    A_reconstructed = U @ S_matrix @ Vt
    print(f"\n重建误差：{np.linalg.norm(A - A_reconstructed):.10f}")

    # 应用：低秩近似
    print("\n" + "-" * 40)
    print("SVD 应用：矩阵压缩")
    print("-" * 40)

    for rank in [1, 2, 3]:
        A_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
        compression = (rank * (U.shape[1] + Vt.shape[0] + 1)) / (A.shape[0] * A.shape[1])
        print(f"Rank {rank}: 相对误差={error:.4f}, 压缩比={compression:.2%}")


def demonstrate_gradient_descent():
    """
    梯度下降法可视化
    """
    print("\n" + "=" * 50)
    print("梯度下降法演示")
    print("=" * 50)

    # 目标函数：f(x, y) = x² + y²
    def f(x, y):
        return x**2 + y**2

    def gradient(x, y):
        return np.array([2*x, 2*y])

    # 梯度下降
    x, y = 3.0, 3.0
    learning_rate = 0.1
    path = [(x, y, f(x, y))]

    for i in range(20):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        path.append((x, y, f(x, y)))

    path = np.array(path)
    print(f"\n初始点：(3.0, 3.0), f=18.0")
    print(f"最终点：({path[-1, 0]:.4f}, {path[-1, 1]:.4f}), f={path[-1, 2]:.6f}")

    # 可视化
    fig = plt.figure(figsize=(12, 5))

    # 3D 表面
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.linspace(-4, 4, 50), np.linspace(-4, 4, 50))
    Z = f(X, Y)
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', linewidth=2, label='梯度下降路径')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D 视图')

    # 等高线
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('等高线视图')
    plt.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.show()


def demonstrate_bayes_theorem():
    """
    贝叶斯定理应用：医学检测
    """
    print("\n" + "=" * 50)
    print("贝叶斯定理应用：疾病检测")
    print("=" * 50)

    # 已知条件
    prevalence = 0.01  # 疾病发病率 1%
    sensitivity = 0.99  # 检测敏感性（真阳性率）
    specificity = 0.95  # 检测特异性（真阴性率 95%）

    # 计算 P(患病 | 阳性)
    # P(A|B) = P(B|A) * P(A) / P(B)
    # P(B) = P(B|A)*P(A) + P(B|¬A)*P(¬A)

    p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    p_disease_given_positive = (sensitivity * prevalence) / p_positive

    print(f"""
已知条件:
  - 疾病发病率：{prevalence*100:.1f}%
  - 检测敏感性：{sensitivity*100:.1f}%
  - 检测特异性：{specificity*100:.1f}%

计算结果:
  - P(阳性) = {p_positive*100:.2f}%
  - P(患病 | 阳性) = {p_disease_given_positive*100:.2f}%

解释：即使检测准确率 99%，由于基础发病率很低，
      检测为阳性时真正患病的概率只有约{p_disease_given_positive*100:.1f}%
    """)

    # 可视化不同发病率下的后验概率
    prevalences = np.linspace(0.001, 0.1, 100)
    posteriors = []

    for prev in prevalences:
        p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
        post = (sensitivity * prev) / p_pos
        posteriors.append(post)

    plt.figure(figsize=(10, 5))
    plt.plot(prevalences * 100, np.array(posteriors) * 100, linewidth=2)
    plt.xlabel('疾病发病率 (%)')
    plt.ylabel('P(患病 | 阳性) (%)')
    plt.title('贝叶斯定理：检测结果为阳性时真正患病的概率')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI 数学基础 - 核心概念可视化")
    print("=" * 60)

    # 运行所有演示
    demonstrate_matrix_multiplication()
    demonstrate_eigenvalues()
    demonstrate_svd()
    demonstrate_gradient_descent()
    demonstrate_bayes_theorem()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
