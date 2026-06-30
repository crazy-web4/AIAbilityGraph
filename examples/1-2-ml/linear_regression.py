"""
1.2 机器学习算法 - 线性回归实战

从零实现 linear regression 并对比 sklearn 版本
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionGD:
    """
    从零实现线性回归（梯度下降法）
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        训练模型

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算损失 (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """计算 R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_synthetic_data(n_samples=100, noise=10):
    """
    生成合成数据用于回归演示

    真实关系：y = 3*x + 50 + noise
    """
    np.random.seed(42)
    X = 2 * np.random.randn(n_samples, 1)
    y = 50 + 3 * X[:, 0] + np.random.randn(n_samples) * noise

    return X, y


def compare_implementations():
    """
    对比从零实现与 sklearn 的实现
    """
    print("=" * 60)
    print("线性回归：从零实现 vs sklearn")
    print("=" * 60)

    # 生成数据
    X, y = generate_synthetic_data(n_samples=100, noise=10)

    # 从零实现
    clf_gd = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    clf_gd.fit(X, y)
    y_pred_gd = clf_gd.predict(X)

    # sklearn 实现
    clf_sklearn = LinearRegression()
    clf_sklearn.fit(X, y)
    y_pred_sklearn = clf_sklearn.predict(X)

    # 对比结果
    print(f"\n真实参数：w=3.000, b=50.000")
    print(f"\n从零实现 (梯度下降):")
    print(f"  w = {clf_gd.weights[0]:.4f}, b = {clf_gd.bias:.4f}")
    print(f"  R² = {clf_gd.score(X, y):.4f}")

    print(f"\nsklearn (正规方程):")
    print(f"  w = {clf_sklearn.coef_[0]:.4f}, b = {clf_sklearn.intercept_:.4f}")
    print(f"  R² = {clf_sklearn.score(X, y):.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 回归线对比
    axes[0].scatter(X, y, alpha=0.6, label='数据点')
    axes[0].plot(X, y_pred_gd, 'r-', linewidth=2, label=f'从零实现 (R²={clf_gd.score(X, y):.3f})')
    axes[0].plot(X, y_pred_sklearn, 'g--', linewidth=2,
                 label=f'sklearn (R²={clf_sklearn.score(X, y):.3f})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('线性回归拟合效果对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 损失曲线
    axes[1].plot(clf_gd.loss_history, linewidth=2)
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('损失 (MSE)')
    axes[1].set_title('梯度下降损失曲线')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_polynomial_regression():
    """
    多项式回归演示过拟合与正则化
    """
    print("\n" + "=" * 60)
    print("多项式回归：过拟合与正则化")
    print("=" * 60)

    # 生成数据
    np.random.seed(42)
    n_samples = 30
    X = np.linspace(0, 10, n_samples)

    # 真实函数：y = sin(x) + noise
    y = np.sin(X) + np.random.randn(n_samples) * 0.1

    # 不同多项式程度
    degrees = [1, 3, 5, 10]

    fig, axes = plt.subplots(1, len(degrees), figsize=(20, 4))

    X_test = np.linspace(0, 10, 100)

    for idx, deg in enumerate(degrees):
        # 多项式特征
        X_poly = np.vander(X, N=deg + 1, increasing=True)
        X_test_poly = np.vander(X_test, N=deg + 1, increasing=True)

        # 拟合
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_test_poly)

        # 计算训练误差
        y_train_pred = model.predict(X_poly)
        train_mse = mean_squared_error(y, y_train_pred)

        axes[idx].scatter(X, y, alpha=0.6, label='训练数据')
        axes[idx].plot(X_test, y_pred, 'r-', linewidth=2, label=f'Degree {deg}')
        axes[idx].plot(X_test, np.sin(X_test), 'g--', alpha=0.5, label='真实函数')
        axes[idx].set_title(f'Degree {deg}\n训练 MSE={train_mse:.4f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n观察:")
    print("  - Degree 1: 欠拟合，无法捕捉正弦曲线")
    print("  - Degree 3: 合适拟合，接近真实函数")
    print("  - Degree 5: 开始过拟合")
    print("  - Degree 10: 严重过拟合，训练 MSE 很小但泛化差")


def demonstrate_regularization():
    """
    正则化技术：Ridge (L2) vs Lasso (L1)
    """
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("正则化：Ridge (L2) vs Lasso (L1)")
    print("=" * 60)

    # 生成高维数据（特征多于样本）
    np.random.seed(42)
    n_samples, n_features = 50, 100

    # 只有前 10 个特征有效
    true_weights = np.zeros(n_features)
    true_weights[:10] = np.random.randn(10) * 10

    X = np.random.randn(n_samples, n_features)
    y = np.dot(X, true_weights) + np.random.randn(n_samples)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 不同模型
    models = {
        '无正则化': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=1.0),
    }

    print("\n模型对比:")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        # 计算有效特征数（权重绝对值 > 0.01）
        if hasattr(model, 'coef_'):
            n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
        else:
            n_nonzero = n_features

        r2 = r2_score(y, y_pred)
        print(f"{name:15s}: R²={r2:.4f}, 有效特征数={n_nonzero}")

    # 可视化权重
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = ['blue', 'red', 'green']
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'coef_'):
            ax.plot(np.abs(model.coef_), color=color, label=name, alpha=0.7)

    ax.axvline(x=10, color='black', linestyle='--', label='真实有效特征边界')
    ax.set_xlabel('特征索引')
    ax.set_ylabel('权重绝对值')
    ax.set_title('正则化对权重的影响 (前 10 个特征应有较大权重)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行所有演示
    compare_implementations()
    demonstrate_polynomial_regression()
    demonstrate_regularization()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
