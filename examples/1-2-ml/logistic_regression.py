"""
1.2 机器学习算法 - 逻辑回归实战

从零实现逻辑回归并应用于二分类任务
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def sigmoid(z):
    """Sigmoid 激活函数（数值稳定版本）"""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


class LogisticRegressionGD:
    """
    从零实现逻辑回归（梯度下降法）
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization=0.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.reg = regularization  # L2 正则化
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            # 计算损失 (交叉熵 + L2 正则化)
            epsilon = 1e-15  # 防止 log(0)
            loss = -np.mean(y * np.log(y_pred + epsilon) +
                           (1 - y) * np.log(1 - y_pred + epsilon))
            loss += (self.reg / 2) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


def visualize_decision_boundary():
    """
    可视化逻辑回归的决策边界
    """
    print("=" * 60)
    print("逻辑回归：决策边界可视化")
    print("=" * 60)

    # 生成数据
    np.random.seed(42)

    # 类别 0:  centered at (2, 2)
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)

    # 类别 1: centered at (-2, -2)
    X1 = np.random.randn(50, 2) + np.array([-2, -2])
    y1 = np.ones(50)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # 训练模型
    clf = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    clf.fit(X, y)

    print(f"\n训练完成!")
    print(f"  权重：w = {clf.weights}")
    print(f"  偏置：b = {clf.bias:.4f}")
    print(f"  训练准确率：{clf.score(X, y):.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 决策边界
    ax = axes[0]
    ax.scatter(X0[:, 0], X0[:, 1], c='blue', marker='o', label='类别 0', alpha=0.6)
    ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='s', label='类别 1', alpha=0.6)

    # 绘制决策边界：w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b) / w2
    x_min, x_max = -4, 4
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(clf.weights[0] * x_vals + clf.bias) / clf.weights[1]
    ax.plot(x_vals, y_vals, 'k-', linewidth=2, label='决策边界')

    # 决策边界两侧的区域
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                         colors=['blue', 'red'], alpha=0.1)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('逻辑回归决策边界')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 损失曲线
    ax = axes[1]
    ax.plot(clf.loss_history, linewidth=2)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('损失 (交叉熵)')
    ax.set_title('训练损失曲线')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_regularization_effect():
    """
    演示正则化对过拟合的影响
    """
    print("\n" + "=" * 60)
    print("正则化效果演示")
    print("=" * 60)

    # 生成数据（带噪声）
    np.random.seed(42)
    n_samples = 100

    # 生成两个有重叠的类别
    X = np.random.randn(n_samples, 2)
    # 添加一些"异常点"
    X = np.vstack([X, [[3, 3]] * 5, [[-3, -3]] * 5])  # 异常点
    y = np.array([0] * 50 + [1] * 50 + [0] * 5 + [1] * 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 不同正则化强度
    reg_values = [0.0, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    results = []

    for idx, reg in enumerate(reg_values):
        clf = LogisticRegressionGD(learning_rate=0.1, n_iterations=500, regularization=reg)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        results.append({
            'reg': reg,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'weights': clf.weights.copy()
        })

        # 绘制决策区域
        ax = axes[idx]

        xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.2)
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                  c='blue', marker='o', alpha=0.6, label='类别 0')
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                  c='red', marker='s', alpha=0.6, label='类别 1')

        ax.set_title(f'正则化 λ={reg}\n训练准确率={train_acc:.3f}, 测试准确率={test_acc:.3f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印结果
    print("\n正则化对模型的影响:")
    print("-" * 50)
    print(f"{'λ':>10} {'训练准确率':>12} {'测试准确率':>12} {'||w||²':>12}")
    print("-" * 50)
    for r in results:
        weight_norm = np.sum(r['weights'] ** 2)
        print(f"{r['reg']:>10.1f} {r['train_acc']:>12.4f} {r['test_acc']:>12.4f} {weight_norm:>12.4f}")


def demonstrate_multiclass_classification():
    """
    多分类：One-vs-Rest 策略
    """
    print("\n" + "=" * 60)
    print("多分类逻辑回归 (One-vs-Rest)")
    print("=" * 60)

    from sklearn.datasets import load_iris

    # 加载鸢尾花数据集
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # 只使用前两个特征便于可视化

    # One-vs-Rest 逻辑回归
    class OneVsRestLogisticRegression:
        def __init__(self, n_classes):
            self.n_classes = n_classes
            self.classifiers = []

        def fit(self, X, y):
            for c in range(self.n_classes):
                # 转换为二分类问题
                y_binary = (y == c).astype(int)
                clf = LogisticRegressionGD(learning_rate=0.1, n_iterations=500)
                clf.fit(X, y_binary)
                self.classifiers.append(clf)
            return self

        def predict(self, X):
            probs = np.array([clf.predict_proba(X) for clf in self.classifiers]).T
            return np.argmax(probs, axis=1)

        def score(self, X, y):
            return np.mean(self.predict(X) == y)

    # 训练
    clf = OneVsRestLogisticRegression(n_classes=3)
    clf.fit(X, y)

    # 评估
    accuracy = clf.score(X, y)
    print(f"训练集准确率：{accuracy:.4f}")

    # 混淆矩阵
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    print("\n混淆矩阵:")
    print(cm)

    # 可视化决策区域
    fig, ax = plt.subplots(figsize=(8, 6))

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    colors = ['red', 'green', 'blue']
    for c in range(3):
        ax.scatter(X[y == c, 0], X[y == c, 1],
                  c=colors[c], label=iris.target_names[c],
                  edgecolors='black', alpha=0.7)

    ax.set_xlabel('花萼长度')
    ax.set_ylabel('花萼宽度')
    ax.set_title(f'多分类逻辑回归\n准确率={accuracy:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行所有演示
    visualize_decision_boundary()
    demonstrate_regularization_effect()
    demonstrate_multiclass_classification()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
