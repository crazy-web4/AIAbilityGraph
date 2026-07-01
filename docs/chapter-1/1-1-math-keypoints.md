# 1.1 数学与统计基础 - 关键知识点详解

> 本节为 1.1 节的补充知识点，包含线性代数、概率论、微积分的核心概念与 AI 应用对应关系。

---

## 知识点 1: 线性代数核心概念速查

### 矩阵运算与 AI 应用对应表

| 概念 | 数学定义 | AI 中的应用 | 代码示例 |
|------|---------|------------|---------|
| **矩阵乘法** | $C = AB$ | 神经网络前向传播 | `torch.matmul(W, x)` |
| **转置** | $A^T$ | 梯度计算、注意力矩阵 | `x.T` |
| **逆矩阵** | $A^{-1}$ | 线性方程组求解 (少用) | `torch.inverse(A)` |
| **特征值分解** | $Av = \lambda v$ | PCA 降维、谱聚类 | `torch.eig(A)` |
| **SVD 分解** | $A = U\Sigma V^T$ | 推荐系统、矩阵压缩 | `torch.svd(A)` |
| **行列式** | $\det(A)$ | 概率密度变换、雅可比 | `torch.det(A)` |

### 关键概念详解

#### 1. 特征值与特征向量

**直观理解：**
- 特征向量是矩阵变换中**方向不变**的向量
- 特征值是该方向上的**缩放比例**

```python
# 特征值分解的应用：PCA 降维
import numpy as np
import torch

def pca_demo():
    """
    使用特征值分解进行 PCA 降维
    """
    # 假设数据 (n_samples, n_features)
    X = np.random.randn(100, 10)
    
    # 1. 中心化
    X_centered = X - X.mean(axis=0)
    
    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    
    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. 选择前 k 个主成分
    k = 3
    top_k_eigenvectors = eigenvectors[:, -k:]  # 最大的 k 个特征值对应特征向量
    
    # 5. 投影降维
    X_reduced = X_centered @ top_k_eigenvectors
    
    print(f"原始维度：{X.shape}, 降维后：{X_reduced.shape}")
    print(f"解释方差比：{eigenvalues[-k:].sum() / eigenvalues.sum():.2%}")
    
    return X_reduced

pca_demo()
```

#### 2. 奇异值分解 (SVD)

**应用场景：**
- 推荐系统（矩阵分解）
- 图像压缩
- 潜在语义分析 (LSA)

```python
def svd_compression_demo():
    """
    SVD 用于图像压缩
    """
    import matplotlib.pyplot as plt
    
    # 模拟灰度图像 (512x512)
    image = np.random.randn(512, 512)
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # 用不同秩近似
    ranks = [5, 20, 50, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, r in enumerate(ranks):
        # 秩 r 近似：只用前 r 个奇异值
        compressed = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        axes[i].imshow(compressed, cmap='gray')
        axes[i].set_title(f'秩 = {r}, 压缩率 = {(r*(512+512+1))/(512*512):.1%}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    svd_compression_demo()
```

---

## 知识点 2: 概率论核心概念

### 常用概率分布速查表

| 分布 | 概率密度/质量函数 | 期望 | 方差 | AI 应用 |
|------|------------------|------|------|--------|
| **伯努利** | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | 二分类标签 |
| **二项** | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | 多次独立试验 |
| **正态** | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | 参数初始化、噪声 |
| **均匀** | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 随机初始化 |
| **泊松** | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 稀有事件计数 |
| **指数** | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | 等待时间、寿命 |
| **狄利克雷** | $\frac{1}{B(\alpha)}\prod x_i^{\alpha_i-1}$ | $\frac{\alpha_i}{\alpha_0}$ | 复杂 | LDA 主题模型 |
| **多项式** | $\frac{n!}{\prod k_i!}\prod p_i^{k_i}$ | $np_i$ | $np_i(1-p_i)$ | 多分类、词袋模型 |

### 贝叶斯定理详解

**公式：**
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**AI 中的应用：**
1. **朴素贝叶斯分类器**
2. **贝叶斯优化** (超参数调优)
3. **贝叶斯神经网络** (不确定性估计)

```python
# 贝叶斯优化示例 (超参数调优)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np


class BayesianOptimization:
    """
    贝叶斯优化：用高斯过程代理 + 采集函数
    
    用于高效搜索超参数空间
    """
    
    def __init__(self, f, bounds, n_init=5, n_iter=20):
        """
        Args:
            f: 目标函数 (越小越好)
            bounds: 参数边界 [(min1, max1), ...]
            n_init: 初始随机采样数
            n_iter: 优化迭代次数
        """
        self.f = f
        self.bounds = bounds
        self.n_init = n_init
        self.n_iter = n_iter
        self.dim = len(bounds)
        
        # 已评估的点
        self.X_sample = None
        self.y_sample = None
        
        # 高斯过程
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    
    def _sample_random_points(self, n):
        """随机采样初始点"""
        points = []
        for _ in range(n):
            point = [np.random.uniform(low, high) for low, high in self.bounds]
            points.append(point)
        return np.array(points)
    
    def _acquisition_function(self, x):
        """
        采集函数：EI (Expected Improvement)
        
        EI(x) = E[max(f(x_best) - f(x), 0)]
        """
        # 预测均值和方差
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        
        if self.y_sample is None:
            return 0
        
        # 当前最优
        f_best = np.min(self.y_sample)
        
        # EI 计算
        with np.errstate(divide='ignore'):
            imp = f_best - mu
            Z = imp / sigma if sigma > 0 else 0
            ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
        
        return ei[0]
    
    def _norm_cdf(self, x):
        """标准正态分布 CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _norm_pdf(self, x):
        """标准正态分布 PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def optimize(self):
        """执行贝叶斯优化"""
        # 1. 初始随机采样
        X_init = self._sample_random_points(self.n_init)
        y_init = np.array([self.f(x) for x in X_init])
        
        self.X_sample = X_init
        self.y_sample = y_init
        
        print(f"初始最优：{np.min(y_init):.4f}")
        
        # 2. 迭代优化
        for i in range(self.n_iter):
            # 拟合高斯过程
            self.gp.fit(self.X_sample, self.y_sample)
            
            # 最大化采集函数找到下一个点
            from scipy.optimize import minimize
            
            def neg_ei(x):
                return -self._acquisition_function(x)
            
            # 多起点优化
            best_x = None
            best_ei = -np.inf
            
            for _ in range(10):
                x0 = self._sample_random_points(1)[0]
                result = minimize(neg_ei, x0, bounds=self.bounds, method='L-BFGS-B')
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x
            
            # 评估新点
            y_new = self.f(best_x)
            
            # 更新样本
            self.X_sample = np.vstack([self.X_sample, best_x])
            self.y_sample = np.append(self.y_sample, y_new)
            
            if i % 5 == 0:
                print(f"Iter {i+1}: 新点 f={y_new:.4f}, 当前最优={np.min(self.y_sample):.4f}")
        
        best_idx = np.argmin(self.y_sample)
        return {
            'best_params': self.X_sample[best_idx],
            'best_value': self.y_sample[best_idx],
            'history': {
                'X': self.X_sample,
                'y': self.y_sample
            }
        }


# 使用示例：优化神经网络超参数
if __name__ == "__main__":
    # 模拟目标函数 (假设是验证集损失)
    def objective(x):
        lr, batch_size = x
        # 模拟：最优 lr≈0.001, batch_size≈32
        loss = (np.log10(lr) + 3)**2 + **(batch_size - 32)2 / 100
        return loss + np.random.randn() * 0.1  # 加点噪声
    
    bounds = [
        (1e-5, 0.1),    # 学习率范围
        (8, 128)        # 批次大小范围
    ]
    
    optimizer = BayesianOptimization(objective, bounds, n_init=5, n_iter=20)
    result = optimizer.optimize()
    
    print(f"\n优化结果:")
    print(f"最佳参数：lr={result['best_params'][0]:.5f}, batch_size={result['best_params'][1]:.1f}")
    print(f"最优值：{result['best_value']:.4f}")
```

---

## 知识点 3: 微积分与优化

### 梯度下降变体对比

| 算法 | 更新公式 | 特点 | 适用场景 |
|------|---------|------|---------|
| **SGD** | $\theta = \theta - \eta\nabla L$ | 简单、可能震荡 | 凸优化、小数据 |
| **Momentum** | $v = \gamma v + \eta\nabla L, \theta = \theta - v$ | 积累动量、加速收敛 | 一般场景 |
| **AdaGrad** | $\theta = \theta - \frac{\eta}{\sqrt{G}+\epsilon}\nabla L$ | 自适应学习率 | 稀疏数据 |
| **RMSProp** | $v = \gamma v + (1-\gamma)(\nabla L)^2$ | 解决 AdaGrad 学习率衰减 | RNN/LSTM |
| **Adam** | 结合 Momentum + RMSProp | 默认选择 | 大多数场景 |

### 反向传播链式法则可视化

```
计算图示例：L = σ(W·x + b)

前向传播:
x ──→ [W·x + b] ──→ σ(·) ──→ L
      │              │
      z              a

反向传播 (链式法则):
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
      = (∂L/∂a) × (σ'(z)) × (x)
```

```python
# 手动实现反向传播演示
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def manual_backprop():
    """
    手动演示反向传播
    
    网络：input(3) -> linear -> sigmoid -> loss
    """
    # 模拟数据
    x = np.array([1.0, 2.0, 3.0])      # 输入
    W = np.array([[0.1, 0.2, 0.3]])    # 权重 (1x3)
    b = np.array([0.0])                 # 偏置
    y_true = 1.0                         # 真实标签
    
    # === 前向传播 ===
    z = np.dot(W, x) + b               # 线性层
    a = sigmoid(z)                      # 激活层
    
    # MSE 损失
    loss = (a - y_true) ** 2
    
    # === 反向传播 ===
    # 1. 损失对 a 的梯度
    d_loss_da = 2 * (a - y_true)
    
    # 2. a 对 z 的梯度 (sigmoid 导数)
    d_a_dz = sigmoid_derivative(z)
    
    # 3. z 对 W 的梯度
    d_z_dW = x  # ∂z/∂W = x
    
    # 组合梯度 (链式法则)
    d_loss_dW = d_loss_da * d_a_dz * d_z_dW
    
    print(f"前向：z={z:.4f}, a={a:.4f}, loss={loss:.4f}")
    print(f"梯度：∂L/∂W = {d_loss_dW}")
    
    # 参数更新
    lr = 0.1
    W_new = W - lr * d_loss_dW
    print(f"更新后 W: {W_new}")
    
    return W_new


if __name__ == "__main__":
    manual_backprop()
```

---

## 练习题

### 练习 1: 矩阵分解应用

给定用户 - 物品评分矩阵（稀疏），实现基于 SVD 的推荐系统：

```python
# 评分矩阵 (用户×物品)，-1 表示未评分
R = np.array([
    [5, 3, -1, 1],
    [4, -1, -1, 2],
    [1, 2, 4, -1],
    [-1, 3, 5, 4],
    [2, 4, 1, 3]
])

# 任务:
# 1. 用 SVD 分解预测缺失评分
# 2. 为用户 0 推荐 Top-2 未评分物品
```

### 练习 2: 贝叶斯定理应用

某疾病发病率为 1%，检测准确率为 99%（患者检出阳性 99%，健康人检出阴性 99%）。
如果某人检测阳性，实际患病的概率是多少？

用代码验证你的计算。

---

## 延伸阅读

- [3Blue1Brown 线性代数系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_abr)
- [统计学习方法](https://book.douban.com/subject/10590856/) - 李航
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) - Christopher Bishop

---

[← 返回 1.1 主文档](1-1-math-basics.md) | [下一节：机器学习算法 →](1-2-ml-algorithms.md)
