# 1.1 数学与统计基础

> AI 的数学基础三大支柱：**线性代数**（数据表示）、**微积分**（优化核心）、**概率统计**（不确定性建模）

## 学习目标

学完本节后，你将能够：

- [ ] 理解向量、矩阵运算及其在神经网络中的应用
- [ ] 掌握导数、梯度、偏导数的概念与优化意义
- [ ] 运用概率分布、贝叶斯定理进行不确定性推理
- [ ] 理解最大似然估计、假设检验等统计方法

---

## 1.1.1 线性代数核心

### 为什么 AI 需要线性代数？

神经网络的每一层都可以表示为矩阵运算：

```
输出 = 激活函数 (输入 × 权重矩阵 + 偏置)
```

### 核心概念

#### 1. 向量与矩阵

| 概念 | 数学表示 | Python/NumPy | AI 应用场景 |
|------|----------|--------------|-------------|
| 标量 | $s \in \mathbb{R}$ | `s = 5.0` | 学习率、损失值 |
| 向量 | $\mathbf{v} \in \mathbb{R}^n$ | `v = np.array([1, 2, 3])` | 特征向量、词嵌入 |
| 矩阵 | $\mathbf{A} \in \mathbb{R}^{m \times n}$ | `A = np.zeros((m, n))` | 权重矩阵、注意力矩阵 |
| 张量 | $\mathbf{T} \in \mathbb{R}^{d_1 \times ... \times d_n}$ | `T = torch.randn(2, 3, 4)` | 图像批次、Transformer 中间态 |

#### 2. 关键矩阵运算

```python
import numpy as np

# 矩阵乘法 - 神经网络前向传播的核心
A = np.random.randn(3, 4)  # 3×4 矩阵
B = np.random.randn(4, 2)  # 4×2 矩阵
C = A @ B                   # 矩阵乘法，结果 3×2

# 转置 - 用于梯度计算
A_T = A.T

# 逆矩阵 - 某些优化算法需要
A_square = np.random.randn(3, 3)
A_inv = np.linalg.inv(A_square)

# 特征值分解 - PCA、谱聚类的核心
eigenvalues, eigenvectors = np.linalg.eig(A_square)

# 奇异值分解 (SVD) - 矩阵压缩、推荐系统
U, S, Vt = np.linalg.svd(np.random.randn(5, 3))
```

#### 3. 向量范数

| 范数 | 公式 | 应用 |
|------|------|------|
| L0 范数 | $\|\mathbf{x}\|_0$ = 非零元素个数 | 稀疏性度量 |
| L1 范数 | $\|\mathbf{x}\|_1 = \sum |x_i|$ | Lasso 正则化 |
| L2 范数 | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ | Ridge 正则化、权重衰减 |
| L∞ 范数 | $\|\mathbf{x}\|_\infty = \max |x_i|$ | 梯度裁剪 |

---

## 1.1.2 微积分与优化

### 导数与梯度

**导数**衡量函数的瞬时变化率，是梯度下降法的基础。

```python
# 数值微分示例
def numerical_gradient(f, x, h=1e-5):
    """计算函数 f 在点 x 的梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 示例：计算 f(x, y) = x² + y² 的梯度
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([3.0, 4.0])
grad = numerical_gradient(f, x)
print(f"梯度：{grad}")  # 应该接近 [6, 8]
```

### 链式法则 - 反向传播的核心

对于复合函数 $y = f(g(x))$，导数为：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**神经网络反向传播**就是链式法则的递归应用：

```
损失 L → 输出层 → 隐藏层 2 → 隐藏层 1 → 输入层
         ↑          ↑           ↑
    每一层都用链式法则传递梯度
```

### 梯度下降法

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, iterations=1000):
    """
    梯度下降优化算法
    
    参数:
        f: 目标函数
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        iterations: 迭代次数
    """
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x.copy())
        
    return x, np.array(history)

# 示例：找到 f(x) = x² 的最小值
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

x0 = np.array([5.0])
minimum, history = gradient_descent(f, grad_f, x0)
print(f"最小值点：{minimum}")  # 接近 [0]
```

---

## 1.1.3 概率与统计

### 基础概率概念

| 概念 | 符号 | 定义 | AI 应用 |
|------|------|------|--------|
| 概率 | $P(A)$ | 事件 A 发生的可能性 | 分类置信度 |
| 联合概率 | $P(A, B)$ | A 和 B 同时发生 | 多标签分类 |
| 条件概率 | $P(A|B)$ | 已知 B 时 A 的概率 | 贝叶斯推理 |
| 边缘概率 | $P(A) = \sum_B P(A, B)$ | 忽略其他变量 | 预测分布 |

### 贝叶斯定理

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**在 AI 中的应用**：
- 朴素贝叶斯分类器
- 贝叶斯优化（超参数调优）
- 变分推断（VAE、贝叶斯神经网络）

### 常见概率分布

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# 正态分布 - 深度学习权重初始化
normal = stats.norm(loc=0, scale=1)
x = np.linspace(-4, 4, 100)
y = normal.pdf(x)

# 伯努利分布 - 二分类问题
bernoulli = stats.bernoulli(p=0.7)
prob = bernoulli.pmf([0, 1])  # P(X=0), P(X=1)

# 多项式分布 - 多分类问题
multinomial = stats.multinomial(n=10, p=[0.3, 0.5, 0.2])
prob = multinomial.pmf([3, 5, 2])
```

### 统计推断

| 方法 | 用途 | AI 场景 |
|------|------|--------|
| 最大似然估计 (MLE) | 参数估计 | 训练损失函数推导 |
| 最大后验估计 (MAP) | 带先验的参数估计 | 正则化解释 |
| 假设检验 | 判断差异是否显著 | A/B 测试、模型对比 |
| 置信区间 | 参数估计的不确定性 | 模型预测区间 |

---

## 1.1.4 实战代码示例

### 代码 1：矩阵运算可视化

见 [`examples/1-1-math/matrix_operations.py`](../../examples/1-1-math/matrix_operations.py)

### 代码 2：梯度下降可视化

见 [`examples/1-1-math/gradient_descent_vis.py`](../../examples/1-1-math/gradient_descent_vis.py)

### 代码 3：贝叶斯分类器实战

见 [`examples/1-1-math/bayes_classifier.py`](../../examples/1-1-math/bayes_classifier.py)

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| "线性代数只要会调用库就行" | 理解矩阵维度变化对调试神经网络至关重要 |
| "微积分必须手算所有导数" | 理解概念即可，框架会自动求导 |
| "概率统计太理论，实战用不上" | 分布假设、置信区间直接影响模型评估 |

---

## 练习题

### 基础题

1. **矩阵乘法维度**：若 $\mathbf{A} \in \mathbb{R}^{3 \times 4}$，$\mathbf{B} \in \mathbb{R}^{4 \times 2}$，求 $\mathbf{AB}$ 的维度。

2. **梯度计算**：求 $f(x, y) = x^2y + \sin(y)$ 关于 $x$ 和 $y$ 的偏导数。

3. **贝叶斯应用**：某疾病发病率 1%，检测准确率 99%，检测为阳性时真正患病的概率是多少？

### 编程题

4. 实现一个批量梯度下降函数，支持学习率衰减。

5. 用 NumPy 实现softmax 函数及其梯度（注意数值稳定性）。

---

## 延伸阅读

- 📘 《Linear Algebra and Its Applications》- Gilbert Strang
- 📘 《Pattern Recognition and Machine Learning》- Christopher Bishop (第 1-2 章)
- 🌐 [3Blue1Brown 线性代数本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_abr)
- 🌐 [MIT 18.06 线性代数公开课](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

---

[← 上一节：章前导引](README.md) | [下一节：1.2 经典机器学习算法 →](1-2-ml-algorithms.md)
