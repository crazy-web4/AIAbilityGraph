# 1.2 经典机器学习算法

> 机器学习是 AI 的核心分支，本章系统讲解**监督学习**、**无监督学习**与**集成方法**三大知识模块。

## 学习目标

学完本节后，你将能够：

- [ ] 理解并实现常见监督学习算法（线性回归、逻辑回归、SVM、决策树）
- [ ] 掌握无监督学习方法（K-Means 聚类、PCA 降维）
- [ ] 运用集成方法提升模型性能（Bagging、Boosting、Random Forest）
- [ ] 根据数据特点选择合适的算法

---

## 1.2.1 监督学习算法

监督学习：从**带标签**的训练数据中学习映射函数 $f: X \rightarrow Y$

### 1. 线性回归 (Linear Regression)

**适用场景**：连续值预测（房价、销量、温度等）

**模型形式**：
$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

**损失函数**（均方误差）：
$$L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

```python
# examples/1-2-ml/linear_regression.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 训练数据
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
X_test = np.array([[6], [7], [8]])
y_pred = model.predict(X_test)

print(f"权重: w = {model.coef_[0]:.3f}, 截距: b = {model.intercept_:.3f}")
print(f"拟合方程：y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
```

### 2. 逻辑回归 (Logistic Regression)

**适用场景**：二分类问题（垃圾邮件检测、疾病诊断）

**核心思想**：用 Sigmoid 函数将线性输出映射到 (0, 1)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

```python
# examples/1-2-ml/logistic_regression.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 生成数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + np.array([2, 2]),   # 类别 0
    np.random.randn(100, 2) + np.array([-2, -2])  # 类别 1
])
y = np.array([0]*100 + [1]*100)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
print(f"准确率：{accuracy_score(y_test, y_pred):.3f}")
print(f"系数: w = {clf.coef_}, 截距: b = {clf.intercept_}")
```

### 3. 支持向量机 (SVM)

**适用场景**：小样本、高维分类（文本分类、图像识别）

**核心思想**：找到最大化间隔的决策超平面

**关键概念**：
- **支持向量**：决定边界的少数关键样本
- **核技巧**：将低维线性不可分问题映射到高维

```python
# examples/1-2-ml/svm_classifier.py
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 创建 SVM 分类器（RBF 核）
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# 交叉验证评估
scores = cross_val_score(clf, X, y, cv=5)
print(f"SVM 5 折交叉验证准确率：{scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# 核函数选择指南
"""
kernel='linear'  - 线性可分数据，特征数>>样本数
kernel='rbf'     - 通用选择，处理非线性
kernel='poly'    - 已知数据有多项式关系
"""
```

### 4. 决策树 (Decision Tree)

**适用场景**：可解释性要求高的场景（风控、医疗诊断）

**核心概念**：
- **信息增益**：ID3 算法的分裂标准
- **信息增益比**：C4.5 算法，减少偏向多值属性
- **基尼不纯度**：CART 算法，计算效率更高

```python
# examples/1-2-ml/decision_tree.py
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# 创建决策树（限制深度防止过拟合）
tree = DecisionTreeClassifier(
    max_depth=4,           # 最大深度
    min_samples_split=10,  # 内部节点再划分所需最小样本数
    criterion='gini'       # 基尼系数
)
tree.fit(X_train, y_train)

# 评估
train_score = tree.score(X_train, y_train)
test_score = tree.score(X_test, y_test)
print(f"训练准确率：{train_score:.3f}, 测试准确率：{test_score:.3f}")

# 特征重要性
importances = sorted(
    zip(data.feature_names, tree.feature_importances_),
    key=lambda x: x[1], reverse=True
)[:5]  # Top 5 重要特征
print("\nTop 5 重要特征:")
for name, imp in importances:
    print(f"  {name}: {imp:.4f}")
```

---

## 1.2.2 无监督学习算法

### 1. K-Means 聚类

**适用场景**：客户分群、图像压缩、异常检测

**算法步骤**：
1. 随机初始化 K 个聚类中心
2. 将每个样本分配到最近的中心
3. 重新计算每个聚类的中心
4. 重复 2-3 直到收敛

```python
# examples/1-2-ml/kmeans_clustering.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成聚类数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Means 聚类
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',    # 智能初始化，加速收敛
    n_init=10,           # 不同初始化运行 10 次取最优
    max_iter=300
)
y_pred = kmeans.fit_predict(X)

# 评估：轮廓系数（-1 到 1，越大越好）
from sklearn.metrics import silhouette_score
score = silhouette_score(X, y_pred)
print(f"轮廓系数：{score:.3f}")

# 聚类中心
centers = kmeans.cluster_centers_
print(f"聚类中心形状：{centers.shape}")
```

### 2. 层次聚类 (Hierarchical Clustering)

**适用场景**：需要层次结构（生物分类、文档层次化）

```python
# examples/1-2-ml/hierarchical_clustering.py
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
np.random.seed(0)
X = np.vstack([
    np.random.randn(50, 2) + [3, 3],
    np.random.randn(50, 2) + [-3, -3]
])

# 层次聚类
linkage_matrix = linkage(X, method='ward')  # ward: 最小化簇内方差

# 绘制树状图
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title("层次聚类树状图")
plt.xlabel("样本索引")
plt.ylabel("距离")
plt.show()
```

### 3. 主成分分析 (PCA)

**适用场景**：数据降维、特征提取、可视化

**核心思想**：找到数据方差最大的正交投影方向

```python
# examples/1-2-ml/pca_dimensionality_reduction.py
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载手写数字数据集（64 维）
digits = load_digits()
X = digits.data
y = digits.target

# 标准化（PCA 前必须步骤）
X_scaled = StandardScaler().fit_transform(X)

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 解释方差比
print(f"第一主成分解释方差：{pca.explained_variance_ratio_[0]:.3f}")
print(f"第二主成分解释方差：{pca.explained_variance_ratio_[1]:.3f}")
print(f"累计解释方差：{sum(pca.explained_variance_ratio_):.3f}")

# 可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='数字类别')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
plt.title('PCA 降维可视化')
plt.show()
```

---

## 1.2.3 集成方法

### Bagging - 并行训练多个模型

**代表算法**：随机森林 (Random Forest)

```python
# examples/1-2-ml/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 随机森林
rf = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=10,          # 树的最大深度
    min_samples_split=5,   # 内部节点再划分所需最小样本数
    max_features='sqrt',   # 最佳特征数 = sqrt(特征总数)
    oob_score=True         # 使用袋外样本评估
)
rf.fit(X_train, y_train)

print(f"袋外分数 (OOB): {rf.oob_score_:.3f}")
print(f"测试集分数：{rf.score(X_test, y_test):.3f}")
```

### Boosting - 串行训练，每轮修正前一轮错误

**代表算法**：AdaBoost、Gradient Boosting、XGBoost

```python
# examples/1-2-ml/gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 梯度提升树
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8  # 每棵树使用 80% 样本
)

# 交叉验证
scores = cross_val_score(gb, X, y, cv=5)
print(f"Gradient Boosting 准确率：{scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

---

## 算法选择指南

| 任务类型 | 数据规模 | 推荐算法 | 理由 |
|----------|----------|----------|------|
| 二分类 | 小样本 | 逻辑回归/SVM | 简单、不易过拟合 |
| 多分类 | 中等 | 随机森林/XGBoost | 准确率高、鲁棒 |
| 回归 | 有线性关系 | 线性回归/Ridge | 可解释性强 |
| 回归 | 非线性 | 梯度提升树 | 拟合复杂模式 |
| 聚类 | 球形簇 | K-Means | 快速、易实现 |
| 降维 | 高维数据 | PCA/t-SNE | PCA 保全局、t-SNE 保局部 |

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| "越复杂的算法越好" | 奥卡姆剃刀：简单模型在小数据上往往更好 |
| "集成方法总是最优" | 集成需要更多计算资源，实时场景可能不适用 |
| "标准化可有可无" | SVM、K-Means、PCA 对特征尺度敏感，必须标准化 |

---

## 练习题

### 基础题

1. **偏差 - 方差权衡**：解释为什么决策树深度过大导致过拟合？如何用交叉验证检测？

2. **SVM 核选择**：什么时候用线性核？什么时候用 RBF 核？

3. **聚类评估**：为什么准确率不能用于聚类评估？轮廓系数如何计算？

### 编程题

4. 实现一个完整的机器学习 pipeline：
   - 数据预处理（缺失值处理 + 标准化）
   - 特征选择（基于重要性或相关性）
   - 模型训练与调参
   - 交叉验证评估

5. 比较随机森林、XGBoost、LightGBM 在同一数据集上的表现。

---

## 延伸阅读

- 📘 《统计学习方法》- 李航 (第 1-9 章)
- 📘 《The Elements of Statistical Learning》- Hastie et al.
- 🌐 [Scikit-learn 机器学习教程](https://scikit-learn.org/stable/supervised_learning.html)
- 🌐 [XGBoost 官方文档](https://xgboost.readthedocs.io/)

---

[← 上一节：1.1 数学基础](1-1-math-basics.md) | [下一节：1.3 深度学习基础 →](1-3-deep-learning.md)
