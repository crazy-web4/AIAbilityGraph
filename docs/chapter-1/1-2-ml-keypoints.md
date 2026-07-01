# 1.2 经典机器学习算法 - 关键知识点详解

> 本节为 1.2 节的补充知识点，包含算法对比表、实战代码、调参指南。

---

## 知识点 1: 监督学习算法对比

### 核心算法对比表

| 算法 | 原理 | 优点 | 缺点 | 适用场景 | sklearn API |
|------|------|------|------|---------|------------|
| **线性回归** | 最小二乘法拟合直线 | 简单、可解释 | 只能拟合线性关系 | 房价预测、销量预测 | `LinearRegression` |
| **逻辑回归** | Sigmoid+ 极大似然 | 简单、输出概率 | 线性边界 | 二分类、风险评估 | `LogisticRegression` |
| **决策树** | 信息增益/基尼系数分裂 | 可解释、无需标准化 | 易过拟合 | 规则清晰的场景 | `DecisionTreeClassifier` |
| **随机森林** | Bagging+ 决策树集成 | 准确、抗过拟合 |  Black-box、慢 | 通用、特征重要性 | `RandomForestClassifier` |
| **GBDT** | Boosting 集成弱学习器 | 准确、处理复杂关系 | 参数敏感 | 结构化数据竞赛 | `GradientBoostingClassifier` |
| **XGBoost** | GBDT 优化版 | 更快更准、支持正则 | 参数多 | 竞赛/工业首选 | `XGBClassifier` |
| **LightGBM** | 基于直方图的 GBDT | 最快、支持大规模 | 小数据易过拟合 | 大数据集 | `LGBMClassifier` |
| **SVM** | 最大化间隔 + 核技巧 | 高维有效、核函数灵活 | 大数据慢、难调参 | 小数据集、高维 | `SVC` |
| **KNN** | 最近邻投票 | 简单、无需训练 | 预测慢、高维失效 | 小规模、局部模式 | `KNeighborsClassifier` |
| **朴素贝叶斯** | 贝叶斯定理 + 独立假设 | 快、适合文本 | 独立性假设强 | 文本分类、垃圾邮件 | `GaussianNB` |

### XGBoost 实战代码

```python
# examples/1-2-ml/xgboost_demo.py
"""
XGBoost 完整实战示例

涵盖:
1. 数据预处理
2. 模型训练
3. 超参数调优
4. 特征重要性分析
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def create_sample_data():
    """创建模拟分类数据"""
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,      # 10 个有效特征
        n_redundant=5,         # 5 个冗余特征
        n_clusters_per_class=2,
        random_state=42
    )
    
    # 转换为 DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names


def xgboost_pipeline():
    """XGBoost 完整流程"""
    
    # 1. 准备数据
    print("=" * 50)
    print("1. 数据准备")
    print("=" * 50)
    
    df, feature_names = create_sample_data()
    
    X = df[feature_names]
    y = df['target']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集：{X_train.shape}, 测试集：{X_test.shape}")
    print(f"正样本比例：{y_train.mean():.2%}")
    
    # 2. 创建 DMatrix (XGBoost 专用数据格式)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 3. 基础模型训练
    print("\n" + "=" * 50)
    print("2. 基础模型训练")
    print("=" * 50)
    
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    model = xgb.train(
        base_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'valid')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # 4. 预测与评估
    print("\n" + "=" * 50)
    print("3. 模型评估")
    print("=" * 50)
    
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 5. 特征重要性
    print("\n" + "=" * 50)
    print("4. 特征重要性分析")
    print("=" * 50)
    
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['feature', 'gain']
    )
    importance_df = importance_df.sort_values('gain', ascending=False)
    
    print("\nTop 10 重要特征:")
    print(importance_df.head(10).to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, x='gain', y='feature')
    plt.title('Top 15 特征重要性 (Gain)')
    plt.xlabel('Gain')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("\n特征重要性图已保存：feature_importance.png")
    
    # 6. 超参数调优示例
    print("\n" + "=" * 50)
    print("5. 超参数调优 (Grid Search)")
    print("=" * 50)
    
    # 使用 sklearn API 进行网格搜索
    xgb_sklearn = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        seed=42,
        use_label_encoder=False
    )
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
    }
    
    grid_search = GridSearchCV(
        xgb_sklearn,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数：{grid_search.best_params_}")
    print(f"最佳 AUC: {grid_search.best_score_:.4f}")
    
    # 用最佳参数重新训练
    best_model = grid_search.best_estimator_
    best_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"测试集 AUC: {best_auc:.4f}")
    
    return model, best_model


if __name__ == "__main__":
    model, best_model = xgboost_pipeline()
```

---

## 知识点 2: 无监督学习算法

### 聚类算法对比

| 算法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **K-Means** | 最小化簇内方差 | 简单、快速 | 需指定 K、对异常值敏感 | 球形簇、大数据 |
| **层次聚类** | 树状聚合/分裂 | 可视化好、无需指定 K | 慢、内存消耗大 | 小数据、探索分析 |
| **DBSCAN** | 基于密度的聚类 | 无需指定 K、发现任意形状 | 对参数敏感 | 非球形簇、异常检测 |
| **GMM** | 高斯混合模型 EM 算法 | 软聚类、概率输出 | 需指定 K、慢 | 重叠簇、概率模型 |

### K-Means 实现与可视化

```python
# examples/1-2-ml/kmeans_demo.py
"""
K-Means 聚类完整示例

包含:
1. 手肘法选择 K
2. 聚类可视化
3. 轮廓系数评估
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns


def create_blobs_data(n_features=2):
    """创建模拟聚类数据"""
    X, y_true = make_blobs(
        n_samples=1000,
        centers=5,
        n_features=n_features,
        cluster_std=0.8,
        random_state=42
    )
    return X, y_true


def find_optimal_k(X, k_range=range(2, 11)):
    """
    用手肘法和轮廓系数找最优 K
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        labels = kmeans.labels_
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        print(f"K={k}: Inertia={inertias[-1]:.2f}, Silhouette={sil_score:.3f}")
    
    return inertias, silhouette_scores


def plot_elbow_and_silhouette(k_range, inertias, silhouette_scores):
    """绘制手肘图和轮廓系数图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 手肘图
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('簇数量 K')
    ax1.set_ylabel('Inertia (SSE)')
    ax1.set_title('手肘法选择 K')
    ax1.grid(True, alpha=0.3)
    
    # 轮廓系数图
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('簇数量 K')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_k_selection.png', dpi=150)
    print("\nK 选择图已保存：kmeans_k_selection.png")


def plot_clusters(X, labels, title):
    """绘制聚类结果"""
    plt.figure(figsize=(8, 6))
    
    if X.shape[1] == 2:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                            alpha=0.6, s=50)
        plt.colorbar(scatter)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title(title)
        plt.savefig(f'kmeans_{title}.png', dpi=150)
    
    plt.show()


def kmeans_full_pipeline():
    """K-Means 完整流程"""
    
    # 1. 创建数据
    print("=" * 50)
    print("K-Means 聚类完整示例")
    print("=" * 50)
    
    X, y_true = create_blobs_data()
    print(f"数据形状：{X.shape}")
    print(f"真实簇数：{len(np.unique(y_true))}")
    
    # 2. 寻找最优 K
    print("\n" + "=" * 50)
    print("1. 寻找最优 K")
    print("=" * 50)
    
    k_range = range(2, 11)
    inertias, silhouette_scores = find_optimal_k(X, k_range)
    plot_elbow_and_silhouette(k_range, inertias, silhouette_scores)
    
    # 3. 用最优 K 训练
    print("\n" + "=" * 50)
    print("2. 训练最终模型")
    print("=" * 50)
    
    optimal_k = 5  # 根据上面的结果选择
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    print(f"\n簇中心形状：{centers.shape}")
    print(f"簇大小分布：{np.bincount(labels)}")
    
    # 计算轮廓系数
    sil_score = silhouette_score(X, labels)
    print(f"轮廓系数：{sil_score:.3f}")
    
    # 4. 可视化
    plot_clusters(X, labels, title=f'K-Means(K={optimal_k})')
    
    # 同时绘制真实标签对比
    plot_clusters(X, y_true, title=f'True Labels')
    
    # 5. 分析聚类质量
    print("\n" + "=" * 50)
    print("3. 聚类质量分析")
    print("=" * 50)
    
    # 用混淆矩阵式的方式查看簇分配
    from sklearn.metrics import contingency_matrix
    
    cm = contingency_matrix(y_true, labels)
    print(f"\n簇分配矩阵 (行：真实标签，列：预测簇):")
    print(cm)
    
    return kmeans


if __name__ == "__main__":
    kmeans = kmeans_full_pipeline()
```

---

## 知识点 3: 模型评估与选择

### 分类指标详解

| 指标 | 公式 | 含义 | 适用场景 |
|------|------|------|---------|
| **准确率** | (TP+TN)/(TP+TN+FP+FN) | 预测正确的比例 | 类别平衡 |
| **精确率** | TP/(TP+FP) | 预测为正的有多少真阳 | 关注假阳性代价 |
| **召回率** | TP/(TP+FN) | 真阳中有多少被找出 | 关注假阴性代价 |
| **F1 分数** | 2×P×R/(P+R) | 精确率和召回率调和平均 | 需要平衡 P 和 R |
| **AUC-ROC** | ROC 曲线下面积 | 区分正负样本能力 | 类别不平衡、模型对比 |

```python
# 评估指标可视化
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)


def plot_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    绘制分类评估可视化
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        threshold: 分类阈值
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('混淆矩阵')
    axes[0, 0].set_xlabel('预测')
    axes[0, 0].set_ylabel('真实')
    
    # 2. ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, f'AUC = {roc_auc:.3f}', color='darkorange', lw=2)
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('假阳性率 (FP/(FP+TN))')
    axes[0, 1].set_ylabel('真阳性率 (TP/(TP+FN))')
    axes[0, 1].set_title('ROC 曲线')
    axes[0, 1].legend(loc='lower right')
    
    # 3. Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[1, 0].plot(recall, precision, color='blue', lw=2)
    axes[1, 0].set_xlabel('召回率')
    axes[1, 0].set_ylabel('精确率')
    axes[1, 0].set_title('Precision-Recall 曲线')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 阈值影响
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_true == 1))
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        fn = np.sum((y_pred_t == 0) & (y_true == 1))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
    
    axes[1, 1].plot(thresholds, precisions, 'b-', label='精确率')
    axes[1, 1].plot(thresholds, recalls, 'r-', label='召回率')
    axes[1, 1].plot(thresholds, f1_scores, 'g-', label='F1 分数')
    axes[1, 1].set_xlabel('阈值')
    axes[1, 1].set_ylabel('分数')
    axes[1, 1].set_title('阈值对指标的影响')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=150)
    print("分类评估图已保存：classification_metrics.png")


if __name__ == "__main__":
    # 测试
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, random_state=42)
    
    # 模拟模型预测概率
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]
    
    plot_classification_metrics(y, y_proba)
```

---

## 练习题

### 练习 1: 特征工程实战

给定以下数据集，完成特征工程并训练模型：

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 加载数据
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 任务:
# 1. 探索性数据分析 (分布、相关性)
# 2. 特征转换 (对数变换、分箱等)
# 3. 特征选择 (相关性、重要性)
# 4. 训练随机森林并评估
```

### 练习 2: 处理类别不平衡

使用 SMOTE、类别权重等方法处理类别不平衡问题：

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# 创建不平衡数据
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],  # 9:1 不平衡
    random_state=42
)

# 任务:
# 1. 比较不同采样方法 (SMOTE、欠采样)
# 2. 比较不同类别权重设置
# 3. 使用 AUC-ROC 和 F1 评估
```

---

## 延伸阅读

- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [XGBoost 论文](https://arxiv.org/abs/1603.02754)

---

[← 返回 1.2 主文档](1-2-ml-algorithms.md) | [下一节：深度学习基础 →](1-3-deep-learning.md)
