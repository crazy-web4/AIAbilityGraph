"""
贝叶斯分类器实战

从零实现朴素贝叶斯分类器，并应用于文本分类任务
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


class NaiveBayesClassifier:
    """
    多项式朴素贝叶斯分类器

    适用于离散特征，如文本分类中的词频统计
    """

    def __init__(self, alpha=1.0):
        """
        参数:
            alpha: 拉普拉斯平滑参数，防止零概率
        """
        self.alpha = alpha
        self.class_priors = {}  # P(C)
        self.feature_probs = {}  # P(F|C)
        self.classes = []
        self.vocab = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练分类器

        参数:
            X: 特征矩阵 (n_samples, n_features) - 词频向量
            y: 标签数组 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.vocab = n_features

        # 计算先验概率 P(C)
        class_counts = Counter(y)
        for c in self.classes:
            self.class_priors[c] = class_counts[c] / n_samples

        # 计算条件概率 P(F|C) - 使用拉普拉斯平滑
        for c in self.classes:
            X_c = X[y == c]  # 该类的所有样本

            # 该类中每个特征的总计数
            feature_counts = X_c.sum(axis=0) + self.alpha

            # 归一化为概率
            total = feature_counts.sum()
            self.feature_probs[c] = feature_counts / total

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测各类别的后验概率

        返回:
            (n_samples, n_classes) 概率矩阵
        """
        log_probs = []

        for c in self.classes:
            # 对数先验
            log_prior = np.log(self.class_priors[c])

            # 对数似然：sum of log P(F|C) for each feature
            # X @ log(P(F|C)) 高效计算
            log_likelihood = X @ np.log(self.feature_probs[c] + 1e-10)

            # 后验概率（对数空间）
            log_posterior = log_prior + log_likelihood
            log_probs.append(log_posterior)

        # 转换为概率（numerical stability: subtract max）
        log_probs = np.vstack(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)


def create_text_dataset():
    """
    创建简单的文本分类数据集

    类别：体育、科技、政治
    """
    # 简化的词袋表示
    vocabulary = [
        '比赛', '球队', '进球', '冠军', '运动', '球员', '裁判', '体育场',
        '手机', '软件', '代码', '数据', '算法', '人工智能', '程序', '网络',
        '政府', '政策', '选举', '法律', '议会', '总统', '改革', '经济'
    ]

    word2idx = {word: idx for idx, word in enumerate(vocabulary)}

    # 训练样本
    texts = [
        # 体育类
        "球队 进球 冠军 球员 比赛 体育场",
        "运动 球员 裁判 比赛 进球",
        "冠军 球队 体育场 比赛 球员",
        "比赛 裁判 进球 冠军 运动",

        # 科技类
        "软件 代码 算法 数据 程序",
        "人工智能 算法 数据 软件 网络",
        "代码 程序 软件 网络 数据",
        "算法 人工智能 程序 软件",

        # 政治类
        "政府 政策 选举 法律",
        "总统 改革 经济 政府 政策",
        "选举 议会 法律 改革",
        "政策 政府 经济 改革 总统"
    ]

    labels = ['体育'] * 4 + ['科技'] * 4 + ['政治'] * 4

    # 转换为词频向量
    X = []
    for text in texts:
        vec = np.zeros(len(vocabulary))
        words = text.split()
        for word in words:
            if word in word2idx:
                vec[word2idx[word]] += 1
        X.append(vec)

    return np.array(X), np.array(labels), vocabulary


def demonstrate_bayes_classification():
    """
    演示贝叶斯分类器工作流程
    """
    print("=" * 60)
    print("朴素贝叶斯文本分类演示")
    print("=" * 60)

    # 加载数据
    X, y, vocabulary = create_text_dataset()

    print(f"\n数据集信息:")
    print(f"  样本数：{len(X)}")
    print(f"  词汇表大小：{len(vocabulary)}")
    print(f"  类别：{np.unique(y)}")

    # 训练分类器
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X, y)

    # 输出先验概率
    print("\n" + "-" * 40)
    print("类别先验概率 P(C):")
    for c, prob in clf.class_priors.items():
        print(f"  P({c}) = {prob:.3f}")

    # 输出条件概率（每个类别的前几个特征）
    print("\n" + "-" * 40)
    print("各类别条件概率 P(F|C) - Top 5 特征:")
    for c in clf.classes:
        probs = clf.feature_probs[c]
        top_idx = np.argsort(probs)[-5:][::-1]
        print(f"\n  {c}:")
        for idx in top_idx:
            print(f"    {vocabulary[idx]}: {probs[idx]:.4f}")

    # 训练集准确率
    train_acc = clf.score(X, y)
    print(f"\n" + "-" * 40)
    print(f"训练集准确率：{train_acc:.3f}")

    # 测试新样本
    print("\n" + "-" * 40)
    print("预测新样本:")

    test_texts = [
        "球队 冠军 比赛 进球",
        "人工智能 软件 算法 代码",
        "政府 政策 选举 改革"
    ]

    for text in test_texts:
        # 转换为特征向量
        vec = np.zeros(len(vocabulary))
        for word in text.split():
            if word in word2idx:
                vec[word2idx[word]] = 1

        pred = clf.predict(vec.reshape(1, -1))[0]
        probs = clf.predict_proba(vec.reshape(1, -1))[0]

        print(f"\n  文本：{text}")
        print(f"  预测：{pred}")
        print(f"  概率分布：")
        for c, p in zip(clf.classes, probs):
            print(f"    {c}: {p:.3f}")


def demonstrate_spam_detection():
    """
    垃圾邮件检测示例
    """
    print("\n" + "=" * 60)
    print("垃圾邮件检测示例")
    print("=" * 60)

    # 简化的垃圾邮件数据集
    vocabulary = ['免费', '赢取', '奖金', '点击', '链接', '优惠', '限时', '活动',
                  '会议', '报告', '项目', '进度', '文档', '附件', '审阅',
                  '你好', '谢谢', '请问', '关于', '希望']

    word2idx = {word: idx for idx, word in enumerate(vocabulary)}

    # 训练数据
    texts = [
        # 垃圾邮件 (1)
        "免费 赢取 奖金 点击 限时 优惠",
        "奖金 活动 点击 链接 免费",
        "限时 优惠 赢取 免费 奖金",
        "点击 链接 免费 活动 优惠",

        # 正常邮件 (0)
        "会议 报告 项目 进度 文档",
        "附件 请 审阅 报告 文档",
        "关于 项目 进度 会议 报告",
        "你好 请问 关于 项目 希望"
    ]

    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # 转换为特征向量
    X = []
    for text in texts:
        vec = np.zeros(len(vocabulary))
        for word in text.split():
            if word in word2idx:
                vec[word2idx[word]] = 1
        X.append(vec)
    X = np.array(X)

    # 训练
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X, labels)

    # 测试
    test_cases = [
        ("免费 赢取 点击 链接", "典型垃圾邮件"),
        ("会议 报告 文档 附件", "典型工作邮件"),
        ("免费 会议 优惠 报告", "混合特征"),
        ("你好 关于 项目 希望", "日常沟通")
    ]

    print("\n测试结果:")
    print("-" * 50)

    for text, description in test_cases:
        vec = np.zeros(len(vocabulary))
        for word in text.split():
            if word in word2idx:
                vec[word2idx[word]] = 1

        pred = clf.predict(vec.reshape(1, -1))[0]
        probs = clf.predict_proba(vec.reshape(1, -1))[0]

        label = "垃圾邮件" if pred == 1 else "正常邮件"
        print(f"\n{description}:")
        print(f"  文本：{text}")
        print(f"  预测：{label}")
        print(f"  P(垃圾)= {probs[1]:.3f}, P(正常)= {probs[0]:.3f}")


if __name__ == "__main__":
    # 运行演示
    demonstrate_bayes_classification()
    demonstrate_spam_detection()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
