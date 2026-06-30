# 1.4 AI 安全与伦理

> 随着 AI 技术广泛应用，**安全性**、**公平性**、**可解释性**和**伦理规范**成为必须考虑的核心议题。

## 学习目标

学完本节后，你将能够：

- [ ] 理解对抗攻击的原理与防御方法
- [ ] 识别并缓解 AI 系统中的偏见问题
- [ ] 应用可解释性工具分析模型决策
- [ ] 掌握 AI 伦理框架与合规要求

---

## 1.4.1 对抗攻击与防御

### 什么是对抗攻击？

**对抗样本**：在输入上添加人眼不可察觉的扰动，使模型产生错误输出。

$$\mathbf{x}_{adv} = \mathbf{x} + \delta, \quad \text{其中} \|\delta\|_\infty \leq \epsilon$$

$$f(\mathbf{x}_{adv}) \neq f(\mathbf{x})$$

### 经典攻击方法

#### 1. FGSM (Fast Gradient Sign Method)

**原理**：沿梯度方向添加扰动

$$\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_\mathbf{x} J(\mathbf{x}, y))$$

```python
# examples/1-4-ethics/adversarial_attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def fgsm_attack(model, image, label, epsilon=0.02):
    """
    Fast Gradient Sign Method (FGSM) 攻击
    
    参数:
        model: 目标模型
        image: 输入图像 (requires_grad=True)
        label: 真实标签
        epsilon: 扰动上限
    
    返回:
        adversarial_image: 对抗样本
    """
    model.eval()
    image.requires_grad = True
    
    # 前向传播
    output = model(image)
    loss = F.cross_entropy(output, label)
    
    # 反向传播获取梯度
    loss.backward()
    
    # 生成对抗样本
    image_adv = image + epsilon * image.grad.data.sign()
    
    # 截断到合法范围 [0, 1]
    image_adv = torch.clamp(image_adv, 0, 1)
    
    return image_adv.detach()

# 使用示例
def evaluate_adrobustness(model, test_loader, epsilon=0.02):
    """评估模型在对抗攻击下的鲁棒性"""
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images_adv = fgsm_attack(model, images, labels, epsilon)
        outputs = model(images_adv)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"对抗准确率 (ε={epsilon}): {accuracy:.4f}")
    return accuracy
```

#### 2. PGD (Projected Gradient Descent)

**原理**：多步迭代攻击，比 FGSM 更强

```python
def pgd_attack(model, image, label, epsilon=0.02, steps=10, alpha=0.005):
    """
    Projected Gradient Descent 攻击
    
    参数:
        steps: 迭代步数
        alpha: 每步大小
    """
    model.eval()
    image_adv = image.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        output = model(image_adv)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        
        with torch.no_grad():
            # 梯度上升
            image_adv = image_adv + alpha * image_adv.grad.sign()
            
            # 投影到 epsilon 球内
            delta = image_adv - image
            delta = torch.clamp(delta, -epsilon, epsilon)
            image_adv = torch.clamp(image + delta, 0, 1)
        
        image_adv.grad.zero_()
    
    return image_adv.detach()
```

### 防御方法

#### 1. 对抗训练 (Adversarial Training)

**核心思想**：将对抗样本加入训练集

```python
def adversarial_training_step(model, images, labels, optimizer, epsilon=0.02):
    """
    对抗训练单步
    
    1. 生成对抗样本
    2. 用对抗样本计算损失
    3. 反向传播更新
    """
    model.train()
    
    # 生成对抗样本
    images_adv = fgsm_attack(model, images, labels, epsilon)
    
    # 用对抗样本训练
    optimizer.zero_grad()
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

#### 2. 防御性蒸馏 (Defensive Distillation)

```python
class DistillationLoss(nn.Module):
    """
    知识蒸馏损失 - 也可用于防御
    
    T: 温度参数，软化输出分布
    """
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha  # 硬标签权重
        self.kl_div = nn.KLDivLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, hard_labels):
        # 软目标损失（分布匹配）
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T ** 2)
        
        # 硬目标损失（分类准确）
        hard_loss = self.ce(student_logits, hard_labels)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

### 对抗攻防对比表

| 攻击方法 | 强度 | 速度 | 防御难度 |
|----------|------|------|----------|
| FGSM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| PGD | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| CW Attack | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 自适应攻击 | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

---

## 1.4.2 AI 公平性与偏见

### 偏见来源

| 来源 | 描述 | 示例 |
|------|------|------|
| 数据偏见 | 训练数据不具代表性 | 人脸数据集以白人为主 |
| 标签偏见 | 标注者的主观判断 | 简历筛选中的性别偏好 |
| 算法偏见 | 目标函数设计不当 | 推荐系统放大极端内容 |
| 评估偏见 | 测试集覆盖不全 | 只在小样本上验证 |

### 公平性指标

```python
# examples/1-4-ethics/fairness_metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """
    AI 公平性评估指标
    """
    
    @staticmethod
    def demographic_parity(y_true, y_pred, sensitive_attr):
        """
        人口统计学均等：不同群体获得正例的概率相同
        
        P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
        """
        groups = np.unique(sensitive_attr)
        rates = []
        
        for g in groups:
            mask = sensitive_attr == g
            rate = np.mean(y_pred[mask] == 1)
            rates.append(rate)
        
        # 差异越小越公平
        return max(rates) - min(rates)
    
    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_attr):
        """
        均等机会：不同群体有相同的 TPR 和 FPR
        """
        groups = np.unique(sensitive_attr)
        tpr_diff = []
        fpr_diff = []
        
        for g in groups:
            mask = sensitive_attr == g
            tn, fp, fn, tp = confusion_matrix(
                y_true[mask], y_pred[mask]
            ).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_diff.append(tpr)
            fpr_diff.append(fpr)
        
        return max(tpr_diff) - min(tpr_diff), max(fpr_diff) - min(fpr_diff)
    
    @staticmethod
    def disparate_impact(y_pred, sensitive_attr):
        """
        不同影响率：保护群体与优势群体正例率的比值
        
        0.8 规则：比值 < 0.8 认为存在歧视
        """
        groups = np.unique(sensitive_attr)
        rates = [np.mean(y_pred[sensitive_attr == g] == 1) for g in groups]
        
        return min(rates) / max(rates) if max(rates) > 0 else 0
```

### 偏见缓解技术

#### 1. 数据层面 - 重采样

```python
from imblearn.over_sampling import SMOTE

def balance_dataset(X, y):
    """使用 SMOTE 平衡不同群体的数据"""
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced
```

#### 2. 算法层面 - 公平性约束

```python
class FairClassifier:
    """
    带公平性约束的分类器
    """
    def __init__(self, base_model, fairness_lambda=0.1):
        self.model = base_model
        self.lambda_ = fairness_lambda  # 公平性权重
    
    def fair_loss(self, y_pred, y_true, sensitive_attr):
        """
        带公平性惩罚的损失函数
        
        L_total = L_classification + λ * L_fairness
        """
        # 分类损失
        cls_loss = F.cross_entropy(y_pred, y_true)
        
        # 公平性损失（不同群体预测差异）
        groups = torch.unique(sensitive_attr)
        group_probs = []
        for g in groups:
            mask = sensitive_attr == g
            group_prob = y_pred[mask].softmax(dim=1)[:, 1].mean()
            group_probs.append(group_prob)
        
        fair_loss = torch.std(torch.stack(group_probs))
        
        return cls_loss + self.lambda_ * fair_loss
```

---

## 1.4.3 模型可解释性

### 1. LIME (Local Interpretable Model-agnostic Explanations)

**原理**：用局部可解释模型近似复杂模型

```python
# examples/1-4-ethics/model_explainability.py
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

def explain_with_lime(model, X_train, X_test, feature_names, class_names):
    """
    使用 LIME 解释模型预测
    """
    # 创建解释器
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # 解释单个样本
    sample_idx = 0
    exp = explainer.explain_instance(
        X_test[sample_idx],
        model.predict_proba,
        num_features=5
    )
    
    # 获取解释
    print("Top 5 重要特征:")
    for feature, importance in exp.as_list():
        print(f"  {feature}: {importance:.4f}")
    
    # 可视化
    exp.show_in_notebook(show_table=True)
    
    return exp
```

### 2. SHAP (SHapley Additive exPlanations)

**原理**：基于博弈论的 Shapley 值分配特征贡献

```python
import shap

def explain_with_shap(model, X_background, X_test, feature_names=None):
    """
    使用 SHAP 解释模型
    """
    # 创建解释器（TreeExplainer 适用于树模型）
    explainer = shap.TreeExplainer(model, X_background)
    
    # 计算 SHAP 值
    shap_values = explainer.shap_values(X_test)
    
    # 可视化
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    shap.dependence_plot(0, shap_values, X_test, feature_names=feature_names)
    
    return shap_values

# 深度学习模型 SHAP
def explain_deep_model(model, X_background, X_test):
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test)
    return shap_values
```

### 3. 注意力可视化

```python
def visualize_attention(model, tokenizer, text):
    """
    可视化 Transformer 注意力权重
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # 获取注意力权重 (layer, head, query, key)
    attentions = outputs.attentions[-1]  # 最后一层
    
    # 获取 token 到词的映射
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # 可视化注意力热力图
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    attn_matrix = attentions[0, 0].cpu().numpy()  # 第一个样本，第一个头
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens)
    plt.title("Attention Map - Head 0")
    plt.show()
```

### 可解释性方法对比

| 方法 | 模型无关 | 局部/全局 | 计算成本 | 适用场景 |
|------|----------|-----------|----------|----------|
| LIME | ✅ | 局部 | 低 | 快速解释单个预测 |
| SHAP | ✅ | 局部 + 全局 | 中 | 特征重要性分析 |
| Attention | ❌ | 全局 | 低 | Transformer 模型 |
| Saliency Map | ❌ | 局部 | 低 | 图像模型 |
| Partial Dependence | ✅ | 全局 | 高 | 特征效应分析 |

---

## 1.4.4 AI 伦理框架

### 核心伦理原则

| 原则 | 描述 | 实践方法 |
|------|------|----------|
| **有益性** | AI 应造福人类 | 影响评估、风险评估 |
| **非maleficence** | 不伤害原则 | 安全测试、红队演练 |
| **自主性** | 尊重人类决策权 | 人机协同、可退出机制 |
| **公正性** | 公平对待所有人 | 偏见检测、多样化团队 |
| **透明性** | 决策过程可追溯 | 文档记录、可解释性 |

### 负责任 AI 检查清单

```markdown
## 模型发布前检查清单

### 数据层面
- [ ] 数据来源合法且符合隐私法规
- [ ] 数据覆盖多样人群，无明显偏差
- [ ] 敏感属性（种族、性别等）已妥善处理

### 模型层面
- [ ] 已评估不同群体上的性能差异
- [ ] 已测试对抗鲁棒性
- [ ] 已生成可解释性报告

### 部署层面
- [ ] 有明确的使用场景和限制说明
- [ ] 有人工审核/申诉机制
- [ ] 有监控和回滚计划

### 合规层面
- [ ] 符合 GDPR/个人信息保护法
- [ ] 通过伦理审查委员会审批
- [ ] 有明确的责任归属
```

### 相关法规

| 法规 | 适用范围 | 关键要求 |
|------|----------|----------|
| **GDPR** (欧盟) | 在欧运营企业 | 解释权、被遗忘权、数据可携带 |
| **AI Act** (欧盟) | AI 系统提供商 | 风险分级、高风险系统严格监管 |
| **个人信息保护法** (中国) | 在中国处理个人信息 | 最小必要、知情同意、目的限制 |
| **算法推荐管理规定** (中国) | 算法推荐服务 | 透明、公平、不得诱导沉迷 |

---

## 实战项目：公平性审计

```python
# examples/1-4-ethics/fairness_audit.py
"""
完整 Fairness Audit 流程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class AIFairnessAudit:
    """
    AI 系统公平性审计工具
    """
    
    def __init__(self, model, sensitive_features):
        """
        参数:
            model: 训练好的模型
            sensitive_features: 敏感特征列表 (如'gender', 'race')
        """
        self.model = model
        self.sensitive_features = sensitive_features
        self.results = {}
    
    def audit(self, X_test, y_test, X_full):
        """
        执行完整审计
        
        返回:
            dict: 包含各项公平性指标
        """
        # 1. 整体性能
        y_pred = self.model.predict(X_test)
        self.results['overall_accuracy'] = accuracy_score(y_test, y_pred)
        self.results['overall_f1'] = f1_score(y_test, y_pred, average='macro')
        
        # 2. 分群体性能
        group_metrics = {}
        for feature in self.sensitive_features:
            groups = X_full[feature].unique()
            group_metrics[feature] = {}
            
            for group in groups:
                mask = X_test[feature] == group
                if mask.sum() > 0:
                    group_acc = accuracy_score(y_test[mask], y_pred[mask])
                    group_f1 = f1_score(y_test[mask], y_pred[mask], average='macro')
                    group_metrics[feature][group] = {
                        'accuracy': group_acc,
                        'f1': group_f1,
                        'sample_size': mask.sum()
                    }
        
        self.results['group_metrics'] = group_metrics
        
        # 3. 公平性指标
        self.results['fairness_gaps'] = self._compute_gaps(group_metrics)
        
        return self.results
    
    def _compute_gaps(self, group_metrics):
        """计算群体间性能差距"""
        gaps = {}
        for feature, groups in group_metrics.items():
            accuracies = [g['accuracy'] for g in groups.values()]
            gaps[f'{feature}_accuracy_gap'] = max(accuracies) - min(accuracies)
        return gaps
    
    def generate_report(self):
        """生成审计报告"""
        print("=" * 50)
        print("AI 系统公平性审计报告")
        print("=" * 50)
        print(f"整体准确率：{self.results['overall_accuracy']:.4f}")
        print(f"整体 F1 分数：{self.results['overall_f1']:.4f}")
        print()
        
        print("分群体性能:")
        for feature, groups in self.results['group_metrics'].items():
            print(f"\n  {feature}:")
            for group, metrics in groups.items():
                print(f"    {group}: Acc={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1']:.4f}, N={metrics['sample_size']}")
        
        print("\n公平性差距:")
        for metric, gap in self.results['fairness_gaps'].items():
            status = "⚠️ 需关注" if gap > 0.05 else "✅ 可接受"
            print(f"  {metric}: {gap:.4f} {status}")
        
        return self.results


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randn(n_samples) * 10000 + 50000,
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['A', 'B', 'C'], n_samples),
    })
    
    # 生成带有偏见的标签
    y = (data['income'] > 50000).astype(int)
    y[data['gender'] == 'F'] = np.random.binomial(1, 0.7, (y[data['gender'] == 'F'] == 1).sum())
    
    X = pd.get_dummies(data, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 执行审计
    audit = AIFairnessAudit(model, ['gender', 'race'])
    audit.audit(X_test, y_test, data)
    audit.generate_report()
```

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| "对抗攻击只是理论问题" | 已有实际攻击案例（自动驾驶误导、内容审核绕过） |
| "公平性只是政治正确" | 偏见导致法律风险、商业损失、用户流失 |
| "可解释性影响性能" | 可解释性与性能可兼得，且有助于调试 |
| "伦理审查拖慢进度" | 早期考虑伦理问题可避免后期返工 |

---

## 练习题

### 思考题

1. **对抗攻防权衡**：为什么提高鲁棒性可能降低正常样本准确率？

2. **公平性冲突**：人口统计学均等和机会均等能同时满足吗？为什么？

3. **可解释性边界**：在什么场景下可解释性比准确率更重要？

### 编程题

4. 实现 PGD 攻击并比较不同ε值下的模型鲁棒性曲线。

5. 使用 SHAP 分析一个真实数据集（如成人收入预测）中的公平性问题。

---

## 延伸阅读

- 📘 《AI Ethics》- Virginia Dignum
- 📘 《Weapons of Math Destruction》- Cathy O'Neil
- 🌐 [Google PAIR - Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
- 🌐 [Fairness Indicators - TensorFlow](https://www.tensorflow.org/tfx/guide/fairness_indicators)
- 🌐 [Responsible AI Toolkit - Microsoft](https://github.com/microsoft/responsible-ai-toolbox)

---

[← 上一节：1.3 深度学习基础](1-3-deep-learning.md) | [第 2 章：大模型核心技术 →](../chapter-2/README.md)
