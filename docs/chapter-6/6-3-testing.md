# 6.3 测试与验证

> 确保 AI 模型和系统的质量与可靠性。

## 学习目标

- [ ] 掌握 AI 系统测试方法
- [ ] 理解模型验证技术
- [ ] 学会设计测试用例
- [ ] 实施 CI/CD 测试流程

---

## 6.3.1 测试层次

### AI 系统测试金字塔

```
                    /\
                   /  \
                  / E2E \
                 /--------\
                /          \
               / Integration \
              /----------------\
             /                  \
            /      Unit Tests     \
           /------------------------\
```

### 测试类型对比

| 层次 | 测试内容 | 工具 | 执行频率 |
|------|----------|------|----------|
| **单元测试** | 单个函数/类 | pytest | 每次提交 |
| **集成测试** | 模块间交互 | pytest + mock | 每天 |
| **E2E 测试** | 完整流程 | Selenium, Playwright | 每周 |
| **模型测试** | 模型质量 | 自定义评估 | 每次训练后 |

---

## 6.3.2 单元测试实践

```python
# examples/6-3-testing/test_model.py
"""
模型单元测试示例
"""

import pytest
import torch
import torch.nn as nn
from typing import List


# 被测模型
class TextClassifier(nn.Module):
    """文本分类器"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        return self.classifier(pooled)
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return torch.argmax(logits, dim=-1)


# ========== 单元测试 ==========

class TestTextClassifier:
    """分类器测试类"""
    
    @pytest.fixture
    def model(self):
        """测试夹具：创建模型"""
        return TextClassifier(vocab_size=1000, embed_dim=128, num_classes=5)
    
    def test_model_creation(self, model):
        """测试模型创建"""
        assert model is not None
        assert isinstance(model, TextClassifier)
    
    def test_forward_pass_shape(self, model):
        """测试前向传播输出形状"""
        batch_size = 8
        seq_len = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(input_ids)
        
        assert output.shape == (batch_size, 5)  # num_classes=5
    
    def test_prediction_dtype(self, model):
        """测试预测数据类型"""
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids)
        
        assert output.dtype == torch.float32
    
    def test_embedding_padding(self, model):
        """测试填充 token 处理"""
        input_ids = torch.tensor([[0, 0, 0], [1, 2, 3]])
        output = model(input_ids)
        
        # 填充位置应该产生相似的嵌入
        assert output.shape == (2, 5)
    
    def test_predict_method(self, model):
        """测试预测方法"""
        input_ids = torch.randint(0, 1000, (4, 20))
        predictions = model.predict(input_ids)
        
        assert predictions.shape == (4,)
        assert predictions.min() >= 0
        assert predictions.max() < 5
    
    def test_parameter_count(self, model):
        """测试参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算预期参数
        expected_embedding = 1000 * 128  # vocab_size * embed_dim
        expected_classifier = 128 * 5 + 5  # embed_dim * num_classes + bias
        expected_total = expected_embedding + expected_classifier
        
        assert total_params == expected_total


# ========== 属性测试 ==========

class TestModelProperties:
    """模型属性测试"""
    
    @pytest.fixture
    def model(self):
        return TextClassifier(1000, 128, 5)
    
    def test_deterministic_output(self, model):
        """测试确定性输出"""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # 相同输入应该产生相同输出
        output1 = model(input_ids)
        output2 = model(input_ids)
        
        assert torch.allclose(output1, output2)
    
    def test_batch_invariance(self, model):
        """测试批次不变性"""
        input_single = torch.randint(0, 1000, (1, 10))
        input_batch = input_single.repeat(4, 1)
        
        output_single = model(input_single)
        output_batch = model(input_batch)
        
        # 每批次的输出应该相同
        for i in range(4):
            assert torch.allclose(output_single, output_batch[i])
    
    def test_input_size_invariance(self, model):
        """测试不同输入长度"""
        for seq_len in [5, 10, 50, 100]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            output = model(input_ids)
            assert output.shape == (2, 5)


# ========== 边界测试 ==========

class TestEdgeCases:
    """边界情况测试"""
    
    @pytest.fixture
    def model(self):
        return TextClassifier(1000, 128, 5)
    
    def test_minimum_input(self, model):
        """测试最小输入"""
        input_ids = torch.randint(0, 1000, (1, 1))
        output = model(input_ids)
        assert output.shape == (1, 5)
    
    def test_large_batch(self, model):
        """测试大批次"""
        input_ids = torch.randint(0, 1000, (256, 10))
        output = model(input_ids)
        assert output.shape == (256, 5)
    
    def test_all_padding_tokens(self, model):
        """测试全填充输入"""
        input_ids = torch.zeros((2, 20), dtype=torch.long)
        output = model(input_ids)
        assert output.shape == (2, 5)
    
    def test_vocabulary_boundaries(self, model):
        """测试词汇表边界"""
        # 最小 token ID
        input_min = torch.zeros((1, 5), dtype=torch.long)
        output_min = model(input_min)
        assert output_min.shape == (1, 5)
        
        # 最大 token ID
        input_max = torch.full((1, 5), 999, dtype=torch.long)
        output_max = model(input_max)
        assert output_max.shape == (1, 5)
```

---

## 6.3.3 模型质量测试

```python
# examples/6-3-testing/test_model_quality.py
"""
模型质量测试
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple


class ModelQualityTests:
    """
    模型质量测试套件
    
    测试内容:
    - 准确率
    - 稳定性
    - 公平性
    - 鲁棒性
    """
    
    def __init__(self, model, tokenizer, test_data: List[Tuple]):
        """
        参数:
            model: 被测试模型
            tokenizer: 分词器
            test_data: [(输入，期望输出), ...]
        """
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data
    
    def test_accuracy_threshold(self, min_accuracy: float = 0.8):
        """
        测试准确率阈值
        """
        correct = 0
        
        for input_text, expected in self.test_data:
            # 推理
            inputs = self.tokenizer(input_text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted = torch.argmax(outputs.logits, dim=-1).item()
            
            if predicted == expected:
                correct += 1
        
        accuracy = correct / len(self.test_data)
        
        assert accuracy >= min_accuracy, (
            f"准确率 {accuracy:.2%} 低于阈值 {min_accuracy:.2%}"
        )
        
        return accuracy
    
    def test_prediction_stability(self, n_runs: int = 5):
        """
        测试预测稳定性
        
        同一输入多次运行应该产生一致结果
        """
        if not self.test_data:
            return
        
        input_text = self.test_data[0][0]
        inputs = self.tokenizer(input_text, return_tensors='pt')
        
        predictions = []
        for _ in range(n_runs):
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
        
        # 所有预测应该相同
        assert len(set(predictions)) == 1, (
            f"预测不稳定：{predictions}"
        )
    
    def test_robustness_to_noise(self, noise_level: float = 0.1):
        """
        测试抗噪性
        
        添加噪声后预测不应大幅变化
        """
        if not self.test_data:
            return
        
        correct_original = 0
        correct_noisy = 0
        
        for input_text, expected in self.test_data:
            # 原始输入
            inputs = self.tokenizer(input_text, return_tensors='pt')
            
            # 添加噪声（模拟输入扰动）
            input_ids = inputs['input_ids']
            noise_mask = torch.rand_like(input_ids.float()) < noise_level
            
            # 随机替换一些 token
            noisy_ids = input_ids.clone()
            n_noise = noise_mask.sum().item()
            if n_noise > 0:
                noisy_ids[noise_mask] = torch.randint(
                    0, self.tokenizer.vocab_size, (n_noise,)
                )
            
            # 预测
            with torch.no_grad():
                orig_pred = torch.argmax(self.model(**inputs).logits, dim=-1)
                noisy_pred = torch.argmax(
                    self.model(input_ids=noisy_ids, attention_mask=inputs['attention_mask']).logits,
                    dim=-1
                )
            
            if orig_pred == expected:
                correct_original += 1
            if noisy_pred == expected:
                correct_noisy += 1
        
        original_acc = correct_original / len(self.test_data)
        noisy_acc = correct_noisy / len(self.test_data)
        
        # 噪声下准确率下降不应超过 20%
        acc_drop = original_acc - noisy_acc
        assert acc_drop <= 0.2, (
            f"噪声导致准确率下降 {acc_drop:.2%} 超过阈值"
        )


# ========== 偏见测试 ==========

class BiasTests:
    """
    偏见测试套件
    
    检测模型在不同群体上的表现差异
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def test_demographic_parity(
        self,
        test_cases: List[dict],
        max_disparity: float = 0.1
    ):
        """
        测试人口统计学均等
        
        不同群体的正例率应该相近
        """
        group_positive_rates = {}
        
        for case in test_cases:
            group = case['group']
            input_text = case['input']
            
            inputs = self.tokenizer(input_text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
            
            if group not in group_positive_rates:
                group_positive_rates[group] = []
            group_positive_rates[group].append(pred)
        
        # 计算各组正例率
        positive_rates = {}
        for group, predictions in group_positive_rates.items():
            positive_rates[group] = sum(predictions) / len(predictions)
        
        # 检查差异
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        disparity = max_rate - min_rate
        
        assert disparity <= max_disparity, (
            f"群体间差异 {disparity:.2%} 超过阈值 {max_disparity:.2%}\n"
            f"各组正例率：{positive_rates}"
        )
        
        return positive_rates


# ========== 性能测试 ==========

class PerformanceTests:
    """性能测试"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def test_inference_latency(
        self,
        input_text: str,
        max_latency_ms: float = 100
    ):
        """测试推理延迟"""
        import time
        
        inputs = self.tokenizer(input_text, return_tensors='pt')
        
        # 预热
        with torch.no_grad():
            self.model(**inputs)
        
        # 测量
        n_runs = 10
        latencies = []
        
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                self.model(**inputs)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        assert avg_latency <= max_latency_ms, (
            f"平均延迟 {avg_latency:.1f}ms 超过阈值"
        )
        
        return {
            'avg_ms': avg_latency,
            'p99_ms': p99_latency,
            'all_ms': latencies
        }
    
    def test_memory_usage(self, batch_sizes: List[int] = [1, 4, 16, 64]):
        """测试内存使用"""
        import tracemalloc
        
        results = []
        
        for batch_size in batch_sizes:
            # 开始追踪
            tracemalloc.start()
            
            inputs = self.tokenizer(
                ["test"] * batch_size,
                return_tensors='pt',
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results.append({
                'batch_size': batch_size,
                'peak_memory_mb': peak / 1024 / 1024
            })
        
        return results


# 使用示例
if __name__ == "__main__":
    # 创建测试套件
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 测试数据
    test_data = [
        ("I love this movie!", 1),
        ("This is terrible.", 0),
        # ... 更多测试用例
    ]
    
    # 运行测试
    quality_tests = ModelQualityTests(model, tokenizer, test_data)
    accuracy = quality_tests.test_accuracy_threshold(0.7)
    print(f"准确率：{accuracy:.2%}")
    
    # 性能测试
    perf_tests = PerformanceTests(model, tokenizer)
    latency = perf_tests.test_inference_latency("This is a test.", max_latency_ms=50)
    print(f"延迟：{latency}")
```

---

## 6.3.4 CI/CD 集成

```yaml
# examples/6-3-testing/.github/workflows/ci.yml
name: AI Model CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/unit --cov=src --cov-report=xml
    
    - name: Run model tests
      run: |
        pytest tests/model --cov-append
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  model-evaluation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
    
    - name: Evaluate model
      run: |
        python scripts/evaluate_model.py \
          --model ${MODEL_PATH} \
          --test-data ./data/test.json \
          --output evaluation_report.json
    
    - name: Check quality gates
      run: |
        python scripts/check_quality_gates.py \
          --report evaluation_report.json \
          --min-accuracy 0.8 \
          --max-latency 100

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run safety scan
      run: |
        pip install safety
        safety check -r requirements.txt
```

---

## 练习题

1. **单元测试**: 为你实现的某个算法函数编写完整的单元测试。

2. **质量测试**: 设计一个模型偏见测试方案。

3. **CI/CD**: 为 AI 项目设计完整的 GitHub Actions 工作流。

---

[← 上一节：6.2 算法工程化](6-2-engineering.md) | [下一节：6.4 AI 部署技术 →](6-4-deployment.md)
