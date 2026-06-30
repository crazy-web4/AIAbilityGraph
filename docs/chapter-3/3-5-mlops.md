# 3.5 MLOps 与模型持续交付

> MLOps 将 DevOps 实践应用于机器学习，实现模型的持续集成、交付和监控。

## 学习目标

学完本节后，你将能够：

- [ ] 构建 ML 流水线与 CI/CD 管道
- [ ] 使用 MLflow 管理实验和模型
- [ ] 实施模型监控与告警
- [ ] 设计模型注册表与版本管理

---

## 3.5.1 MLflow 实验管理

### 实验追踪

```python
# examples/3-5-mlops/mlflow_tracking.py
"""
使用 MLflow 追踪实验
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def setup_experiment(experiment_name="llm-training"):
    """设置实验"""
    
    # 设置追踪服务器
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    return mlflow


def log_training_run(config, metrics, params, artifacts=None):
    """记录训练运行"""
    
    with mlflow.start_run():
        # 记录参数
        mlflow.log_params({
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model_type": config.model_type,
        })
        
        # 记录指标
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # 记录模型
        mlflow.pytorch.log_model(model, "model")
        
        # 记录 artifact
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)
        
        # 记录标签
        mlflow.set_tag("team", "nlp")
        mlflow.set_tag("project", "llm-pretrain")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        
        return run_id


class ModelRegistry:
    """模型注册表管理"""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.client = MlflowClient(tracking_uri=tracking_uri)
    
    def register_model(self, model_name, run_id, stage="Staging"):
        """注册模型到注册表"""
        
        # 从 run 创建模型版本
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(model_uri, model_name)
        
        # 转换阶段
        if stage:
            self.client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )
        
        return result
    
    def get_latest_version(self, model_name, stage="Production"):
        """获取最新生产版本"""
        versions = self.client.search_model_versions(
            f"name='{model_name}' AND stage='{stage}'"
        )
        if versions:
            return max(versions, key=lambda v: int(v.version))
        return None
    
    def compare_versions(self, model_name, versions=None):
        """比较不同版本的性能"""
        if versions is None:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        
        results = []
        for v in versions:
            run = self.client.get_run(v.run_id)
            results.append({
                "version": v.version,
                "stage": v.current_stage,
                "metrics": run.data.metrics,
                "params": run.data.params,
            })
        
        return results
```

---

## 3.5.2 模型版本管理

```python
# examples/3-5-mlops/model_versioning.py
"""
模型版本管理最佳实践
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """模型版本信息"""
    name: str
    version: str
    created_at: str
    model_hash: str
    metrics: dict
    config: dict
    training_data_version: str
    code_version: str
    notes: str = ""


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, registry_path="./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.manifest_file = self.registry_path / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """加载清单文件"""
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_manifest(self):
        """保存清单"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
    
    def compute_model_hash(self, model_path: str) -> str:
        """计算模型文件哈希"""
        import hashlib
        
        hasher = hashlib.sha256()
        with open(model_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()[:16]
    
    def register_version(self, 
                         model_name: str,
                         model_path: str,
                         metrics: dict,
                         config: dict,
                         training_data_hash: str,
                         code_version: str,
                         notes: str = "") -> ModelVersion:
        """
        注册新版本
        
        返回:
            ModelVersion 对象
        """
        # 生成版本号
        if model_name not in self.manifest["models"]:
            version = "1.0.0"
            self.manifest["models"][model_name] = []
        else:
            # 自动递增 patch 版本
            last = self.manifest["models"][model_name][-1]
            major, minor, patch = map(int, last["version"].split("."))
            version = f"{major}.{minor}.{patch + 1}"
        
        # 计算模型哈希
        model_hash = self.compute_model_hash(model_path)
        
        # 创建版本记录
        version_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_hash": model_hash,
            "metrics": metrics,
            "config": config,
            "training_data_version": training_data_hash,
            "code_version": code_version,
            "notes": notes,
            "model_path": model_path,
        }
        
        self.manifest["models"][model_name].append(version_info)
        self._save_manifest()
        
        # 保存模型副本
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        import shutil
        shutil.copy(model_path, version_dir / "model.pt")
        
        # 保存元数据
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"已注册 {model_name} v{version}")
        
        return ModelVersion(
            name=model_name,
            version=version,
            created_at=version_info["created_at"],
            model_hash=model_hash,
            metrics=metrics,
            config=config,
            training_data_version=training_data_hash,
            code_version=code_version,
            notes=notes
        )
    
    def list_versions(self, model_name: str) -> list:
        """列出模型的所有版本"""
        return self.manifest["models"].get(model_name, [])
    
    def get_version(self, model_name: str, version: str) -> dict:
        """获取特定版本信息"""
        versions = self.list_versions(model_name)
        for v in versions:
            if v["version"] == version:
                return v
        return None
    
    def promote_to_production(self, model_name: str, version: str):
        """将版本提升为生产版本"""
        # 更新 manifest 标记生产版本
        for v in self.manifest["models"].get(model_name, []):
            v["is_production"] = (v["version"] == version)
        
        self._save_manifest()
        print(f"{model_name} v{version} 已提升为生产版本")
```

---

## 3.5.3 CI/CD 管道

```yaml
# examples/3-5-mlops/.github/workflows/ml-pipeline.yml
name: ML Training Pipeline

on:
  push:
    branches: [main]
    paths: ['src/', 'config/', 'data/']
  pull_request:
    branches: [main]
  schedule:
    # 每周日凌晨 2 点重新训练
    - cron: '0 2 * * 0'

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      
      - name: Validate data
        run: python scripts/validate_data.py
      
      - name: Run data tests
        run: pytest tests/test_data.py

  train:
    needs: data-validation
    runs-on: [self-hosted, gpu]
    resources:
      gpu: 1
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup environment
        run: |
          pip install -r requirements.txt
          pip install mlflow
      
      - name: Train model
        run: |
          mlflow run . \
            -P config=config/train.yaml \
            -P experiment_id=${{ github.sha }}
      
      - name: Evaluate model
        run: python scripts/evaluate.py
      
      - name: Register model
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/register_model.py \
            --name llm-model \
            --run-id ${{ env.MLFLOW_RUN_ID }}

  deploy:
    needs: train
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          ./scripts/deploy.sh staging
      
      - name: Run integration tests
        run: pytest tests/integration/
      
      - name: Deploy to production
        run: |
          ./scripts/deploy.sh production
```

---

## 3.5.4 模型监控

```python
# examples/3-5-mlops/monitoring.py
"""
模型监控与告警
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MonitoringMetrics:
    """监控指标"""
    request_count: int = 0
    latency_p50: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    input_drift: float = 0.0
    output_distribution: Dict = None


class ModelMonitor:
    """模型监控器"""
    
    def __init__(self, 
                 model_name: str,
                 baseline_stats: Dict = None,
                 drift_threshold: float = 0.1,
                 latency_threshold_ms: float = 500):
        
        self.model_name = model_name
        self.baseline = baseline_stats or {}
        self.drift_threshold = drift_threshold
        self.latency_threshold = latency_threshold_ms
        
        # 存储最近的请求
        self.requests: List[Dict] = []
        self.max_history = 10000
    
    def log_request(self, 
                    inputs: np.ndarray, 
                    outputs: np.ndarray,
                    latency_ms: float,
                    error: bool = False):
        """记录请求"""
        
        self.requests.append({
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs.tolist() if hasattr(inputs, 'tolist') else inputs,
            "outputs": outputs.tolist() if hasattr(outputs, 'tolist') else outputs,
            "latency_ms": latency_ms,
            "error": error,
        })
        
        # 保持历史大小
        if len(self.requests) > self.max_history:
            self.requests = self.requests[-self.max_history:]
    
    def compute_metrics(self) -> MonitoringMetrics:
        """计算当前指标"""
        
        if not self.requests:
            return MonitoringMetrics()
        
        # 延迟统计
        latencies = [r["latency_ms"] for r in self.requests]
        
        # 错误率
        errors = sum(1 for r in self.requests if r["error"])
        error_rate = errors / len(self.requests)
        
        # 输入漂移检测
        input_drift = self._compute_drift()
        
        return MonitoringMetrics(
            request_count=len(self.requests),
            latency_p50=np.percentile(latencies, 50),
            latency_p99=np.percentile(latencies, 99),
            error_rate=error_rate,
            input_drift=input_drift,
        )
    
    def _compute_drift(self) -> float:
        """计算输入分布漂移"""
        if len(self.requests) < 100 or not self.baseline:
            return 0.0
        
        # 简化实现：比较均值漂移
        recent_inputs = [
            np.mean(r["inputs"]) 
            for r in self.requests[-100:]
        ]
        
        baseline_mean = self.baseline.get("input_mean", 0)
        baseline_std = self.baseline.get("input_std", 1)
        
        drift = abs(np.mean(recent_inputs) - baseline_mean) / (baseline_std + 1e-8)
        
        return drift
    
    def check_alerts(self) -> List[Dict]:
        """检查是否需要告警"""
        alerts = []
        metrics = self.compute_metrics()
        
        # 延迟告警
        if metrics.latency_p99 > self.latency_threshold:
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"P99 延迟：{metrics.latency_p99:.1f}ms > {self.latency_threshold}ms",
            })
        
        # 错误率告警
        if metrics.error_rate > 0.05:
            alerts.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"错误率：{metrics.error_rate*100:.1f}%",
            })
        
        # 数据漂移告警
        if metrics.input_drift > self.drift_threshold:
            alerts.append({
                "type": "data_drift",
                "severity": "warning",
                "message": f"输入漂移：{metrics.input_drift:.2f}",
            })
        
        return alerts
    
    def get_dashboard_data(self) -> dict:
        """获取仪表盘数据"""
        metrics = self.compute_metrics()
        
        return {
            "model_name": self.model_name,
            "last_updated": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "alerts": self.check_alerts(),
            "request_trend": self._get_request_trend(),
        }
    
    def _get_request_trend(self) -> list:
        """获取请求趋势（按小时）"""
        if not self.requests:
            return []
        
        hourly_counts = {}
        for r in self.requests:
            hour = r["timestamp"][:13]  # YYYY-MM-DDTHH
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        return sorted(hourly_counts.items())[-24:]  # 最近 24 小时


# 告警通知
class AlertManager:
    """告警管理器"""
    
    def __init__(self, webhooks: List[str] = None):
        self.webhooks = webhooks or []
    
    def send_alert(self, alert: Dict, model_name: str):
        """发送告警"""
        
        message = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "alert": alert,
        }
        
        # 发送到 Slack/钉钉等
        for webhook in self.webhooks:
            self._send_webhook(webhook, message)
    
    def _send_webhook(self, webhook: str, message: dict):
        """发送 webhook 通知"""
        import requests
        
        try:
            requests.post(webhook, json=message, timeout=10)
        except Exception as e:
            print(f"发送告警失败：{e}")
```

---

## 3.5.5 MLOps 工具对比

| 工具 | 用途 | 核心功能 | 集成 |
|------|------|----------|------|
| **MLflow** | 实验追踪 | 参数、指标、模型注册 | PyTorch、TF、Scikit |
| **Weights & Biases** | 实验可视化 | 实时图表、协作 | 主流框架 |
| **Kubeflow** | 管道编排 | K8s 原生 ML 流水线 | 云原生 |
| **DVC** | 数据版本 | Git for Data | Git 集成 |
| **Feast** | 特征存储 | 特征注册、服务 | Spark、Flink |

---

## 练习题

1. **版本策略**：如何设计模型版本号规范（语义化版本号）？

2. **漂移检测**：除了均值漂移，还有哪些漂移检测方法？

3. **CI/CD 设计**：为对话模型设计完整的 CI/CD 流水线。

---

[← 上一节：3.4 训练数据处理](3-4-data-processing.md) | [下一节：3.6 模型评估 →](3-6-evaluation.md)
