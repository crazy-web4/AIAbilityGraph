# 6.2 算法工程化

> 将算法模型转换为可部署、可维护的生产系统。

## 学习目标

- [ ] 理解算法工程化的核心挑战
- [ ] 掌握模型序列化与版本管理
- [ ] 学会设计可扩展的算法架构
- [ ] 实施性能监控与日志

---

## 6.2.1 模型序列化与保存

### PyTorch 模型保存最佳实践

```python
# examples/6-2-engineering/model_serialization.py
"""
模型序列化最佳实践
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    version: str
    created_at: str
    architecture: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    training_config: Dict[str, Any]


class ModelCheckpoint:
    """
    模型检查点管理器
    
    功能:
    - 保存模型权重
    - 保存优化器状态
    - 保存元数据
    - 支持断点续训
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        metadata: ModelMetadata
    ) -> str:
        """
        保存完整检查点
        
        返回:
            保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'metadata': asdict(metadata)
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"checkpoint_{timestamp}_epoch{epoch}.pt"
        filepath = self.save_dir / filename
        
        # 保存
        torch.save(checkpoint, filepath)
        
        # 同时保存模型元数据
        metadata_path = self.save_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        print(f"已保存检查点：{filepath}")
        
        return str(filepath)
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None
    ) -> Dict:
        """
        加载检查点
        
        返回:
            {epoch, model_state_dict, optimizer_state_dict, metrics}
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics'],
            'metadata': checkpoint.get('metadata')
        }
    
    def get_latest_checkpoint(self) -> str:
        """获取最新的检查点"""
        checkpoints = list(self.save_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    
    def export_for_deployment(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        output_path: str
    ):
        """
        导出部署用模型
        
        只保存模型权重，移除训练相关状态
        """
        # 切换到评估模式
        model.eval()
        
        # 保存 state_dict
        deploy_package = {
            'model_state_dict': model.state_dict(),
            'metadata': asdict(metadata),
            'model_class': model.__class__.__name__
        }
        
        torch.save(deploy_package, output_path)
        print(f"已导出部署模型：{output_path}")
        
        return output_path


# 使用示例
class SimpleClassifier(nn.Module):
    """简单分类器示例"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def demo_checkpoint():
    """检查点使用演示"""
    
    # 创建模型和优化器
    model = SimpleClassifier(input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 元数据
    metadata = ModelMetadata(
        model_name="simple_classifier",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        architecture="MLP",
        input_dim=784,
        output_dim=10,
        hidden_dim=256,
        training_config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    )
    
    # 创建检查点管理器
    checkpoint_mgr = ModelCheckpoint(save_dir="./checkpoints")
    
    # 模拟训练几轮后保存
    metrics = {'train_loss': 0.5, 'val_loss': 0.6, 'accuracy': 0.85}
    
    checkpoint_path = checkpoint_mgr.save(
        model=model,
        optimizer=optimizer,
        epoch=5,
        metrics=metrics,
        metadata=metadata
    )
    
    # 加载检查点
    new_model = SimpleClassifier(784, 256, 10)
    loaded = checkpoint_mgr.load(checkpoint_path, new_model, optimizer)
    
    print(f"从 epoch {loaded['epoch']} 恢复训练")
    print(f"指标：{loaded['metrics']}")
    
    # 导出部署模型
    checkpoint_mgr.export_for_deployment(
        model=model,
        metadata=metadata,
        output_path="./deploy/model.pt"
    )


if __name__ == "__main__":
    demo_checkpoint()
```

---

## 6.2.2 模型部署模式

### 1. TorchScript 部署

```python
# examples/6-2-engineering/torchscript_deploy.py
"""
使用 TorchScript 部署模型
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class deployableModel(nn.Module):
    """
    可部署模型
    
    要求:
    - 使用 TorchScript 兼容的操作
    - 避免动态控制流
    - 明确类型注解
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Traceable 操作
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回预测和概率
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1)
        return predicted, probs


def export_to_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str
) -> torch.jit.ScriptModule:
    """
    导出为 TorchScript
    
    两种方式:
    1. Tracing: 通过示例输入追踪执行路径
    2. Scripting: 直接编译 Python 代码
    """
    model.eval()
    
    # 方式 1: Tracing (适合静态计算图)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 方式 2: Scripting (适合有控制流的模型)
    # scripted_model = torch.jit.script(model)
    
    # 保存
    traced_model.save(output_path)
    print(f"已保存 TorchScript 模型：{output_path}")
    
    return traced_model


def load_torchscript(path: str) -> torch.jit.ScriptModule:
    """加载 TorchScript 模型"""
    return torch.jit.load(path)


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = deployableModel(input_dim=784, output_dim=10)
    
    # 导出
    dummy_input = torch.randn(1, 784)
    export_to_torchscript(model, dummy_input, "./deploy/model.pt")
    
    # 加载推理
    deployed_model = load_torchscript("./deploy/model.pt")
    
    # 推理
    test_input = torch.randn(4, 784)
    with torch.no_grad():
        predictions, probs = deployed_model(test_input)
    
    print(f"预测：{predictions}")
    print(f"概率：{probs}")
```

### 2. ONNX 格式导出

```python
# examples/6-2-engineering/onnx_export.py
"""
导出模型到 ONNX 格式
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 13
):
    """
    导出模型到 ONNX
    
    参数:
        model: PyTorch 模型
        input_shape: 输入形状 (batch, channels, height, width)
        output_path: 输出路径
        opset_version: ONNX 算子版本
    """
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"已导出 ONNX 模型：{output_path}")
    
    # 验证模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过")


def run_onnx_inference(onnx_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    使用 ONNX Runtime 推理
    """
    # 创建推理会话
    session = ort.InferenceSession(onnx_path)
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 推理
    result = session.run([output_name], {input_name: input_data})
    
    return result[0]


# 使用示例
if __name__ == "__main__":
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 导出
    export_to_onnx(
        model,
        input_shape=(1, 784),
        output_path="./deploy/model.onnx"
    )
    
    # 测试推理
    test_input = np.random.randn(4, 784).astype(np.float32)
    result = run_onnx_inference("./deploy/model.onnx", test_input)
    
    print(f"推理结果形状：{result.shape}")
```

---

## 6.2.3 性能监控

```python
# examples/6-2-engineering/model_monitoring.py
"""
模型性能监控
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import json


@dataclass
class InferenceRecord:
    """推理记录"""
    timestamp: float
    latency_ms: float
    input_shape: tuple
    success: bool
    error_message: str = None


class ModelPerformanceMonitor:
    """
    模型性能监控器
    
    监控指标:
    - 延迟 (P50, P95, P99)
    - 吞吐量 (QPS)
    - 错误率
    - 资源使用率
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Dict[str, float] = None
    ):
        self.window_size = window_size
        self.records: deque = deque(maxlen=window_size)
        
        self.alert_thresholds = alert_thresholds or {
            'p99_latency_ms': 500,
            'error_rate': 0.05,
            'min_qps': 10
        }
        
        self.logger = logging.getLogger(__name__)
    
    def record(
        self,
        latency_ms: float,
        input_shape: tuple,
        success: bool,
        error_message: str = None
    ):
        """记录一次推理"""
        record = InferenceRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            input_shape=input_shape,
            success=success,
            error_message=error_message
        )
        self.records.append(record)
    
    def get_metrics(self) -> Dict:
        """获取当前指标"""
        if not self.records:
            return {}
        
        latencies = [r.latency_ms for r in self.records]
        latencies_sorted = sorted(latencies)
        
        # 计算百分位数
        p50_idx = int(len(latencies_sorted) * 0.5)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p99_idx = int(len(latencies_sorted) * 0.99)
        
        # 计算 QPS
        if len(self.records) >= 2:
            time_span = self.records[-1].timestamp - self.records[0].timestamp
            qps = len(self.records) / time_span if time_span > 0 else 0
        else:
            qps = 0
        
        # 错误率
        n_errors = sum(1 for r in self.records if not r.success)
        error_rate = n_errors / len(self.records)
        
        return {
            'count': len(self.records),
            'latency_p50': latencies_sorted[p50_idx],
            'latency_p95': latencies_sorted[p95_idx],
            'latency_p99': latencies_sorted[p99_idx],
            'qps': qps,
            'error_rate': error_rate,
            'avg_latency': sum(latencies) / len(latencies)
        }
    
    def check_alerts(self) -> List[Dict]:
        """检查告警"""
        metrics = self.get_metrics()
        alerts = []
        
        if metrics.get('latency_p99', 0) > self.alert_thresholds['p99_latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"P99 延迟 {metrics['latency_p99']:.1f}ms 超过阈值"
            })
        
        if metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"错误率 {metrics['error_rate']*100:.1f}% 超过阈值"
            })
        
        return alerts
    
    def export_metrics(self, output_path: str):
        """导出指标到文件"""
        metrics = self.get_metrics()
        metrics['timestamp'] = time.time()
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"指标已导出：{output_path}")


# 装饰器：自动记录推理性能
def monitor_inference(monitor: ModelPerformanceMonitor):
    """推理监控装饰器"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                input_shape = args[0].shape if args and hasattr(args[0], 'shape') else ()
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                input_shape = ()
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                monitor.record(
                    latency_ms=latency_ms,
                    input_shape=input_shape,
                    success=success,
                    error_message=error_message
                )
        
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    monitor = ModelPerformanceMonitor(window_size=1000)
    
    @monitor_inference(monitor)
    def dummy_inference(input_tensor):
        time.sleep(0.01)  # 模拟 10ms 推理
        return input_tensor * 2
    
    import torch
    
    # 模拟推理
    for i in range(100):
        x = torch.randn(1, 784)
        _ = dummy_inference(x)
    
    # 获取指标
    metrics = monitor.get_metrics()
    print("性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 检查告警
    alerts = monitor.check_alerts()
    if alerts:
        print("\n告警:")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['message']}")
```

---

## 练习题

1. **模型导出**: 将一个训练好的分类器导出为 TorchScript 和 ONNX 两种格式。

2. **监控告警**: 设计一个完整的监控告警流程，包括指标收集、阈值判断、告警通知。

3. **性能优化**: 分析某个推理服务的性能瓶颈并提出优化方案。

---

[← 上一节：6.1 Python 进阶](6-1-python-advanced.md) | [下一节：6.3 测试与验证 →](6-3-testing.md)
