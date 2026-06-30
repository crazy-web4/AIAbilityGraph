#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型序列化与部署

功能:
- 模型检查点管理
-  TorchScript 导出
- ONNX 格式转换
- 部署模型打包
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional


# ========== 模型元数据 ==========

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


# ========== 模型检查点管理器 ==========

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

        # 同时保存元数据
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

    def get_latest_checkpoint(self) -> Optional[str]:
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


# ========== 示例模型 ==========

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


# ========== TorchScript 部署 ==========

class DeployableModel(nn.Module):
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

    def predict(self, x: torch.Tensor) -> tuple:
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


# ========== ONNX 导出 ==========

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
        input_shape: 输入形状 (batch, ...)
        output_path: 输出路径
        opset_version: ONNX 算子版本
    """
    try:
        import onnx
    except ImportError:
        print("请安装 onnx: pip install onnx")
        return

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


def run_onnx_inference(onnx_path: str, input_data) -> any:
    """
    使用 ONNX Runtime 推理
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("请安装 onnxruntime: pip install onnxruntime")
        return None

    # 创建推理会话
    session = ort.InferenceSession(onnx_path)

    # 获取输入名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 推理
    result = session.run([output_name], {input_name: input_data})

    return result[0]


# ========== 演示 ==========

def demo_checkpoint():
    """检查点使用演示"""
    print("=" * 60)
    print("模型检查点演示")
    print("=" * 60)

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


def demo_torchscript():
    """TorchScript 导出演示"""
    print("\n" + "=" * 60)
    print("TorchScript 导出演示")
    print("=" * 60)

    # 创建模型
    model = DeployableModel(input_dim=784, output_dim=10)
    model.eval()

    # 导出
    dummy_input = torch.randn(1, 784)
    export_to_torchscript(model, dummy_input, "./deploy/model_torchscript.pt")

    # 加载推理
    deployed_model = load_torchscript("./deploy/model_torchscript.pt")

    # 推理
    test_input = torch.randn(4, 784)
    with torch.no_grad():
        predictions, probs = deployed_model.predict(test_input)

    print(f"预测结果：{predictions}")
    print(f"概率分布：{probs[0]}")


# ========== 主程序 ==========

if __name__ == "__main__":
    import os

    # 创建目录
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./deploy", exist_ok=True)

    # 运行演示
    demo_checkpoint()
    demo_torchscript()

    print("\n" + "=" * 60)
    print("模型序列化演示完成")
    print("=" * 60)
    print("""
【关键要点】
1. 检查点应包含: 模型权重 + 优化器状态 + 元数据
2. 部署前切换到 eval 模式
3. TorchScript 适合静态图优化
4. ONNX 适合跨平台部署
5. 始终验证导出后的模型输出
    """)
