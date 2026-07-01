# 6.2 算法工程化 - 关键知识点详解

> 本节为 6.2 节的补充知识点，包含代码规范、重构技巧、性能优化指南。

---

## 知识点 1: AI 项目代码规范

### 目录结构规范

```
project_name/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖列表
├── pyproject.toml           # 项目配置 (推荐)
├── setup.py                 # 安装脚本 (可选)
│
├── configs/                 # 配置文件
│   ├── default.yaml        # 默认配置
│   ├── training.yaml       # 训练配置
│   └── inference.yaml      # 推理配置
│
├── src/                     # 源代码
│   ├── __init__.py
│   ├── data/               # 数据相关
│   │   ├── __init__.py
│   │   ├── dataset.py      # 数据集类
│   │   ├── dataloader.py   # 数据加载
│   │   └── transforms.py   # 数据变换
│   │
│   ├── models/             # 模型定义
│   │   ├── __init__.py
│   │   ├── base.py         # 基类
│   │   ├── transformer.py  # 具体模型
│   │   └── unet.py
│   │
│   ├── training/           # 训练逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py      # 训练器
│   │   ├── losses.py       # 损失函数
│   │   └── metrics.py      # 评估指标
│   │
│   ├── inference/          # 推理逻辑
│   │   ├── __init__.py
│   │   ├── predictor.py    # 预测器
│   │   └── pipelines.py    # 推理流水线
│   │
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── logging.py      # 日志配置
│       └── misc.py         # 杂项工具
│
├── scripts/                 # 可执行脚本
│   ├── train.py            # 训练入口
│   ├── evaluate.py         # 评估入口
│   └── inference.py        # 推理入口
│
├── tests/                   # 测试代码
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── notebooks/               # Jupyter Notebook
│   ├── 01_eda.ipynb
│   └── 02_baseline.ipynb
│
├── data/                    # 数据目录 (通常.gitignore)
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── splits/             # 数据划分
│
├── models/                  # 模型检查点 (通常.gitignore)
│   ├── checkpoints/
│   └── exported/
│
└── logs/                    # 日志目录 (通常.gitignore)
```

### 代码模板

```python
# src/models/base.py
"""
模型基类模块

约定:
1. 所有模型继承自 BaseModule
2. 统一的 forward 接口
3. 内置常见工具方法
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseModule(nn.Module, ABC):
    """
    模型基类
    
    提供:
    - 统一的初始化
    - 设备管理
    - 保存/加载
    - 推理模式切换
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._device = None
    
    @property
    def device(self):
        """获取当前设备"""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def to(self, *args, **kwargs):
        """重写 to 方法，跟踪设备"""
        self._device = None  # 重置，下次访问时重新获取
        return super().to(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """前向传播 (必须实现)"""
        pass
    
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Any:
        """
        推理模式 (自动 no_grad)
        
        子类可重写此方法添加后处理
        """
        self.eval()
        return self.forward(*args, **kwargs)
    
    def save_checkpoint(self, path: str, extra: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        if extra:
            checkpoint.update(extra)
        
        torch.save(checkpoint, path)
        print(f"检查点已保存至：{path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        config: Optional[Dict] = None,
        device: str = 'cpu'
    ) -> 'BaseModule':
        """加载检查点"""
        checkpoint = torch.load(path, map_location=device)
        
        # 使用保存的配置或传入的配置
        _config = config or checkpoint['config']
        
        # 创建模型
        model = cls(_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


# ==================== 使用示例 ====================

class TransformerClassifier(BaseModule):
    """Transformer 分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embed_dim']
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['dim_ff'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )
        
        self.classifier = nn.Linear(
            config['embed_dim'],
            config['num_classes']
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = self.encoder(x)            # (batch, seq_len, embed_dim)
        
        # 全局池化 (取 [CLS] 或 mean)
        pooled = x.mean(dim=1)         # (batch, embed_dim)
        
        logits = self.classifier(pooled)
        return logits
    
    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(input_ids)
        return torch.softmax(logits, dim=-1)
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        proba = self.predict_proba(input_ids)
        return torch.argmax(proba, dim=-1)
```

---

## 知识点 2: 性能优化技巧

### PyTorch 性能优化清单

```python
# examples/6-2-engineering/performance_tips.py
"""
PyTorch 性能优化技巧汇总
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time


# ========== 1. 数据加载优化 ==========

class OptimizedDataset(Dataset):
    """
    优化数据集
    
    技巧:
    1. 预加载到内存 (如果可能)
    2. 使用共享内存
    3. 避免在 __getitem__ 中做重型操作
    """
    
    def __init__(self, data_paths: list, preload: bool = True):
        self.data_paths = data_paths
        
        if preload:
            # 预加载数据
            self.data = [self._load(p) for p in data_paths]
        else:
            self.data = None
    
    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        return self._load(self.data_paths[idx])
    
    def __len__(self):
        return len(self.data_paths)


def create_dataloader(dataset, batch_size=32, num_workers=4):
    """
    创建优化的 DataLoader
    
    关键参数:
    - num_workers: 多进程加载 (推荐 4-8)
    - pin_memory: 固定内存加速 CPU->GPU 传输
    - persistent_workers: 复用 worker 进程
    - prefetch_factor: 预取批次数量
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,          # 加速 GPU 传输
        persistent_workers=True,   # PyTorch 1.7+
        prefetch_factor=2,         # 每个 worker 预取 2 批
        shuffle=True,
        drop_last=True
    )


# ========== 2. 混合精度训练 ==========

class MixedPrecisionTrainer:
    """
    混合精度训练示例
    
    使用 torch.cuda.amp 实现自动混合精度 (AMP)
    """
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # 梯度缩放器 (防止下溢)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch):
        """单个训练步骤"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # 自动混合精度
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # 缩放梯度后反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()


# ========== 3. 梯度累积 ==========

class GradientAccumulationTrainer:
    """
    梯度累积
    
    适用场景:
    - 显存不足，无法使用大批次
    - 需要大的有效批次大小稳定训练
    
    原理:
    累积多个小批次的梯度后再更新
    """
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
    
    def train_step(self, batch, step_idx):
        """
        带梯度累积的训练步骤
        
        有效批次大小 = batch_size × accumulation_steps
        """
        # 批次索引用于判断是否更新
        should_update = (step_idx + 1) % self.accumulation_steps == 0
        
        with torch.set_grad_enabled(True):
            outputs = self.model(batch['input'])
            loss = nn.CrossEntropyLoss()(outputs, batch['target'])
            
            # 缩放损失以保持一致性
            loss = loss / self.accumulation_steps
            
            loss.backward()
            
            if should_update:
                # 梯度裁剪 (重要！)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps


# ========== 4. 模型编译加速 (PyTorch 2.0+) ==========

class CompiledModel(nn.Module):
    """
    使用 torch.compile 编译模型
    
    PyTorch 2.0+ 特性
    加速比：1.5-3x (取决于模型)
    """
    
    def __init__(self, model):
        super().__init__()
        # 编译模型
        self.model = torch.compile(model)
    
    def forward(self, x):
        return self.model(x)


# ========== 5. 零拷贝 Tensor 传输 ==========

def zero_copy_example():
    """
    零拷贝示例
    
    避免不必要的 Tensor 设备传输
    """
    
    # ❌ 慢：每次创建新 Tensor
    def slow_version(data_list):
        results = []
        for data in data_list:
            tensor = torch.tensor(data).cuda()  # 每次都转移
            results.append(tensor)
        return results
    
    # ✅ 快：批量转移
    def fast_version(data_list):
        # CPU 栈叠后一次性转移
        tensors = [torch.tensor(d) for d in data_list]
        stacked = torch.stack(tensors).cuda()
        return stacked


# ========== 6. 内存优化 ==========

class MemoryEfficientModel(nn.Module):
    """
    内存优化模型
    
    技巧:
    1. 梯度检查点
    2. 激活重计算
    3. 及时释放中间结果
    """
    
    def __init__(self, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # 深层网络
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(24)
        ])
    
    def forward(self, x):
        if self.use_checkpointing:
            # 使用梯度检查点节省内存
            return self._forward_checkpoint(x)
        else:
            return self._forward_normal(x)
    
    def _forward_normal(self, x):
        """标准前向 (占用更多内存)"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _forward_checkpoint(self, x):
        """
        使用梯度检查点
        
        牺牲计算时间换取内存节省
        适合显存受限场景
        """
        from torch.utils.checkpoint import checkpoint
        
        def custom_forward(*inputs):
            for layer in self.layers[:4]:  # 每 4 层检查点一次
                x = layer(inputs[0])
            return x
        
        # 每 4 层做一次检查点
        for i in range(0, len(self.layers), 4):
            x = checkpoint(
                lambda *args: self._run_layers(args[0], i, min(i+4, len(self.layers))),
                x.requires_grad_()
            )
        
        return x
    
    def _run_layers(self, x, start, end):
        for layer in self.layers[start:end]:
            x = layer(x)
        return x


# ========== 性能对比 ==========

def benchmark():
    """性能对比示例"""
    
    # 数据
    batch_size = 64
    seq_len = 512
    data = torch.randn(batch_size, seq_len, 768).cuda()
    
    # 模型
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(768, 8, 2048, batch_first=True),
        num_layers=12
    ).cuda()
    
    # 基准
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(data)
    torch.cuda.synchronize()
    baseline_time = time.time() - start
    
    # 编译后
    compiled_model = torch.compile(model)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = compiled_model(data)
    torch.cuda.synchronize()
    compiled_time = time.time() - start
    
    print(f"基准时间：{baseline_time:.3f}s")
    print(f"编译后时间：{compiled_time:.3f}s")
    print(f"加速比：{baseline_time/compiled_time:.2f}x")


if __name__ == "__main__":
    benchmark()
```

---

## 知识点 3: 代码重构检查清单

### 重构 Checklist

```
□ 代码组织
  □ 是否有重复代码可以提取为函数？
  □ 函数是否职责单一（<50 行）？
  □ 类是否有单一职责？
  □ 是否有魔法数字需要提取为常量？

□ 命名规范
  □ 变量名是否清晰描述用途？
  □ 函数名是否是动词开头？
  □ 类名是否是大驼峰？
  □ 常量名是否全大写？

□ 类型注解
  □ 函数参数是否有类型注解？
  □ 返回值是否有类型注解？
  □ 复杂类型是否使用 TypeAlias？

□ 错误处理
  □ 是否有适当的 try-except？
  □ 异常信息是否清晰有用？
  □ 是否有日志记录关键操作？

□ 测试覆盖
  □ 核心逻辑是否有单元测试？
  □ 边界情况是否有测试？
  □ 是否有集成测试？
```

### 重构前后对比示例

```python
# ==================== 重构前 ====================

def train_model(data, model, lr=0.001, epochs=10, batch_size=32):
    # 问题：所有参数都在函数签名中，难以管理
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = batch[:, :-1]
            targets = batch[:, -1]
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}, Acc: {100*correct/total:.2f}%")


# ==================== 重构后 ====================

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置 (集中管理超参数)"""
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5


class Trainer:
    """
    训练器类
    
    职责:
    - 管理训练循环
    - 处理检查点
    - 记录指标
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.metrics = {'train_loss': [], 'train_acc': [], 'val_loss': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """单个 epoch 训练"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向 + 反向
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累积指标
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        return {
            'loss': total_loss / len(dataloader.dataset),
            'accuracy': 100 * correct / total
        }
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
            
            # 记录
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['train_acc'].append(train_metrics['accuracy'])
            if val_metrics:
                self.metrics['val_loss'].append(val_metrics['loss'])
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics.get('loss', 'N/A')}"
            )
            
            # 早停
            if val_loader:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("触发早停条件")
                        break
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.model(inputs.to(self.device))
                loss = self.loss_fn(outputs, targets.to(self.device))
                total_loss += loss.item() * inputs.size(0)
        
        return {'loss': total_loss / len(dataloader.dataset)}
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }, path)
        logger.info(f"检查点已保存：{path}")


# ==================== 使用方式 ====================

def main():
    # 配置
    config = TrainingConfig(
        learning_rate=1e-4,
        epochs=50,
        batch_size=64,
        early_stopping_patience=10
    )
    
    # 创建组件
    model = MyModel()
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    # 训练
    trainer = Trainer(model, config)
    trainer.fit(dataloader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## 练习题

### 练习 1: 代码重构

重构以下代码：

```python
# 问题代码
def process(d1, d2, d3, m='cuda'):
    t = torch.tensor(d1).to(m)
    t2 = torch.tensor(d2).to(m)
    t3 = torch.tensor(d3).to(m)
    r1 = t + t2
    r2 = r1 * t3
    r3 = torch.sum(r2)
    print(r3)
    return r3
```

要求:
1. 改进参数命名
2. 添加类型注解
3. 提取为类
4. 添加日志和错误处理

### 练习 2: 性能分析

使用 `cProfile` 或 `torch.profiler` 分析以下代码的性能瓶颈:

```python
def train_loop(model, data, epochs=10):
    for epoch in range(epochs):
        for batch in data:
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

---

## 延伸阅读

- [PyTorch 性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Python 代码规范](https://peps.python.org/pep-0008/)
- [重构入门](https://martinfowler.com/books/refactoring.html)

---

[← 返回 6.2 主文档](6-2-engineering.md) | [下一节：测试与验证 →](6-3-testing.md)
