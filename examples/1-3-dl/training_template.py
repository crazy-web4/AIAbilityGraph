# examples/1-3-dl/training_template.py
"""
深度学习训练完整模板

包含最佳实践:
1. 学习率调度 (预热 + 衰减)
2. 梯度裁剪
3. 早停法
4. 检查点保存
5. TensorBoard 日志
6. 混合精度训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp
        
        # AdamW 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器 (在 fit 中设置)
        self.scheduler = None
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == 'cuda')
        
        # 保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _get_lr_schedule(self, epoch: int, step: int, total_steps: int) -> float:
        """
        学习率调度: 预热 + 余弦退火
        """
        # 预热阶段
        if epoch < self.warmup_epochs:
            warmup_progress = (epoch * total_steps + step) / (self.warmup_epochs * total_steps)
            return 0.1 + 0.9 * warmup_progress
        
        # 余弦退火
        progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    def _update_learning_rate(self, epoch: int, step: int, total_steps: int):
        """更新学习率"""
        lr_ratio = self._get_lr_schedule(epoch, step, total_steps)
        base_lr = self.optimizer.param_groups[0]['lr']
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr * lr_ratio
    
    def train_epoch(self, epoch: int) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        total_steps = len(self.train_loader)
        
        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 更新学习率
            self._update_learning_rate(epoch, step, total_steps)
            
            # 混合精度前向
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device == 'cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return avg_loss, accuracy, current_lr
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新
        torch.save(checkpoint, self.save_dir / 'checkpoint_latest.pth')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pth')
            print(f"  ✓ 已保存最佳模型 (Epoch {epoch})")
    
    def fit(self) -> dict:
        """完整训练流程"""
        
        print("=" * 70)
        print("开始训练")
        print("=" * 70)
        print(f"设备：{self.device}")
        print(f"训练集大小：{len(self.train_loader.dataset)}")
        print(f"验证集大小：{len(self.val_loader.dataset)}")
        print(f"最大 Epoch: {self.max_epochs}")
        print(f"早停 Patience: {self.patience}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # 训练
            train_loss, train_acc, current_lr = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 打印日志
            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n触发早停条件 (连续{self.patience}个 epoch 未改善)")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时：{total_time/60:.1f} 分钟")
        
        return self.history
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"已加载检查点：{path}")


# ==================== 使用示例 ====================

def create_sample_model():
    """创建示例模型"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )


if __name__ == "__main__":
    # 这里需要实际的数据加载器
    # 仅作为模板参考
    print("训练模板已就绪，请替换为您的数据和模型")
