# 第 1.3 节 深度学习代码

深度学习基础代码示例。

## 文件说明

| 文件 | 说明 |
|------|------|
| `mlp_pytorch.py` | PyTorch MLP 实现与 MNIST 训练 |
| `cnn_architectures.py` | CNN 架构实现（LeNet/VGG/ResNet） |
| `activation_functions.py` | 激活函数对比、梯度流动测试与 Dead ReLU 演示 |
| `optimizer_comparison.py` | SGD/SGD+momentum/Adam 等优化器对比实验 |
| `training_template.py` | 完整训练流程模板（含验证、调度、检查点） |
| `transformer_from_scratch.py` | 从零实现 Transformer（Encoder-Decoder）+ 序列反转任务训练 |

## 运行方式

```bash
pip install torch torchvision matplotlib

python mlp_pytorch.py
python cnn_architectures.py
python transformer_from_scratch.py
```

`transformer_from_scratch.py` 配合文档 `docs/chapter-1/1-3-deep-learning.md`
的 1.3.4 节使用：涵盖多头注意力（Self/Cross-Attention）、正弦位置编码、
Padding/Causal Mask、Pre-Norm、权重绑定与贪心解码，CPU 上约 1-2 分钟收敛。
