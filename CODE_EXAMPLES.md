# 代码示例索引

> 本教程配套的所有代码示例，按章节组织。

---

## 第 1 章：AI 基础理论

### 1.1 数学与统计基础

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/1-1-math/matrix_operations.py` | 矩阵运算可视化 | `python matrix_operations.py` |
| `examples/1-1-math/gradient_descent_vis.py` | 梯度下降可视化 | `python gradient_descent_vis.py` |
| `examples/1-1-math/bayes_classifier.py` | 贝叶斯分类器实战 | `python bayes_classifier.py` |

**依赖安装**:
```bash
pip install numpy matplotlib scipy scikit-learn
```

### 1.2 经典机器学习算法

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/1-2-ml/linear_regression.py` | 线性回归（从零实现 + sklearn） | `python linear_regression.py` |
| `examples/1-2-ml/logistic_regression.py` | 逻辑回归与分类实战 | `python logistic_regression.py` |

**依赖安装**:
```bash
pip install numpy scikit-learn matplotlib seaborn
```

### 1.3 深度学习基础

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/1-3-dl/mlp_pytorch.py` | PyTorch MLP 实现与 MNIST 训练 | `python mlp_pytorch.py` |
| `examples/1-3-dl/cnn_architectures.py` | CNN 架构实现（LeNet/VGG/ResNet） | `python cnn_architectures.py` |

**依赖安装**:
```bash
pip install torch torchvision matplotlib
```

---

## 第 2 章：大模型核心技术

### 2.1 大模型架构设计

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/2-1-architecture/transformer_components.py` | Transformer 组件实现 | `python transformer_components.py` |

**依赖安装**:
```bash
pip install torch torchtext
```

### 2.3 参数高效微调

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/2-3-peft/lora_finetuning.py` | LoRA/QLoRA 实战代码 | `python lora_finetuning.py` |

**依赖安装**:
```bash
pip install transformers peft bitsandbytes accelerate
```

---

## 第 3 章：大模型工程化

### 3.2 模型压缩与量化

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/3-2-compression/quantization_demo.py` | 量化技术演示 | `python quantization_demo.py` |

**依赖安装**:
```bash
pip install torch transformers bitsandbytes
```

---

## 第 4 章：大模型应用开发

### 4.3 RAG 应用开发

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/4-3-rag/rag_system.py` | 完整 RAG 系统实现 | `python rag_system.py` |

**依赖安装**:
```bash
pip install chromadb sentence-transformers openai
```

### 4.4 Agent 应用开发

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/4-4-agent/agent_framework.py` | Function Calling 与 ReAct Agent | `python agent_framework.py` |

**依赖安装**:
```bash
pip install openai
```

---

## 番外篇示例

### Special-06：智能体平台（模型网关 / Durable Agent / MCP）

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/special-06/model_router.py` | 多 Provider 路由 + 熔断/重试/fallback + 缓存（离线 Mock 可跑） | `python model_router.py` |
| `examples/special-06/durable_agent.py` | Durable Agent Loop：检查点、崩溃恢复、幂等工具、预算护栏 | `python durable_agent.py` |
| `examples/special-06/mcp_server.py` | MCP Server 最小示例（订单查询/退款工单，Streamable HTTP） | `pip install "mcp[cli]" && python mcp_server.py` |

### Special-07：推理模型

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `examples/special-07/reasoning_api.py` | 推理模型调用：思考过程分离、思考预算（离线 Mock 可跑） | `python reasoning_api.py` |
| `examples/special-07/distill_reasoning.py` | 推理蒸馏：生成可验证轨迹、过滤、导出 SFT 数据集 | `python distill_reasoning.py` |

---

## 依赖汇总

### 核心依赖

```bash
# 深度学习
pip install torch torchvision torchaudio

# 数据处理
pip install numpy pandas matplotlib scipy

# 机器学习
pip install scikit-learn xgboost lightgbm

# 大模型
pip install transformers peft accelerate bitsandbytes

# 向量数据库
pip install chromadb faiss-cpu

# API 调用
pip install openai anthropic httpx

# Web 框架
pip install fastapi uvicorn pydantic

# 工具库
pip install jupyter notebook pytest black
```

### 完整 requirements.txt

```
# examples/requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
peft>=0.5.0
accelerate>=0.20.0
bitsandbytes>=0.41.0

numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
scikit-learn>=1.2.0

chromadb>=0.4.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0

openai>=1.0.0
anthropic>=0.5.0

fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

jupyter>=1.0.0
pytest>=7.0.0
```

---

## 运行所有示例

```bash
# 克隆仓库
git clone https://github.com/your-username/AIAbilityGraph.git
cd AIAbilityGraph

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r examples/requirements.txt

# 运行示例
python examples/1-1-math/matrix_operations.py
python examples/1-3-dl/mlp_pytorch.py
# ...
```

---

## 开发环境配置

### VS Code 配置

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "ms-python.black-formatter"
}
```

### Jupyter Notebook 配置

```bash
# 启用 Jupyter 扩展
pip install jupyter ipykernel
python -m ipykernel install --user --name AIAbilityGraph
```

---

## 问题排查

### 常见问题

**1. CUDA 版本不匹配**
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. 内存不足**
```bash
# 限制 GPU 显存
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**3. 依赖冲突**
```bash
# 创建干净环境
conda create -n ai-tutorial python=3.10
conda activate ai-tutorial
pip install -r examples/requirements.txt
```

---

## 贡献代码示例

欢迎贡献更多代码示例！请遵循以下规范：

1. **文件命名**: 使用小写字母和下划线
2. **文档字符串**: 每个函数/类都要有 docstring
3. **类型注解**: 使用 Python 类型提示
4. **示例数据**: 使用小样本，避免大文件
5. **可运行性**: 确保代码可以直接运行

提交前请运行：
```bash
python -m black examples/
python -m pytest examples/
```

---

[← 返回主目录](README.md)
