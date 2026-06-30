# 6.4 AI 工程化与部署

> 将 AI 模型部署到生产环境，实现稳定、可扩展的服务。

## 学习目标

- [ ] 掌握模型服务化技术
- [ ] 理解容器化部署流程
- [ ] 学会设计高可用架构
- [ ] 实施监控与日志系统

---

## 6.4.1 模型服务化

### FastAPI 模型服务

```python
# examples/6-4-deployment/model_server.py
"""
FastAPI 模型服务
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import time
import logging
from contextlib import asynccontextmanager

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== 数据模型 ==========

class PredictionRequest(BaseModel):
    """预测请求"""
    input_text: str = Field(..., description="输入文本", max_length=512)
    return_probabilities: bool = Field(False, description="是否返回概率")


class PredictionResponse(BaseModel):
    """预测响应"""
    prediction: int
    label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    inputs: List[str] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    predictions: List[int]
    labels: List[str]
    confidences: List[float]
    total_latency_ms: float


# ========== 生命周期管理 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    
    # 启动：加载模型
    logger.info("正在加载模型...")
    global model, tokenizer, device
    
    # 模拟模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()  # 实际的模型加载函数
    tokenizer = load_tokenizer()
    
    logger.info(f"模型已加载到 {device}")
    
    yield
    
    # 关闭：清理资源
    logger.info("正在释放模型资源...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 创建应用
app = FastAPI(
    title="AI Model Service",
    description="模型推理 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
tokenizer = None
device = None


# ========== API 端点 ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "AI Model Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    单次预测
    
    Args:
        request: 预测请求
    
    Returns:
        预测结果
    """
    start_time = time.perf_counter()
    
    try:
        # 分词
        inputs = tokenizer(
            request.input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
        
        # 计算延迟
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 获取标签
        label_map = {0: "negative", 1: "positive"}
        
        response = PredictionResponse(
            prediction=prediction,
            label=label_map.get(prediction, f"class_{prediction}"),
            confidence=probabilities[0, prediction].item(),
            latency_ms=latency_ms
        )
        
        if request.return_probabilities:
            response.probabilities = {
                label_map.get(i, f"class_{i}"): prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        
        return response
    
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    批量预测
    
    Args:
        request: 批量预测请求
    
    Returns:
        批量预测结果
    """
    start_time = time.perf_counter()
    
    try:
        # 批处理
        inputs = tokenizer(
            request.inputs,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        label_map = {0: "negative", 1: "positive"}
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            labels=[label_map.get(p, f"class_{p}") for p in predictions.tolist()],
            confidences=[
                probabilities[i, p].item()
                for i, p in enumerate(predictions)
            ],
            total_latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"批量预测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    import psutil
    import torch.cuda
    
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    }
    
    if torch.cuda.is_available():
        metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        metrics["gpu_utilization"] = torch.cuda.utilization()
    
    return metrics


# 辅助函数
def load_model():
    """加载模型"""
    # 实际实现
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    ).to(device)


def load_tokenizer():
    """加载分词器"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


# 运行服务
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # 模型服务通常单 worker
    )
```

---

## 6.4.2 容器化部署

### Dockerfile

```dockerfile
# examples/6-4-deployment/Dockerfile
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 下载模型（可优化为多阶段构建）
RUN python scripts/download_model.py --model-name distilbert-base-uncased

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Docker Compose

```yaml
# examples/6-4-deployment/docker-compose.yml
version: '3.8'

services:
  model-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - model-data:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - model-service

volumes:
  model-data:
  redis-data:
```

---

## 6.4.3 高可用架构

### 多副本部署

```yaml
# examples/6-4-deployment/kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-inference
spec:
  replicas: 3  # 3 个副本
  selector:
    matchLabels:
      app: model-inference
  template:
    metadata:
      labels:
        app: model-inference
    spec:
      containers:
      - name: model-service
        image: my-registry/model-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_PATH
          value: /models
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-inference
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 6.4.4 监控与日志

### Prometheus + Grafana 监控

```yaml
# examples/6-4-deployment/monitoring.yaml
# Prometheus 配置
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-service
spec:
  selector:
    matchLabels:
      app: model-inference
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
```

### 自定义指标导出

```python
# examples/6-4-deployment/metrics_exporter.py
"""
Prometheus 指标导出
"""

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response

# 定义指标
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total model requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Model request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_LOAD = Gauge(
    'model_gpu_utilization',
    'GPU utilization',
    ['device_id']
)

ACTIVE_CONNECTIONS = Gauge(
    'model_active_connections',
    'Number of active connections'
)


# FastAPI 集成
from fastapi import FastAPI

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """导出 Prometheus 指标"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# 装饰器：自动记录指标
from functools import wraps
import time

def track_metrics(endpoint: str):
    """指标追踪装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(endpoint=endpoint, status="started").inc()
            ACTIVE_CONNECTIONS.inc()
            
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
                REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
                ACTIVE_CONNECTIONS.dec()
        
        return wrapper
    return decorator
```

---

## 6.4.5 部署检查清单

```markdown
# AI 模型部署检查清单

## 预部署检查
- [ ] 模型性能验证（准确率、延迟）
- [ ] 压力测试（并发、峰值）
- [ ] 安全扫描（依赖漏洞）
- [ ] 日志配置正确
- [ ] 监控指标配置

## 部署过程
- [ ] 镜像构建成功
- [ ] 健康检查通过
- [ ] 滚动更新无中断
- [ ] 回滚方案可用

## 部署后验证
- [ ] 生产流量验证
- [ ] 指标监控正常
- [ ] 告警配置正确
- [ ] 日志收集正常

## 运维准备
- [ ] 运维文档完整
- [ ] 故障处理流程明确
- [ ] 备份策略配置
- [ ] 扩容方案就绪
```

---

## 练习题

1. **服务部署**: 将一个训练好的模型封装为 FastAPI 服务，并编写 Dockerfile。

2. **K8s 配置**: 设计 Kubernetes 部署配置，支持自动扩缩容。

3. **监控方案**: 为模型服务设计完整的监控指标体系。

---

[← 上一节：6.3 测试与验证](6-3-testing.md) | [第 7 章：AI 资源与成本 →](../chapter-7/README.md)
