#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 模型服务

功能:
- RESTful API 端点
- 批量推理
- 健康检查
- 指标导出
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import json

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
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    inputs: List[str] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    predictions: List[int]
    labels: List[str]
    confidences: List[float]
    total_latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    device: str
    model_loaded: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """指标响应"""
    total_requests: int
    avg_latency_ms: float
    requests_per_minute: float
    device: str


# ========== 示例模型 ==========

class SimpleClassifier(nn.Module):
    """简单分类器（用于演示）"""

    def __init__(self, input_dim: int = 768, output_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ========== 全局状态 ==========

class ServiceState:
    """服务状态管理"""

    def __init__(self):
        self.start_time = datetime.now()
        self.model = None
        self.device = None
        self.request_count = 0
        self.total_latency_ms = 0
        self.requests_per_minute = []

    def record_request(self, latency_ms: float):
        """记录请求"""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.requests_per_minute.append(datetime.now())

        # 保留最近 1 分钟的请求
        cutoff = datetime.now()
        self.requests_per_minute = [
            t for t in self.requests_per_minute
            if (datetime.now() - t).total_seconds() < 60
        ]

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0
        return self.total_latency_ms / self.request_count

    @property
    def rpm(self) -> float:
        return len(self.requests_per_minute)


state = ServiceState()


# ========== 生命周期管理 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""

    # 启动：加载模型
    logger.info("正在加载模型...")

    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟模型加载（实际应加载真实模型）
    model = SimpleClassifier(input_dim=768, output_dim=5)
    model.to(device)
    model.eval()

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


# ========== API 端点 ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "AI Model Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        device=str(state.device) if state.device else "not loaded",
        model_loaded=state.model is not None,
        uptime_seconds=(datetime.now() - state.start_time).total_seconds()
    )


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
        # 模拟推理（实际应使用真实模型）
        # 这里用随机向量模拟
        with torch.no_grad():
            # 模拟输入嵌入
            input_embedding = torch.randn(1, 768).to(state.device)
            outputs = state.model(input_embedding)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()

        # 计算延迟
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 记录指标
        state.record_request(latency_ms)

        # 获取标签
        label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "joy", 4: "anger"}

        response = PredictionResponse(
            prediction=prediction,
            label=label_map.get(prediction, f"class_{prediction}"),
            confidence=probabilities[0, prediction].item(),
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
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
        batch_size = len(request.inputs)

        # 批量推理（模拟）
        with torch.no_grad():
            input_embeddings = torch.randn(batch_size, 768).to(state.device)
            outputs = state.model(input_embeddings)
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        latency_ms = (time.perf_counter() - start_time) * 1000

        label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "joy", 4: "anger"}

        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            labels=[label_map.get(p, f"class_{p}") for p in predictions.tolist()],
            confidences=[
                probabilities[i, p].item()
                for i, p in enumerate(predictions)
            ],
            total_latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"批量预测失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """获取服务指标"""
    return MetricsResponse(
        total_requests=state.request_count,
        avg_latency_ms=state.avg_latency_ms,
        requests_per_minute=state.rpm,
        device=str(state.device) if state.device else "not loaded"
    )


@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {
                "name": "sentiment-classifier",
                "version": "1.0.0",
                "input_dim": 768,
                "output_dim": 5,
                "device": str(state.device) if state.device else "cpu"
            }
        ]
    }


# ========== 运行入口 ==========

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # 模型服务通常单 worker
    )
