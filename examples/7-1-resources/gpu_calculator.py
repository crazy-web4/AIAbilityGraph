#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 资源需求计算器

功能:
- 显存需求估算
- GPU 型号推荐
- 训练时间预估
- 成本对比分析
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json


# ========== GPU 配置定义 ==========

@dataclass
class GPUConfig:
    """GPU 配置"""
    name: str
    memory_gb: int
    tflops_fp16: int
    tflops_fp32: int
    power_w: int
    nvlink_bw: int  # GB/s, 0 表示不支持
    price_per_hour: float  # 租赁价格


# 主流 GPU 配置
GPUS: Dict[str, GPUConfig] = {
    "A100-40": GPUConfig(
        name="A100-40G", memory_gb=40, tflops_fp16=312,
        tflops_fp32=19, power_w=400, nvlink_bw=600,
        price_per_hour=2.5
    ),
    "A100-80": GPUConfig(
        name="A100-80G", memory_gb=80, tflops_fp16=312,
        tflops_fp32=19, power_w=400, nvlink_bw=600,
        price_per_hour=3.0
    ),
    "H100": GPUConfig(
        name="H100", memory_gb=80, tflops_fp16=989,
        tflops_fp32=67, power_w=700, nvlink_bw=900,
        price_per_hour=5.0
    ),
    "A10": GPUConfig(
        name="A10", memory_gb=24, tflops_fp16=125,
        tflops_fp32=31, power_w=150, nvlink_bw=0,
        price_per_hour=0.8
    ),
    "L4": GPUConfig(
        name="L4", memory_gb=24, tflops_fp16=250,
        tflops_fp32=31, power_w=72, nvlink_bw=0,
        price_per_hour=0.6
    ),
}


@dataclass
class ModelConfig:
    """模型配置"""
    params_b: float       # 参数量 (B)
    seq_len: int          # 序列长度
    batch_size: int       # 批次大小
    embed_dim: Optional[int] = None  # 嵌入维度


# ========== 显存计算 ==========

def estimate_memory_requirement(model_config: ModelConfig) -> Dict[str, float]:
    """
    估算显存需求 (GB)

    简化公式:
    总显存 = 模型权重 + 梯度 + 优化器状态 + 激活值 + KV Cache + 其他

    返回:
        各部分显存占用的详细分解
    """
    params_b = model_config.params_b
    batch_size = model_config.batch_size
    seq_len = model_config.seq_len

    # FP16 权重 (2 bytes/param)
    weights_memory = params_b * 2  # GB

    # 梯度 (2 bytes/param)
    gradients_memory = params_b * 2  # GB

    # Adam 优化器状态 (8-12 bytes/param，取 10)
    optimizer_memory = params_b * 10  # GB

    # 激活值 (与 batch_size 和 seq_len 相关)
    # 简化估算：每 1B 参数约需 0.5GB * batch_size/32
    activation_memory = params_b * 0.5 * (batch_size / 32)  # GB

    # KV Cache (推理时主要占用)
    # 每 token 约需 2 * hidden_dim * n_layers bytes
    # 简化：每 1B 参数约需 0.5GB
    kv_cache_memory = params_b * 0.5  # GB

    # 其他 (临时 buffer、碎片等)
    other_memory = params_b * 0.3

    total_training = weights_memory + gradients_memory + optimizer_memory + activation_memory + other_memory
    total_inference = weights_memory + kv_cache_memory + other_memory

    return {
        "weights_gb": weights_memory,
        "gradients_gb": gradients_memory,
        "optimizer_gb": optimizer_memory,
        "activation_gb": activation_memory,
        "kv_cache_gb": kv_cache_memory,
        "other_gb": other_memory,
        "total_training_gb": total_training,
        "total_inference_gb": total_inference,
        "total_with_10_percent_buffer": total_training * 1.1
    }


def recommend_gpu(model_config: ModelConfig) -> Tuple[str, int, str]:
    """
    根据模型配置推荐 GPU

    返回:
        (GPU 型号，GPU 数量，推荐原因)
    """
    memory_breakdown = estimate_memory_requirement(model_config)
    memory_needed = memory_breakdown["total_training_gb"]

    recommendations = []

    # 检查单卡是否可训练
    for gpu_name, gpu_config in GPUS.items():
        if gpu_config.memory_gb >= memory_needed:
            return (
                gpu_config.name,
                1,
                f"单卡显存足够 ({gpu_config.memory_gb}GB >= {memory_needed:.1f}GB)"
            )

    # 需要多卡
    for gpu_name, gpu_config in GPUS.items():
        if gpu_config.nvlink_bw > 0:  # 优先推荐支持 NVLink 的
            n_gpus = int(memory_needed / gpu_config.memory_gb * 1.2) + 1  # 20% 冗余
            if n_gpus <= 8:  # 单机上限
                return (
                    gpu_config.name,
                    n_gpus,
                    f"多卡训练：{n_gpus}×{gpu_config.name} (需要 {memory_needed:.1f}GB)"
                )

    # 超大规模需要多机
    gpu_config = GPUS["A100-80"]
    n_gpus = int(memory_needed / gpu_config.memory_gb * 1.2) + 1
    return (
        gpu_config.name,
        n_gpus,
        f"大规模训练：需要 {n_gpus}+ {gpu_config.name}，建议多机分布式"
    )


# ========== 训练时间估算 ==========

def estimate_training_time(
    model_config: ModelConfig,
    n_gpus: int,
    gpu_type: str = "A100-80",
    n_epochs: int = 3,
    dataset_size: int = 1000000
) -> Dict:
    """
    估算训练时间

    参数:
        model_config: 模型配置
        n_gpus: GPU 数量
        gpu_type: GPU 类型
        n_epochs: 训练轮数
        dataset_size: 数据集大小 (样本数)

    返回:
        训练时间估算
    """
    params_b = model_config.params_b
    gpu_config = GPUS.get(gpu_type, GPUS["A100-80"])

    # 简化计算：基于 FLOPs 估算
    # 每次前向传播约需 2 * params * tokens FLOPs
    # 每次反向传播约需 4 * params * tokens FLOPs
    # 总计约 6 * params * tokens

    tokens_per_sample = model_config.seq_len
    total_tokens = dataset_size * tokens_per_sample * n_epochs
    total_flops = 6 * params_b * 1e9 * total_tokens

    # GPU 算力 (考虑利用率 0.5-0.7)
    gpu_tflops = gpu_config.tflops_fp16
    effective_tflops = gpu_tflops * n_gpus * 0.6  # 60% 利用率

    # 训练时间 (秒)
    training_seconds = total_flops / (effective_tflops * 1e12)
    training_hours = training_seconds / 3600

    # 成本估算
    compute_cost = training_hours * gpu_config.price_per_hour

    return {
        "total_tokens": total_tokens,
        "total_flops": total_flops,
        "training_hours": training_hours,
        "training_days": training_hours / 24,
        "estimated_compute_cost": compute_cost,
        "gpu_config": f"{n_gpus}× {gpu_config.name}"
    }


# ========== 成本对比 ==========

def compare_cloud_vs_onprem(
    n_gpus: int,
    gpu_type: str,
    usage_hours_per_day: float,
    project_months: int,
    electricity_cost: float = 1.0
) -> Dict:
    """
    对比云服务与自建成本

    参数:
        n_gpus: GPU 数量
        gpu_type: GPU 型号
        usage_hours_per_day: 每天使用小时
        project_months: 项目周期 (月)
        electricity_cost: 电费 (元/度)

    返回:
        成本对比分析
    """
    gpu_config = GPUS.get(gpu_type, GPUS["A100-80"])

    # 云租赁成本
    cloud_monthly = gpu_config.price_per_hour * n_gpus * usage_hours_per_day * 30
    cloud_total = cloud_monthly * project_months

    # 自建成本
    gpu_price = 100000 if "A100" in gpu_type else 50000  # 简化估算
    hardware_cost = gpu_price * n_gpus
    infrastructure_cost = hardware_cost * 0.3  # 服务器/网络等

    # 电费 (GPU + 散热，按 500W/GPU 计)
    power_kwh_per_hour = n_gpus * 0.5
    electricity_monthly = power_kwh_per_hour * usage_hours_per_day * 30 * electricity_cost
    electricity_total = electricity_monthly * project_months

    # 运维成本
    maintenance_cost = hardware_cost * 0.1

    # 总成本
    onprem_total = hardware_cost + infrastructure_cost + electricity_total + maintenance_cost

    # 残值 (3 年后 30%)
    resale_value = hardware_cost * 0.3
    onprem_net = onprem_total - resale_value

    return {
        "cloud": {
            "monthly": cloud_monthly,
            "total": cloud_total
        },
        "onprem": {
            "hardware": hardware_cost,
            "infrastructure": infrastructure_cost,
            "electricity": electricity_total,
            "maintenance": maintenance_cost,
            "resale": -resale_value,
            "total": onprem_net,
            "net": onprem_net
        },
        "recommendation": "自建" if onprem_net < cloud_total else "租赁",
        "savings": abs(onprem_net - cloud_total)
    }


# ========== 使用示例 ==========

def main():
    """演示 GPU 资源计算器"""
    print("=" * 60)
    print("GPU 资源需求计算器")
    print("=" * 60)
    print()

    # ===== 1. 显存需求估算 =====
    print("【1. 显存需求估算】")

    test_models = [
        ModelConfig(params_b=7, seq_len=4096, batch_size=32),
        ModelConfig(params_b=13, seq_len=4096, batch_size=32),
        ModelConfig(params_b=70, seq_len=4096, batch_size=16),
    ]

    for model in test_models:
        breakdown = estimate_memory_requirement(model)
        print(f"\n{model.params_b}B 模型:")
        print(f"  训练显存：{breakdown['total_training_gb']:.1f} GB")
        print(f"  推理显存：{breakdown['total_inference_gb']:.1f} GB")

    # ===== 2. GPU 推荐 =====
    print("\n\n【2. GPU 推荐】")

    for model in test_models:
        gpu_name, n_gpus, reason = recommend_gpu(model)
        print(f"\n{model.params_b}B 模型:")
        print(f"  推荐：{n_gpus}× {gpu_name}")
        print(f"  原因：{reason}")

    # ===== 3. 训练时间估算 =====
    print("\n\n【3. 训练时间估算】")

    model_7b = ModelConfig(params_b=7, seq_len=4096, batch_size=32)
    time_est = estimate_training_time(
        model_config=model_7b,
        n_gpus=8,
        gpu_type="A100-80",
        n_epochs=3,
        dataset_size=100000
    )

    print(f"7B 模型训练 (8×A100, 10 万样本，3 epochs):")
    print(f"  训练时长：{time_est['training_hours']:.1f} 小时 ({time_est['training_days']:.1f} 天)")
    print(f"  计算成本：¥{time_est['estimated_compute_cost']:.0f}")

    # ===== 4. 成本对比 =====
    print("\n\n【4. 成本对比：自建 vs 租赁】")

    comparison = compare_cloud_vs_onprem(
        n_gpus=8,
        gpu_type="A100-80",
        usage_hours_per_day=8,
        project_months=12
    )

    print(f"8×A100, 使用 12 个月，每天 8 小时:")
    print(f"  云租赁：¥{comparison['cloud']['total']:,.0f}")
    print(f"  自建：  ¥{comparison['onprem']['net']:,.0f}")
    print(f"  推荐：  {comparison['recommendation']} (节省 ¥{comparison['savings']:,.0f})")

    print("\n" + "=" * 60)
    print("计算完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
