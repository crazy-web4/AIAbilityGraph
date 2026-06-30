"""
3.2 模型压缩与量化 - 量化技术实战

实现内容:
- 训练后量化 (PTQ)
- 量化感知训练 (QAT)
- BitsAndBytes 集成
"""

import torch
import torch.nn as nn
from typing import Dict, List


# ========== 基础量化 ==========

class QuantizationDemo:
    """
    量化基础演示

    量化类型:
    - INT8: 8 比特量化
    - INT4: 4 比特量化
    - NF4: Normal Float 4 (QLoRA)
    """

    @staticmethod
    def min_max_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        Min-Max 量化

        公式:
        scale = (max - min) / (2^bits - 1)
        x_int = round((x - min) / scale)
        """
        bits = 2 ** bits - 1

        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / bits

        # 量化
        x_int = torch.round((x - min_val) / scale).clamp(0, bits)

        # 反量化
        x_dequant = x_int * scale + min_val

        return x_dequant

    @staticmethod
    def symmetric_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        对称量化

        以 0 为中心，适合权重
        """
        bits = 2 ** (bits - 1) - 1  # 对称范围

        abs_max = x.abs().max()
        scale = abs_max / bits

        # 量化
        x_int = torch.round(x / scale).clamp(-bits, bits)

        # 反量化
        x_dequant = x_int * scale

        return x_dequant

    @staticmethod
    def quantize_tensor(x: torch.Tensor, qmin: int, qmax: int) -> tuple:
        """
        张量量化

        返回:
            (量化值，scale, zero_point)
        """
        n_bits = qmax - qmin

        # 计算 scale 和 zero_point
        rmin = x.min().item()
        rmax = x.max().item()

        scale = (rmax - rmin) / n_bits
        zero_point = qmin - round(rmin / scale)

        # 量化
        x_int = torch.round(x / scale) + zero_point
        x_int = x_int.clamp(qmin, qmax).to(torch.int8)

        return x_int, scale, zero_point


# ========== PyTorch 量化 ==========

class QuantizedModel(nn.Module):
    """
    量化模型示例

    使用 PyTorch 内建量化
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        # 浮点模型
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 量化配置
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # 设置量化配置
        self.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        self.qconfig_backend = 'fbgemm'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def prepare_quantization(self):
        """准备量化"""
        torch.ao.quantization.prepare(self, inplace=True)

    def convert_to_quantized(self):
        """转换为量化模型"""
        torch.ao.quantization.convert(self, inplace=True)


# ========== BitsAndBytes 量化 ==========

def create_bnb_config(
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4"
) -> Dict:
    """
    创建 BitsAndBytes 量化配置

    参数:
        load_in_4bit: 是否 4-bit 量化
        use_double_quant: 是否双重量化
        quant_type: 量化类型 (nf4/fp4)

    返回:
        配置字典
    """
    config = {
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_compute_dtype": torch.float16,
    }

    if use_double_quant:
        config["bnb_4bit_use_double_quant"] = True

    return config


def load_quantized_model(
    model_name: str,
    quant_config: Dict = None
):
    """
    加载量化模型

    使用 HuggingFace + BitsAndBytes
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if quant_config is None:
        quant_config = create_bnb_config()

    # 创建配置
    bnb_config = BitsAndBytesConfig(**quant_config)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return model


# ========== 量化效果对比 ==========

def compare_quantization_methods():
    """比较不同量化方法"""

    # 创建测试张量
    x = torch.randn(1000, 1000)

    print("=" * 50)
    print("量化方法对比")
    print("=" * 50)

    # 原始 (FP32)
    original_size = x.element_size() * x.numel()
    print(f"\n原始 (FP32):")
    print(f"  大小：{original_size / 1024 / 1024:.2f} MB")
    print(f"  精度：32-bit")

    # INT8
    x_int8 = QuantizationDemo.symmetric_quantize(x, bits=8)
    int8_error = (x - x_int8).abs().mean().item()
    print(f"\nINT8 量化:")
    print(f"  大小：{original_size / 1024 / 1024 / 4:.2f} MB (压缩 4x)")
    print(f"  误差：{int8_error:.6f}")

    # INT4
    x_int4 = QuantizationDemo.symmetric_quantize(x, bits=4)
    int4_error = (x - x_int4).abs().mean().item()
    print(f"\nINT4 量化:")
    print(f"  大小：{original_size / 1024 / 1024 / 8:.2f} MB (压缩 8x)")
    print(f"  误差：{int4_error:.6f}")

    # 模拟 NF4
    print(f"\nNF4 (QLoRA):")
    print(f"  大小：{original_size / 1024 / 1024 / 8:.2f} MB (压缩 8x)")
    print(f"  误差：~0.001 (正常化浮点)")


# ========== 性能测试 ==========

def benchmark_quantization():
    """量化性能基准测试"""

    import time

    # 创建模型
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024)
    )

    # 输入
    x = torch.randn(32, 1024)

    # FP32 基准
    model_fp32 = model
    model_fp32.eval()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            _ = model_fp32(x)
        fp32_time = (time.perf_counter() - start) / 100

    print("=" * 50)
    print("性能基准测试")
    print("=" * 50)
    print(f"FP32 推理时间：{fp32_time * 1000:.2f} ms")
    print(f"FP32 显存占用：{sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.2f} MB")

    # INT8 量化
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            _ = model_int8(x)
        int8_time = (time.perf_compiler() - start) / 100

    # 注意：实际加速需要在支持 INT8 的硬件上测试
    print(f"\nINT8 理论加速：2-4x")
    print(f"INT8 显存节省：4x")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 量化方法对比
    compare_quantization_methods()

    # 性能基准
    # benchmark_quantization()

    # 模型量化示例
    print("\n" + "=" * 50)
    print("模型量化示例")
    print("=" * 50)

    # 创建模型
    model = QuantizedModel(input_size=512, hidden_size=256, output_size=10)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 准备量化
    model.prepare_quantization()
    print("量化准备完成")

    # 模拟校准
    model.eval()
    with torch.no_grad():
        x = torch.randn(32, 512)
        _ = model(x)

    # 转换
    model.convert_to_quantized()
    print("量化转换完成")

    # 测试推理
    with torch.no_grad():
        output = model(torch.randn(1, 512))
    print(f"输出形状：{output.shape}")
