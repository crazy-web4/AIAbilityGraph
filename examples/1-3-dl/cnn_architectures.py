"""
1.3 深度学习基础 - CNN 架构实战

实现经典 CNN 架构：LeNet-5, VGG, ResNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== LeNet-5 (1998) ====================

class LeNet5(nn.Module):
    """
    LeNet-5 架构

    历史意义：第一个成功的卷积神经网络
    应用：手写数字识别 (MNIST)

    架构:
        Input(32×32) → Conv(6@28×28) → Pool(6@14×14) → Conv(16@10×10) → Pool(16@5×5) → FC
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 特征提取部分
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)  # 32→28
        self.pool1 = nn.AvgPool2d(2)  # 28→14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14→10
        self.pool2 = nn.AvgPool2d(2)  # 10→5

        # 分类部分
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== VGG-16 (2014) ====================

class VGGBlock(nn.Module):
    """VGG 卷积块"""

    def __init__(self, in_channels, out_channels, num_layers=2):
        super(VGGBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class VGG16(nn.Module):
    """
    VGG-16 架构

    特点:
    - 统一使用 3×3 卷积
    - 深度增加 (16 层)
    - 使用 MaxPool 降采样

    配置：[64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M]
    """

    def __init__(self, num_classes=1000, input_channels=3):
        super(VGG16, self).__init__()

        # 使用 VGGBlock 构建网络
        self.block1 = VGGBlock(input_channels, 64, num_layers=2)       # 224→112
        self.block2 = VGGBlock(64, 128, num_layers=2)                   # 112→56
        self.block3 = VGGBlock(128, 256, num_layers=3)                  # 56→28
        self.block4 = VGGBlock(256, 512, num_layers=3)                  # 28→14
        self.block5 = VGGBlock(512, 512, num_layers=3)                  # 14→7

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== ResNet (2015) ====================

class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)

    核心思想：恒等映射 F(x) + x
    解决：深度网络的退化问题
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 通用实现

    通过配置不同 block 数量创建 ResNet-18/34/50/101
    """

    def __init__(self, block, layers, num_classes=1000, input_channels=3):
        """
        参数:
            block: 残差块类型
            layers: 每层 block 数量 [2, 2, 2, 2] → ResNet-18
        """
        super(ResNet, self).__init__()

        self.in_channels = 64

        # 初始卷积
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个 residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """创建残差层"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ==================== 预定义 ResNet 变体 ====================

def resnet18(num_classes=1000):
    """ResNet-18: [2, 2, 2, 2]"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    """ResNet-34: [3, 4, 6, 3]"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


# ==================== 模型对比与测试 ====================

def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """打印模型参数量统计"""

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()

    # 前向传播测试
    with torch.no_grad():
        x = torch.randn(*input_size)
        _ = model(x)

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)

    print(f"\n模型参数量统计:")
    print(f"  总参数量：{total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数：{trainable_params:,} ({trainable_params/1e6:.2f}M)")

    return total_params


def compare_architectures():
    """比较不同 CNN 架构"""

    print("=" * 60)
    print("CNN 架构对比")
    print("=" * 60)

    models = {
        'LeNet-5': (LeNet5(num_classes=10), (1, 1, 32, 32)),
        'VGG-16': (VGG16(num_classes=1000), (1, 3, 224, 224)),
        'ResNet-18': (resnet18(num_classes=1000), (1, 3, 224, 224)),
    }

    results = []

    for name, (model, input_size) in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        params = print_model_summary(model, input_size)
        results.append((name, params))

    # 对比表格
    print("\n" + "=" * 60)
    print("架构对比总结")
    print("=" * 60)
    print(f"{'模型':<15} {'参数量 (M)':>15} {'年份':>10} {'特点':<30}")
    print("-" * 60)
    print(f"{'LeNet-5':<15} {0.04:>15.2f} {1998:>10} {'首个成功 CNN，手写数字识别':<30}")
    print(f"{'VGG-16':<15} {138.0:>15.2f} {2014:>10} {'统一 3×3 卷积，深度增加':<30}")
    print(f"{'ResNet-18':<15} {11.7:>15.2f} {2015:>10} {'残差连接，解决梯度消失':<30}")


def demonstrate_feature_extraction():
    """
    演示 CNN 特征提取可视化
    """
    print("\n" + "=" * 60)
    print("CNN 特征提取可视化")
    print("=" * 60)

    # 创建一个简单的 CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32→16
            x = self.pool(F.relu(self.conv2(x)))  # 16→8
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN()

    # 打印每层输出尺寸
    print("\n前向传播尺寸变化:")
    print("-" * 40)

    x = torch.randn(4, 3, 32, 32)  # batch=4, RGB, 32×32
    print(f"Input: {x.shape}")

    x = model.conv1(x)
    print(f"Conv1 后：{x.shape}")

    x = model.pool(F.relu(x))
    print(f"Pool1 后：{x.shape}")

    x = model.conv2(x)
    print(f"Conv2 后：{x.shape}")

    x = model.pool(F.relu(x))
    print(f"Pool2 后：{x.shape}")

    x = x.view(x.size(0), -1)
    print(f"展平后：{x.shape}")


if __name__ == "__main__":
    compare_architectures()
    demonstrate_feature_extraction()

    print("\n" + "=" * 60)
    print("CNN 架构演示完成!")
    print("=" * 60)
