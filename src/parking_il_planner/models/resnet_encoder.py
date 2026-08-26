"""
ResNet特征提取器

从3通道occupancy grid图像中提取空间特征
使用预训练的ResNet架构，但修改输入通道为3

输入: (B, 3, 384, 384)
输出: (B, 256, 24, 24) 特征图
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet特征提取器
    
    架构:
    - 使用ResNet18/34的前4个stage（去掉全局平均池化和全连接层）
    - 输入: (B, 3, 384, 384)
    - 输出: (B, 256, 24, 24)
    
    计算过程:
    384×384 → conv1 → 192×192 → layer1 → 96×96 → layer2 → 48×48 
    → layer3 → 24×24 → layer4(去掉) → 最终24×24×256
    """
    
    def __init__(self, resnet_type: str = "resnet18", out_channels: int = 256):
        """
        初始化
        
        Args:
            resnet_type: 'resnet18' 或 'resnet34'
            out_channels: 输出通道数
        """
        super().__init__()
        
        self.resnet_type = resnet_type
        
        # 加载预训练的ResNet
        if resnet_type == "resnet18":
            resnet = models.resnet18(weights=None)
        elif resnet_type == "resnet34":
            resnet = models.resnet34(weights=None)
        else:
            raise ValueError(f"不支持的ResNet类型: {resnet_type}")
        
        # 修改第一层：输入通道改为3（occupancy grid: 当前车/目标/障碍物）
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 提取特征提取部分（去掉池化和全连接层）
        # 我们只需要到layer3，输出24×24×256
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 输出64通道，96×96
        self.layer2 = resnet.layer2  # 输出128通道，48×48
        self.layer3 = resnet.layer3  # 输出256通道，24×24
        
        # 1×1卷积调整通道数（如果需要）
        if out_channels != 256:
            self.channel_adapter = nn.Conv2d(256, out_channels, kernel_size=1)
        else:
            self.channel_adapter = nn.Identity()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, 3, 384, 384) 输入图像
        
        Returns:
            (B, 256, 24, 24) 特征图
        """
        # Stage 1: 384×384 → 192×192
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 192×192
        
        # Stage 2: 192×192 → 96×96
        x = self.layer1(x)
        
        # Stage 3: 96×96 → 48×48
        x = self.layer2(x)
        
        # Stage 4: 48×48 → 24×24
        x = self.layer3(x)
        
        # 调整通道数
        x = self.channel_adapter(x)
        
        return x


def build_resnet_encoder(config) -> ResNetFeatureExtractor:
    """
    构建ResNet特征提取器
    
    Args:
        config: ModelConfig对象
    
    Returns:
        ResNetFeatureExtractor
    """
    return ResNetFeatureExtractor(
        resnet_type=config.resnet_type,
        out_channels=config.resnet_out_channels
    )
