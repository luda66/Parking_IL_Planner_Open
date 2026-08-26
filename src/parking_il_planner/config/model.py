"""
模型超参数配置
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """模型架构配置"""
    # 输入配置
    image_size: Tuple[int, int] = (384, 384)
    in_channels: int = 3
    pixels_per_meter: float = 20.0

    # ResNet配置
    resnet_type: str = "resnet18"
    resnet_out_channels: int = 256
    resnet_feature_size: Tuple[int, int] = (24, 24)

    # Transformer配置
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # 分类头配置
    num_classes: int = 7

    # Phase 3: 辅助输入
    use_state_vector: bool = True
    state_vector_dim: int = 6
    use_action_history: bool = True
    action_history_steps: int = 2
    action_history_dim: int = 14  # 2 * num_classes

    # Phase 3: 正则化
    resnet_dropout: float = 0.3
    label_smoothing: float = 0.1

    # 序列长度 (由特征图尺寸计算)
    @property
    def sequence_length(self) -> int:
        return self.resnet_feature_size[0] * self.resnet_feature_size[1]  # 24*24=576


# 默认模型配置
MODEL_CONFIG = ModelConfig()
