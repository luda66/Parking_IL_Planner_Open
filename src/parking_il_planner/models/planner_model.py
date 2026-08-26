"""
ResNet + BERT 融合模型

完整的APA动作预测模型：
1. ResNet提取空间特征
2. BERT进行空间关系推理（含状态向量+动作历史辅助输入）
3. 输出7个动作的概率分布

输入: (B, 3, 384, 384) occupancy grid + 可选 state_vector (B, 6) + 可选 action_history (B, 14)
输出: (B, 7) 动作logits
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from parking_il_planner.config.model import ModelConfig
from parking_il_planner.models.resnet_encoder import build_resnet_encoder
from parking_il_planner.models.transformer_classifier import (
    build_bert_classifier,
)


class APAPlannerImitationModel(nn.Module):
    """
    APA模仿学习模型

    架构:
    [Input: 3×384×384]
      → [ResNet18/34]
      → [Features: 256×24×24]
      → [Flatten + Positional Encoding]
      → [Optional: prepend state_vector token + action_history token]
      → [Transformer Encoder ×N]
      → [Global Pooling]
      → [Classifier Head]
      → [Output: 7 action logits]
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config

        # 1. ResNet特征提取器
        self.resnet_encoder = build_resnet_encoder(config)

        # 2. BERT分类器（含辅助输入支持）
        self.bert_classifier = build_bert_classifier(config)

        print("模型架构: ResNet+BERT")
        print(f"  ResNet: {config.resnet_type}")
        print(f"  特征通道: {config.resnet_out_channels}")
        print(f"  特征尺寸: {config.resnet_feature_size}")
        print(f"  Transformer层数: {config.transformer_num_layers}")
        print(f"  注意力头数: {config.transformer_nhead}")
        print(f"  输出类别: {config.num_classes}")
        if getattr(config, 'use_state_vector', False):
            print(f"  状态向量: {config.state_vector_dim}D → {config.resnet_out_channels}D token")
        if getattr(config, 'use_action_history', False):
            print(f"  动作历史: {config.action_history_dim}D → {config.resnet_out_channels}D token")
        if getattr(config, 'resnet_dropout', 0) > 0:
            print(f"  ResNet dropout: {config.resnet_dropout}")

    def forward(
        self,
        images: torch.Tensor,
        state_vector: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            images: (B, 3, 384, 384) 输入occupancy grid
            state_vector: (B, 6) 相对目标状态 [dx, dy, dheading, dist, cos_dh, sin_dh]
            action_history: (B, 14) 最近2步动作 one-hot

        Returns:
            (B, 7) 动作logits
        """
        features = self.resnet_encoder(images)  # (B, 256, 24, 24)
        logits = self.bert_classifier(features, state_vector, action_history)
        return logits

    def predict(self, images: torch.Tensor, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        logits = self.forward(images, **kwargs)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs

    def predict_action(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.forward(images, **kwargs)
        actions = torch.argmax(logits, dim=-1)
        return actions

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_summary(self) -> Dict:
        return {
            'total_params': self.get_num_params(),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'resnet_type': self.config.resnet_type,
            'transformer_layers': self.config.transformer_num_layers,
            'num_classes': self.config.num_classes
        }

    @staticmethod
    def compute_state_vector(current_state, target_state) -> torch.Tensor:
        """Compute 6D relative state vector from current and target states."""
        dx = target_state[..., 3] - current_state[..., 0] if current_state.dim() > 1 else target_state[3] - current_state[0]
        dy = target_state[..., 4] - current_state[..., 1] if current_state.dim() > 1 else target_state[4] - current_state[1]
        dh = target_state[..., 5] - current_state[..., 2] if current_state.dim() > 1 else target_state[5] - current_state[2]
        dist = torch.sqrt(dx**2 + dy**2)
        cos_dh = torch.cos(dh)
        sin_dh = torch.sin(dh)
        return torch.stack([dx, dy, dh, dist, cos_dh, sin_dh], dim=-1)


def build_apa_model(config: Optional[ModelConfig] = None, device: str = 'cpu') -> APAPlannerImitationModel:
    model = APAPlannerImitationModel(config)
    model = model.to(device)
    return model


if __name__ == "__main__":
    from parking_il_planner.config.model import MODEL_CONFIG

    print("=" * 70)
    print("测试APA模型架构")
    print("=" * 70)

    model = build_apa_model(MODEL_CONFIG, device='cpu')

    summary = model.get_model_summary()
    print("\n模型参数:")
    print(f"  总参数数: {summary['total_params']:,}")
    print(f"  可训练参数: {summary['trainable_params']:,}")

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 384, 384)
    dummy_state = torch.randn(batch_size, 6)
    dummy_action_hist = torch.zeros(batch_size, 14)

    print("\n测试前向传播 (with state_vector + action_history):")
    print(f"  输入形状: {dummy_input.shape}")

    logits = model(dummy_input, state_vector=dummy_state, action_history=dummy_action_hist)
    print(f"  输出形状: {logits.shape}")
    print(f"  输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    print("\n测试前向传播 (image only, backward compatible):")
    logits2 = model(dummy_input)
    print(f"  输出形状: {logits2.shape}")

    print("\n✅ 模型测试通过！")
