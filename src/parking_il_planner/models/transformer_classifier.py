"""
BERT风格的空间关系推理模块

将ResNet提取的2D特征图展平为序列，使用Transformer编码器
学习车辆、目标、障碍物之间的空间关系

输入: (B, 256, 24, 24) 特征图 + 可选辅助 token
输出: (B, 7) 动作分类logits
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from parking_il_planner.models.positional_encoding import LearnablePositionalEncoding


class BertSpatialReasoning(nn.Module):
    """
    BERT风格的空间关系推理模块

    架构:
    1. 将2D特征图展平为序列: (B, 256, 24, 24) → (B, 576, 256)
    2. 可选：prepend state_vector token 和 action_history token
    3. 添加位置编码
    4. 通过N层Transformer编码器
    5. 全局池化 + 分类头
    """

    def __init__(
        self,
        in_channels: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 7,
        use_state_vector: bool = False,
        state_vector_dim: int = 6,
        use_action_history: bool = False,
        action_history_dim: int = 14,
        resnet_dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_state_vector = use_state_vector
        self.use_action_history = use_action_history

        # Dropout after ResNet features (before Transformer)
        self.feature_dropout = nn.Dropout(resnet_dropout) if resnet_dropout > 0 else nn.Identity()

        # State vector projection: 6D → in_channels
        if use_state_vector:
            self.state_proj = nn.Sequential(
                nn.Linear(state_vector_dim, in_channels),
                nn.GELU(),
                nn.Linear(in_channels, in_channels),
            )

        # Action history projection: 14D → in_channels
        if use_action_history:
            self.action_proj = nn.Sequential(
                nn.Linear(action_history_dim, in_channels),
                nn.GELU(),
                nn.Linear(in_channels, in_channels),
            )

        # 位置编码（可学习）
        self.pos_encoding = LearnablePositionalEncoding(
            dim=in_channels,
            max_positions=1000,
            dropout=dropout
        )

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 全局池化 + 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        state_vector: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (B, C, H, W) ResNet特征图
            state_vector: (B, 6) 相对目标状态向量 [dx, dy, dheading, dist, cos_dh, sin_dh]
            action_history: (B, 14) 最近2步动作 one-hot

        Returns:
            (B, num_classes) 动作logits
        """
        B, C, H, W = features.shape

        # 1. 展平空间维度: (B, C, H, W) → (B, H*W, C)
        x = features.flatten(2).transpose(1, 2)  # (B, N, C)

        # 2. Apply dropout to visual features
        x = self.feature_dropout(x)

        # 3. Prepend auxiliary tokens
        prefix_tokens = []
        if self.use_state_vector and state_vector is not None:
            state_token = self.state_proj(state_vector).unsqueeze(1)  # (B, 1, C)
            prefix_tokens.append(state_token)
        if self.use_action_history and action_history is not None:
            action_token = self.action_proj(action_history).unsqueeze(1)  # (B, 1, C)
            prefix_tokens.append(action_token)

        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)  # (B, N+k, C)

        # 4. 添加位置编码
        x = self.pos_encoding(x)

        # 5. Transformer编码
        x = self.transformer_encoder(x)

        # 6. 全局平均池化
        x = x.mean(dim=1)  # (B, C)

        # 7. 分类头
        logits = self.classifier(x)

        return logits


def build_bert_classifier(config) -> BertSpatialReasoning:
    """
    构建BERT分类器
    """
    return BertSpatialReasoning(
        in_channels=config.resnet_out_channels,
        nhead=config.transformer_nhead,
        num_layers=config.transformer_num_layers,
        dim_feedforward=config.transformer_dim_feedforward,
        dropout=config.transformer_dropout,
        num_classes=config.num_classes,
        use_state_vector=getattr(config, 'use_state_vector', False),
        state_vector_dim=getattr(config, 'state_vector_dim', 6),
        use_action_history=getattr(config, 'use_action_history', False),
        action_history_dim=getattr(config, 'action_history_dim', 14),
        resnet_dropout=getattr(config, 'resnet_dropout', 0.0),
    )
