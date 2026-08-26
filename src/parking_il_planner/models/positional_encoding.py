"""
位置编码模块

为序列化的空间特征添加位置信息
使用正弦/余弦位置编码（类似Transformer原始论文）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """
    2D位置编码
    
    将位置信息编码为每个token的附加特征
    使用正弦和余弦函数的不同频率
    """
    
    def __init__(self, dim: int, max_h: int = 50, max_w: int = 50, dropout: float = 0.1):
        """
        初始化
        
        Args:
            dim: 特征维度（必须能被4整除）
            max_h: 最大高度
            max_w: 最大宽度
            dropout: Dropout比率
        """
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.dropout = nn.Dropout(dropout)
        
        assert dim % 4 == 0, "特征维度必须能被4整除"
        dim_half = dim // 4
        
        # 计算位置编码
        pe = torch.zeros(max_h, max_w, dim)
        
        # Y方向编码
        pos_y = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)
        div_term_y = torch.exp(torch.arange(0, dim_half, 2, dtype=torch.float32) * 
                               -(math.log(10000.0) / dim_half))
        pe[:, :, 0::4] = torch.sin(pos_y * div_term_y).unsqueeze(1).expand(-1, max_w, -1)
        pe[:, :, 1::4] = torch.cos(pos_y * div_term_y).unsqueeze(1).expand(-1, max_w, -1)
        
        # X方向编码
        pos_x = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)
        div_term_x = torch.exp(torch.arange(0, dim_half, 2, dtype=torch.float32) * 
                               -(math.log(10000.0) / dim_half))
        pe[:, :, 2::4] = torch.sin(pos_x * div_term_x).unsqueeze(0).expand(max_h, -1, -1)
        pe[:, :, 3::4] = torch.cos(pos_x * div_term_x).unsqueeze(0).expand(max_h, -1, -1)
        
        # 注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0)  # (1, H, W, D)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: (B, D, H, W) 输入特征图
        
        Returns:
            (B, D, H, W) 带位置编码的特征
        """
        B, D, H, W = x.shape
        
        # 调整位置编码尺寸以匹配输入
        pe = self.pe[:, :H, :W, :].permute(0, 3, 1, 2)  # (1, D, H, W)
        
        x = x + pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    使用可学习的嵌入作为位置信息
    """
    
    def __init__(self, dim: int, max_positions: int = 1000, dropout: float = 0.1):
        """
        初始化
        
        Args:
            dim: 特征维度
            max_positions: 最大位置数
            dropout: Dropout比率
        """
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的位置编码
        self.position_embeddings = nn.Embedding(max_positions, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: (B, N, D) 输入序列
        
        Returns:
            (B, N, D) 带位置编码的序列
        """
        B, N, D = x.shape
        
        # 创建位置索引
        positions = torch.arange(N, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
        
        # 获取位置编码
        pos_embeddings = self.position_embeddings(positions)
        
        x = x + pos_embeddings
        return self.dropout(x)
