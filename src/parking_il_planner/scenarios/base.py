"""
场景定义模块

定义泊车场景的抽象基类和数据结构
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from parking_il_planner.geometry.collision import Rectangle
from parking_il_planner.geometry.kinematics import VehicleState


@dataclass
class Obstacle:
    """障碍物"""
    rect: Rectangle           # 矩形障碍物
    name: str = ""            # 名称（可选）

    @classmethod
    def from_center(cls, x: float, y: float, length: float, width: float,
                    heading: float = 0.0, name: str = "") -> Obstacle:
        """从中心位置创建障碍物"""
        return cls(
            rect=Rectangle(center_x=x, center_y=y, length=length, width=width, heading=heading),
            name=name
        )

    @classmethod
    def from_wall(cls, x: float, y: float, length: float, width: float,
                  heading: float = 0.0, name: str = "") -> Obstacle:
        """创建墙壁障碍物（别名）"""
        return cls.from_center(x, y, length, width, heading, name)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.rect.center_x, self.rect.center_y)

    @property
    def corners(self) -> np.ndarray:
        return self.rect.get_corners()


@dataclass
class TargetPose:
    """目标位姿"""
    x: float                  # 目标X位置
    y: float                  # 目标Y位置
    heading: float = 0.0      # 目标航向角 (弧度)

    def to_state(self) -> VehicleState:
        """转换为VehicleState"""
        return VehicleState(x=self.x, y=self.y, heading=self.heading)


@dataclass
class ScenarioBounds:
    """场景边界"""
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class ParkingScenario(ABC):
    """
    泊车场景抽象基类
    
    每个具体场景需要实现:
    - 生成障碍物列表
    - 生成初始状态
    - 生成目标位姿
    """

    def __init__(self, bounds: Optional[ScenarioBounds] = None):
        self.bounds = bounds or ScenarioBounds()
        self._obstacles: List[Obstacle] = []
        self._initial_state: Optional[VehicleState] = None
        self._target_pose: Optional[TargetPose] = None

    @abstractmethod
    def generate(self, seed: int = 0) -> bool:
        """
        生成场景
        
        Args:
            seed: 随机种子
            
        Returns:
            是否生成成功
        """
        pass

    @abstractmethod
    def get_obstacles(self) -> List[Obstacle]:
        """获取障碍物列表"""
        pass

    @abstractmethod
    def get_initial_state(self) -> VehicleState:
        """获取初始状态"""
        pass

    @abstractmethod
    def get_target_pose(self) -> TargetPose:
        """获取目标位姿"""
        pass

    def get_rectangles(self) -> List[Rectangle]:
        """获取所有障碍物的Rectangle列表（用于碰撞检测）"""
        return [obs.rect for obs in self.get_obstacles()]

    def is_valid(self) -> bool:
        """验证场景是否有效"""
        return (
            self._initial_state is not None and
            self._target_pose is not None and
            len(self._obstacles) > 0
        )

    @property
    def name(self) -> str:
        """场景名称"""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(valid={self.is_valid()})"


def create_wall(x: float, y: float, length: float, width: float,
                heading: float = 0.0, name: str = "") -> Obstacle:
    """便捷函数：创建墙壁障碍物"""
    return Obstacle.from_wall(x, y, length, width, heading, name)
