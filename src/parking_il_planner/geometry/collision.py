"""
碰撞检测模块

实现车辆与障碍物的碰撞检测
使用OBB（Oriented Bounding Box）碰撞检测算法
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from parking_il_planner.config.vehicle import VEHICLE_CONFIG
from parking_il_planner.geometry.kinematics import VehicleState


@dataclass
class Rectangle:
    """矩形障碍物/车辆"""
    center_x: float    # 中心X坐标
    center_y: float    # 中心Y坐标
    length: float      # 长度
    width: float       # 宽度
    heading: float = 0.0  # 航向角 (弧度)

    def get_corners(self) -> np.ndarray:
        """
        获取矩形的四个角点坐标
        
        Returns:
            (4, 2) 角点坐标数组
        """
        # 计算半长半宽
        half_l = self.length / 2
        half_w = self.width / 2
        
        # 局部坐标系下的角点（逆时针）
        local_corners = np.array([
            [-half_l, -half_w],  # 左下
            [half_l, -half_w],   # 右下
            [half_l, half_w],    # 右上
            [-half_l, half_w],   # 左上
        ])
        
        # 旋转矩阵
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        rotation = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])
        
        # 旋转并平移
        corners = local_corners @ rotation.T
        corners[:, 0] += self.center_x
        corners[:, 1] += self.center_y
        
        return corners


@dataclass
class CollisionResult:
    """碰撞检测结果"""
    collision: bool = False           # 是否发生碰撞
    collision_point: Optional[Tuple[float, float]] = None  # 碰撞点
    penetration_depth: float = 0.0    # 穿透深度 (近似)
    message: str = ""                 # 描述信息


class CollisionDetector:
    """
    碰撞检测器
    
    使用SAT（Separating Axis Theorem，分离轴定理）进行OBB-OBB碰撞检测
    """

    def __init__(self, safety_margin: float = 0.0):
        """
        初始化
        
        Args:
            safety_margin: 安全裕度 (米)，增加此值可以使检测更保守
        """
        self.safety_margin = safety_margin

    def get_vehicle_rect(self, state: VehicleState) -> Rectangle:
        """
        根据车辆状态获取车辆矩形

        Args:
            state: 车辆状态

        Returns:
            Rectangle 车辆矩形
            
        关键：state.x, state.y 是后轴中心坐标，需要转换为几何中心坐标
        后轴中心到几何中心的距离 = length/2 - rear_overhang
        """
        # 后轴中心 -> 几何中心的偏移
        offset = VEHICLE_CONFIG.length / 2 - VEHICLE_CONFIG.rear_overhang
        center_x = state.x + offset * math.cos(state.heading)
        center_y = state.y + offset * math.sin(state.heading)
        
        return Rectangle(
            center_x=center_x,
            center_y=center_y,
            length=VEHICLE_CONFIG.length + self.safety_margin * 2,
            width=VEHICLE_CONFIG.width + self.safety_margin * 2,
            heading=state.heading
        )

    def get_axes(self, rect: Rectangle) -> List[np.ndarray]:
        """
        获取矩形的两个主轴方向
        
        Args:
            rect: 矩形
            
        Returns:
            两个轴方向的单位向量列表
        """
        cos_h = math.cos(rect.heading)
        sin_h = math.sin(rect.heading)
        
        # 长轴方向
        axis1 = np.array([cos_h, sin_h])
        # 宽轴方向（垂直）
        axis2 = np.array([-sin_h, cos_h])
        
        return [axis1, axis2]

    def project_polygon(self, polygon: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        """
        将多边形投影到轴上
        
        Args:
            polygon: (N, 2) 多边形顶点
            axis: 投影轴
            
        Returns:
            (min_proj, max_proj): 投影区间
        """
        projections = polygon @ axis
        return projections.min(), projections.max()

    def intervals_overlap(
        self, 
        min1: float, max1: float, 
        min2: float, max2: float
    ) -> bool:
        """检查两个区间是否重叠"""
        return max1 >= min2 and max2 >= min1

    def check_obb_collision(self, rect1: Rectangle, rect2: Rectangle) -> CollisionResult:
        """
        检测两个OBB是否碰撞
        
        Args:
            rect1: 第一个矩形
            rect2: 第二个矩形
            
        Returns:
            CollisionResult 碰撞结果
        """
        corners1 = rect1.get_corners()
        corners2 = rect2.get_corners()
        
        # 获取所有需要测试的轴（两个矩形的4个主轴）
        axes = self.get_axes(rect1) + self.get_axes(rect2)
        
        for axis in axes:
            min1, max1 = self.project_polygon(corners1, axis)
            min2, max2 = self.project_polygon(corners2, axis)
            
            if not self.intervals_overlap(min1, max1, min2, max2):
                # 找到分离轴，无碰撞
                return CollisionResult(
                    collision=False,
                    message="No collision: separating axis found"
                )
        
        # 所有轴上投影都重叠，发生碰撞
        # 计算近似碰撞点（两个中心连线的中点）
        collision_point = (
            (rect1.center_x + rect2.center_x) / 2,
            (rect1.center_y + rect2.center_y) / 2
        )
        
        # 近似穿透深度
        penetration = min(
            min(max1 - min2, max2 - min1) for axis in axes
            for min1, max1 in [self.project_polygon(corners1, axis)]
            for min2, max2 in [self.project_polygon(corners2, axis)]
        )
        
        return CollisionResult(
            collision=True,
            collision_point=collision_point,
            penetration_depth=penetration,
            message="Collision detected"
        )

    def check_vehicle_obstacle(
        self, 
        state: VehicleState, 
        obstacle: Rectangle
    ) -> CollisionResult:
        """
        检测车辆与障碍物是否碰撞
        
        Args:
            state: 车辆状态
            obstacle: 障碍物矩形
            
        Returns:
            CollisionResult 碰撞结果
        """
        vehicle_rect = self.get_vehicle_rect(state)
        return self.check_obb_collision(vehicle_rect, obstacle)

    def check_vehicle_obstacles(
        self,
        state: VehicleState,
        obstacles: List[Rectangle]
    ) -> CollisionResult:
        """
        检测车辆与多个障碍物是否碰撞
        
        Args:
            state: 车辆状态
            obstacles: 障碍物列表
            
        Returns:
            CollisionResult 第一个碰撞结果，无碰撞则返回无碰撞结果
        """
        for obstacle in obstacles:
            result = self.check_vehicle_obstacle(state, obstacle)
            if result.collision:
                return result
        
        return CollisionResult(collision=False, message="No collision with any obstacle")

    def check_trajectory(
        self,
        trajectory: List[VehicleState],
        obstacles: List[Rectangle]
    ) -> List[Tuple[int, CollisionResult]]:
        """
        检测整个轨迹是否与障碍物碰撞
        
        Args:
            trajectory: 轨迹点列表
            obstacles: 障碍物列表
            
        Returns:
            碰撞点列表 [(索引, CollisionResult), ...]
        """
        collisions = []
        for i, state in enumerate(trajectory):
            result = self.check_vehicle_obstacles(state, obstacles)
            if result.collision:
                collisions.append((i, result))
        
        return collisions


def create_wall(x: float, y: float, length: float, width: float, heading: float = 0.0) -> Rectangle:
    """
    便捷函数：创建墙壁障碍物
    
    Args:
        x, y: 中心位置
        length, width: 尺寸
        heading: 朝向
        
    Returns:
        Rectangle 墙壁
    """
    return Rectangle(center_x=x, center_y=y, length=length, width=width, heading=heading)


def create_vehicle_rect(state: VehicleState, safety_margin: float = 0.0) -> Rectangle:
    """
    便捷函数：从车辆状态创建矩形
    
    Args:
        state: 车辆状态
        safety_margin: 安全裕度
        
    Returns:
        Rectangle 车辆矩形
    """
    detector = CollisionDetector(safety_margin=safety_margin)
    return detector.get_vehicle_rect(state)
