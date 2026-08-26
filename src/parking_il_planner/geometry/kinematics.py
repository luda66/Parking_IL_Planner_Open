"""
车辆运动学模型

基于阿克曼转向几何（Ackermann Steering Geometry）
实现车辆状态更新和轨迹计算

核心公式:
    x_dot = v * cos(θ)
    y_dot = v * sin(θ)
    θ_dot = v * tan(δ) / L

其中:
    (x, y): 后轴中心位置
    θ: 航向角 (heading)
    v: 速度
    δ: 前轮转角
    L: 轴距
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from parking_il_planner.config.vehicle import VEHICLE_CONFIG


@dataclass
class VehicleState:
    """车辆状态"""
    x: float = 0.0          # 后轴中心X坐标 (米)
    y: float = 0.0          # 后轴中心Y坐标 (米)
    heading: float = 0.0    # 航向角 (弧度)，0表示沿X轴正方向
    velocity: float = 0.0   # 速度 (米/秒)，负值表示后退

    def copy(self) -> VehicleState:
        """深拷贝"""
        return VehicleState(
            x=self.x,
            y=self.y,
            heading=self.heading,
            velocity=self.velocity
        )

    @property
    def position(self) -> Tuple[float, float]:
        """位置元组"""
        return (self.x, self.y)

    def distance_to(self, other: VehicleState) -> float:
        """计算到另一状态的距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_diff(self, other: VehicleState) -> float:
        """计算到另一状态的角度差 (绝对值，弧度)"""
        diff = abs(self.heading - other.heading)
        # 归一化到 [0, π]
        while diff > math.pi:
            diff = 2 * math.pi - diff
        return diff

    def __repr__(self) -> str:
        return (f"VehicleState(x={self.x:.3f}, y={self.y:.3f}, "
                f"heading={math.degrees(self.heading):.1f}°, "
                f"velocity={self.velocity:.2f})")


class VehicleKinematics:
    """
    车辆运动学模型
    
    使用简化的自行车模型（Bicycle Model）
    """

    def __init__(self, config=None):
        """
        初始化
        
        Args:
            config: VehicleConfig实例，默认使用全局配置
        """
        self.config = config or VEHICLE_CONFIG
        self.L = self.config.wheelbase  # 轴距

    def kinematic_equations(
        self, 
        state: VehicleState, 
        steering_angle: float
    ) -> Tuple[float, float, float]:
        """
        运动学方程
        
        Args:
            state: 当前车辆状态
            steering_angle: 前轮转角 (弧度)
            
        Returns:
            (x_dot, y_dot, theta_dot): 状态导数
        """
        v = state.velocity
        theta = state.heading
        
        x_dot = v * math.cos(theta)
        y_dot = v * math.sin(theta)
        theta_dot = v * math.tan(steering_angle) / self.L
        
        return x_dot, y_dot, theta_dot

    def step(
        self, 
        state: VehicleState, 
        steering_angle: float, 
        dt: float = 0.1
    ) -> VehicleState:
        """
        单步状态更新（欧拉积分）
        
        Args:
            state: 当前状态
            steering_angle: 前轮转角 (弧度)
            dt: 时间步长 (秒)
            
        Returns:
            新状态
        """
        x_dot, y_dot, theta_dot = self.kinematic_equations(state, steering_angle)
        
        new_state = state.copy()
        new_state.x += x_dot * dt
        new_state.y += y_dot * dt
        new_state.heading += theta_dot * dt
        
        # 航向角归一化到 [-π, π]
        new_state.heading = math.atan2(
            math.sin(new_state.heading), 
            math.cos(new_state.heading)
        )
        
        return new_state

    def simulate(
        self,
        initial_state: VehicleState,
        steering_angles: List[float],
        velocities: List[float],
        dt: float = 0.1
    ) -> List[VehicleState]:
        """
        仿真轨迹
        
        Args:
            initial_state: 初始状态
            steering_angles: 转向角序列 (弧度)
            velocities: 速度序列 (米/秒)
            dt: 时间步长 (秒)
            
        Returns:
            状态序列（包含初始状态）
        """
        assert len(steering_angles) == len(velocities), \
            "转向角和速度序列长度必须相同"
        
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        
        for steering, velocity in zip(steering_angles, velocities, strict=False):
            current_state.velocity = velocity
            current_state = self.step(current_state, steering, dt)
            states.append(current_state.copy())
        
        return states

    @staticmethod
    def heading_to_vector(heading: float, length: float = 1.0) -> Tuple[float, float]:
        """将航向角转换为方向向量"""
        return (length * math.cos(heading), length * math.sin(heading))

    @staticmethod
    def angle_between(headings: float, target: float) -> float:
        """计算两个航向角之间的差值 (带符号，弧度)"""
        diff = target - headings
        # 归一化到 [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-π, π]
    
    Args:
        angle: 输入角度 (弧度)
        
    Returns:
        归一化后的角度
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def deg2rad(degrees: float) -> float:
    """角度转弧度"""
    return degrees * math.pi / 180.0


def rad2deg(radians: float) -> float:
    """弧度转角度"""
    return radians * 180.0 / math.pi
