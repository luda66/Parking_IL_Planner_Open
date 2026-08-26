"""
车辆运动学参数配置

基于算法文章中的车辆参数定义
"""
import math
from dataclasses import dataclass


@dataclass
class VehicleConfig:
    """车辆运动学参数"""
    # 基本尺寸 (米)
    length: float = 4.360       # 车长
    width: float = 1.785        # 车宽
    wheelbase: float = 2.535    # 轴距
    front_track: float = 1.525  # 前轮距
    rear_track: float = 1.535   # 后轮距
    front_overhang: float = 0.840   # 前悬
    rear_overhang: float = 0.985    # 后悬
    
    # 运动学参数
    min_turning_radius: float = 5.5   # 最小转弯半径 (米)
    
    # 计算属性
    @property
    def max_steering_angle(self) -> float:
        """最大转向角 (弧度) - 基于最小转弯半径计算"""
        return math.atan(self.wheelbase / self.min_turning_radius)
    
    @property
    def rear_to_center(self) -> float:
        """后轴中心到车辆几何中心的距离 (米)"""
        return (self.length - self.rear_overhang) - self.length / 2


# 默认车辆配置实例
VEHICLE_CONFIG = VehicleConfig()


# 动作空间定义
class ActionSpace:
    """7类离散动作空间"""
    S0 = 0       # 停车
    S_PLUS = 1   # 直行前进
    S_MINUS = 2  # 直行后退
    L_PLUS = 3   # 左转前进
    L_MINUS = 4  # 左转后退
    R_PLUS = 5   # 右转前进
    R_MINUS = 6  # 右转后退
    
    NUM_ACTIONS = 7
    ACTION_NAMES = {
        S0: "S0",
        S_PLUS: "S+",
        S_MINUS: "S-",
        L_PLUS: "L+",
        L_MINUS: "L-",
        R_PLUS: "R+",
        R_MINUS: "R-",
    }
    ACTION_DESC = {
        S0: "停车",
        S_PLUS: "直行前进单位距离",
        S_MINUS: "直行后退单位距离",
        L_PLUS: "左转前进单位角度",
        L_MINUS: "左转后退单位角度",
        R_PLUS: "右转前进单位角度",
        R_MINUS: "右转后退单位角度",
    }


# 动作参数
UNIT_DISTANCE = 0.05    # 单位距离 (米)
UNIT_ANGLE = 0.5        # 单位角度 (度)
UNIT_ANGLE_RAD = math.radians(UNIT_ANGLE)  # 单位角度 (弧度)
