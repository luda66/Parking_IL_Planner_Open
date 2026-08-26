"""
动作执行器

将7类离散动作转换为车辆控制量（速度、转向角）
并执行动作，更新车辆状态

动作空间:
    S0:  停车
    S+:  直行前进单位距离
    S-:  直行后退单位距离
    L+:  左转前进单位角度
    L-:  左转后退单位角度
    R+:  右转前进单位角度
    R-:  右转后退单位角度
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

from parking_il_planner.config.vehicle import (
    UNIT_ANGLE_RAD,
    UNIT_DISTANCE,
    VEHICLE_CONFIG,
    ActionSpace,
)
from parking_il_planner.geometry.kinematics import VehicleKinematics, VehicleState


class Action(IntEnum):
    """7类离散动作枚举"""
    S0 = ActionSpace.S0       # 停车
    S_PLUS = ActionSpace.S_PLUS   # 直行前进
    S_MINUS = ActionSpace.S_MINUS  # 直行后退
    L_PLUS = ActionSpace.L_PLUS   # 左转前进
    L_MINUS = ActionSpace.L_MINUS  # 左转后退
    R_PLUS = ActionSpace.R_PLUS   # 右转前进
    R_MINUS = ActionSpace.R_MINUS  # 右转后退

    @property
    def name(self) -> str:
        return ActionSpace.ACTION_NAMES[self]

    @property
    def description(self) -> str:
        return ActionSpace.ACTION_DESC[self]


@dataclass
class ControlCommand:
    """控制指令"""
    velocity: float = 0.0      # 速度 (米/秒)
    steering_angle: float = 0.0  # 转向角 (弧度)
    duration: float = 0.0      # 执行时间 (秒)

    @property
    def distance(self) -> float:
        """执行距离"""
        return abs(self.velocity) * self.duration

    @property
    def angle_change(self) -> float:
        """航向角变化"""
        return abs(self.velocity * math.tan(self.steering_angle) / VEHICLE_CONFIG.wheelbase * self.duration)


@dataclass
class ExecutionResult:
    """动作执行结果"""
    initial_state: VehicleState          # 初始状态
    final_state: VehicleState            # 最终状态
    action: Action                        # 执行的动作
    control: ControlCommand              # 控制指令
    trajectory: List[VehicleState] = field(default_factory=list)  # 轨迹点
    success: bool = True                 # 是否成功执行
    error_message: str = ""              # 错误信息


class ActionExecutor:
    """
    动作执行器
    
    将离散动作转换为控制指令并执行
    """

    # 默认速度 (米/秒) - 可根据需要调整
    FORWARD_SPEED = 0.5    # 前进速度
    BACKWARD_SPEED = -0.3  # 后退速度（较慢以保证安全）

    def __init__(self, kinematics: Optional[VehicleKinematics] = None):
        """
        初始化
        
        Args:
            kinematics: 运动学模型实例
        """
        self.kinematics = kinematics or VehicleKinematics()
        self.max_steering = VEHICLE_CONFIG.max_steering_angle

    def action_to_control(self, action: Action) -> ControlCommand:
        """
        将动作转换为控制指令
        
        Args:
            action: 离散动作
            
        Returns:
            ControlCommand 控制指令
        """
        if action == Action.S0:
            return ControlCommand(velocity=0, steering_angle=0, duration=0)

        # 计算转向角
        steering_angle = 0.0
        if action in (Action.L_PLUS, Action.L_MINUS):
            steering_angle = self.max_steering  # 左转：方向盘打到底
        elif action in (Action.R_PLUS, Action.R_MINUS):
            steering_angle = -self.max_steering  # 右转：方向盘打到底

        # 计算速度和持续时间
        if action in (Action.S_PLUS, Action.L_PLUS, Action.R_PLUS):
            velocity = self.FORWARD_SPEED
        else:  # S_MINUS, L_MINUS, R_MINUS
            velocity = self.BACKWARD_SPEED

        # 对于平移动作（S+, S-），执行固定距离
        if action in (Action.S_PLUS, Action.S_MINUS):
            duration = UNIT_DISTANCE / abs(velocity)
        else:
            # 对于旋转动作（L+, L-, R+, R-），计算达到单位角度所需时间
            # θ_dot = v * tan(δ) / L
            omega = abs(velocity * math.tan(steering_angle) / VEHICLE_CONFIG.wheelbase)
            duration = UNIT_ANGLE_RAD / omega

        return ControlCommand(
            velocity=velocity,
            steering_angle=steering_angle,
            duration=duration
        )

    def execute(
        self, 
        state: VehicleState, 
        action: Action,
        dt: float = 0.001
    ) -> ExecutionResult:
        """
        执行动作
        
        Args:
            state: 当前状态
            action: 要执行的动作
            dt: 仿真时间步长 (秒)
            
        Returns:
            ExecutionResult 执行结果
        """
        control = self.action_to_control(action)
        
        if action == Action.S0:
            return ExecutionResult(
                initial_state=state.copy(),
                final_state=state.copy(),
                action=action,
                control=control
            )

        # 仿真执行过程
        current_state = state.copy()
        current_state.velocity = control.velocity  # 设置速度
        trajectory = [current_state.copy()]
        elapsed_time = 0.0

        while elapsed_time < control.duration:
            current_state = self.kinematics.step(
                current_state,
                control.steering_angle,
                dt
            )
            trajectory.append(current_state.copy())
            elapsed_time += dt

        return ExecutionResult(
            initial_state=state.copy(),
            final_state=current_state.copy(),
            action=action,
            control=control,
            trajectory=trajectory
        )

    def execute_sequence(
        self,
        initial_state: VehicleState,
        actions: List[Action],
        dt: float = 0.001
    ) -> Tuple[List[VehicleState], List[ExecutionResult]]:
        """
        执行动作序列
        
        Args:
            initial_state: 初始状态
            actions: 动作序列
            dt: 仿真时间步长
            
        Returns:
            (states, results): 状态序列和执行结果列表
        """
        states = [initial_state.copy()]
        results = []
        current_state = initial_state.copy()

        for action in actions:
            if action == Action.S0:
                break  # 遇到停车动作停止
            
            result = self.execute(current_state, action, dt)
            results.append(result)
            current_state = result.final_state
            states.append(current_state.copy())

        return states, results


def get_action_info() -> dict:
    """获取所有动作的详细信息"""
    info = {}
    executor = ActionExecutor()
    
    for action in Action:
        control = executor.action_to_control(action)
        info[action.name] = {
            "description": action.description,
            "velocity": control.velocity,
            "steering_angle_deg": math.degrees(control.steering_angle),
            "duration": control.duration,
            "distance": control.distance,
        }
    
    return info
