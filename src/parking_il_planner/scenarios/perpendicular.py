"""
垂直泊车场景

简化布局：车位在上方，车辆在下方通道
"""
import math
import random
from typing import List

from parking_il_planner.config.vehicle import VEHICLE_CONFIG
from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.scenarios.base import (
    Obstacle,
    ParkingScenario,
    TargetPose,
    create_wall,
)


class PerpendicularParkingScenario(ParkingScenario):
    """
    垂直泊车场景
    
    布局:
    ┌──────────────────────────┐  y=8
    │    ┌────────┐             │
    │    │  车位   │             │  y=5
    │    │ (目标)  │             │
    │    └────────┘             │
    │                           │
    │      通道 (4米宽)           │
    │                           │
    │         车辆               │  y=0
    │        (初始)              │
    └──────────────────────────┘  y=-2
    """

    def __init__(self, bounds=None):
        super().__init__(bounds)

    def generate(self, seed: int = 0) -> bool:
        rng = random.Random(seed)

        # 车位参数
        slot_x = rng.uniform(-1.0, 1.0)
        slot_y = 3.0
        # 尾入：车头朝外（朝通道方向），车尾朝里（深入车位）
        slot_heading = math.radians(-90)

        slot_width = VEHICLE_CONFIG.width + 2.5    # 左右余量（RS倒车入库需要更大摆动空间）

        # 邻车作为障碍物（车位左右两侧已停车辆）
        neighbor_offset = slot_width / 2 + VEHICLE_CONFIG.width / 2 + 0.5
        self._obstacles = [
            # 左侧邻车
            create_wall(
                x=slot_x - neighbor_offset,
                y=slot_y,
                length=VEHICLE_CONFIG.length,
                width=VEHICLE_CONFIG.width,
                heading=math.radians(90),
                name="left_car"
            ),
            # 右侧邻车
            create_wall(
                x=slot_x + neighbor_offset,
                y=slot_y,
                length=VEHICLE_CONFIG.length,
                width=VEHICLE_CONFIG.width,
                heading=math.radians(90),
                name="right_car"
            ),
        ]

        # 车辆初始位置 - 在车位前方通道
        # 初始x偏移不能超过邻车位置（避免初始碰撞）
        max_lateral = neighbor_offset - VEHICLE_CONFIG.length / 2 - 0.5
        side = rng.choice([-1, 1])
        init_x = slot_x + side * rng.uniform(0.5, min(3.0, max_lateral))
        init_y = slot_y - rng.uniform(2.0, 4.0)
        init_heading = rng.uniform(math.radians(-15), math.radians(15))

        self._initial_state = VehicleState(
            x=init_x,
            y=init_y,
            heading=init_heading
        )

        self._target_pose = TargetPose(
            x=slot_x,
            y=slot_y,
            heading=slot_heading
        )

        return True

    def get_obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def get_initial_state(self) -> VehicleState:
        return self._initial_state

    def get_target_pose(self) -> TargetPose:
        return self._target_pose
