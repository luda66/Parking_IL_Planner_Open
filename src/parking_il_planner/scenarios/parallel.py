"""
侧方位泊车场景

车位在车辆前后方，沿路边平行车位
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


class ParallelParkingScenario(ParkingScenario):
    """
    侧方位泊车场景
    
    简化布局
    """

    SLOT_LENGTH = VEHICLE_CONFIG.length + 1.0
    SLOT_WIDTH = VEHICLE_CONFIG.width + 0.6

    def __init__(self, bounds=None):
        super().__init__(bounds)

    def generate(self, seed: int = 0) -> bool:
        rng = random.Random(seed)

        curb_y = 2.0  # 路边Y坐标

        # 车位位置
        slot_x = rng.uniform(-0.5, 0.5)
        slot_heading = 0.0  # 平行于路边

        # 前车和后车
        gap = VEHICLE_CONFIG.length + rng.uniform(3.0, 4.0)  # 前后车间距（RS路径需要摆动空间）
        front_car_x = slot_x + gap / 2 + VEHICLE_CONFIG.length / 2
        rear_car_x = slot_x - gap / 2 - VEHICLE_CONFIG.length / 2

        self._obstacles = [
            # 路边（上方墙壁）—— 远离目标位姿，避免碰撞
            create_wall(
                x=0,
                y=curb_y + VEHICLE_CONFIG.width / 2 + 0.8,
                length=14,
                width=0.15,
                name="curb"
            ),
            # 前车
            create_wall(
                x=front_car_x,
                y=curb_y,
                length=VEHICLE_CONFIG.length,
                width=VEHICLE_CONFIG.width,
                name="front_car"
            ),
            # 后车
            create_wall(
                x=rear_car_x,
                y=curb_y,
                length=VEHICLE_CONFIG.length,
                width=VEHICLE_CONFIG.width,
                name="rear_car"
            ),
        ]

        # 车辆初始位置 - 在车位侧前方（侧方位典型起始位）
        init_x = slot_x + rng.uniform(2.0, 4.0)
        init_y = curb_y - rng.uniform(1.5, 3.0)
        init_heading = rng.uniform(math.radians(-15), math.radians(15))

        self._initial_state = VehicleState(
            x=init_x,
            y=init_y,
            heading=init_heading
        )

        self._target_pose = TargetPose(
            x=slot_x,
            y=curb_y,
            heading=slot_heading
        )

        return True

    def get_obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def get_initial_state(self) -> VehicleState:
        return self._initial_state

    def get_target_pose(self) -> TargetPose:
        return self._target_pose
