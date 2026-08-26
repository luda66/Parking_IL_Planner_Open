"""
直线进退场景

生成S+和S-动作样本。车辆几乎与目标对齐，只需前进或后退。
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


class StraightApproachScenario(ParkingScenario):
    """
    直线进退场景

    车辆在目标正前方或正后方，通过直行/倒车到达目标。
    两侧有墙壁提供视觉上下文。
    """

    def __init__(self, bounds=None):
        super().__init__(bounds)

    def generate(self, seed: int = 0) -> bool:
        rng = random.Random(seed)

        target_x = rng.uniform(-1.0, 1.0)
        target_y = rng.uniform(1.0, 3.0)
        target_heading = rng.uniform(math.radians(-30), math.radians(30))

        # 车辆在目标的前方或后方（沿target heading方向）
        dist = rng.uniform(1.5, 4.0)
        direction = rng.choice([-1, 1])  # -1: behind target (need S+), 1: ahead (need S-)

        init_x = target_x - direction * dist * math.cos(target_heading)
        init_y = target_y - direction * dist * math.sin(target_heading)
        # 微小航向偏差，让路径不是纯直线（更自然）
        init_heading = target_heading + rng.uniform(math.radians(-5), math.radians(5))

        # 两侧墙壁（平行于行驶方向，提供视觉上下文）
        wall_offset = VEHICLE_CONFIG.width + 1.0
        wall_length = dist + VEHICLE_CONFIG.length + 2.0
        mid_x = (target_x + init_x) / 2
        mid_y = (target_y + init_y) / 2

        perp_angle = target_heading + math.pi / 2

        self._obstacles = [
            create_wall(
                x=mid_x + wall_offset * math.cos(perp_angle),
                y=mid_y + wall_offset * math.sin(perp_angle),
                length=wall_length,
                width=0.2,
                heading=target_heading,
                name="left_wall"
            ),
            create_wall(
                x=mid_x - wall_offset * math.cos(perp_angle),
                y=mid_y - wall_offset * math.sin(perp_angle),
                length=wall_length,
                width=0.2,
                heading=target_heading,
                name="right_wall"
            ),
        ]

        self._initial_state = VehicleState(x=init_x, y=init_y, heading=init_heading)
        self._target_pose = TargetPose(x=target_x, y=target_y, heading=target_heading)

        return True

    def get_obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def get_initial_state(self) -> VehicleState:
        return self._initial_state

    def get_target_pose(self) -> TargetPose:
        return self._target_pose
