"""
斜方位泊车场景

车位倾斜一定角度（30-60度）
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


class AngledParkingScenario(ParkingScenario):
    """
    斜方位泊车场景

    障碍物放置在不会被RS路径穿透的位置：
    - 后墙在目标深处（远离路径）
    - 通道对面墙壁（在初始位置的另一侧）
    """

    MIN_ANGLE = 30
    MAX_ANGLE = 60

    def __init__(self, bounds=None):
        super().__init__(bounds)

    def generate(self, seed: int = 0) -> bool:
        rng = random.Random(seed)

        angle_deg = rng.uniform(self.MIN_ANGLE, self.MAX_ANGLE)
        angle_rad = math.radians(angle_deg)

        slot_x = rng.uniform(-1.0, 1.0)
        slot_y = 4.0
        # 尾入：车头朝外（通道方向），车尾朝深处
        # 车位内部方向 = angle_rad，尾入时车头朝反方向
        slot_heading = angle_rad - math.pi

        # 通道对面的墙壁（初始位置下方远处）
        aisle_wall_y = -5.0

        # 后方停车位边界（在目标后方远处，沿车尾方向）
        back_wall_dist = VEHICLE_CONFIG.length + 4.0
        back_wall_x = slot_x + back_wall_dist * math.cos(angle_rad)
        back_wall_y = slot_y + back_wall_dist * math.sin(angle_rad)

        self._obstacles = [
            # 通道对面墙壁
            create_wall(
                x=slot_x,
                y=aisle_wall_y,
                length=12.0,
                width=0.2,
                heading=0.0,
                name="opposite_wall"
            ),
            # 后方墙壁
            create_wall(
                x=back_wall_x,
                y=back_wall_y,
                length=6.0,
                width=0.2,
                heading=angle_rad + math.pi / 2,
                name="back_wall"
            ),
        ]

        # 车辆初始位置 - 在车位侧前方，heading≈0°促进尾入
        side = rng.choice([-1, 1])
        init_x = slot_x + side * rng.uniform(1.5, 3.0)
        init_y = slot_y - rng.uniform(2.0, 4.0)
        init_heading = rng.uniform(math.radians(-15), math.radians(15))

        self._initial_state = VehicleState(x=init_x, y=init_y, heading=init_heading)
        self._target_pose = TargetPose(x=slot_x, y=slot_y, heading=slot_heading)

        return True

    def get_obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def get_initial_state(self) -> VehicleState:
        return self._initial_state

    def get_target_pose(self) -> TargetPose:
        return self._target_pose
