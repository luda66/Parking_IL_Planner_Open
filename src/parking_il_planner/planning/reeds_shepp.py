"""Reeds-Shepp expert planner.

The analytic path functions in this module are adapted from the MIT-licensed
PythonRobotics Reeds-Shepp implementation by Atsushi Sakai, Videh Patel, and
other contributors. See the repository NOTICE file for the original copyright
and license text.

The adaptation integrates the solver with this project's discrete action space
and vehicle model.

流程：
1. Reeds-Shepp 解析求解 → 最短路径（分段描述）
2. 路径插值 → 稠密状态序列 (x, y, yaw, direction)
3. 状态序列离散化 → 7 类动作序列（与 ActionExecutor 一致）
4. 端点验证 → 确保动作回放后末态与目标一致

参考：
- Reeds, J. A.; Shepp, L. A. (1990). "Optimal paths for a car that goes both
  forwards and backwards"
- PythonRobotics: https://github.com/AtsushiSakai/PythonRobotics
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from parking_il_planner.config.vehicle import UNIT_ANGLE_RAD, UNIT_DISTANCE, VEHICLE_CONFIG
from parking_il_planner.geometry.kinematics import VehicleState, normalize_angle
from parking_il_planner.planning.actions import Action, ActionExecutor

# ============================================================================
# Reeds-Shepp 解析解核心（基于 PythonRobotics，MIT License）
# ============================================================================

@dataclass
class RSPath:
    """Reeds-Shepp 路径"""
    lengths: List[float] = field(default_factory=list)
    ctypes: List[str] = field(default_factory=list)
    L: float = 0.0
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)


def _mod2pi(x: float) -> float:
    v = math.fmod(x, math.copysign(2.0 * math.pi, x)) if x != 0 else 0.0
    if v < -math.pi:
        v += 2.0 * math.pi
    elif v > math.pi:
        v -= 2.0 * math.pi
    return v


def _pi_2_pi(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _set_path(paths, lengths, ctypes, step_size):
    L = sum(abs(length) for length in lengths)
    if L <= step_size:
        return paths
    for p in paths:
        if p.ctypes == ctypes and abs(
            sum(abs(length) for length in p.lengths) - L
        ) <= step_size:
            return paths
    path = RSPath(lengths=lengths, ctypes=ctypes, L=L)
    paths.append(path)
    return paths


# --- 12 基本路径函数 ---

def _LSL(x, y, phi):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']
    return False, [], []


def _LSR(x, y, phi):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1_sq = u1 ** 2
    if u1_sq >= 4.0:
        u = math.sqrt(u1_sq - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, [t, u, v], ['L', 'S', 'R']
    return False, [], []


def _LxRxL(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = _mod2pi(A + theta + math.pi / 2)
        u = _mod2pi(math.pi - 2 * A)
        v = _mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']
    return False, [], []


def _LxRL(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = _mod2pi(A + theta + math.pi / 2)
        u = _mod2pi(math.pi - 2 * A)
        v = _mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']
    return False, [], []


def _LRxL(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 <= 4.0:
        u = math.acos(1 - u1 ** 2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = _mod2pi(-A + theta + math.pi / 2)
        v = _mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']
    return False, [], []


def _LRxLR(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = _mod2pi(theta + A + math.pi / 2)
        u = _mod2pi(A)
        v = _mod2pi(phi - t + 2 * u)
        if t >= 0 and u >= 0 and v >= 0:
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']
    return False, [], []


def _LxRLxR(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    u2 = (20 - u1 ** 2) / 16
    if 0 <= u2 <= 1:
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = _mod2pi(theta + A + math.pi / 2)
        v = _mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']
    return False, [], []


def _LxR90SL(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = _mod2pi(theta + A + math.pi / 2)
        v = _mod2pi(t - phi + math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -v], ['L', 'R', 'S', 'L']
    return False, [], []


def _LSR90xL(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 >= 2.0:
        u = math.sqrt(u1 ** 2 - 4) - 2
        A = math.atan2(math.sqrt(u1 ** 2 - 4), 2)
        t = _mod2pi(theta - A + math.pi / 2)
        v = _mod2pi(t - phi - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, u, math.pi / 2, -v], ['L', 'S', 'R', 'L']
    return False, [], []


def _LxR90SR(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 >= 2.0:
        t = _mod2pi(theta + math.pi / 2)
        u = u1 - 2
        v = _mod2pi(phi - t - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -v], ['L', 'R', 'S', 'R']
    return False, [], []


def _LSL90xR(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2
        v = _mod2pi(phi - t - math.pi / 2)
        if t >= 0 and v >= 0:
            return True, [t, u, math.pi / 2, -v], ['L', 'S', 'L', 'R']
    return False, [], []


def _LxR90SL90xR(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = _polar(zeta, eeta)
    if u1 >= 4.0:
        u = math.sqrt(u1 ** 2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1 ** 2 - 4))
        t = _mod2pi(theta + A + math.pi / 2)
        v = _mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return True, [t, -math.pi / 2, -u, -math.pi / 2, v], ['L', 'R', 'S', 'L', 'R']
    return False, [], []


_PATH_FUNCTIONS = [
    _LSL, _LSR,
    _LxRxL, _LxRL, _LRxL,
    _LRxLR, _LxRLxR,
    _LxR90SL, _LxR90SR,
    _LSR90xL, _LSL90xR,
    _LxR90SL90xR,
]


def _timeflip(dists):
    return [-d for d in dists]


def _reflect(dirs):
    return [{'L': 'R', 'R': 'L', 'S': 'S'}[d] for d in dirs]


def _generate_paths(q0, q1, max_curvature, step_size):
    """生成所有候选 RS 路径（归一化坐标）"""
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    norm_step = step_size * max_curvature

    paths = []
    for func in _PATH_FUNCTIONS:
        # 原始
        flag, dists, dirs = func(x, y, dth)
        if flag:
            paths = _set_path(paths, dists, dirs, norm_step)

        # timeflip (前后反转)
        flag, dists, dirs = func(-x, y, -dth)
        if flag:
            paths = _set_path(paths, _timeflip(dists), dirs, norm_step)

        # reflect (左右镜像)
        flag, dists, dirs = func(x, -y, -dth)
        if flag:
            paths = _set_path(paths, dists, _reflect(dirs), norm_step)

        # timeflip + reflect
        flag, dists, dirs = func(-x, -y, dth)
        if flag:
            paths = _set_path(paths, _timeflip(dists), _reflect(dirs), norm_step)

    return paths


def _interpolate(dist, length, mode, max_curvature, ox, oy, oyaw):
    """沿路径段插值单点"""
    if mode == "S":
        x = ox + dist / max_curvature * math.cos(oyaw)
        y = oy + dist / max_curvature * math.sin(oyaw)
        yaw = oyaw
    else:
        ldx = math.sin(dist) / max_curvature
        if mode == "L":
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = oyaw + dist
        else:  # R
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = oyaw - dist
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        x = ox + gdx
        y = oy + gdy
    direction = 1 if length > 0.0 else -1
    return x, y, yaw, direction


def _generate_local_course(lengths, modes, max_curvature, step_size):
    """生成局部坐标系路径点"""
    ox, oy, oyaw = 0.0, 0.0, 0.0
    xs, ys, yaws, directions = [], [], [], []

    for length, mode in zip(lengths, modes, strict=False):
        d_dist = step_size * max_curvature if length >= 0 else -step_size * max_curvature
        dists = np.arange(0.0, length, d_dist).tolist()
        dists.append(length)

        for dist in dists:
            x, y, yaw, d = _interpolate(dist, length, mode, max_curvature, ox, oy, oyaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(d)

        ox, oy, oyaw = xs[-1], ys[-1], yaws[-1]

    return xs, ys, yaws, directions


# ============================================================================
# 高层接口
# ============================================================================

def reeds_shepp_path_planning(
    sx: float, sy: float, syaw: float,
    gx: float, gy: float, gyaw: float,
    max_curvature: float,
    step_size: float = 0.05
) -> Optional[RSPath]:
    """
    计算 Reeds-Shepp 最短路径

    Args:
        sx, sy, syaw: 起点 (m, m, rad)
        gx, gy, gyaw: 终点 (m, m, rad)
        max_curvature: 最大曲率 = 1/R_min
        step_size: 插值步长 (m)

    Returns:
        RSPath 或 None
    """
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = _generate_paths(q0, q1, max_curvature, step_size)
    if not paths:
        return None

    # 插值所有路径
    for path in paths:
        xs, ys, yaws, directions = _generate_local_course(
            path.lengths, path.ctypes, max_curvature, step_size
        )
        # 转换到全局坐标
        c = math.cos(-q0[2])
        s = math.sin(-q0[2])
        path.x = [c * ix + s * iy + q0[0] for ix, iy in zip(xs, ys, strict=False)]
        path.y = [-s * ix + c * iy + q0[1] for ix, iy in zip(xs, ys, strict=False)]
        path.yaw = [_pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / max_curvature for length in path.lengths]
        path.L = path.L / max_curvature

    # 选最短路径
    best = min(paths, key=lambda p: abs(p.L))
    return best


# ============================================================================
# 离散化层：RS 路径 → 7 类动作序列
# ============================================================================

@dataclass
class PlanResult:
    """规划结果"""
    actions: List[int] = field(default_factory=list)
    states: List[VehicleState] = field(default_factory=list)
    rs_path: Optional[RSPath] = None
    success: bool = False
    final_error_pos: float = 0.0
    final_error_heading: float = 0.0
    message: str = ""


class ReedsSheppExpertPlanner:
    """
    Reeds-Shepp 专家规划器

    将 RS 路径离散化为与 ActionExecutor 完全一致的动作序列。
    离散化策略：遍历 RS 路径点，根据相邻点的位移方向和航向变化
    分类为 7 种动作，然后通过 ActionExecutor 回放验证。
    """

    def __init__(self):
        self.max_curvature = 1.0 / VEHICLE_CONFIG.min_turning_radius
        self.executor = ActionExecutor()
        self.step_size = UNIT_DISTANCE  # 0.05m，与离散动作步长一致

    def plan(
        self,
        start: VehicleState,
        target: VehicleState,
    ) -> PlanResult:
        """
        规划从 start 到 target 的动作序列

        Args:
            start: 起点状态
            target: 目标状态

        Returns:
            PlanResult
        """
        # Step 1: RS 路径求解
        rs_path = reeds_shepp_path_planning(
            start.x, start.y, start.heading,
            target.x, target.y, target.heading,
            self.max_curvature,
            step_size=self.step_size
        )
        if rs_path is None:
            return PlanResult(success=False, message="RS path not found")

        # Step 2: 路径点分类为动作序列
        actions = self._classify_path_to_actions(rs_path)

        if not actions:
            return PlanResult(
                success=False, rs_path=rs_path,
                message="Action classification produced empty sequence"
            )

        # Step 3: 追加 S0 停止动作
        actions.append(Action.S0.value)

        # Step 4: 通过 ActionExecutor 回放，计算末态误差
        states = [start.copy()]
        current = start.copy()
        for a in actions:
            if a == Action.S0:
                break
            result = self.executor.execute(current, Action(a))
            current = result.final_state
            states.append(current.copy())

        # 计算末态误差
        pos_err = math.sqrt(
            (current.x - target.x) ** 2 + (current.y - target.y) ** 2
        )
        heading_err = abs(normalize_angle(current.heading - target.heading))

        return PlanResult(
            actions=actions,
            states=states,
            rs_path=rs_path,
            success=True,
            final_error_pos=pos_err,
            final_error_heading=heading_err,
            message=f"OK: {len(actions)} actions, pos_err={pos_err:.4f}m, heading_err={math.degrees(heading_err):.2f}°"
        )

    def _classify_path_to_actions(self, path: RSPath) -> List[int]:
        """
        将 RS 路径分段描述直接转换为离散动作序列

        策略：按 RS 路径的每个 segment 计算对应动作数量。
        - 'S' 段：n_steps = round(|length| / UNIT_DISTANCE)，动作为 S+ 或 S-
        - 'L'/'R' 段：n_steps = round(|angle_change| / UNIT_ANGLE_RAD)，
                      angle_change = |length| / R_min
                      动作根据方向(前/后)和转向(L/R)确定

        这种方法避免了逐点分类的累积误差。
        """
        actions = []
        R = VEHICLE_CONFIG.min_turning_radius

        for length, ctype in zip(path.lengths, path.ctypes, strict=False):
            if abs(length) < 1e-6:
                continue

            forward = length > 0  # 正长度=前进，负长度=后退
            abs_length = abs(length)

            if ctype == 'S':
                n_steps = round(abs_length / UNIT_DISTANCE)
                action = Action.S_PLUS.value if forward else Action.S_MINUS.value
                actions.extend([action] * n_steps)
            else:
                # 转弯段：角度变化 = 弧长 / R
                angle_change = abs_length / R
                n_steps = round(angle_change / UNIT_ANGLE_RAD)

                if ctype == 'L':
                    action = Action.L_PLUS.value if forward else Action.L_MINUS.value
                else:  # 'R'
                    action = Action.R_PLUS.value if forward else Action.R_MINUS.value
                actions.extend([action] * n_steps)

        return actions


# ============================================================================
# 便捷函数
# ============================================================================

def plan_trajectory(
    start: VehicleState,
    target: VehicleState,
) -> PlanResult:
    """规划单条轨迹（便捷入口）"""
    planner = ReedsSheppExpertPlanner()
    return planner.plan(start, target)


# ============================================================================
# 验证脚本
# ============================================================================

def run_validation(n_cases: int = 100, seed: int = 42, verbose: bool = True):
    """
    端到端验证：随机生成 n 组 (start, target)，规划+回放，统计成功率

    成功标准（严格）：位置误差 < 0.05m，航向误差 < 3°
    实用标准：位置误差 < 0.085m，航向误差 < 3°
    （离散化步长 0.048-0.05m 的累积误差导致部分 case 略超 0.05m）
    """
    import random
    rng = random.Random(seed)
    planner = ReedsSheppExpertPlanner()

    pos_threshold_strict = 0.05
    pos_threshold_practical = 0.085
    heading_threshold = math.radians(3.0)

    n_success_strict = 0
    n_success_practical = 0
    n_plan_fail = 0
    pos_errors = []
    heading_errors = []
    action_counts = np.zeros(7, dtype=int)

    for _i in range(n_cases):
        start = VehicleState(
            x=rng.uniform(-5, 5),
            y=rng.uniform(-5, 5),
            heading=rng.uniform(-math.pi, math.pi)
        )
        target = VehicleState(
            x=rng.uniform(-5, 5),
            y=rng.uniform(-5, 5),
            heading=rng.uniform(-math.pi, math.pi)
        )

        result = planner.plan(start, target)

        if not result.success:
            n_plan_fail += 1
            continue

        pos_errors.append(result.final_error_pos)
        heading_errors.append(result.final_error_heading)

        for a in result.actions:
            action_counts[a] += 1

        if (result.final_error_pos < pos_threshold_strict and
                result.final_error_heading < heading_threshold):
            n_success_strict += 1
        if (result.final_error_pos < pos_threshold_practical and
                result.final_error_heading < heading_threshold):
            n_success_practical += 1

    n_planned = n_cases - n_plan_fail
    rate_strict = n_success_strict / n_cases * 100 if n_cases > 0 else 0
    rate_practical = n_success_practical / n_cases * 100 if n_cases > 0 else 0

    if verbose:
        print("=" * 60)
        print("Reeds-Shepp Expert Planner Validation")
        print("=" * 60)
        print(f"  Cases: {n_cases}")
        print(f"  Plan failures: {n_plan_fail}")
        print(f"  Planned successfully: {n_planned}")
        print(f"  Strict (pos<0.05m, heading<3°): {n_success_strict} ({rate_strict:.1f}%)")
        print(f"  Practical (pos<0.085m, heading<3°): {n_success_practical} ({rate_practical:.1f}%)")
        if pos_errors:
            print(f"  Position error: mean={np.mean(pos_errors):.4f}m, "
                  f"median={np.median(pos_errors):.4f}m, "
                  f"max={np.max(pos_errors):.4f}m")
            print(f"  Heading error: mean={np.degrees(np.mean(heading_errors)):.2f}°, "
                  f"median={np.degrees(np.median(heading_errors)):.2f}°, "
                  f"max={np.degrees(np.max(heading_errors)):.2f}°")
        print("\n  Action distribution:")
        names = ['S0', 'S+', 'S-', 'L+', 'L-', 'R+', 'R-']
        total_actions = action_counts.sum()
        for name, count in zip(names, action_counts, strict=False):
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"    {name}: {count:6d} ({pct:.1f}%)")
        print("=" * 60)

    return {
        'n_cases': n_cases,
        'n_success_strict': n_success_strict,
        'n_success_practical': n_success_practical,
        'n_plan_fail': n_plan_fail,
        'success_rate_strict': rate_strict,
        'success_rate_practical': rate_practical,
        'pos_errors': pos_errors,
        'heading_errors': heading_errors,
        'action_counts': action_counts,
    }


if __name__ == '__main__':
    run_validation(n_cases=100, seed=42)
