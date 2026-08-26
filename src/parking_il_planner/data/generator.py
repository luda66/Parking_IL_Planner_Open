"""
场景驱动数据生成器

用结构化泊车场景 + Reeds-Shepp 规划器生成训练数据。
替代旧的模板驱动方式（data_generator.py），产出更真实、
动作更平衡的训练样本。

流程：
1. 随机选择场景类型（垂直/平行/斜方位）
2. 生成场景实例（车位、障碍物、初始位姿、目标位姿）
3. RS 规划器生成动作序列
4. 碰撞检测验证轨迹安全
5. 渲染 occupancy grid
6. 保存为训练样本

用法：
    source .venv/bin/activate
    python -m data.synthesis.scenario_data_generator --num-trajectories 3000
    python -m data.synthesis.scenario_data_generator --num-trajectories 3000 --output-dir data/generated
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from parking_il_planner.geometry.collision import CollisionDetector
from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.planning.actions import Action, ActionExecutor
from parking_il_planner.planning.reeds_shepp import ReedsSheppExpertPlanner
from parking_il_planner.scenarios.angled import AngledParkingScenario
from parking_il_planner.scenarios.parallel import ParallelParkingScenario
from parking_il_planner.scenarios.perpendicular import PerpendicularParkingScenario
from parking_il_planner.scenarios.straight import StraightApproachScenario
from parking_il_planner.simulation.renderer import SceneRenderer


@dataclass
class GenerationStats:
    """生成统计"""
    n_attempted: int = 0
    n_success: int = 0
    n_plan_fail: int = 0
    n_collision: int = 0
    n_too_short: int = 0
    n_not_reverse: int = 0
    total_samples: int = 0


class ScenarioDataGenerator:
    """
    场景驱动数据生成器

    配比：垂直 30%, 平行 30%, 斜方位 15%, 直线进退 25%
    """

    def __init__(
        self,
        safety_margin: float = 0.0,
        min_trajectory_length: int = 5,
        max_trajectory_length: int = 600,
        collision_penetration_threshold: float = 0.2,
    ):
        self.planner = ReedsSheppExpertPlanner()
        self.executor = ActionExecutor()
        self.detector = CollisionDetector(safety_margin=safety_margin)
        self.renderer = SceneRenderer()
        self.safety_margin = safety_margin
        self.min_traj_len = min_trajectory_length
        self.max_traj_len = max_trajectory_length
        self.collision_penetration_threshold = collision_penetration_threshold

    def generate(
        self,
        num_trajectories: int = 1000,
        seed: int = 42,
        scenario_ratios: dict = None,
    ) -> Tuple[List[dict], GenerationStats]:
        """
        生成训练数据

        Args:
            num_trajectories: 目标轨迹数量
            seed: 随机种子
            scenario_ratios: 场景配比，默认 {'perp': 0.4, 'parallel': 0.4, 'angled': 0.2}

        Returns:
            (samples_list, stats)
            samples_list: 每个元素是一条轨迹的所有样本
        """
        if scenario_ratios is None:
            scenario_ratios = {'perp': 0.40, 'parallel': 0.35, 'angled': 0.25}

        rng = random.Random(seed)
        stats = GenerationStats()
        all_trajectories = []

        # 按配比分配轨迹数
        n_perp = int(num_trajectories * scenario_ratios.get('perp', 0.4))
        n_parallel = int(num_trajectories * scenario_ratios.get('parallel', 0.35))
        n_angled = num_trajectories - n_perp - n_parallel

        # 生成场景任务列表
        tasks = []
        tasks.extend([('perp', i) for i in range(n_perp)])
        tasks.extend([('parallel', i) for i in range(n_parallel)])
        tasks.extend([('angled', i) for i in range(n_angled)])
        rng.shuffle(tasks)

        for task_idx, (scenario_type, _variant_seed) in enumerate(tasks):
            stats.n_attempted += 1
            traj_seed = seed * 10000 + task_idx

            result = self._generate_one_trajectory(
                scenario_type, traj_seed, rng
            )

            if result is None:
                continue

            traj_samples, fail_reason = result
            if traj_samples is None:
                if fail_reason == 'plan':
                    stats.n_plan_fail += 1
                elif fail_reason == 'collision':
                    stats.n_collision += 1
                elif fail_reason == 'short':
                    stats.n_too_short += 1
                elif fail_reason == 'not_reverse':
                    stats.n_not_reverse += 1
                continue

            stats.n_success += 1
            stats.total_samples += len(traj_samples)
            all_trajectories.append(traj_samples)

            if stats.n_success % 100 == 0:
                print(f"  Generated {stats.n_success}/{num_trajectories} trajectories "
                      f"({stats.total_samples} samples)")

        return all_trajectories, stats

    def _generate_one_trajectory(
        self, scenario_type: str, seed: int, rng: random.Random
    ):
        """
        生成单条轨迹

        Returns:
            (samples, None) on success
            (None, fail_reason) on failure
        """
        # 创建场景
        scenario = self._create_scenario(scenario_type)
        scenario.generate(seed=seed)

        initial_state = scenario.get_initial_state()
        target_pose = scenario.get_target_pose()
        target_state = target_pose.to_state()
        obstacles = scenario.get_rectangles()

        # 检查初始位置是否与障碍物碰撞
        if obstacles:
            for obs in obstacles:
                check = self.detector.check_vehicle_obstacle(initial_state, obs)
                if check.collision and check.penetration_depth > self.collision_penetration_threshold:
                    return (None, 'collision')

        # RS 规划
        plan_result = self.planner.plan(initial_state, target_state)
        if not plan_result.success:
            return (None, 'plan')

        actions = plan_result.actions
        if len(actions) < self.min_traj_len:
            return (None, 'short')
        if len(actions) > self.max_traj_len:
            actions = actions[:self.max_traj_len]

        # 回放轨迹，检测严重穿透
        states = [initial_state.copy()]
        current = initial_state.copy()
        max_penetration = 0.0

        for a in actions:
            if a == Action.S0.value:
                break
            result = self.executor.execute(current, Action(a))
            current = result.final_state

            if obstacles:
                for obs in obstacles:
                    check = self.detector.check_vehicle_obstacle(current, obs)
                    if check.collision:
                        max_penetration = max(max_penetration, check.penetration_depth)
            states.append(current.copy())

        if max_penetration > self.collision_penetration_threshold:
            return (None, 'collision')

        # 尾入验证：轨迹后半段倒车动作(S-=2, L-=4, R-=6)占比须>50%
        action_list_no_s0 = [a for a in actions if a != Action.S0.value]
        if len(action_list_no_s0) > 0:
            half_idx = len(action_list_no_s0) // 2
            tail_actions = action_list_no_s0[half_idx:]
            if len(tail_actions) > 0:
                reverse_actions = {Action.S_MINUS.value, Action.L_MINUS.value, Action.R_MINUS.value}
                reverse_count = sum(1 for a in tail_actions if a in reverse_actions)
                if reverse_count / len(tail_actions) < 0.5:
                    return (None, 'not_reverse')

        # 构建样本：每步 = (state, target_state, obstacles, action)
        # 不含最后一步 S0 的状态（S0 样本单独处理）
        samples = []
        action_list = [a for a in actions if a != Action.S0.value]

        for i, action in enumerate(action_list):
            if i >= len(states):
                break
            samples.append({
                'state': states[i],
                'target_state': target_state,
                'obstacles': obstacles,
                'action': action,
            })

        # 追加 S0 样本：目标状态和目标附近微扰动状态
        if len(states) > 1:
            # 精确到达目标时应停止
            samples.append({
                'state': states[-1],
                'target_state': target_state,
                'obstacles': obstacles,
                'action': Action.S0.value,
            })
            # 目标附近微扰动也应停止（±0.03m, ±1.5°内）
            import random as _rnd
            _rng = _rnd.Random(seed)
            for _ in range(7):
                dx = _rng.uniform(-0.03, 0.03)
                dy = _rng.uniform(-0.03, 0.03)
                dh = _rng.uniform(-math.radians(1.5), math.radians(1.5))
                perturbed = VehicleState(
                    x=target_state.x + dx,
                    y=target_state.y + dy,
                    heading=target_state.heading + dh,
                )
                samples.append({
                    'state': perturbed,
                    'target_state': target_state,
                    'obstacles': obstacles,
                    'action': Action.S0.value,
                })

        return (samples, None)

    def _create_scenario(self, scenario_type: str):
        if scenario_type == 'perp':
            return PerpendicularParkingScenario()
        elif scenario_type == 'parallel':
            return ParallelParkingScenario()
        elif scenario_type == 'angled':
            return AngledParkingScenario()
        elif scenario_type == 'straight':
            return StraightApproachScenario()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")


def synthesize_and_save(
    num_trajectories: int = 1000,
    output_dir: str = "data/generated",
    seed: int = 42,
    max_samples: int | None = None,
):
    """
    生成数据并保存为 npy 文件（使用内存映射避免OOM）

    Args:
        num_trajectories: 目标轨迹数
        output_dir: 输出目录
        seed: 随机种子
        max_samples: optional cap for lightweight smoke datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Scenario-Driven Data Generator")
    print("=" * 60)
    print(f"  Target trajectories: {num_trajectories}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {seed}")
    print()

    generator = ScenarioDataGenerator()
    t0 = time.time()

    trajectories, stats = generator.generate(num_trajectories, seed=seed)

    # 展平所有轨迹为样本列表
    all_samples = []
    for traj in trajectories:
        all_samples.extend(traj)

    if max_samples is not None and len(all_samples) > max_samples:
        print(f"  Capping rendered samples: {len(all_samples)} -> {max_samples}")
        all_samples = all_samples[:max_samples]

    elapsed = time.time() - t0
    print(f"\n  Generation done in {elapsed:.1f}s")
    print(f"  Attempted: {stats.n_attempted}")
    print(f"  Success: {stats.n_success}")
    print(f"  Plan failures: {stats.n_plan_fail}")
    print(f"  Collisions: {stats.n_collision}")
    print(f"  Too short: {stats.n_too_short}")
    print(f"  Not reverse-in: {stats.n_not_reverse}")
    print(f"  Total samples: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No samples generated!")
        return 0, 0

    n_samples = len(all_samples)
    img_path = os.path.join(output_dir, "images.npy")

    # 先保存 actions/states/obstacles（小数据）
    actions = np.array([s['action'] for s in all_samples], dtype=np.int32)
    states_data = np.array([
        [s['state'].x, s['state'].y, s['state'].heading,
         s['target_state'].x, s['target_state'].y, s['target_state'].heading]
        for s in all_samples
    ], dtype=np.float32)

    obs_data = []
    for s in all_samples:
        obs_list = []
        for obs in s['obstacles']:
            obs_list.append([obs.center_x, obs.center_y, obs.length, obs.width, obs.heading])
        while len(obs_list) < 20:
            obs_list.append([0, 0, 0, 0, 0])
        obs_data.append(obs_list[:20])
    obs_data = np.array(obs_data, dtype=np.float32)

    np.save(os.path.join(output_dir, "actions.npy"), actions)
    np.save(os.path.join(output_dir, "states.npy"), states_data)
    np.save(os.path.join(output_dir, "obstacles.npy"), obs_data)

    # 渲染图像（使用内存映射文件分块写入）
    renderer = SceneRenderer()
    print(f"\n  Rendering {n_samples} images to disk (memory-mapped)...")
    t_render = time.time()

    # 创建内存映射 npy 文件
    img_shape = (n_samples, 3, 384, 384)
    # 写 npy header
    fp = np.lib.format.open_memmap(
        img_path, mode='w+', dtype=np.uint8, shape=img_shape
    )

    CHUNK = 1000
    for chunk_start in range(0, n_samples, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_samples)
        for i in range(chunk_start, chunk_end):
            sample = all_samples[i]
            img = renderer.render(
                sample['state'], sample['target_state'], sample['obstacles']
            )
            fp[i] = (img * 255).astype(np.uint8)

        if chunk_end % 5000 == 0 or chunk_end == n_samples:
            print(f"    Rendered {chunk_end}/{n_samples}")

    del fp  # flush to disk
    render_time = time.time() - t_render
    print(f"    Done: {n_samples} images in {render_time:.1f}s "
          f"({n_samples/render_time:.0f} img/s)")

    # 统计
    print("\n  Action distribution:")
    names = ['S0', 'S+', 'S-', 'L+', 'L-', 'R+', 'R-']
    counts = np.bincount(actions, minlength=7)
    for name, count in zip(names, counts, strict=False):
        pct = count / len(actions) * 100
        print(f"    {name}: {count:6d} ({pct:.1f}%)")

    # 距离统计
    dists = np.sqrt(
        (states_data[:, 0] - states_data[:, 3]) ** 2 +
        (states_data[:, 1] - states_data[:, 4]) ** 2
    )
    print("\n  Start-to-target distance:")
    print(f"    min={dists.min():.2f}m, mean={dists.mean():.2f}m, max={dists.max():.2f}m")

    img_size_mb = n_samples * 3 * 384 * 384 / 1024 / 1024
    print(f"\n  Files saved to: {output_dir}/")
    print(f"    images.npy: {img_size_mb:.0f} MB")
    print(f"    actions.npy: {actions.nbytes / 1024:.0f} KB")
    print(f"    states.npy: {states_data.nbytes / 1024:.0f} KB")
    print(f"    obstacles.npy: {obs_data.nbytes / 1024 / 1024:.1f} MB")
    print("=" * 60)

    return stats.n_success, len(all_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scenario-driven data generator')
    parser.add_argument('--num-trajectories', type=int, default=1000,
                       help='Number of trajectories to generate')
    parser.add_argument('--output-dir', type=str, default='data/generated',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    synthesize_and_save(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        seed=args.seed,
    )
