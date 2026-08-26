"""Hybrid closed-loop inference: RS expert planner (long-range) + neural network (fine-grained)

The neural network alone can't handle long-range approach (25% success rate), because it enters
states far from training distribution and gets stuck. The RS expert planner can reach the goal
perfectly but lacks obstacle awareness.

Strategy:
  - When dist_to_goal > switch_threshold: use RS expert planner for the next action
  - When dist_to_goal <= switch_threshold: use neural network (DAgger-trained model)
  - Always apply collision detection regardless of controller source

Usage:
    source .venv/bin/activate
    python scripts/hybrid_closed_loop.py --from-dataset 8703 --max-steps 500
    python scripts/hybrid_closed_loop.py --from-dataset 8703 --switch-dist 1.0
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from parking_il_planner.config.model import MODEL_CONFIG
from parking_il_planner.geometry.collision import CollisionDetector, Rectangle
from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.models.planner_model import build_apa_model
from parking_il_planner.planning.actions import Action, ActionExecutor
from parking_il_planner.planning.reeds_shepp import ReedsSheppExpertPlanner
from parking_il_planner.simulation.renderer import SceneRenderer

ACTION_NAMES = ['S0', 'S+', 'S-', 'L+', 'L-', 'R+', 'R-']
EVAL_DIR = 'artifacts/evaluation/closed_loop'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_hybrid_inference(
    start_state: VehicleState,
    target_state: VehicleState,
    obstacles_list: list,
    model: torch.nn.Module,
    planner: ReedsSheppExpertPlanner,
    max_steps: int = 500,
    goal_threshold: float = 0.095,
    angle_threshold: float = 5.0,
    collision_threshold: float = 0.3,
    repeat_limit: int = 200,
    no_progress_limit: int = 50,
    switch_dist: float = 1.5,
    dataset_id=None,
    verbose: bool = True,
) -> dict:
    """Run hybrid closed-loop: RS for approach, NN for fine maneuver."""
    renderer = SceneRenderer()
    executor = ActionExecutor()
    detector = CollisionDetector(safety_margin=0.0)

    current_state = start_state
    action_buffer = [-1, -1]
    last_action = -1
    repeat_count = 0
    best_dist = float('inf')
    no_progress_count = 0
    reached_goal = False
    collision = False
    stuck = False
    rs_steps = 0
    nn_steps = 0

    for step in range(max_steps):
        dist_to_goal = math.sqrt((current_state.x - target_state.x)**2 +
                                 (current_state.y - target_state.y)**2)
        heading_diff = abs(math.degrees(current_state.heading - target_state.heading))
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        # Check goal reached
        if dist_to_goal < goal_threshold and heading_diff < angle_threshold:
            reached_goal = True
            if verbose:
                print(f"  Step {step+1}: GOAL REACHED (dist={dist_to_goal:.3f}m, hdg={heading_diff:.1f}°)")
            break

        # Decide controller: RS for long range, NN for close range
        if dist_to_goal > switch_dist:
            # RS expert planner with collision checking
            plan_result = planner.plan(current_state, target_state)
            rs_action = None
            if plan_result.success and len(plan_result.actions) > 0:
                # Find first non-S0 action from plan
                for a in plan_result.actions:
                    if a != 0:
                        rs_action = a
                        break
                if rs_action is None:
                    rs_action = plan_result.actions[0]

            if rs_action is not None and rs_action != 0:
                # Simulate RS action — check if it causes collision
                sim_result = executor.execute(current_state, Action(rs_action))
                sim_state = sim_result.final_state
                safe = True
                if obstacles_list:
                    col_check = detector.check_vehicle_obstacles(sim_state, obstacles_list)
                    if col_check.collision and col_check.penetration_depth > collision_threshold * 0.5:
                        safe = False

                if safe:
                    pred_action = rs_action
                    controller = "RS"
                    rs_steps += 1
                else:
                    # RS primary action collides — try all 6 movement actions, pick
                    # the one that reduces distance most without collision
                    best_cand = None
                    best_cand_dist = dist_to_goal
                    for cand_a in [1, 2, 3, 4, 5, 6]:
                        sim_r = executor.execute(current_state, Action(cand_a))
                        sim_s = sim_r.final_state
                        cand_safe = True
                        if obstacles_list:
                            cc = detector.check_vehicle_obstacles(sim_s, obstacles_list)
                            if cc.collision and cc.penetration_depth > collision_threshold * 0.5:
                                cand_safe = False
                        if cand_safe:
                            cand_dist = math.sqrt((sim_s.x - target_state.x)**2 +
                                                  (sim_s.y - target_state.y)**2)
                            if cand_dist < best_cand_dist:
                                best_cand_dist = cand_dist
                                best_cand = cand_a

                    if best_cand is not None:
                        pred_action = best_cand
                        controller = "RS(safe)"
                        rs_steps += 1
                    else:
                        # All actions collide — use NN as last resort
                        pred_action = _nn_predict(model, renderer, current_state, target_state,
                                                 obstacles_list, action_buffer)
                        controller = "NN(avoid)"
                        nn_steps += 1
            else:
                # RS planning failed or only S0 — use NN
                pred_action = _nn_predict(model, renderer, current_state, target_state,
                                         obstacles_list, action_buffer)
                controller = "NN(fallback)"
                nn_steps += 1
        else:
            # Neural network (fine-grained maneuvering)
            pred_action = _nn_predict(model, renderer, current_state, target_state,
                                     obstacles_list, action_buffer)
            controller = "NN"
            nn_steps += 1

        if verbose and step < 20:
            print(f"  Step {step+1}: [{controller}] {ACTION_NAMES[pred_action]} (dist={dist_to_goal:.3f}m)")

        # Stop action check
        if pred_action == 0:
            if dist_to_goal < goal_threshold and heading_diff < angle_threshold:
                reached_goal = True
            if verbose:
                print(f"  Step {step+1}: STOP (dist={dist_to_goal:.3f}m, hdg={heading_diff:.1f}°)")
            break

        # Circuit breaker: consecutive same action
        if pred_action == last_action:
            repeat_count += 1
        else:
            repeat_count = 1
            last_action = pred_action
        if repeat_count >= repeat_limit:
            stuck = True
            if verbose:
                print(f"  Step {step+1}: STUCK (repeat {ACTION_NAMES[pred_action]} x{repeat_count})")
            break

        # No-progress detection (more lenient when RS is driving since RS paths
        # legitimately move away from target before converging)
        if dist_to_goal < best_dist - 0.01:
            best_dist = dist_to_goal
            no_progress_count = 0
        else:
            no_progress_count += 1
        effective_limit = no_progress_limit * 3 if "RS" in controller else no_progress_limit
        if no_progress_count >= effective_limit:
            stuck = True
            if verbose:
                print(f"  Step {step+1}: STUCK (no progress for {no_progress_count} steps, best={best_dist:.3f}m)")
            break

        # Execute action
        result = executor.execute(current_state, Action(pred_action))
        current_state = result.final_state
        action_buffer.append(pred_action)

        # Collision detection
        if obstacles_list:
            col_result = detector.check_vehicle_obstacles(current_state, obstacles_list)
            if col_result.collision and col_result.penetration_depth > collision_threshold:
                collision = True
                if verbose:
                    print(f"  Step {step+1}: COLLISION (penetration={col_result.penetration_depth:.3f}m)")
                break

    # Final metrics
    final_dist = math.sqrt((current_state.x - target_state.x)**2 +
                           (current_state.y - target_state.y)**2)
    final_heading = abs(math.degrees(current_state.heading - target_state.heading))
    if final_heading > 180:
        final_heading = 360 - final_heading

    return {
        'dataset_id': dataset_id,
        'reached': reached_goal,
        'collision': collision,
        'stuck': stuck,
        'steps': step + 1 if step < max_steps else max_steps,
        'final_dist': round(final_dist, 4),
        'final_heading_error': round(final_heading, 2),
        'rs_steps': rs_steps,
        'nn_steps': nn_steps,
        'goal_threshold': goal_threshold,
        'angle_threshold': angle_threshold,
        'switch_dist': switch_dist,
    }


def _nn_predict(model, renderer, current_state, target_state, obstacles_list, action_buffer):
    """Get neural network prediction for current state."""
    img = renderer.render(current_state, target_state, obstacles_list)
    img_t = torch.tensor(img[np.newaxis, ...]).to(DEVICE)

    state_vector = None
    action_history = None
    if getattr(model.config, 'use_state_vector', False):
        dx = target_state.x - current_state.x
        dy = target_state.y - current_state.y
        dh = target_state.heading - current_state.heading
        dist = math.sqrt(dx**2 + dy**2)
        state_vector = torch.tensor([[dx, dy, dh, dist, math.cos(dh), math.sin(dh)]],
                                    dtype=torch.float32).to(DEVICE)
    if getattr(model.config, 'use_action_history', False):
        ah = torch.zeros(1, 14)
        if action_buffer[-1] >= 0:
            ah[0, action_buffer[-1]] = 1.0
        if action_buffer[-2] >= 0:
            ah[0, 7 + action_buffer[-2]] = 1.0
        action_history = ah.to(DEVICE)

    with torch.no_grad():
        logits = model(img_t, state_vector=state_vector, action_history=action_history)
        return torch.argmax(logits, -1).item()


def main():
    parser = argparse.ArgumentParser(description='Hybrid closed-loop inference (RS + NN)')
    parser.add_argument('--from-dataset', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--goal-threshold', type=float, default=0.095)
    parser.add_argument('--angle-threshold', type=float, default=5.0)
    parser.add_argument('--collision-threshold', type=float, default=0.3)
    parser.add_argument('--repeat-limit', type=int, default=200)
    parser.add_argument('--no-progress-limit', type=int, default=50)
    parser.add_argument('--switch-dist', type=float, default=1.5,
                       help='Distance threshold to switch from RS to NN (meters)')
    parser.add_argument('--batch', action='store_true', help='Run batch eval on 20 fixed scenarios')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    cp = torch.load('checkpoints/best.pt', map_location=DEVICE, weights_only=False)
    model = build_apa_model(MODEL_CONFIG, device=DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    print(f"  Model: {model.get_num_params():,} params")

    # RS planner
    planner = ReedsSheppExpertPlanner()

    if args.batch:
        _run_batch(model, planner, args)
    else:
        _run_single(model, planner, args)


def _run_single(model, planner, args):
    """Run single scenario."""
    from parking_il_planner.data.dataset import ParkingDataset
    ds = ParkingDataset('data/generated', load_images=False)

    idx = args.from_dataset if args.from_dataset is not None else 8703
    s = ds.states[idx]
    start_state = VehicleState(x=s[0], y=s[1], heading=s[2])
    target_state = VehicleState(x=s[3], y=s[4], heading=s[5])

    obstacles_list = []
    for obs in ds.obstacles[idx]:
        if obs[0] != 0:
            obstacles_list.append(Rectangle(
                center_x=obs[0], center_y=obs[1],
                length=obs[2], width=obs[3], heading=obs[4]))

    dist = math.sqrt((start_state.x - target_state.x)**2 + (start_state.y - target_state.y)**2)
    print(f"\nScenario #{idx}: dist={dist:.2f}m, obstacles={len(obstacles_list)}")
    print(f"  Switch: RS when dist > {args.switch_dist}m, NN when closer")

    result = run_hybrid_inference(
        start_state, target_state, obstacles_list, model, planner,
        max_steps=args.max_steps,
        goal_threshold=args.goal_threshold,
        angle_threshold=args.angle_threshold,
        collision_threshold=args.collision_threshold,
        repeat_limit=args.repeat_limit,
        no_progress_limit=args.no_progress_limit,
        switch_dist=args.switch_dist,
        dataset_id=idx,
        verbose=True,
    )

    print(f"\nResult: {'REACHED' if result['reached'] else 'COLLISION' if result['collision'] else 'STUCK' if result['stuck'] else 'TIMEOUT'}")
    print(f"  Steps: {result['steps']} (RS={result['rs_steps']}, NN={result['nn_steps']})")
    print(f"  Final: dist={result['final_dist']:.4f}m, heading={result['final_heading_error']:.1f}°")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)


def _run_batch(model, planner, args):
    """Run batch evaluation on 20 fixed scenarios."""
    from parking_il_planner.data.dataset import ParkingDataset
    ds = ParkingDataset('data/generated', load_images=False)

    EVAL_INDICES = [0, 957, 2240, 3313, 4109, 5045, 6244, 7585, 8652, 9985,
                    10904, 12233, 13318, 14527, 15656, 16832, 17835, 18978, 20214, 21329]

    print(f"\n{'='*70}")
    print(f"HYBRID BATCH EVALUATION (RS + NN, switch at {args.switch_dist}m)")
    print(f"{'='*70}")
    print(f"  Scenarios: {len(EVAL_INDICES)}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Goal: {args.goal_threshold}m / {args.angle_threshold}°")
    print(f"  Switch: RS when dist > {args.switch_dist}m")

    results = []
    for i, idx in enumerate(EVAL_INDICES):
        s = ds.states[idx]
        start_state = VehicleState(x=s[0], y=s[1], heading=s[2])
        target_state = VehicleState(x=s[3], y=s[4], heading=s[5])

        obstacles_list = []
        for obs in ds.obstacles[idx]:
            if obs[0] != 0:
                obstacles_list.append(Rectangle(
                    center_x=obs[0], center_y=obs[1],
                    length=obs[2], width=obs[3], heading=obs[4]))

        dist = math.sqrt((start_state.x - target_state.x)**2 + (start_state.y - target_state.y)**2)

        r = run_hybrid_inference(
            start_state, target_state, obstacles_list, model, planner,
            max_steps=args.max_steps,
            goal_threshold=args.goal_threshold,
            angle_threshold=args.angle_threshold,
            collision_threshold=args.collision_threshold,
            repeat_limit=args.repeat_limit,
            no_progress_limit=args.no_progress_limit,
            switch_dist=args.switch_dist,
            dataset_id=idx,
            verbose=False,
        )
        results.append(r)

        status = "REACHED" if r['reached'] else ("COLLISION" if r['collision'] else ("STUCK" if r['stuck'] else "TIMEOUT"))
        print(f"  [{i+1:2d}/20] #{idx:5d} dist={dist:.2f}m → {status:9s} "
              f"(steps={r['steps']:3d}, RS={r['rs_steps']:3d}, NN={r['nn_steps']:3d}, "
              f"final={r['final_dist']:.3f}m/{r['final_heading_error']:.1f}°)")

    # Summary
    n = len(results)
    reached = sum(1 for r in results if r['reached'])
    collisions = sum(1 for r in results if r['collision'])
    stuck_count = sum(1 for r in results if r['stuck'])
    timeout = n - reached - collisions - stuck_count

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Reached:    {reached}/{n} ({100*reached/n:.0f}%)")
    print(f"  Collision:  {collisions}/{n}")
    print(f"  Stuck:      {stuck_count}/{n}")
    print(f"  Timeout:    {timeout}/{n}")
    print(f"\n  SUCCESS RATE: {100*reached/n:.1f}% (target: >60%)")

    if reached > 0:
        reached_results = [r for r in results if r['reached']]
        avg_steps = np.mean([r['steps'] for r in reached_results])
        avg_rs = np.mean([r['rs_steps'] for r in reached_results])
        avg_nn = np.mean([r['nn_steps'] for r in reached_results])
        print(f"\n  Among reached ({reached} scenarios):")
        print(f"    Avg steps: {avg_steps:.1f} (RS={avg_rs:.1f}, NN={avg_nn:.1f})")
        print(f"    Avg final dist: {np.mean([r['final_dist'] for r in reached_results]):.4f}m")
        print(f"    Avg heading error: {np.mean([r['final_heading_error'] for r in reached_results]):.2f}°")

    # Save
    os.makedirs(EVAL_DIR, exist_ok=True)
    batch_path = os.path.join(EVAL_DIR, 'hybrid_batch_results.json')
    with open(batch_path, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'total': n, 'reached': reached, 'collision': collisions,
                'stuck': stuck_count, 'timeout': timeout,
                'success_rate': reached / n, 'switch_dist': args.switch_dist,
            }
        }, f, indent=2)
    print(f"\n  Saved: {batch_path}")


if __name__ == "__main__":
    main()
