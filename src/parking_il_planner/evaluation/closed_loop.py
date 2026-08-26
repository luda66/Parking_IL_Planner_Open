"""APA Closed-Loop Inference - Model Actually Drives the Vehicle

================================================================================
QUICK START (copy-paste ready):
================================================================================

# STEP 1: Activate environment
source .venv/bin/activate
cd Parking_IL_Planner_Open

# STEP 2: Pick one command below

================================================================================
USAGE EXAMPLES:
================================================================================

--- From Dataset (recommended - real parking scenarios) ---

# Use initial state from dataset sample #8703 (longest trajectory):
python scripts/closed_loop_inference.py --from-dataset 8703

# Try different starting points:
python scripts/closed_loop_inference.py --from-dataset 20344  # Scene 2 start
python scripts/closed_loop_inference.py --from-dataset 16000  # Scene 3 start
python scripts/closed_loop_inference.py --from-dataset 0      # Scene 4 start
python scripts/closed_loop_inference.py --from-dataset 8622   # Scene 5 start

# Allow more steps for longer parking maneuvers:
python scripts/closed_loop_inference.py --from-dataset 8703 --max-steps 500

# Custom goal threshold (default 0.3m):
python scripts/closed_loop_inference.py --from-dataset 8703 --goal-threshold 0.5

--- Custom Scenarios ---

# Custom start and target (heading in degrees):
python scripts/closed_loop_inference.py --start-x 0 --start-y 0 --start-h 0 \
    --target-x 5 --target-y 3 --target-h -90

# More steps for complex maneuvers:
python scripts/closed_loop_inference.py --max-steps 200

--- Viewing Results ---

# Open GIF animation (normal speed):
eog artifacts/evaluation/closed_loop/closed_loop_result.gif

# Open GIF animation (fast version, every 10th frame):
eog artifacts/evaluation/closed_loop/closed_loop_result_fast.gif

# Open individual frames with arrow-key navigation:
eog artifacts/evaluation/closed_loop/

# List all closed-loop outputs:
ls -lh artifacts/evaluation/closed_loop/

# View trajectory plot:
eog artifacts/evaluation/closed_loop/trajectory.png

# Check result summary from terminal output:
python scripts/closed_loop_inference.py --from-dataset 8703 2>&1 | grep -A10 "CLOSED-LOOP RESULTS"

================================================================================
HOW CLOSED-LOOP INFERENCE WORKS:
================================================================================

This is CLOSED-LOOP (the real thing):

Loop:
  1. Render occupancy grid from current_state (includes obstacles!)
  2. Model predicts action (S0/S+/S-/L+/L-/R+/R-)
  3. Execute action → update vehicle state (position + heading)
  4. Check: reached goal? collision? max steps?
  5. Repeat with NEW state

Key: current_state changes EVERY step. Target state NEVER changes.
Obstacles from dataset are rendered in every frame for visualization.

Unlike run_inference.py (open-loop, classifies pre-existing frames),
this script makes the model ACTUALLY DRIVE from start to goal.

================================================================================
OUTPUT FILES:
================================================================================

artifacts/evaluation/closed_loop/
├── closed_loop_result.gif        # Normal speed (all frames, 600ms each)
├── closed_loop_result_fast.gif   # Fast speed (every 10th frame, 300ms each)
├── trajectory.png                # Top-down trajectory view
├── step_001.png                  # Individual frame 1 (with obstacles)
├── step_002.png                  # Individual frame 2 (with obstacles)
└── ...                           # One PNG per step

================================================================================
TROUBLESHOOTING:
================================================================================

"ModuleNotFoundError: No module named 'torch'"
  → conda activate apa_planner

"Timeout after 10 minutes"
  → 500 steps takes ~5-8 minutes on GPU
  → Run with: timeout 600 python scripts/closed_loop_inference.py ...

"Reached goal: NO, dist=X.XXm"
  → Model may not have learned this type of maneuver
  → Try a different starting point from dataset
  → Increase max-steps to allow more time

"""

import argparse
import math
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from parking_il_planner.config.model import MODEL_CONFIG
from parking_il_planner.geometry.collision import CollisionDetector, Rectangle
from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.models.planner_model import build_apa_model
from parking_il_planner.planning.actions import Action, ActionExecutor
from parking_il_planner.simulation.renderer import SceneRenderer

ACTION_NAMES = ['S0', 'S+', 'S-', 'L+', 'L-', 'R+', 'R-']
ACTION_COLORS_MAP = {0: '#808080', 1: '#00AA00', 2: '#DD0000', 3: '#0066FF', 4: '#9900AA', 5: '#CCAA00', 6: '#00AAAA'}
EVAL_DIR = 'artifacts/evaluation'
CLOSED_LOOP_DIR = os.path.join(EVAL_DIR, 'closed_loop')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    global CLOSED_LOOP_DIR

    print("=" * 70)
    print("APA Closed-Loop Inference")
    print("=" * 70)

    # ============================================================================
    # Parse arguments
    # ============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-dataset', type=int, default=None,
                       help='Use start/target from a dataset sample index')
    parser.add_argument('--start-x', type=float, default=0.0)
    parser.add_argument('--start-y', type=float, default=0.0)
    parser.add_argument('--start-h', type=float, default=0.0, help='Heading in degrees')
    parser.add_argument('--target-x', type=float, default=5.0)
    parser.add_argument('--target-y', type=float, default=3.0)
    parser.add_argument('--target-h', type=float, default=-90.0, help='Heading in degrees')
    parser.add_argument('--max-steps', type=int, default=60)
    parser.add_argument('--goal-threshold', type=float, default=0.095, help='Meters to consider "arrived"')
    parser.add_argument('--angle-threshold', type=float, default=5.0, help='Degrees heading error to consider "arrived"')
    parser.add_argument('--repeat-limit', type=int, default=200, help='Max consecutive same action before stuck')
    parser.add_argument('--no-progress-limit', type=int, default=50, help='Steps without distance decrease before stuck')
    parser.add_argument('--collision-threshold', type=float, default=0.25, help='Penetration depth (m) before declaring collision')
    parser.add_argument('--data-dir', default='data/generated', help='Generated dataset directory')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt', help='Model checkpoint')
    parser.add_argument('--output-dir', default='artifacts/evaluation/closed_loop', help='Output directory')
    args = parser.parse_args()

    CLOSED_LOOP_DIR = args.output_dir
    os.makedirs(CLOSED_LOOP_DIR, exist_ok=True)

    # ============================================================================
    # Load model
    # ============================================================================
    print("\nLoading model...")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}; train a model or provide --checkpoint")
    cp = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model = build_apa_model(MODEL_CONFIG, device=DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    print(f"  OK: {model.get_num_params():,} params")

    # ============================================================================
    # Setup scenario
    # ============================================================================
    print("\nSetting up scenario...")

    if args.from_dataset is not None:
        from parking_il_planner.data.dataset import ParkingDataset
        ds = ParkingDataset(args.data_dir, load_images=True)
        s = ds.states[args.from_dataset]
        start_state = VehicleState(x=s[0], y=s[1], heading=s[2])  # heading already in radians
        target_state = VehicleState(x=s[3], y=s[4], heading=s[5])  # heading already in radians

        # Load obstacles from dataset (for visualization)
        obstacles_list = []
        for obs in ds.obstacles[args.from_dataset]:
            if obs[0] != 0:  # Non-empty obstacle
                obstacles_list.append(Rectangle(
                    center_x=obs[0], center_y=obs[1],
                    length=obs[2], width=obs[3], heading=obs[4]
                ))

        dataset_id = args.from_dataset
        print(f"  From dataset sample #{dataset_id}")
        print(f"  Obstacles: {len(obstacles_list)}")
    else:
        start_state = VehicleState(x=args.start_x, y=args.start_y, heading=math.radians(args.start_h))
        target_state = VehicleState(x=args.target_x, y=args.target_y, heading=math.radians(args.target_h))
        obstacles_list = []
        dataset_id = "custom"
        print("  Custom scenario (no obstacles)")

    print(f"  Start: ({start_state.x:.2f}, {start_state.y:.2f}, {math.degrees(start_state.heading):.0f}°)")
    print(f"  Target: ({target_state.x:.2f}, {target_state.y:.2f}, {math.degrees(target_state.heading):.0f}°)")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Goal threshold: {args.goal_threshold}m")

    # ============================================================================
    # Closed-loop inference
    # ============================================================================
    print("\n" + "=" * 50)
    print("Starting closed-loop inference...")
    print("=" * 50)

    renderer = SceneRenderer()
    executor = ActionExecutor()
    detector = CollisionDetector(safety_margin=0.0)

    current_state = start_state
    history = []  # Store (state, action, confidence) for visualization
    gif_images = []  # All frames for normal GIF
    gif_fast_images = []  # Every 10th frame for fast GIF
    frame_num = 0
    reached_goal = False
    collision = False
    stuck = False
    last_action = -1
    repeat_count = 0
    best_dist = float('inf')
    no_progress_count = 0
    action_buffer = [-1, -1]  # last 2 predicted actions (-1 = no history yet)

    for step in range(args.max_steps):
        frame_num = step + 1

        # 1. Render current state → occupancy grid (WITH obstacles!)
        img = renderer.render(current_state, target_state, obstacles_list)
        dist_to_goal = math.sqrt((current_state.x - target_state.x)**2 + (current_state.y - target_state.y)**2)
        heading_diff = abs(math.degrees(current_state.heading - target_state.heading))
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        # 2. Model inference: predict action from occupancy grid
        # Note: renderer returns float32 [0, 1], NO need to divide by 255!
        img_t = torch.tensor(img[np.newaxis, ...]).to(DEVICE)

        # Compute auxiliary inputs if model supports them
        state_vector = None
        action_history = None
        if getattr(model.config, 'use_state_vector', False):
            dx = target_state.x - current_state.x
            dy = target_state.y - current_state.y
            dh = target_state.heading - current_state.heading
            dist = math.sqrt(dx**2 + dy**2)
            state_vector = torch.tensor([[dx, dy, dh, dist, math.cos(dh), math.sin(dh)]], dtype=torch.float32).to(DEVICE)
        if getattr(model.config, 'use_action_history', False):
            ah = torch.zeros(1, 14)
            if action_buffer[-1] >= 0:
                ah[0, action_buffer[-1]] = 1.0
            if action_buffer[-2] >= 0:
                ah[0, 7 + action_buffer[-2]] = 1.0
            action_history = ah.to(DEVICE)

        with torch.no_grad():
            logits = model(img_t, state_vector=state_vector, action_history=action_history)
            probs = torch.softmax(logits, -1)
            pred_action = torch.argmax(logits, -1).item()
            confidence = probs[0, pred_action].item() * 100

        # 3. Create visualization frame (shows CURRENT state and predicted action)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        combined = img.transpose(1, 2, 0)
        ax.imshow(combined)
        ax.set_title(
            f'Step {frame_num}\n'
            f'Action: {ACTION_NAMES[pred_action]} ({confidence:.0f}%)\n'
            f'Dist: {dist_to_goal:.2f}m',
            fontsize=10, color='#00AA00', fontweight='bold'
        )
        ax.axis('off')
        plt.tight_layout()

        frame_path = os.path.join(CLOSED_LOOP_DIR, f'step_{frame_num:03d}.png')
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        plt.close(fig)

        # PIL for GIF
        pil_img = Image.open(frame_path).resize((400, 400), Image.LANCZOS)
        gif_images.append(pil_img)

        # Fast GIF: save every 10th frame
        if frame_num % 10 == 1:
            gif_fast_images.append(pil_img.copy())

        print(f"  Step {frame_num}: {ACTION_NAMES[pred_action]} (conf={confidence:.0f}%, dist={dist_to_goal:.2f}m)")

        # 4. Check if reached goal (AFTER saving frame)
        reached = dist_to_goal < args.goal_threshold and heading_diff < args.angle_threshold
        if reached:
            print(f"  ★ GOAL REACHED! (dist={dist_to_goal:.3f}m, heading_diff={heading_diff:.1f}°)")
            reached_goal = True
            break

        # 5. Check for stop action
        if pred_action == 0:  # S0 = Stop
            print(f"  Step {frame_num}: Model predicts STOP (confidence: {confidence:.0f}%)")
            break

        # 6. Circuit breaker: consecutive same action OR no progress
        if pred_action == last_action:
            repeat_count += 1
        else:
            repeat_count = 1
            last_action = pred_action
        if repeat_count >= args.repeat_limit:
            print(f"  ⚠ STUCK: action {ACTION_NAMES[pred_action]} repeated {repeat_count} times")
            stuck = True
            break

        if dist_to_goal < best_dist - 0.01:
            best_dist = dist_to_goal
            no_progress_count = 0
        else:
            no_progress_count += 1
        if no_progress_count >= args.no_progress_limit:
            print(f"  ⚠ STUCK: no progress for {no_progress_count} steps (best_dist={best_dist:.3f}m)")
            stuck = True
            break

        # 7. Execute action → update vehicle state (this is the closed-loop!)
        action = Action(pred_action)
        result = executor.execute(current_state, action)
        current_state = result.final_state

        # 8. Collision detection (with penetration threshold)
        if obstacles_list:
            col_result = detector.check_vehicle_obstacles(current_state, obstacles_list)
            if col_result.collision and col_result.penetration_depth > args.collision_threshold:
                print(f"  ✕ COLLISION at step {frame_num} (penetration={col_result.penetration_depth:.3f}m > threshold={args.collision_threshold}m)")
                collision = True
                break

        # 9. Update action buffer for next step
        action_buffer.append(pred_action)

        # 10. Record history
        history.append((current_state, pred_action, confidence))

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 50)
    print("CLOSED-LOOP RESULTS")
    print("=" * 50)

    final_dist = math.sqrt((current_state.x - target_state.x)**2 + (current_state.y - target_state.y)**2)
    final_heading_diff = abs(math.degrees(current_state.heading - target_state.heading))
    if final_heading_diff > 180:
        final_heading_diff = 360 - final_heading_diff

    print(f"  Dataset ID: {dataset_id}")
    print(f"  Start: ({start_state.x:.2f}, {start_state.y:.2f}, {math.degrees(start_state.heading):.0f}°)")
    print(f"  Target: ({target_state.x:.2f}, {target_state.y:.2f}, {math.degrees(target_state.heading):.0f}°)")
    print(f"  Steps executed: {frame_num}")
    print(f"  Reached goal: {'YES' if reached_goal else 'NO'}")
    print(f"  Collision: {'YES' if collision else 'NO'}")
    print(f"  Stuck: {'YES' if stuck else 'NO'}")
    print(f"  Final position: ({current_state.x:.2f}, {current_state.y:.2f}, {math.degrees(current_state.heading):.0f}°)")
    print(f"  Final dist to goal: {final_dist:.3f}m")
    print(f"  Final heading diff: {final_heading_diff:.1f}°")
    if history:
        print(f"  Action sequence: {' -> '.join([ACTION_NAMES[h[1]] for h in history[:30]])}")
        if len(history) > 30:
            print(f"    ... ({len(history)} total)")

    # Save structured result JSON
    import json
    result_json = {
        'dataset_id': dataset_id,
        'reached': reached_goal,
        'collision': collision,
        'stuck': stuck,
        'steps': frame_num,
        'final_dist': round(final_dist, 4),
        'final_heading_error': round(final_heading_diff, 2),
        'goal_threshold': args.goal_threshold,
        'angle_threshold': args.angle_threshold,
    }
    json_path = os.path.join(CLOSED_LOOP_DIR, 'result.json')
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\n  Result JSON: {json_path}")

    # ============================================================================
    # Create GIFs (normal + fast)
    # ============================================================================
    if gif_images:
        # Normal speed GIF (all frames, 600ms each)
        gif_path = os.path.join(CLOSED_LOOP_DIR, 'closed_loop_result.gif')
        gif_images[0].save(
            gif_path, save_all=True, append_images=gif_images[1:],
            duration=600, loop=0, optimize=True
        )
        print(f"\n  Normal GIF: {gif_path} ({len(gif_images)} frames, 600ms)")

        # Fast speed GIF (every 10th frame, 300ms each)
        if gif_fast_images:
            gif_fast_path = os.path.join(CLOSED_LOOP_DIR, 'closed_loop_result_fast.gif')
            gif_fast_images[0].save(
                gif_fast_path, save_all=True, append_images=gif_fast_images[1:],
                duration=300, loop=0, optimize=True
            )
            print(f"  Fast GIF:  {gif_fast_path} ({len(gif_fast_images)} frames, 300ms)")

        print(f"  Individual frames: {CLOSED_LOOP_DIR}/step_001.png ~ step_{frame_num:03d}.png")

    # ============================================================================
    # Create trajectory plot
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot trajectory
    traj_x = [start_state.x] + [h[0].x for h in history]
    traj_y = [start_state.y] + [h[0].y for h in history]
    ax.plot(traj_x, traj_y, 'b-', lw=2, alpha=0.7, label='Trajectory')
    ax.plot(traj_x[0], traj_y[0], 'go', ms=12, label='Start')
    ax.plot(traj_x[-1], traj_y[-1], 'rs', ms=10, label='End')
    ax.plot(target_state.x, target_state.y, 'k*', ms=20, label='Target')

    # Color-code actions
    for i, h in enumerate(history):
        state, action, conf = h
        color = ACTION_COLORS_MAP.get(action, 'gray')
        ax.plot(state.x, state.y, 'o', color=color, markersize=5, alpha=0.7)
        if i % 5 == 0:
            ax.annotate(f'{ACTION_NAMES[action]}', (state.x, state.y), fontsize=6, ha='center', va='bottom')

    ax.set_title(f'Closed-Loop Trajectory (Dataset ID: {dataset_id})\nSteps: {frame_num}, Final dist: {final_dist:.2f}m, Goal: {"Reached" if reached_goal else "Not Reached"}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()
    plt.savefig(os.path.join(CLOSED_LOOP_DIR, 'trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trajectory plot: {CLOSED_LOOP_DIR}/trajectory.png")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
