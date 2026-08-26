"""DAgger (Dataset Aggregation) for APA Planner

Collects on-policy rollout states and labels them with the RS expert oracle,
then aggregates with original training data for retraining.

Usage:
    source .venv/bin/activate
    python training/dagger.py --rounds 5 --rollouts-per-round 50 --epochs-per-round 10
"""

import argparse
import gc
import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch

from parking_il_planner.config.model import MODEL_CONFIG
from parking_il_planner.data.dataset import ParkingDataset
from parking_il_planner.geometry.collision import CollisionDetector, Rectangle
from parking_il_planner.geometry.kinematics import VehicleState
from parking_il_planner.models.planner_model import build_apa_model
from parking_il_planner.planning.actions import Action, ActionExecutor
from parking_il_planner.planning.reeds_shepp import ReedsSheppExpertPlanner
from parking_il_planner.simulation.renderer import SceneRenderer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_INDICES = [355, 1357, 1648, 2756, 6719, 8286, 10375, 11327, 12598, 18281,
                19155, 25180, 27673, 29120, 29972, 30348, 33755, 36798, 38555, 38740]

GOAL_THRESHOLD = 0.095
ANGLE_THRESHOLD = 5.0
COLLISION_THRESHOLD = 0.25


def get_oracle_action(planner: ReedsSheppExpertPlanner, current: VehicleState, target: VehicleState) -> int:
    """Ask RS oracle for the best action from current state to target."""
    result = planner.plan(current, target)
    if result.success and len(result.actions) > 0:
        action = result.actions[0]
        if action == Action.S0.value:
            return 0
        return action
    # If planning fails, check if already at goal
    dist = math.sqrt((current.x - target.x)**2 + (current.y - target.y)**2)
    if dist < 0.1:
        return 0  # S0
    return -1  # Failed


def rollout_and_label(
    model: torch.nn.Module,
    planner: ReedsSheppExpertPlanner,
    renderer: SceneRenderer,
    executor: ActionExecutor,
    start: VehicleState,
    target: VehicleState,
    obstacles: List[Rectangle],
    max_steps: int = 200,
) -> List[Tuple[np.ndarray, int, np.ndarray, np.ndarray]]:
    """
    Roll out the model from start, label each visited state with oracle action.

    Returns: list of (image, oracle_action, state_6d, action_history_14d) tuples
    """
    model.eval()
    samples = []
    current = start
    action_buffer = [-1, -1]
    detector = CollisionDetector(safety_margin=0.0)

    for _step in range(max_steps):
        # Render current state
        img = renderer.render(current, target, obstacles)

        # Get oracle label for this state
        oracle_action = get_oracle_action(planner, current, target)
        if oracle_action < 0:
            break  # Oracle can't plan from here

        # Compute state vector
        dx = target.x - current.x
        dy = target.y - current.y
        dh = target.heading - current.heading
        dist = math.sqrt(dx**2 + dy**2)
        state_6d = np.array([current.x, current.y, current.heading,
                             target.x, target.y, target.heading], dtype=np.float32)

        # Compute action history
        ah = np.zeros(14, dtype=np.float32)
        if action_buffer[-1] >= 0:
            ah[action_buffer[-1]] = 1.0
        if action_buffer[-2] >= 0:
            ah[7 + action_buffer[-2]] = 1.0

        # Store sample (image as uint8 for storage)
        img_uint8 = (img * 255).astype(np.uint8)
        samples.append((img_uint8, oracle_action, state_6d, ah))

        # Check if at goal — still record S0 state, then stop rollout
        heading_diff = abs(math.degrees(current.heading - target.heading))
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        if dist < 0.095 and heading_diff < 5.0:
            # Record one more S0 state at goal for "stop" signal
            goal_img = renderer.render(current, target, obstacles)
            goal_img_uint8 = (goal_img * 255).astype(np.uint8)
            goal_ah = np.zeros(14, dtype=np.float32)
            if action_buffer[-1] >= 0:
                goal_ah[action_buffer[-1]] = 1.0
            if action_buffer[-2] >= 0:
                goal_ah[7 + action_buffer[-2]] = 1.0
            goal_state = np.array([current.x, current.y, current.heading,
                                   target.x, target.y, target.heading], dtype=np.float32)
            samples.append((goal_img_uint8, 0, goal_state, goal_ah))
            break

        # Model predicts action (for state evolution — DAgger uses model's trajectory)
        img_t = torch.tensor(img[np.newaxis, ...]).to(DEVICE)
        state_vector = torch.tensor([[dx, dy, dh, dist, math.cos(dh), math.sin(dh)]],
                                    dtype=torch.float32).to(DEVICE)
        ah_t = torch.tensor(ah[np.newaxis, ...], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(img_t, state_vector=state_vector, action_history=ah_t)
            pred_action = torch.argmax(logits, -1).item()

        # If model predicts S0, use oracle action to keep trajectory going
        if pred_action == 0 and oracle_action != 0:
            pred_action = oracle_action

        # Execute model's action (this is the key DAgger property — follow model's distribution)
        result = executor.execute(current, Action(pred_action))
        current = result.final_state
        action_buffer.append(pred_action)

        # Collision check
        if obstacles:
            col = detector.check_vehicle_obstacles(current, obstacles)
            if col.collision and col.penetration_depth > 0.3:
                break

    return samples


def run_dagger_round(
    model: torch.nn.Module,
    planner: ReedsSheppExpertPlanner,
    renderer: SceneRenderer,
    executor: ActionExecutor,
    dataset: ParkingDataset,
    num_rollouts: int = 50,
    max_steps: int = 200,
    output_dir: str = 'data/generated/dagger_round',
) -> int:
    """
    Run DAgger rollouts and save oracle-labeled data incrementally to disk.

    Returns: number of samples collected
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find trajectory start indices
    targets = dataset.states[:, 3:]
    diffs = np.abs(targets[1:] - targets[:-1]).sum(axis=1)
    boundaries = [0] + list(np.where(diffs > 0.001)[0] + 1)

    # Sample rollout starting points
    rng = np.random.default_rng()
    selected_indices = rng.choice(boundaries, size=min(num_rollouts, len(boundaries)), replace=False)

    # Collect samples incrementally — save to disk in chunks
    all_actions = []
    all_states = []
    all_histories = []
    image_chunks = []
    chunk_idx = 0
    chunk_images = []
    CHUNK_SIZE = 500  # Save every 500 images to limit memory

    for i, idx in enumerate(selected_indices):
        s = dataset.states[idx]
        start = VehicleState(x=s[0], y=s[1], heading=s[2])
        target = VehicleState(x=s[3], y=s[4], heading=s[5])

        # Load obstacles
        obstacles = []
        for obs in dataset.obstacles[idx]:
            if obs[0] != 0:
                obstacles.append(Rectangle(
                    center_x=obs[0], center_y=obs[1],
                    length=obs[2], width=obs[3], heading=obs[4]
                ))

        samples = rollout_and_label(model, planner, renderer, executor,
                                    start, target, obstacles, max_steps)

        for img, action, state, ah in samples:
            chunk_images.append(img)
            all_actions.append(action)
            all_states.append(state)
            all_histories.append(ah)

            if len(chunk_images) >= CHUNK_SIZE:
                chunk_path = os.path.join(output_dir, f'images_chunk_{chunk_idx}.npy')
                np.save(chunk_path, np.array(chunk_images, dtype=np.uint8))
                image_chunks.append(chunk_path)
                chunk_images = []
                chunk_idx += 1

        if (i + 1) % 10 == 0:
            print(f"    Rollout {i+1}/{len(selected_indices)}: {len(all_actions)} samples", flush=True)

    # Save remaining images
    if chunk_images:
        chunk_path = os.path.join(output_dir, f'images_chunk_{chunk_idx}.npy')
        np.save(chunk_path, np.array(chunk_images, dtype=np.uint8))
        image_chunks.append(chunk_path)
        chunk_images = []

    if not all_actions:
        return 0

    # Concatenate all chunks into single images.npy
    all_image_arrays = [np.load(p) for p in image_chunks]
    combined_images = np.concatenate(all_image_arrays, axis=0)
    np.save(os.path.join(output_dir, 'images.npy'), combined_images)
    del all_image_arrays, combined_images

    # Clean up chunks
    for p in image_chunks:
        os.remove(p)

    # Save other arrays
    np.save(os.path.join(output_dir, 'actions.npy'), np.array(all_actions, dtype=np.int32))
    np.save(os.path.join(output_dir, 'states.npy'), np.array(all_states, dtype=np.float32))
    np.save(os.path.join(output_dir, 'obstacles.npy'),
            np.zeros((len(all_actions), 20, 5), dtype=np.float32))

    n = len(all_actions)
    print(f"    Total: {n} samples saved to {output_dir}/", flush=True)
    print(f"    Action dist: {np.bincount(np.array(all_actions), minlength=7)}", flush=True)
    return n


def retrain_with_dagger_data(
    dagger_dir: str,
    model: torch.nn.Module = None,
    num_epochs: int = 10,
    batch_size: int = 64,
):
    """Fine-tune model on DAgger + sampled original data (replay buffer approach)."""
    from torch.utils.data import DataLoader

    from parking_il_planner.config.training import TRAINING_CONFIG
    from parking_il_planner.training.trainer import APATrainer

    dagger_actions = np.load(os.path.join(dagger_dir, 'actions.npy'))
    n_dagger = len(dagger_actions)
    print(f"  DAgger samples: {n_dagger}", flush=True)
    print(f"  DAgger action dist: {np.bincount(dagger_actions, minlength=7)}", flush=True)

    # Sample a subset of original data to mix with DAgger (prevents catastrophic forgetting)
    # Use 1:1 ratio to give DAgger data equal weight
    orig_images_mmap = np.load('data/generated/images.npy', mmap_mode='r')
    orig_actions = np.load('data/generated/actions.npy')
    orig_states = np.load('data/generated/states.npy')

    n_orig_sample = min(n_dagger, len(orig_actions))
    rng = np.random.default_rng(42)
    orig_indices = np.sort(rng.choice(len(orig_actions), size=n_orig_sample, replace=False))

    # Load original subset
    orig_sample_images = np.array(orig_images_mmap[orig_indices])
    orig_sample_actions = orig_actions[orig_indices]
    orig_sample_states = orig_states[orig_indices]
    del orig_images_mmap

    # Load DAgger data
    dagger_images = np.load(os.path.join(dagger_dir, 'images.npy'))
    dagger_states = np.load(os.path.join(dagger_dir, 'states.npy'))

    # Combine
    all_images = np.concatenate([dagger_images, orig_sample_images], axis=0)
    all_actions = np.concatenate([dagger_actions, orig_sample_actions], axis=0)
    all_states = np.concatenate([dagger_states, orig_sample_states], axis=0)
    del dagger_images, orig_sample_images

    print(f"  Original subset: {n_orig_sample} samples", flush=True)
    print(f"  Combined training: {len(all_actions)} samples", flush=True)
    print(f"  Combined action dist: {np.bincount(all_actions, minlength=7)}", flush=True)

    # Save combined to disk
    train_dir = 'data/generated/dagger_combined'
    os.makedirs(train_dir, exist_ok=True)
    np.save(os.path.join(train_dir, 'images.npy'), all_images)
    np.save(os.path.join(train_dir, 'actions.npy'), all_actions)
    np.save(os.path.join(train_dir, 'states.npy'), all_states)
    np.save(os.path.join(train_dir, 'obstacles.npy'),
            np.zeros((len(all_actions), 20, 5), dtype=np.float32))
    del all_images, all_actions, all_states
    gc.collect()

    # Create datasets
    train_ds = ParkingDataset(train_dir, split=None, augment=True)
    val_ds = ParkingDataset('data/generated', split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

    print(f"  Train: {len(train_ds)} samples (with augment), Val: {len(val_ds)} samples", flush=True)

    # Build or use passed model
    if model is None:
        model = build_apa_model(MODEL_CONFIG, device=DEVICE)
        cp = torch.load('checkpoints/best.pt', map_location=DEVICE, weights_only=False)
        model.load_state_dict(cp['model_state_dict'])

    # Fine-tune with low LR, no class weights
    config = TRAINING_CONFIG
    config.num_epochs = num_epochs
    config.learning_rate = 5e-5
    config.checkpoint_dir = 'checkpoints/dagger'

    trainer = APATrainer(model, config, device=DEVICE)

    label_smoothing = getattr(MODEL_CONFIG, 'label_smoothing', 0.1)
    trainer.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainer.train(train_loader, val_loader)

    return model


def evaluate_closed_loop(model: torch.nn.Module, dataset: ParkingDataset,
                         scenario_indices: list = None) -> dict:
    """Closed-loop evaluation on fixed evaluation scenarios.

    Matches batch_closed_loop_eval.py logic: repeat limit + no progress limit.
    """
    if scenario_indices is None:
        scenario_indices = EVAL_INDICES

    renderer = SceneRenderer()
    executor = ActionExecutor()
    detector = CollisionDetector(safety_margin=0.0)

    REPEAT_LIMIT = 200
    NO_PROGRESS_LIMIT = 200

    results = []
    for idx in scenario_indices:
        s = dataset.states[idx]
        start = VehicleState(x=s[0], y=s[1], heading=s[2])
        target = VehicleState(x=s[3], y=s[4], heading=s[5])

        obstacles = []
        for obs in dataset.obstacles[idx]:
            if obs[0] != 0:
                obstacles.append(Rectangle(
                    center_x=obs[0], center_y=obs[1],
                    length=obs[2], width=obs[3], heading=obs[4]))

        current = start
        action_buffer = [-1, -1]
        success = False
        collision = False
        stuck = False
        last_action = -1
        repeat_count = 0
        best_dist = float('inf')
        no_progress_count = 0

        for _step in range(500):
            img = renderer.render(current, target, obstacles)
            dx = target.x - current.x
            dy = target.y - current.y
            dh = target.heading - current.heading
            dist = math.sqrt(dx**2 + dy**2)
            heading_diff = abs(math.degrees(dh))
            if heading_diff > 180:
                heading_diff = 360 - heading_diff

            if dist < 0.095 and heading_diff < 5.0:
                success = True
                break

            img_t = torch.tensor(img[np.newaxis, ...]).to(DEVICE)
            sv = torch.tensor([[dx, dy, dh, dist, math.cos(dh), math.sin(dh)]],
                             dtype=torch.float32).to(DEVICE)
            ah = torch.zeros(1, 14).to(DEVICE)
            if action_buffer[-1] >= 0:
                ah[0, action_buffer[-1]] = 1.0
            if action_buffer[-2] >= 0:
                ah[0, 7 + action_buffer[-2]] = 1.0

            with torch.no_grad():
                logits = model(img_t, state_vector=sv, action_history=ah)
                pred = torch.argmax(logits, -1).item()

            if pred == 0:
                break  # Model predicts stop

            # Stuck detection: repeat limit
            if pred == last_action:
                repeat_count += 1
            else:
                repeat_count = 1
                last_action = pred
            if repeat_count >= REPEAT_LIMIT:
                stuck = True
                break

            # Stuck detection: no progress
            if dist < best_dist - 0.01:
                best_dist = dist
                no_progress_count = 0
            else:
                no_progress_count += 1
            if no_progress_count >= NO_PROGRESS_LIMIT:
                stuck = True
                break

            result = executor.execute(current, Action(pred))
            current = result.final_state
            action_buffer.append(pred)

            if obstacles:
                col = detector.check_vehicle_obstacles(current, obstacles)
                if col.collision and col.penetration_depth > COLLISION_THRESHOLD:
                    collision = True
                    break

        if success:
            reached = True
        elif collision:
            reached = False
        else:
            reached = False
            stuck = True

        results.append({
            'dataset_id': idx,
            'reached': reached,
            'collision': collision,
            'stuck': stuck,
        })

    reached = sum(1 for r in results if r['reached'])
    return {
        'reached': reached,
        'total': len(scenario_indices),
        'rate': reached / len(scenario_indices),
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='DAgger training for APA planner')
    parser.add_argument('--rounds', type=int, default=5, help='Number of DAgger rounds')
    parser.add_argument('--rollouts-per-round', type=int, default=40, help='Rollouts per round')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per rollout')
    parser.add_argument('--epochs-per-round', type=int, default=15, help='Training epochs per round')
    parser.add_argument('--eval-scenarios', type=int, default=20, help='Scenarios for evaluation')
    args = parser.parse_args()

    print("=" * 70)
    print("DAgger Training for APA Planner")
    print("=" * 70)
    print(f"  Rounds: {args.rounds}")
    print(f"  Rollouts/round: {args.rollouts_per_round}")
    print(f"  Max steps/rollout: {args.max_steps}")
    print(f"  Epochs/round: {args.epochs_per_round}")

    # Initialize
    renderer = SceneRenderer()
    executor = ActionExecutor()
    planner = ReedsSheppExpertPlanner()
    dataset = ParkingDataset('data/generated', load_images=False)

    # Load model
    print("\nLoading model...")
    cp = torch.load('checkpoints/best.pt', map_location=DEVICE, weights_only=False)
    model = build_apa_model(MODEL_CONFIG, device=DEVICE)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    print(f"  Model loaded ({model.get_num_params():,} params)")

    # Initial evaluation
    print("\n--- Initial evaluation ---")
    eval_indices = EVAL_INDICES[:args.eval_scenarios] if args.eval_scenarios < len(EVAL_INDICES) else EVAL_INDICES
    eval_result = evaluate_closed_loop(model, dataset, eval_indices)
    print(f"  Success rate: {eval_result['reached']}/{eval_result['total']} ({eval_result['rate']:.1%})")

    # DAgger rounds (disk-based to avoid OOM)
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"DAgger Round {round_num}/{args.rounds}")
        print(f"{'=' * 70}")

        # Collect on-policy data with oracle labels (saved to disk)
        round_dir = f'data/generated/dagger_round_{round_num}'
        print(f"\n  Collecting rollouts → {round_dir}/")
        n_samples = run_dagger_round(
            model, planner, renderer, executor, dataset,
            num_rollouts=args.rollouts_per_round,
            max_steps=args.max_steps,
            output_dir=round_dir,
        )

        if n_samples == 0:
            print("  No samples collected, skipping round")
            continue

        # Retrain from disk (pass current model for chain fine-tuning)
        print(f"\n  Retraining ({args.epochs_per_round} epochs)...")
        model = retrain_with_dagger_data(
            round_dir,
            model=model,
            num_epochs=args.epochs_per_round,
            batch_size=64,
        )
        model.eval()

        # Evaluate
        print(f"\n  Evaluating round {round_num}...")
        eval_result = evaluate_closed_loop(model, dataset, eval_indices)
        print(f"  Success rate: {eval_result['reached']}/{eval_result['total']} ({eval_result['rate']:.1%})")

        # Save result
        os.makedirs('checkpoints', exist_ok=True)
        result_path = f'checkpoints/dagger_round_{round_num}.json'
        with open(result_path, 'w') as f:
            json.dump({'round': round_num, **eval_result,
                      'dagger_samples': n_samples}, f, indent=2)

        if eval_result['rate'] >= 0.6:
            print(f"\n  ★ Target reached! ({eval_result['rate']:.1%} >= 60%)")
            break

    print(f"\n{'=' * 70}")
    print("DAgger Complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
