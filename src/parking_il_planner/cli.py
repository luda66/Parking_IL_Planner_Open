"""Unified command-line entry point with lazy optional-ML imports."""

from __future__ import annotations

import argparse
import sys


def _expert_demo() -> None:
    from parking_il_planner.geometry.kinematics import VehicleState
    from parking_il_planner.planning.reeds_shepp import ReedsSheppExpertPlanner

    start = VehicleState(x=0.0, y=0.0, heading=0.0)
    target = VehicleState(x=1.0, y=0.5, heading=0.0)
    result = ReedsSheppExpertPlanner().plan(start, target)
    print(result.message)
    if not result.success:
        raise SystemExit(1)


def _generate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="parking-il generate")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--output", default="data/generated/demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Cap rendered samples for a small smoke dataset; use 0 for no cap",
    )
    args = parser.parse_args(argv)

    from parking_il_planner.data.generator import synthesize_and_save

    synthesize_and_save(
        num_trajectories=args.trajectories,
        output_dir=args.output,
        seed=args.seed,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
    )


def _delegate(argv: list[str], module_main) -> None:
    sys.argv = [sys.argv[0], *argv]
    module_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parking-il",
        description="Automatic-parking imitation-learning toolkit",
    )
    parser.add_argument(
        "command",
        choices=["expert-demo", "generate", "train", "open-loop", "closed-loop"],
    )
    args, remainder = parser.parse_known_args()

    if args.command == "expert-demo":
        _expert_demo()
    elif args.command == "generate":
        _generate(remainder)
    elif args.command == "train":
        from parking_il_planner.training.run import main as command_main

        _delegate(remainder, command_main)
    elif args.command == "open-loop":
        from parking_il_planner.evaluation.open_loop import main as command_main

        _delegate(remainder, command_main)
    else:
        from parking_il_planner.evaluation.closed_loop import main as command_main

        _delegate(remainder, command_main)
