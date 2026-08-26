# Parking IL Planner Open — Repository Rules

## Purpose

This repository publishes the reusable ideas and code for an imitation-learning
automatic parking planner. It must remain reproducible, portable, and safe to
share publicly.

## Repository layout

- `src/parking_il_planner/`: installable Python package and all runtime code.
- `configs/`: versioned, portable YAML configurations.
- `tests/`: deterministic unit and smoke tests; tests generate their own inputs.
- `docs/`: architecture, data generation, evaluation, reproduction, and limits.
- `assets/`: a small curated set of public documentation media only.
- `data/generated/`: local generated datasets; never commit.
- `checkpoints/`: local model weights; never commit.
- `artifacts/`: local evaluation outputs; never commit.

## Development rules

1. Never add datasets, checkpoints, logs, per-frame renders, credentials, local
   permission files, machine-specific paths, personal email addresses, or tokens.
2. Runtime paths must be supplied by CLI/configuration or resolved relative to
   the current working directory. Never hard-code a user home or repository path.
3. Keep core algorithms under `src/parking_il_planner/`; CLI entry points should
   delegate to package functions instead of duplicating logic.
4. Public claims must include the evaluation protocol, sample count, thresholds,
   controller type, and seed. Open-loop accuracy is not closed-loop success.
5. Generated outputs go under ignored directories. Only deliberately curated,
   compact documentation media may enter `assets/`.
6. Preserve third-party license notices and attribution. PythonRobotics-derived
   Reeds-Shepp code must retain its MIT attribution in `NOTICE` and source headers.
7. Add or update tests for behavior changes. Run tests and packaging checks before
   every release-facing commit.

## Git rules

- Default branch: `main`.
- Commit messages: concise English descriptions of intent.
- Configure author identity locally for this repository; do not inherit a company
  email address.
- Do not add a remote or push unless the repository owner explicitly requests it.
- Before staging, inspect `git status`; before committing, inspect the staged diff.

## Cleanup

- Temporary compatibility or migration scripts must state when they can be removed.
- Do not retain duplicated legacy modules after parity tests pass.
- Keep the repository clone small; external datasets and weights are referenced by
  URL and checksum rather than stored in Git.
