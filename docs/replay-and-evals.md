# Replay And Eval Suite

## Goal

Turn the lab into a repeatable evaluation environment with saved replays and deterministic benchmark runs.

## User-facing scope

- Save short replay traces for the current best model and selected checkpoints
- Add a replay viewer with step controls, speed controls, and frame-by-frame inspection
- Add benchmark suites that run the same seed pack against multiple models
- Report aggregate metrics such as best score, mean score, deaths by cause, and time-to-food

## Technical notes

- Replays should store seeds and actions, not full frame snapshots, to keep files small
- Benchmark runs should execute headless and reuse the same reward and game logic as training
- The UI should keep replays separate from live training so the lab remains responsive

## First implementation slice

1. Record action traces for champion checkpoints.
2. Add a simple replay player panel in the lab.
3. Add a deterministic benchmark command that compares two checkpoints over a fixed seed list.
