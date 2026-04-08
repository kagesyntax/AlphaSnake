# Curriculum Arenas

## Goal

Train policies across multiple arena types instead of a single empty grid so the swarm develops stronger and more transferable behavior.

## User-facing scope

- Add arena presets such as sparse food, obstacle fields, narrow corridors, and survival mode
- Show the current curriculum stage in the lab
- Track champion performance per arena type instead of only one global score
- Let the operator pin a model to a specific arena for evaluation

## Technical notes

- Arena configuration should be data-driven so new presets do not require rewriting the game loop
- Training should advance to harder arenas only after the active champion reaches a threshold
- Snapshot and checkpoint metadata should include the current curriculum stage

## First implementation slice

1. Add an arena config type and feed it into `GameState::new`.
2. Introduce one obstacle arena and one sparse-food arena.
3. Store the current arena stage in swarm stats and surface it in the lab UI.
