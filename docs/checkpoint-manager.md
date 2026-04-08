# Checkpoint Manager

## Goal

Add a proper checkpoint management workflow to the lab so saved models can be inspected, compared, loaded, and exported without leaving the app.

## User-facing scope

- Show the current active checkpoint and the best archived checkpoint side by side
- Add a `Load Checkpoint` action that promotes any archived model into the active slot
- Add a `Compare` action that evaluates two checkpoints over the same fixed seed set
- Add an `Export` action that writes a release-ready bundle for a selected checkpoint

## Technical notes

- Keep the current artifact format and extend it with any extra metadata instead of replacing it
- Add deterministic evaluation seeds so comparison runs are stable between sessions
- Separate archive browsing from active-model state mutation to keep the UI predictable

## First implementation slice

1. Add checkpoint loading from the archive directory into the current slot.
2. Surface current-vs-selected checkpoint metadata in the lab.
3. Add a simple compare runner that reports score, foods eaten, and fitness averages.
