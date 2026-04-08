# Model Registry

## Goal

Add a durable registry for all meaningful models so the project stops treating checkpoints as loose files and starts treating them as named assets with lineage.

## User-facing scope

- Register promoted models with stable ids, labels, tags, and notes
- Show parent-child lineage for evolved models
- Mark models as `candidate`, `champion`, `retired`, or `baseline`
- Search and filter the registry from the lab UI

## Technical notes

- Keep the current artifact files as the binary payload and store registry metadata separately
- Do not make the UI depend on directory names for identity
- Model promotion should update registry state and the active model slot together

## First implementation slice

1. Add a registry file under `artifacts/` for model metadata.
2. Assign a stable model id to every saved checkpoint.
3. Surface a small registry table in the lab with status and lineage columns.
