# Plan archive

Immutable snapshots of each **approved per-stage plan** — the detailed design + rationale agreed
at the plan-approval (ExitPlanMode) gate, frozen at approval time.

## Why this exists

- [`PLANS.md`](../../PLANS.md) is the **living tracker** (Tier 1 roadmap + Tier 2 items with
  status and commit hashes); it is edited in place and its history lives in git.
- The detailed approved plan is authored in an **ephemeral** plan file that is overwritten each
  stage. These snapshots preserve that design record in the repo.

## Convention

- One file per approved plan: `NNN_<slug>.md`, zero-padded sequence + a slug from the plan title
  (lowercased, spaces → `-`). Example: `002_stage-2-model-factory-and-forecaster-abstractions.md`.
- Written **at approval time**, before implementation starts.
- **Immutable**: do not edit a snapshot after the fact. Course corrections during a stage are
  tracked in `PLANS.md` and git, not by rewriting the archived plan.

## Index

- `001_stage-1-data-architecture-and-config-schema.md`
- `002_stage-2-model-factory-and-forecaster-abstractions.md`
- `003_stage-3-experiment-tracking-and-doe-framework.md`
- `004_stage-4-managed-pipelines-and-cloud-orchestration.md`
- `005_stage-4-redesign-train-inference-split.md`
- `006_stage-5-standardized-evaluation-and-entry-point.md`
- `007_post-stage5-eval-audit-and-tooling.md`
