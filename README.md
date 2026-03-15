# AutoResearch at Scale

Argus is a distributed system for autonomous machine learning experimentation.
It continuously proposes, executes, evaluates, and promotes experiments across SLURM GPU clusters.

AutoResearch proposes training experiments, coordinates workers through a shared
filesystem state layer, prevents duplicate work, ingests structured results
deterministically, tracks a global best run, and verifies a promotion path for
larger-scale follow-up.

## Published Site

Live report:
`https://yacob-kidane.github.io/Argus/`

Dashboard:
`https://yacob-kidane.github.io/Argus/dashboard/`

## Repo Structure

- `docs/` contains the GitHub Pages site.
- `docs/index.html` is the report and experiment write-up.
- `docs/dashboard/` is the static dashboard snapshot.
- `docs/data/` contains the published JSON and JSONL snapshot files.
- `orchestrator/`, `jobs/`, `state/`, and `scripts/` contain the experiment system itself.

## Notes Before Publishing

- Point GitHub Pages at the `/docs` folder on `main`.
