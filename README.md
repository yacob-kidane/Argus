# AutoResearch at Scale

Compute-aware autonomous experiment orchestration on GPU clusters.

AutoResearch proposes training experiments, coordinates workers through a shared
filesystem state layer, prevents duplicate work, ingests structured results
deterministically, tracks a global best run, and verifies a promotion path for
larger-scale follow-up.

## Published Site

Live report:
`https://YOUR_USERNAME.github.io/autoresearch-scale/`

Dashboard:
`https://YOUR_USERNAME.github.io/autoresearch-scale/dashboard/`

## Repo Structure

- `docs/` contains the GitHub Pages site.
- `docs/index.html` is the report and experiment write-up.
- `docs/dashboard/` is the static dashboard snapshot.
- `docs/data/` contains the published JSON and JSONL snapshot files.
- `orchestrator/`, `jobs/`, `state/`, and `scripts/` contain the experiment system itself.

## Notes Before Publishing

- Replace `YOUR_USERNAME` in `docs/index.html`, `docs/dashboard/index.html`, and this file.
- Point GitHub Pages at the `/docs` folder on `main`.
