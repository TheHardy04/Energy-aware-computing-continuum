# Experiments

This folder stores generated outputs from placement runs, benchmark sweeps, and analysis artifacts.

## Subfolders

- `results/`: primary CSV outputs from current runs.
- `results_infra10/`: scenario-specific benchmark outputs.

### Pull experiment artifacts from GCP to your local machine

Copies the latest plot and summary CSV from the master VM back to your local `experiments/results/` directory using `gcloud compute scp`.

```bash
python ./experiments/pull_experiment_artifacts.py
```

If your VM, zone, or destination directory are different, pass them explicitly:

```bash
python ./experiments/pull_experiment_artifacts.py --instance storm-nimbus --zone europe-west9-a --dest ~/Downloads/energy-results
```

## Notes

- Keep raw CSV outputs here instead of inside source directories.
- Figures, tables, and aggregated analysis can also live here if you want a single research-artifacts location.