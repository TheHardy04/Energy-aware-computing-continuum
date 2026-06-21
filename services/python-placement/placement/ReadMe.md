# Python Placement Framework

This package contains the core placement engine, graph parsers, evaluator, and CSV exporters used by the research workflow.

## Important Paths

- Infrastructure inputs: `configs/infra/`
- Application inputs: `configs/app/`
- Energy settings: `configs/energy/Energy_GCP.properties`
- Outputs: `experiments/results/`

## Run

```bash
python main.py --strategy CSP --infra ../../configs/infra/Infra_5nodes_GCP.properties --app ../../configs/app/Appli_5comps_GCP.properties
```

## Notes

- The current code keeps its internal `src/` package for compatibility.
- See [../README.md](../README.md) for the repository-level layout and workflow.