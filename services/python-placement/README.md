# Python Placement Service

This service contains the placement algorithms, graph models, evaluation logic, and CSV export code used by the research workflow.

## Main Entry Point

- `placement/main.py`: command-line runner for CSP, LLM, and greedy strategies.

## Layout

- `placement/`: Python package with the core implementation.
- `placement/src/`: graph parsing and placement support modules.
- `placement/energy_calculus.py`: post-processing utilities for experiment results.
- `placement/mappingUnitTest.py`: runtime validation helpers.

## Default Paths

- Infrastructure inputs: `configs/infra/`
- Application inputs: `configs/app/`
- Energy settings: `configs/energy/Energy_GCP.properties`
- Outputs: `experiments/results/`

## Example

```bash
python placement/main.py --strategy CSP --infra ../../configs/infra/Infra_5nodes_GCP.properties --app ../../configs/app/Appli_5comps_GCP.properties
```