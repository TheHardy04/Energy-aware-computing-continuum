# First Algorithm

## Prerequisites

- Python 3.12 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository.
2. Go to the `python_algo` directory
```bash
cd python_algo
```
1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

To run the placement demonstration, execute the `main.py` script:

```bash
python main.py
```

### Command Line Arguments

You can customize the execution with the following arguments:

- `--plot`: Enable visualization of the network and service graphs.
- `--verbose`: Enable detailed graph info and results
- `--infra <path>`: Path to the infrastructure properties file (default: `properties/Infra_8nodes.properties`).
- `--app <path>`: Path to the application properties file (default: `properties/Appli_4comps.properties`).
- `--strategy <strategy>`: Choose the placement strategy. Options:
  - `CSP`: Constraint Satisfaction Problem (default).
  - `GreedyFirstFit`: Greedy First Fit algorithm.
  - `GreedyFirstIterate`: Greedy First Iterate algorithm.
- `--to-csv <path>`: Path to export the placement results as a CSV file (default: `results/placement.csv`).

### Examples

Run and plot graphs with placement details:
```bash
python main.py --plot --verbose
```

Run with a different infrastructure and application file using the GreedyFirstFit strategy:
```bash
python main.py --infra properties/Infra_16nodes_fog3tier.properties --app properties/Appli_8comps_smartbuilding.properties --strategy GreedyFirstFit
```

## Features

- **Infrastructure Modeling**: Represents the computing continuum infrastructure as a network graph.
- **Application Modeling**: Represents applications as service graphs with component dependencies.
- **Placement Algorithms**: Implements multiple placement strategies:
  - **CSP**: Constraint Satisfaction Problem-based placement.
  - **Greedy Algorithms**: Greedy First Fit and Greedy First Iterate strategies.
- **Visualization**: Generates visual representations of the network and service graphs.
- **Unit Testing**: Includes mapping unit tests to verify the validity of the placement.
- **Result Exporter**: Implements a placement result exporter to export result in a CSV file.