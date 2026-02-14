# First Algorithm

## Prerequisites

- Python 3.12 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository.
2. Go the `first_algo` directory
```bash
cd first_algo
```
3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

To run the placement demonstration, execute the `main.py` script from the root directory of the repository:

```bash
python main.py
```

You can optionally specify a starting host for the placement:

```bash
python main.py --start-host <node_id>
```

## Features

- **Infrastructure Modeling**: Represents the computing continuum infrastructure as a network graph.
- **Application Modeling**: Represents applications as service graphs with component dependencies.
- **Placement Algorithm**: Implements a 2 Greedy algorithms to map service components to infrastructure nodes
- **Visualization**: Generates visual representations of the network and service graphs.
- **Unit Testing**: Includes mapping unit tests to verify the validity of the placement.
- **Result Exporter**: Implements a placement result exporter to export result in a CSV file.