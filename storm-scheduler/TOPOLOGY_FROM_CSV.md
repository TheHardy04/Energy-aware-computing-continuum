# TopologyFromCSV - Storm Topology Generator

## Overview

`TopologyFromCSV` is a Java class that dynamically generates Apache Storm topologies from application properties files. It reads the component specifications and links from `.properties` files (like those in `../python_algo/properties/`) and creates a corresponding Storm topology.

## Properties File Format

The properties files should follow this format:

```ini
# Number of components
application.components = 4

# Component requirements: {CPU, RAM, lambda (data flow rate), mu (processing rate)}
components.requirements = \
{1,1000,400,900}, \
{1,1000,400,220}, \
{2,2000,400,260}, \
{2,2000,400,760}

# Link specifications: {id, source, destination, bandwidth, max_latency}
links.description = \
{0,0,1,400,260}, \
{1,1,2,400,160}, \
{2,2,3,400,160}

links.nb = 3

# Optional: Placement constraints {componentId, nodeId}
component.nbDZ = 1
component.DZ = \
{0, 5}
```

## How It Works

1. **Component Parsing**: Reads component specifications including CPU, RAM, lambda (data flow rate), and mu (processing rate)
2. **Link Parsing**: Reads link descriptions that define how components are connected
3. **Source Detection**: Automatically identifies the source component (one with no incoming links)
4. **Topology Building**:
   - Creates a **Spout** for the source component with the specified lambda (emission rate)
   - Creates **Bolts** for all other components with CPU work proportional to their mu (processing rate)
   - Wires components together according to the links specification
5. **Placement Constraints**: Applies optional placement constraints for specific components

## Usage

### From the storm-scheduler directory:

```bash
# Using the launch script
./scripts/launch-topology-from-csv.sh <properties_file> [topology_name]

# Examples:
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_4comps.properties MyTopology
./scripts/launch-topology-from-csv.sh ../python_algo/properties/Appli_10comps_dcns.properties DCNS
```

### Manual submission:

```bash
# Build the project
mvn clean package

# Submit to Storm cluster
storm jar target/storm-scheduler-1.0-SNAPSHOT.jar \
  fr.dvrc.thardy.topology.TopologyFromCSV \
  ../python_algo/properties/Appli_4comps.properties \
  MyTopology
```

## Available Properties Files

The project includes several pre-configured application properties:

- `Appli_4comps.properties` - Simple 4-component application
- `Appli_10comps_dcns.properties` - Intelligent surveillance DCNS (10 components)
- `Appli_11comps_smartcity.properties` - Smart city application
- `Appli_12comps_ehealth.properties` - E-health monitoring application
- `Appli_8comps_smartbuilding.properties` - Smart building application

## Topology Structure

The generated topology consists of:

- **DataSourceSpout**: Emits data at a rate specified by the lambda parameter
- **ProcessingBolt**: Simulates CPU work based on the mu (processing rate) parameter
- All components are connected according to the links specification
- Uses shuffle grouping for data distribution

## Configuration

The topology is configured with:
- **Debug**: false (disable debug logging)
- **Workers**: 3 (distributed processing)
- **Ackers**: 0 (reliability disabled for performance)

## Monitoring

Once submitted, you can monitor the topology using:

- Storm UI: http://localhost:8080 (by default)
- Storm CLI: `storm list` to see running topologies

To kill a topology:
```bash
storm kill <topology_name>
```

## Notes

- The CPU work in each bolt is calculated as: `cpuMillis = max(1, 1000 / max(mu, 100))`
- Higher mu values result in more CPU work per tuple
- The spout emission rate is controlled by the lambda parameter (events per second)
- Placement constraints are stored in the topology configuration but require a custom scheduler to be enforced

