# Energy-aware Computing Continuum

This repository contains the code and research paper for an Energy-aware Service Placement framework in the Cloud-Edge Computing Continuum.

## Structure

- **[python_algo](/python_algo/)**: Source code for the placement algorithms, graph models, and main execution script.
- **[gcp_automations](/gcp_automations/)**: Windows-friendly scripts to deploy/reuse GCP master and worker VMs from infrastructure properties.
- **[paper](/paper/)**: LaTeX source code for the research paper "Energy-Aware Service Placement in Edge".
- **[storm-scheduler](/storm-scheduler/)**: Implementation of the placement algorithms integrated with Apache Storm for real-time stream processing.

## GCP Deployment (Properties -> VMs)

Use the deployment script to provision the Storm cluster directly from a properties file:

```bash
python .\gcp_automations\deploy_gcp_from_properties.py .\python_algo\properties\Infra_5nodes_GCP.properties
```

Current behavior:

- Idempotent deployment: existing VMs are detected and reused.
- Automatic firewall setup for Storm internal ports and optional external UI access on `8080`.
- Dedicated Nimbus master + worker nodes created from host CPU/RAM tuples.
- Worker startup is optimized for Storm runtime and no longer installs Python/venv dependencies.

## Acknowledgement

TBD