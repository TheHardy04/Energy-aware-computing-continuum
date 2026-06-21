# GCP Automations

Helpers to provision GCP VMs from infra properties files and collect runtime metrics from Google Cloud Monitoring.

## Install

```powershell
pip install -r gcp_automations/requirements.txt
```

## Prerequisites

- Google Cloud SDK installed
- `gcloud auth login`
- `gcloud auth application-default login`
- Access to the target GCP project
- `monitoring.googleapis.com` enabled for monitoring exports

## Deploy Infrastructure

Creates or reuses a dedicated Nimbus VM plus worker VMs inferred from `hosts.configuration` and `hosts.zones`.

```powershell
python gcp_automations/deploy_gcp_from_properties.py configs/infra/Infra_5nodes_GCP.properties
```

Example with a larger topology:

```powershell
python gcp_automations/deploy_gcp_from_properties.py configs/infra/infra_10nodes_smartcity_GCP.properties
```

What the deploy script does:

- reuses existing VMs when names already exist
- creates Storm firewall rules if missing
- creates one `storm-nimbus` master VM
- creates worker VMs from CPU and RAM tuples in the properties file
- applies optional per-node NetEm shaping when `latency_ms` and `bandwidth_mbit` are present in the infra file

### NetEm inputs

The deploy script now derives per-node network shaping from the existing `network.topology` block in the infra properties file. It uses the topology latency and bandwidth values to compute a host-level NetEm profile, then passes that profile to VM metadata and re-applies it over SSH for reused VMs.

If `network.topology` is missing, the script falls back to optional node-level latency and bandwidth lists for backward compatibility.

The startup scripts clear any existing root qdisc before adding a new `netem` rule.

## Collect VM Metrics

Fetch CPU and network metrics for the VMs described by an infra properties file.

```powershell
python gcp_automations/gcp_vm_monitoring.py --infra configs/infra/Infra_5nodes_GCP.properties --window-minutes 15
```

Example with explicit project:

```powershell
python gcp_automations/gcp_vm_monitoring.py --project-id <gcp-project> --infra configs/infra/Infra_5nodes_GCP.properties --window-minutes 30
```

## Files

- [deploy_gcp_from_properties.py](deploy_gcp_from_properties.py): infra-to-VM deployment
- [gcp_vm_monitoring.py](gcp_vm_monitoring.py): Cloud Monitoring export
- [master_vm_startup.sh](master_vm_startup.sh): master node bootstrap
- [vm_startup.sh](vm_startup.sh): worker node bootstrap