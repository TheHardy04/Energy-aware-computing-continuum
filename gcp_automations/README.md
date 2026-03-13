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
python gcp_automations/deploy_gcp_from_properties.py python_algo/properties/Infra_5nodes_GCP.properties
```

Example with a larger topology:

```powershell
python gcp_automations/deploy_gcp_from_properties.py python_algo/properties/infra_10nodes_smartcity_GCP.properties
```

What the deploy script does:

- reuses existing VMs when names already exist
- creates Storm firewall rules if missing
- creates one `storm-nimbus` master VM
- creates worker VMs from CPU and RAM tuples in the properties file

## Collect VM Metrics

Fetch CPU and network metrics for the VMs described by an infra properties file.

```powershell
python gcp_automations/gcp_vm_monitoring.py --infra python_algo/properties/Infra_5nodes_GCP.properties --window-minutes 15
```

Example with explicit project:

```powershell
python gcp_automations/gcp_vm_monitoring.py --project-id <gcp-project> --infra python_algo/properties/Infra_5nodes_GCP.properties --window-minutes 30
```

## Files

- [deploy_gcp_from_properties.py](deploy_gcp_from_properties.py): infra-to-VM deployment
- [gcp_vm_monitoring.py](gcp_vm_monitoring.py): Cloud Monitoring export
- [master_vm_startup.sh](master_vm_startup.sh): master node bootstrap
- [vm_startup.sh](vm_startup.sh): worker node bootstrap