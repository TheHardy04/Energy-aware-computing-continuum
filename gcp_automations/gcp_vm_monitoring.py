"""Fetch GCE VM utilization metrics from Google Cloud Monitoring.

Authentication (local development):
1) Install Google Cloud SDK and log in:
   gcloud auth application-default login
2) Set your default project (optional if provided via --project-id):
   gcloud config set project <YOUR_PROJECT_ID>
3) Ensure this API is enabled in the project:
   monitoring.googleapis.com

Example:
  python gcp_automations/gcp_vm_monitoring.py \
	  --properties-file python_algo/properties/Infra_5nodes_GCP.properties \
	  --window-minutes 15
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from google.auth import default as google_auth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import monitoring_v3


LOGGER = logging.getLogger("gcp_vm_monitoring")

MASTER_NAME = "storm-nimbus"

MACHINE_MAP: Dict[Tuple[int, int], Dict[str, str]] = {
	(4, 16000): {"type": "n2-standard-4", "prefix": "cloud-core"},
	(2, 8000): {"type": "e2-standard-2", "prefix": "fog-gateway"},
	(2, 4000): {"type": "e2-medium", "prefix": "worker-edge"},
	(1, 2000): {"type": "e2-small", "prefix": "worker-edge-light"},
	(1, 1000): {"type": "e2-micro", "prefix": "worker-iot"},
}


@dataclass(frozen=True)
class MetricSpec:
	metric_type: str
	dataframe_column: str
	aligner: monitoring_v3.Aggregation.Aligner


METRICS: Sequence[MetricSpec] = (
	MetricSpec(
		metric_type="compute.googleapis.com/instance/cpu/utilization",
		dataframe_column="CPU_Usage_Percent",
		aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
	),
	MetricSpec(
		metric_type="compute.googleapis.com/instance/network/received_bytes_count",
		dataframe_column="Network_RX_Bytes",
		aligner=monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
	),
	MetricSpec(
		metric_type="compute.googleapis.com/instance/network/sent_bytes_count",
		dataframe_column="Network_TX_Bytes",
		aligner=monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
	),
)


def parse_hosts_configuration(properties_file: Path) -> List[Tuple[int, int]]:
	"""Parse hosts.configuration tuples from the infra properties file."""
	try:
		content = properties_file.read_text(encoding="utf-8")
	except OSError as exc:
		raise ValueError(f"Failed to read properties file: {properties_file}") from exc

	match = re.search(
		r"hosts\.configuration\s*=\s*(.*?)(?:\n\n|\n[a-zA-Z]|$)",
		content,
		re.DOTALL,
	)
	if not match:
		raise ValueError("Could not find 'hosts.configuration' in properties file.")

	hosts_str = match.group(1).replace("\n", "").replace("\\", "").strip()
	host_pairs = re.findall(r"\{(\d+),\s*(\d+)\}", hosts_str)
	if not host_pairs:
		raise ValueError("No host tuples found in 'hosts.configuration'.")

	return [(int(cpu), int(ram)) for cpu, ram in host_pairs]


def build_worker_vm_names(hosts: Iterable[Tuple[int, int]]) -> List[str]:
	"""Recreate worker VM naming strategy used by deploy_gcp_from_properties.py."""
	counters = {spec["prefix"]: 1 for spec in MACHINE_MAP.values()}
	names: List[str] = []

	for cpu, ram in hosts:
		spec = MACHINE_MAP.get((cpu, ram))
		if spec is None:
			LOGGER.warning(
				"Unknown host configuration (%s CPU, %s RAM). Skipping this host.",
				cpu,
				ram,
			)
			continue
		prefix = spec["prefix"]
		names.append(f"{prefix}-{counters[prefix]}")
		counters[prefix] += 1

	return names


def build_vm_name_filter(vm_names: Sequence[str]) -> str:
	"""Build filter expression for VM names in Monitoring metadata labels."""
	escaped = [name.replace('"', r'\"') for name in vm_names]
	clauses = [f'metadata.system_labels.name = "{name}"' for name in escaped]
	return " OR ".join(clauses)


def resolve_project_id(project_id_arg: Optional[str]) -> str:
	"""Resolve GCP project ID from CLI or ADC context."""
	if project_id_arg:
		return project_id_arg

	_, detected_project = google_auth_default()
	if not detected_project:
		raise ValueError(
			"Could not resolve GCP project ID. Pass --project-id or configure ADC project."
		)
	return detected_project


def fetch_metric_timeseries(
	client: monitoring_v3.MetricServiceClient,
	project_id: str,
	vm_names: Sequence[str],
	metric_spec: MetricSpec,
	start_time: datetime,
	end_time: datetime,
	alignment_seconds: int,
) -> pd.DataFrame:
	"""Fetch one metric for all target VMs and return a normalized DataFrame."""
	if not vm_names:
		return pd.DataFrame(columns=["Timestamp", "VM_Name", metric_spec.dataframe_column])

	name_filter = build_vm_name_filter(vm_names)
	filter_expr = (
		'resource.type = "gce_instance" '
		f'AND metric.type = "{metric_spec.metric_type}" '
		f"AND ({name_filter})"
	)

	LOGGER.debug("Filter: %s", filter_expr)

	interval = monitoring_v3.TimeInterval(
		{
			"start_time": {"seconds": int(start_time.timestamp())},
			"end_time": {"seconds": int(end_time.timestamp())},
		}
	)

	aggregation = monitoring_v3.Aggregation(
		{
			"alignment_period": {"seconds": alignment_seconds},
			"per_series_aligner": metric_spec.aligner,
		}
	)

	request = monitoring_v3.ListTimeSeriesRequest(
		{
			"name": f"projects/{project_id}",
			"filter": filter_expr,
			"interval": interval,
			"view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
			"aggregation": aggregation,
		}
	)

	rows: List[Dict[str, object]] = []
	try:
		for series in client.list_time_series(request=request):
			# Extract VM name from resource labels (instance_id)
			# GCE resource labels contain: project_id, instance_id, zone
			try:
				vm_name = series.resource.labels.get("instance_id", "unknown-vm")
			except AttributeError:
				vm_name = "unknown-vm"
			
			LOGGER.debug("Processing time series for VM: %s", vm_name)
			
			for point in series.points:
				# end_time is a DatetimeWithNanoseconds object (already a datetime)
				# Make sure it's timezone-aware
				ts = point.interval.end_time
				if ts.tzinfo is None:
					timestamp = ts.replace(tzinfo=timezone.utc)
				else:
					timestamp = ts

				value = point.value.double_value
				if metric_spec.dataframe_column == "CPU_Usage_Percent":
					value *= 100.0

				rows.append(
					{
						"Timestamp": timestamp,
						"VM_Name": vm_name,
						metric_spec.dataframe_column: float(value),
					}
				)
	except Exception as exc:
		LOGGER.error(
			"API error fetching %s: %s %s",
			metric_spec.metric_type,
			type(exc).__name__,
			str(exc),
		)
		raise RuntimeError(
			f"Monitoring API request failed for metric: {metric_spec.metric_type}"
		) from exc

	return pd.DataFrame(rows)


def build_merged_dataframe(metric_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
	"""Merge all metric frames on Timestamp + VM_Name and normalize missing values."""
	base = pd.DataFrame(columns=["Timestamp", "VM_Name"])
	for frame in metric_frames:
		if frame.empty:
			continue
		if base.empty:
			base = frame
		else:
			base = base.merge(frame, on=["Timestamp", "VM_Name"], how="outer")

	if base.empty:
		return pd.DataFrame(
			columns=[
				"Timestamp",
				"VM_Name",
				"CPU_Usage_Percent",
				"Network_RX_Bytes",
				"Network_TX_Bytes",
			]
		)

	for col in ["CPU_Usage_Percent", "Network_RX_Bytes", "Network_TX_Bytes"]:
		if col not in base.columns:
			base[col] = 0.0

	base["CPU_Usage_Percent"] = base["CPU_Usage_Percent"].fillna(0.0)
	base["Network_RX_Bytes"] = base["Network_RX_Bytes"].fillna(0.0)
	base["Network_TX_Bytes"] = base["Network_TX_Bytes"].fillna(0.0)

	base = base.sort_values(["Timestamp", "VM_Name"]).reset_index(drop=True)
	return base[
		[
			"Timestamp",
			"VM_Name",
			"CPU_Usage_Percent",
			"Network_RX_Bytes",
			"Network_TX_Bytes",
		]
	]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch GCP VM metrics (CPU + network bytes) for experiment VMs."
	)
	parser.add_argument(
		"--properties-file",
		required=True,
		type=Path,
		help="Path to infra properties file used for deployment.",
	)
	parser.add_argument(
		"--window-minutes",
		type=int,
		default=15,
		help="Lookback window in minutes (default: 15).",
	)
	parser.add_argument(
		"--project-id",
		type=str,
		default=None,
		help="GCP project ID. If omitted, resolved from ADC.",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path("gcp_experiment_metrics.csv"),
		help="Output CSV path (default: gcp_experiment_metrics.csv).",
	)
	parser.add_argument(
		"--include-master",
		action="store_true",
		help="Include storm-nimbus in monitored VM list.",
	)
	parser.add_argument(
		"--alignment-seconds",
		type=int,
		default=60,
		help="Cloud Monitoring alignment period in seconds (default: 60).",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Enable debug logging and validation checks.",
	)
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	
	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(
		level=log_level,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)

	try:
		if args.window_minutes <= 0:
			raise ValueError("--window-minutes must be > 0")
		if args.alignment_seconds <= 0:
			raise ValueError("--alignment-seconds must be > 0")

		hosts = parse_hosts_configuration(args.properties_file)
		vm_names = build_worker_vm_names(hosts)
		if args.include_master:
			vm_names = [MASTER_NAME, *vm_names]

		if not vm_names:
			raise ValueError("No VM names were resolved from the provided properties file.")

		project_id = resolve_project_id(args.project_id)
		LOGGER.info("Resolved project: %s", project_id)
		LOGGER.info("Monitoring %d VMs: %s", len(vm_names), ", ".join(vm_names))

		end_time = datetime.now(timezone.utc)
		start_time = end_time - timedelta(minutes=args.window_minutes)
		LOGGER.info(
			"Querying metrics from %s to %s (window: %d min)",
			start_time.isoformat(),
			end_time.isoformat(),
			args.window_minutes,
		)

		client = monitoring_v3.MetricServiceClient()

		metric_frames = []
		for metric in METRICS:
			LOGGER.info("Fetching metric: %s", metric.metric_type)
			frame = fetch_metric_timeseries(
				client=client,
				project_id=project_id,
				vm_names=vm_names,
				metric_spec=metric,
				start_time=start_time,
				end_time=end_time,
				alignment_seconds=args.alignment_seconds,
			)
			LOGGER.info("Retrieved %d data points for %s", len(frame), metric.metric_type)
			metric_frames.append(frame)

		result_df = build_merged_dataframe(metric_frames)
		result_df.to_csv(args.output_csv, index=False)

		LOGGER.info(
			"Done. Wrote %d rows to %s",
			len(result_df),
			args.output_csv,
		)
		return 0

	except DefaultCredentialsError as exc:
		LOGGER.error(
			"No Application Default Credentials found. Set up ADC:\n"
			"  gcloud auth application-default login\n"
			"  gcloud auth application-default set-quota-project <PROJECT_ID>\n"
			"Or use: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
		)
		LOGGER.error("Original error: %s", str(exc))
		return 1
	except Exception as exc:  # pylint: disable=broad-except
		LOGGER.error("Failed to collect VM metrics: %s", exc, exc_info=True)
		return 1


if __name__ == "__main__":
	sys.exit(main())
