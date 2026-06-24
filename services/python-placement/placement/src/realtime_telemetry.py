"""Real-time telemetry daemon for GCP and Storm metrics.

The daemon polls Google Cloud Monitoring and the local Apache Storm REST API
every 30 seconds, then appends telemetry rows to separate CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from google.api_core.exceptions import GoogleAPIError
from google.cloud import monitoring_v3


CURRENT_DIR = Path(__file__).resolve().parent
PLACEMENT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PLACEMENT_ROOT.parents[2]

if str(PLACEMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLACEMENT_ROOT))

from src.gcpEnergyModel import _load_energy_settings, node_power_w  # noqa: E402
from src.gcp_telemetry import GcpTelemetryProvider  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_WINDOW_SECONDS = 30
DEFAULT_STORM_BASE_URL = "http://localhost:8080"
DEFAULT_STORM_CSV = REPO_ROOT / "experiments" / "raw" / "storm_metrics.csv"
DEFAULT_GCP_CSV = REPO_ROOT / "experiments" / "raw" / "gcp_metrics.csv"
DEFAULT_VCPU_CAPACITY = 1
DEFAULT_HTTP_TIMEOUT_SECONDS = 8.0
POWER_SCALE = 1000
E_BIT = float(os.environ.get("REALTIME_TELEMETRY_E_BIT", "1e-7"))

STORM_CSV_COLUMNS = [
    "Timestamp",
    "Topology_ID",
    "Tuples_Emitted",
    "Average_Latency_ms",
]

GCP_CSV_COLUMNS = [
    "Exact_GCP_Timestamp",
    "Node_Name",
    "CPU_Utilization",
    "Bytes_Sent",
    "Compute_Power_W",
    "Network_Power_W",
    "Total_Power_W",
]


@dataclass(frozen=True)
class StormSnapshot:
    """Storm application telemetry extracted from the REST API."""

    topology_id: str = ""
    tuples_emitted: int = 0
    average_latency_ms: float = 0.0


@dataclass(frozen=True)
class GcpSample:
    """Merged GCP metrics for one monitored instance."""

    instance_id: str
    exact_gcp_timestamp: str
    node_name: str
    cpu_utilization: float
    bytes_sent: float
    compute_power_w: float
    network_power_w: float
    total_power_w: float


class StormApiClient:
    """Client for the local Apache Storm REST API."""

    def __init__(self, base_url: str = DEFAULT_STORM_BASE_URL, timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected Storm response from {url}: expected a JSON object.")
        return payload

    def fetch_snapshot(self) -> StormSnapshot:
        summary_payload = self._get_json("/api/v1/topology/summary")
        topology_id = _select_topology_id(summary_payload)
        detail_payload = self._get_json(f"/api/v1/topology/{topology_id}")

        emitted = _search_nested_value(detail_payload, ("emitted", "tuplesEmitted", "tuples_emitted"))
        latency = _search_nested_value(detail_payload, ("completeLatency", "complete_latency", "averageLatency"))

        return StormSnapshot(
            topology_id=topology_id,
            tuples_emitted=_to_int(emitted, 0),
            average_latency_ms=_to_float(latency, 0.0),
        )


class RealtimeGcpTelemetryProvider(GcpTelemetryProvider):
    """Extends the repository telemetry provider with network-byte collection."""

    def __init__(self, project_id: str, window_seconds: int = DEFAULT_WINDOW_SECONDS, default_vcpu_capacity: int = DEFAULT_VCPU_CAPACITY):
        super().__init__(project_id)
        self.window_seconds = max(1, int(window_seconds))
        self.default_vcpu_capacity = max(1, int(default_vcpu_capacity))
        self.network_cache: Dict[str, float] = {}
        self.node_name_cache: Dict[str, str] = {}
        self.point_timestamp_cache: Dict[str, str] = {}

    def _fetch_metric_map(self, metric_type: str, aligner: monitoring_v3.Aggregation.Aligner) -> Dict[str, float]:
        now = time.time()
        project_name = f"projects/{self.project_id}"
        interval = monitoring_v3.TimeInterval(
            {
                "start_time": {"seconds": int(now - 240)},
                "end_time": {"seconds": int(now)},
            }
        )
        aggregation = monitoring_v3.Aggregation(
            {
                "alignment_period": {"seconds": self.window_seconds},
                "per_series_aligner": aligner,
            }
        )

        results = self.client.list_time_series(
            request={
                "name": project_name,
                "filter": f'metric.type = "{metric_type}" AND resource.type="gce_instance"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": aggregation,
            }
        )

        metric_map: Dict[str, float] = {}
        for result in results:
            instance_id = _extract_instance_id(result)
            metric_value, exact_timestamp = _latest_point_value(result)
            metric_map[instance_id] = metric_value
            self.point_timestamp_cache[instance_id] = exact_timestamp
            self.node_name_cache[instance_id] = _extract_node_name(result)
        return metric_map

    def refresh_window(self) -> None:
        """Refresh CPU utilization and network-out caches for the current window."""
        self.cpu_cache = self._fetch_metric_map(
            "compute.googleapis.com/instance/cpu/utilization",
            monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
        )
        self.network_cache = self._fetch_metric_map(
            "compute.googleapis.com/instance/network/sent_bytes_count",
            monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
        )
        self.cache_timestamp = time.time()

    def collect_snapshot(self) -> List[GcpSample]:
        """Build merged GCP telemetry for the latest window."""
        self.refresh_window()
        cfg = _load_energy_settings()
        cpu_capacity = self.default_vcpu_capacity * POWER_SCALE

        instance_ids = sorted(set(self.cpu_cache.keys()) | set(self.network_cache.keys()))
        samples: List[GcpSample] = []

        for instance_id in instance_ids:
            cpu_utilization = max(0.0, _to_float(self.cpu_cache.get(instance_id, 0.0), 0.0))
            bytes_sent = max(0.0, _to_float(self.network_cache.get(instance_id, 0.0), 0.0))
            cpu_used = int(round(cpu_utilization * cpu_capacity))
            exact_timestamp = self.point_timestamp_cache.get(
                instance_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

            compute_power_w = node_power_w(cpu_used, cpu_capacity, cfg) / POWER_SCALE
            network_power_w = (bytes_sent * 8.0 * E_BIT) / float(self.window_seconds)

            samples.append(
                GcpSample(
                    instance_id=instance_id,
                    exact_gcp_timestamp=exact_timestamp,
                    node_name=self.node_name_cache.get(instance_id, instance_id),
                    cpu_utilization=cpu_utilization,
                    bytes_sent=bytes_sent,
                    compute_power_w=compute_power_w,
                    network_power_w=network_power_w,
                    total_power_w=compute_power_w + network_power_w,
                )
            )

        return samples


def resolve_project_id() -> str:
    """Resolve the GCP project ID from common environment variables."""
    for env_name in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        "Could not resolve the GCP project ID. Set GCP_PROJECT_ID, GOOGLE_CLOUD_PROJECT, or GCLOUD_PROJECT."
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _latest_point_value(series: Any) -> Tuple[float, str]:
    """Return the latest numeric value and end timestamp from a Monitoring time series."""
    points = getattr(series, "points", None) or []
    if not points:
        return 0.0, datetime.fromtimestamp(0, tz=timezone.utc).isoformat(timespec="seconds")

    def sort_key(point: Any) -> Tuple[int, int]:
        interval = getattr(point, "interval", None)
        end_time = getattr(interval, "end_time", None)
        seconds = int(getattr(end_time, "seconds", 0) or 0)
        nanos = int(getattr(end_time, "nanos", 0) or 0)
        return seconds, nanos

    point = max(points, key=sort_key)
    interval = getattr(point, "interval", None)
    end_time = getattr(interval, "end_time", None)
    end_seconds = int(getattr(end_time, "seconds", 0) or 0)
    exact_timestamp = datetime.fromtimestamp(end_seconds, tz=timezone.utc).isoformat(timespec="seconds")
    value = getattr(point, "value", None)
    if value is None:
        return 0.0, exact_timestamp

    value_pb = getattr(value, "_pb", None)
    oneof = value_pb.WhichOneof("value") if value_pb is not None else None
    if oneof == "double_value":
        return float(value.double_value), exact_timestamp
    if oneof == "int64_value":
        return float(value.int64_value), exact_timestamp
    if oneof == "bool_value":
        return (1.0 if value.bool_value else 0.0), exact_timestamp
    if getattr(value, "double_value", None) is not None:
        return float(value.double_value), exact_timestamp
    if getattr(value, "int64_value", None) is not None:
        return float(value.int64_value), exact_timestamp
    return 0.0, exact_timestamp


def _extract_instance_id(series: Any) -> str:
    resource = getattr(series, "resource", None)
    labels = getattr(resource, "labels", {}) or {}
    return str(labels.get("instance_id", "unknown-instance"))


def _extract_node_name(series: Any) -> str:
    metadata = getattr(series, "metadata", None)
    system_labels = getattr(metadata, "system_labels", {}) or {}
    node_name = str(system_labels.get("name", "")).strip()
    if node_name:
        return node_name
    return _extract_instance_id(series)


def _select_topology_id(summary_payload: Dict[str, Any]) -> str:
    topologies = summary_payload.get("topologies") or summary_payload.get("Topologies") or []
    if not isinstance(topologies, list) or not topologies:
        raise ValueError("Storm summary did not return any topologies.")

    def topology_score(item: Dict[str, Any]) -> Tuple[int, float]:
        status = str(item.get("status") or item.get("topologyStatus") or "").upper()
        active_score = 1 if status in {"ACTIVE", "RUNNING", "STARTED"} else 0
        uptime = _to_float(item.get("uptimeSeconds") or item.get("uptimeSeconds", 0.0), 0.0)
        return active_score, uptime

    chosen = max(topologies, key=topology_score)
    topology_id = (
        chosen.get("id")
        or chosen.get("topologyId")
        or chosen.get("topology_id")
        or chosen.get("topologyID")
    )
    if not topology_id:
        raise ValueError("Storm summary topology entry did not contain an ID.")
    return str(topology_id)


def _search_nested_value(payload: Any, keys: Iterable[str]) -> Optional[Any]:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        for value in payload.values():
            found = _search_nested_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _search_nested_value(item, keys)
            if found is not None:
                return found
    return None


def write_rows(output_csv: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Append telemetry rows to the output CSV and create the header when needed."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0

    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cycle(
    gcp_provider: RealtimeGcpTelemetryProvider,
    storm_client: StormApiClient,
    storm_csv: Path,
    gcp_csv: Path,
) -> None:
    """Fetch both telemetry sources in parallel and persist each stream independently."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        gcp_future = executor.submit(gcp_provider.collect_snapshot)
        storm_future = executor.submit(storm_client.fetch_snapshot)

        storm_snapshot = StormSnapshot()
        gcp_samples: List[GcpSample] = []

        try:
            storm_snapshot = storm_future.result()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Storm API is unavailable for this cycle: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Unexpected Storm API error: %s", exc)

        try:
            gcp_samples = gcp_future.result()
        except GoogleAPIError as exc:
            LOGGER.warning("GCP Monitoring is unavailable for this cycle: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Unexpected GCP telemetry error: %s", exc)

    if storm_snapshot.topology_id:
        storm_rows = [
            {
                "Timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "Topology_ID": storm_snapshot.topology_id,
                "Tuples_Emitted": storm_snapshot.tuples_emitted,
                "Average_Latency_ms": storm_snapshot.average_latency_ms,
            }
        ]
        write_rows(storm_csv, storm_rows, STORM_CSV_COLUMNS)
        LOGGER.info("Wrote %d Storm telemetry rows to %s", len(storm_rows), storm_csv)
    else:
        LOGGER.warning("No Storm telemetry row was collected during this cycle.")

    if not gcp_samples:
        LOGGER.warning("No GCP telemetry rows were collected during this cycle.")
        return

    gcp_rows: List[Dict[str, Any]] = []
    for sample in gcp_samples:
        gcp_rows.append(
            {
                "Exact_GCP_Timestamp": sample.exact_gcp_timestamp,
                "Node_Name": sample.node_name,
                "CPU_Utilization": sample.cpu_utilization,
                "Bytes_Sent": sample.bytes_sent,
                "Compute_Power_W": sample.compute_power_w,
                "Network_Power_W": sample.network_power_w,
                "Total_Power_W": sample.total_power_w,
            }
        )

    write_rows(gcp_csv, gcp_rows, GCP_CSV_COLUMNS)
    LOGGER.info("Wrote %d GCP telemetry rows to %s", len(gcp_rows), gcp_csv)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the daemon."""
    parser = argparse.ArgumentParser(description="Real-time GCP and Storm telemetry daemon")
    parser.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", ""), help="GCP project ID. Defaults to GCP_PROJECT_ID.")
    parser.add_argument("--storm-url", default=DEFAULT_STORM_BASE_URL, help="Base URL for the Storm REST API.")
    parser.add_argument("--storm-csv", default=str(DEFAULT_STORM_CSV), help="Output CSV path for Storm metrics.")
    parser.add_argument("--gcp-csv", default=str(DEFAULT_GCP_CSV), help="Output CSV path for GCP metrics.")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS, help="Loop interval in seconds.")
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS, help="Monitoring window in seconds.")
    parser.add_argument("--default-vcpu-capacity", type=int, default=DEFAULT_VCPU_CAPACITY, help="Fallback vCPU capacity per monitored VM.")
    parser.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS, help="HTTP timeout for Storm requests.")
    parser.add_argument("--once", action="store_true", help="Run a single telemetry cycle and exit.")
    return parser.parse_args()


def main() -> int:
    """Run the telemetry daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    project_id = args.project_id.strip() or resolve_project_id()
    storm_csv = Path(args.storm_csv).expanduser().resolve()
    gcp_csv = Path(args.gcp_csv).expanduser().resolve()

    gcp_provider = RealtimeGcpTelemetryProvider(
        project_id=project_id,
        window_seconds=args.window_seconds,
        default_vcpu_capacity=args.default_vcpu_capacity,
    )
    storm_client = StormApiClient(
        base_url=args.storm_url,
        timeout_seconds=args.http_timeout_seconds,
    )

    LOGGER.info("Starting telemetry daemon. Storm output: %s", storm_csv)
    LOGGER.info("Starting telemetry daemon. GCP output: %s", gcp_csv)

    while True:
        cycle_start = time.time()
        try:
            run_cycle(gcp_provider, storm_client, storm_csv, gcp_csv)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Telemetry cycle failed: %s", exc)

        if args.once:
            break

        elapsed = time.time() - cycle_start
        sleep_seconds = max(0.0, float(args.interval_seconds) - elapsed)
        time.sleep(sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
