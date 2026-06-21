import time
from google.cloud import monitoring_v3
from google.api_core.exceptions import GoogleAPIError

class GcpTelemetryProvider:
    """
    Fetches and caches real-time metrics from Google Cloud Monitoring.
    Leverages Application Default Credentials (ADC) when running on a GCP VM.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.cpu_cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 300  # Cache duration in seconds (5 minutes)

    def refresh_cache(self):
        """Fetches the latest CPU metrics for all instances in a single API call."""
        now = time.time()
        # Return immediately if the cache is still valid
        if now - self.cache_timestamp < self.cache_ttl and self.cpu_cache:
            return

        project_name = f"projects/{self.project_id}"
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(now)},
            "start_time": {"seconds": int(now - 300)},
        })

        try:
            # Query GCP for CPU utilization of all compute instances
            results = self.client.list_time_series(
                request={
                    "name": project_name,
                    "filter": 'metric.type = "compute.googleapis.com/instance/cpu/utilization" AND resource.type="gce_instance"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            self.cpu_cache.clear()
            for result in results:
                # The resource labels contain the GCP internal instance_id
                instance_id = result.resource.labels.get("instance_id", "unknown")
                if result.points:
                    # Metric value is a double between 0.0 (0%) and 1.0 (100%)
                    cpu_val = result.points[0].value.double_value
                    self.cpu_cache[instance_id] = cpu_val

            self.cache_timestamp = now
            print(f"[Telemetry] CPU cache updated for {len(self.cpu_cache)} GCP nodes.")

        except GoogleAPIError as e:
            print(f"[Telemetry API Error] Could not fetch metrics: {e}")

    def get_cpu_utilization(self, instance_id: str, default_val: float = 0.1) -> float:
        """
        Retrieves the CPU utilization for a specific node from the cache.
        If the instance is not found, returns a fallback default value.
        """
        self.refresh_cache()
        return self.cpu_cache.get(instance_id, default_val)