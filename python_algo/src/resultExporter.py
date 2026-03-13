from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Dict, Any, Optional

from src.placementAlgo import PlacementResult

if TYPE_CHECKING:
    from src.evaluation import EvaluationMetrics


class ResultExporter:
    """
    Utility class to export placement results and evaluation metrics.
    """

    @staticmethod
    def export_placement_to_csv(placement_result: PlacementResult, filename: str) -> None:
        """
        Exports the placement mapping to a CSV file with columns: Component, Host.

        :param placement_result: The placement result containing the mapping to export
        :type placement_result: PlacementResult
        :param filename: The name of the CSV file to write to
        :type filename: str
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Component', 'Host'])
            for comp, host in placement_result.mapping.items():
                writer.writerow([comp, host])
        print(f"Placement mapping exported to {filename}")

    @staticmethod
    def export_metrics_to_csv(
        metrics: EvaluationMetrics,
        filename: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        append: bool = True,
    ) -> None:
        """
        Exports evaluation metrics to a CSV file.

        Each call writes one row of scalar metrics. When ``append=True`` the row is
        appended to an existing file (useful for multi-run benchmarks); otherwise the
        file is overwritten. Any dict-valued fields (host_cpu_usage, host_ram_usage)
        and the violations list are serialised as compact strings.

        :param metrics: The evaluation metrics to export.
        :param filename: Destination CSV file path.
        :param extra_fields: Optional dict of additional columns to prepend, e.g.
            ``{'strategy': 'GreedyFirstFit', 'infra': 'Infra_5nodes_GCP', 'app': 'Appli_5comps_GCP'}``.
        :param append: If True, append to an existing file and skip the header when the
            file already exists. If False, always overwrite.
        """
        if not filename.endswith('.csv'):
            filename += '.csv'

        extra_fields = extra_fields or {}

        SCALAR_COLUMNS = [
            ('total_energy_w',        metrics.total_energy),
            ('energy_node_w',         metrics.energy_node),
            ('energy_link_w',         metrics.energy_link),
            ('avg_latency_ms',        metrics.avg_latency),
            ('worst_latency_ms',      metrics.worst_latency),
            ('total_latency_ms',      metrics.total_latency),
            ('active_hosts',          metrics.active_hosts_count),
            ('solver_energy_wh',      metrics.solver_energy_wh),
            ('solver_energy_min_wh',  metrics.solver_energy_min_wh),
            ('solver_energy_max_wh',  metrics.solver_energy_max_wh),
            ('solver_energy_j',       metrics.solver_energy_j),
            ('solver_energy_min_j',   metrics.solver_energy_min_j),
            ('solver_energy_max_j',   metrics.solver_energy_max_j),
            ('solver_energy_model',   metrics.solver_energy_model),
            ('host_cpu_usage',        str(dict(metrics.host_cpu_usage))),
            ('host_ram_usage',        str(dict(metrics.host_ram_usage))),
            ('violations',            '; '.join(metrics.violations) if metrics.violations else ''),
        ]

        all_columns = list(extra_fields.keys()) + [col for col, _ in SCALAR_COLUMNS]
        all_values  = list(extra_fields.values()) + [val for _, val in SCALAR_COLUMNS]

        file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
        write_header = not (append and file_exists)
        mode = 'a' if append else 'w'

        with open(filename, mode=mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(all_columns)
            writer.writerow(all_values)

        print(f"Evaluation metrics exported to {filename}")