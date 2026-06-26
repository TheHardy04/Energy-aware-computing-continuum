#!/usr/bin/env python3

"""Pull the latest experiment artifacts from a GCP VM.

Copies the plot and summary CSV produced by the analysis scripts from the
remote VM into a local destination directory using gcloud compute scp.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_INSTANCE_NAME = "storm-nimbus"
DEFAULT_ZONE = "europe-west9-a"
REMOTE_PROJECT_ROOT = "/home/storm/Energy-aware-computing-continuum"
REMOTE_RESULTS_DIR = f"{REMOTE_PROJECT_ROOT}/experiments/results"
REMOTE_STAGE_DIR = "/tmp/energy-aware-computing-continuum-artifacts"
PLOT_NAME = "algo_comparison_plot.png"
SUMMARY_NAME = "summary_kpi_comparison.csv"


def resolve_gcp_project_id() -> str | None:
    """Resolve the active GCP project from the environment or gcloud config."""

    for env_var in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = os.environ.get(env_var)
        if value:
            return value

    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    project_id = result.stdout.strip()
    if project_id and project_id != "(unset)":
        return project_id
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the latest plot and summary CSV from a GCP VM to your machine.",
    )
    parser.add_argument("--instance", default=DEFAULT_INSTANCE_NAME, help=f"GCP VM name (default: {DEFAULT_INSTANCE_NAME})")
    parser.add_argument("--zone", default=DEFAULT_ZONE, help=f"GCP zone of the VM (default: {DEFAULT_ZONE})")
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parents[1] / "experiments" / "results"),
        help="Local destination directory",
    )
    parser.add_argument(
        "--remote",
        default=REMOTE_RESULTS_DIR,
        help=f"Remote results directory on the VM (default: {REMOTE_RESULTS_DIR})",
    )
    parser.add_argument("--project-id", help="Explicit GCP project ID")
    return parser.parse_args()


def stage_remote_artifacts(project_id: str, instance: str, zone: str, remote_dir: str) -> None:
    stage_command = " && ".join(
        [
            f"rm -rf {shlex.quote(REMOTE_STAGE_DIR)}",
            f"mkdir -p {shlex.quote(REMOTE_STAGE_DIR)}",
            f"cp {shlex.quote(f'{remote_dir}/{PLOT_NAME}')} {shlex.quote(REMOTE_STAGE_DIR)}/",
            f"cp {shlex.quote(f'{remote_dir}/{SUMMARY_NAME}')} {shlex.quote(REMOTE_STAGE_DIR)}/",
            f"chmod 0644 {shlex.quote(REMOTE_STAGE_DIR)}/*",
        ]
    )
    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            f"--project={project_id}",
            f"--zone={zone}",
            "--quiet",
            instance,
            "--command",
            f"sudo -n sh -c {shlex.quote(stage_command)}",
        ],
        check=True,
    )


def copy_artifact(project_id: str, instance: str, zone: str, local_dir: str, artifact_name: str) -> None:
    source = f"{instance}:{REMOTE_STAGE_DIR}/{artifact_name}"
    target = f"{local_dir}/"
    subprocess.run(
        [
            "gcloud",
            "compute",
            "scp",
            f"--project={project_id}",
            f"--zone={zone}",
            "--quiet",
            source,
            target,
        ],
        check=True,
    )


def cleanup_remote_stage(project_id: str, instance: str, zone: str) -> None:
    subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            f"--project={project_id}",
            f"--zone={zone}",
            "--quiet",
            instance,
            "--command",
            f"sudo -n rm -rf {shlex.quote(REMOTE_STAGE_DIR)}",
        ],
        check=False,
    )


def main() -> int:
    args = parse_args()

    project_id = args.project_id or resolve_gcp_project_id()
    if not project_id:
        print(
            "Could not determine the GCP project ID. Set GCP_PROJECT_ID, GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or pass --project-id.",
            file=sys.stderr,
        )
        return 1

    dest_dir = Path(args.dest).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling experiment artifacts from {args.instance} in {args.zone}...")
    print(f"Source: {args.remote}")
    print(f"Destination: {dest_dir}")

    stage_remote_artifacts(project_id, args.instance, args.zone, args.remote)
    try:
        copy_artifact(project_id, args.instance, args.zone, str(dest_dir), PLOT_NAME)
        copy_artifact(project_id, args.instance, args.zone, str(dest_dir), SUMMARY_NAME)
    finally:
        cleanup_remote_stage(project_id, args.instance, args.zone)

    print(f"Copied: {dest_dir / PLOT_NAME}")
    print(f"Copied: {dest_dir / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())