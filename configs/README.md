# Configuration Files

This folder stores versioned experiment inputs and shared model settings.

## Subfolders

- `infra/`: infrastructure `.properties` files and VM mapping CSVs.
- `app/`: application `.properties` files.
- `energy/`: shared GCP energy-model properties.

## Usage

The Python placement service, GCP automation scripts, and Storm topology launcher all read from this folder.