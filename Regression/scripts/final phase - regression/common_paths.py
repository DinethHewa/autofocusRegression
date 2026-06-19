#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

DATA_DIR = "/home/dineth/focus_measure/journal/Regression/data"
TABLES_DIR = "/home/dineth/focus_measure/journal/Regression/tables"
PHASE3_MANIFEST_CREATION_DIR = "/home/dineth/focus_measure/journal/Regression/scripts/phase 3 - sign selection/manifest_creation"


def dataset_to_tables_subdir(dataset: str) -> str:
    return dataset.upper()


def manifest_path(dataset: str) -> str:
    return str(Path(DATA_DIR) / f"manifest_{dataset}.csv")


def table_path(dataset: str) -> str:
    table_name = f"{dataset}_table.csv"
    return str(Path(TABLES_DIR) / dataset_to_tables_subdir(dataset) / table_name)
