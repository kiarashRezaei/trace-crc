#!/usr/bin/env python3
"""Validate the portable TRACE-CRC reproducibility artifact."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DIU_DIR = ROOT / "DiU"
OUTPUT_DIR = DIU_DIR / "final_comparison_revision_all_in_one_reproducibility"
NOTEBOOK_PATH = DIU_DIR / "diu_conformal_all_in_one_reproducibility.ipynb"

INFERENCE_SEEDS = (
    20260724,
    20260824,
    20260924,
    20261024,
    20261124,
)
EXPECTED_CACHE_METADATA = {
    "dataset_files": ["data/csi/CSI_Dataset_1000_diverse.pt"],
    "max_samples_total": 1000,
    "checkpoint_path": "DiU/models/models_900.pt",
    "config_path": "DiU/config.yml",
    "T_in": 10,
    "T_out": 20,
    "inference_snr": 20.0,
    "sampling_steps": 20,
    "batch_size": 16,
}
EXPECTED_METHOD_COUNT = 19
EXPECTED_PRIMARY_ROWS = EXPECTED_METHOD_COUNT * len(INFERENCE_SEEDS)
EXPECTED_TRACE_SPLIT_ROWS = 50


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_required_files() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "CITATION.cff",
        ROOT / "requirements.txt",
        ROOT / "conformal" / "core" / "LICENSE",
        ROOT / "conformal" / "core" / "README.md",
        DIU_DIR / "PARAMETER_NOTES.md",
        DIU_DIR / "config.yml",
        DIU_DIR / "inference_helpers.py",
        NOTEBOOK_PATH,
        OUTPUT_DIR / "parameter_selection.json",
        OUTPUT_DIR / "robustness_summary.json",
        OUTPUT_DIR / "manifest.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        fail(f"Missing required files: {missing}")


def validate_notebook() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell.get("source", [])), filename=f"code-cell-{index}")
    if any(cell.get("outputs") for cell in code_cells):
        fail("The committed notebook must not contain execution outputs.")
    serialized = NOTEBOOK_PATH.read_text(encoding="utf-8")
    if "/Users/" in serialized or "Projects/cp-csi" in serialized:
        fail("The notebook contains a machine-specific absolute path.")
    if "latex_figure_dir" in serialized or "PGFPLOTS/TIKZ FIGURE DATA" in serialized:
        fail("The GitHub notebook still contains LaTeX-figure export code.")
    if (OUTPUT_DIR / "latex_figures").exists():
        fail("The GitHub artifact must not contain a latex_figures directory.")


def validate_caches() -> dict[str, str]:
    hashes = {}
    cache_dir = OUTPUT_DIR / "compact_seed_inference"
    for seed in INFERENCE_SEEDS:
        path = cache_dir / f"compact_seed_{seed}.npz"
        if not path.exists():
            fail(f"Missing compact cache: {path.relative_to(ROOT)}")
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
            expected = {**EXPECTED_CACHE_METADATA, "seed": seed}
            if metadata != expected:
                fail(f"Cache metadata mismatch for seed {seed}: {metadata}")
            if cache["fro_residual"].shape != (1000, 20):
                fail(f"Residual shape mismatch for seed {seed}.")
            if cache["predicted_norm"].shape != (1000, 20):
                fail(f"Predicted-norm shape mismatch for seed {seed}.")
            if cache["nmse_db"].shape != (20,):
                fail(f"NMSE shape mismatch for seed {seed}.")
        hashes[path.name] = sha256(path)
    return hashes


def validate_parameter_selection() -> None:
    path = OUTPUT_DIR / "parameter_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("test_data_used_for_tuning") is not False:
        fail("Parameter-selection metadata does not confirm held-out test data.")
    frozen = selection["trace_frozen_for_evaluation"]
    expected = {
        "profile_fraction": 0.10,
        "cp_fraction_within_rest": 0.15,
        "num_groups": 2,
        "smooth_window": 3,
        "floor_ratio": 0.15,
        "ridge_alpha": 1.0,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            fail(f"Frozen TRACE parameter mismatch: {key}={frozen.get(key)!r}")
    if frozen.get("empirical_fallback_allowed") is not False:
        fail("TRACE metadata permits an empirical fallback.")


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def validate_result_tables() -> None:
    primary_path = OUTPUT_DIR / "tables/method_inference_seed_metrics.csv"
    split_path = OUTPUT_DIR / "tables/trace_crc_split_metrics.csv"
    primary_rows = count_csv_rows(primary_path)
    split_rows = count_csv_rows(split_path)
    if primary_rows != EXPECTED_PRIMARY_ROWS:
        fail(
            f"Expected {EXPECTED_PRIMARY_ROWS} primary rows, found {primary_rows}."
        )
    if split_rows != EXPECTED_TRACE_SPLIT_ROWS:
        fail(
            f"Expected {EXPECTED_TRACE_SPLIT_ROWS} TRACE split rows, "
            f"found {split_rows}."
        )


def validate_metadata_files() -> None:
    citation = yaml.safe_load(
        (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    if citation.get("license") != "MIT":
        fail("CITATION.cff does not declare the MIT license.")

    project_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if (
        "MIT License" not in project_license
        or "Copyright (c) 2026 Kiarash Rezaei" not in project_license
    ):
        fail("The project-level MIT license is incomplete.")

    upstream_license = (
        ROOT / "conformal" / "core" / "LICENSE"
    ).read_text(encoding="utf-8")
    if (
        "MIT License" not in upstream_license
        or "Copyright (c) 2021 Anastasios Angelopoulos"
        not in upstream_license
    ):
        fail("The vendored LTT copyright or license notice is missing.")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    required_notice_fragments = (
        "https://github.com/aangelopoulos/ltt",
        "3ad7a64adad9f356e29f40e73935a6114f896396",
        "10.1214/24-AOAS1998",
    )
    if any(fragment not in notices for fragment in required_notice_fragments):
        fail("The LTT third-party provenance notice is incomplete.")

    manifest = json.loads(
        (OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("output_root") != (
        "DiU/final_comparison_revision_all_in_one_reproducibility"
    ):
        fail("Manifest output_root is not repository-relative.")


def main() -> int:
    validate_required_files()
    validate_notebook()
    cache_hashes = validate_caches()
    validate_parameter_selection()
    validate_result_tables()
    validate_metadata_files()

    print("TRACE-CRC repository validation: PASS")
    print(f"Repository root: {ROOT}")
    print(f"Validated inference caches: {len(cache_hashes)}")
    print(f"Validated primary result rows: {EXPECTED_PRIMARY_ROWS}")
    print(f"Validated TRACE seed/split rows: {EXPECTED_TRACE_SPLIT_ROWS}")
    for filename, digest in cache_hashes.items():
        print(f"{digest}  {filename}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"TRACE-CRC repository validation: FAIL\n{error}", file=sys.stderr)
        sys.exit(1)
