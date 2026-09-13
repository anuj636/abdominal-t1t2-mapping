#!/usr/bin/env python3
"""
Validate a candidate ImDataParamsBMRR_subspace.h5 against a baseline/reference file.

Checks (strict by default):
- Missing required top-level groups in candidate.
- Signal shape mismatch between baseline and candidate.
- Optional CSV chunk-summary row count mismatch.

Exit code:
- 0: pass
- 1: one or more checks failed
- 2: usage/runtime error
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:
    import h5py
except Exception as exc:  # pragma: no cover
    print(f"ERROR: failed to import h5py: {exc}", file=sys.stderr)
    sys.exit(2)


@dataclass
class ValidationResult:
    ok: bool
    messages: List[str]


def parse_groups(value: str) -> List[str]:
    groups = [item.strip() for item in value.split(",") if item.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("required groups list must not be empty")
    return groups


def parse_shape(value: str) -> Tuple[int, ...]:
    try:
        nums = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape list: {value}") from exc
    if not nums:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    if any(n <= 0 for n in nums):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return tuple(nums)


def top_level_groups(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        return sorted([name for name in f.keys()])


def signal_shape(h5_path: str) -> Tuple[int, ...]:
    with h5py.File(h5_path, "r") as f:
        ds_path = "ImDataParams/signal"
        if ds_path not in f:
            raise KeyError(f"missing dataset: {ds_path} in {h5_path}")
        return tuple(int(x) for x in f[ds_path].shape)


def count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def expected_chunk_rows(shape: Sequence[int]) -> int:
    # For compare-format signal tensors in this workflow:
    # (x, y, z, dyn, echo) -> rows = z * dyn.
    # Example observed: (140, 105, 141, 28, 2) -> 141*28 = 3948.
    if len(shape) < 4:
        raise ValueError(f"signal shape must have at least 4 dims, got {shape}")
    return int(shape[2]) * int(shape[3])


def validate(
    baseline_h5: str | None,
    candidate_h5: str,
    required_groups: Iterable[str],
    require_shape_match: bool,
    expected_candidate_shape: Tuple[int, ...] | None,
    baseline_csv: str | None,
    candidate_csv: str | None,
    require_csv_match: bool,
) -> ValidationResult:
    messages: List[str] = []
    ok = True

    cand_groups = top_level_groups(candidate_h5)
    base_groups: List[str] = []
    base_shape: Tuple[int, ...] | None = None

    if baseline_h5:
        base_groups = top_level_groups(baseline_h5)
        base_shape = signal_shape(baseline_h5)

    missing_required = sorted([g for g in required_groups if g not in cand_groups])
    missing_vs_baseline = sorted([g for g in base_groups if g not in cand_groups]) if base_groups else []

    if base_groups:
        messages.append(f"BASE groups: {base_groups}")
    messages.append(f"CAND groups: {cand_groups}")
    messages.append(f"Missing required in candidate: {missing_required if missing_required else 'None'}")
    if base_groups:
        messages.append(f"Missing vs baseline in candidate: {missing_vs_baseline if missing_vs_baseline else 'None'}")

    if missing_required:
        ok = False

    cand_shape = signal_shape(candidate_h5)
    delta = None
    if base_shape is not None and len(base_shape) == len(cand_shape):
        delta = tuple(b - c for b, c in zip(base_shape, cand_shape))

    if base_shape is not None:
        messages.append(f"BASE signal shape: {base_shape}")
    messages.append(f"CAND signal shape: {cand_shape}")
    if base_shape is not None and delta is None:
        messages.append("Signal shape delta (base-cand): n/a (rank mismatch)")
    elif delta is not None:
        messages.append(f"Signal shape delta (base-cand): {delta}")

    if base_shape is not None and require_shape_match and base_shape != cand_shape:
        ok = False
        messages.append("FAIL: candidate signal shape does not match baseline")

    if expected_candidate_shape is not None and cand_shape != expected_candidate_shape:
        ok = False
        messages.append(
            f"FAIL: candidate signal shape {cand_shape} does not match expected {expected_candidate_shape}"
        )

    cand_expected_rows = expected_chunk_rows(cand_shape)
    if base_shape is not None:
        base_expected_rows = expected_chunk_rows(base_shape)
        messages.append(f"Expected chunk rows from BASE signal (dyn*z): {base_expected_rows}")
    messages.append(f"Expected chunk rows from CAND signal (dyn*z): {cand_expected_rows}")

    if candidate_csv:
        if not os.path.exists(candidate_csv):
            ok = False
            messages.append(f"FAIL: candidate CSV not found: {candidate_csv}")
        else:
            cand_rows = count_csv_rows(candidate_csv)
            messages.append(f"CAND CSV rows: {cand_rows}")
            if cand_rows != cand_expected_rows:
                ok = False
                messages.append(
                    f"FAIL: candidate CSV rows ({cand_rows}) do not match CAND expected dyn*z ({cand_expected_rows})"
                )

    if baseline_csv and candidate_csv:
        if not os.path.exists(baseline_csv):
            ok = False
            messages.append(f"FAIL: baseline CSV not found: {baseline_csv}")
        elif os.path.exists(candidate_csv):
            base_rows = count_csv_rows(baseline_csv)
            cand_rows = count_csv_rows(candidate_csv)
            messages.append(f"BASE CSV rows: {base_rows}")
            messages.append(f"Missing CSV rows in candidate (base-cand): {base_rows - cand_rows}")

            if base_shape is not None:
                base_expected_rows = expected_chunk_rows(base_shape)
                if base_rows != base_expected_rows:
                    messages.append(
                        f"WARN: baseline CSV rows ({base_rows}) do not match BASE expected dyn*z ({base_expected_rows})"
                    )

            if require_csv_match and base_rows != cand_rows:
                ok = False
                messages.append("FAIL: candidate CSV row count does not match baseline")

    return ValidationResult(ok=ok, messages=messages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate candidate H5 output against baseline/reference output"
    )
    parser.add_argument("--baseline-h5", required=False, default=None, help="Reference H5 path")
    parser.add_argument("--candidate-h5", required=True, help="Candidate H5 path")
    parser.add_argument(
        "--required-groups",
        default="ImDataParams,MotionParams,RelaxParams",
        type=parse_groups,
        help="Comma-separated list of required top-level groups in candidate",
    )
    parser.add_argument(
        "--no-require-shape-match",
        action="store_true",
        help="Do not fail when candidate signal shape differs from baseline",
    )
    parser.add_argument(
        "--expected-candidate-shape",
        type=parse_shape,
        default=None,
        help="Expected candidate shape as comma-separated ints, e.g. 2,4,139,73,200",
    )
    parser.add_argument("--baseline-csv", default=None, help="Optional baseline summary CSV")
    parser.add_argument("--candidate-csv", default=None, help="Optional candidate summary CSV")
    parser.add_argument(
        "--no-require-csv-match",
        action="store_true",
        help="Do not fail when candidate CSV row count differs from baseline",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    check_paths = [args.candidate_h5]
    if args.baseline_h5:
        check_paths.append(args.baseline_h5)
    for path in check_paths:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2

    result = validate(
        baseline_h5=args.baseline_h5,
        candidate_h5=args.candidate_h5,
        required_groups=args.required_groups,
        require_shape_match=not args.no_require_shape_match,
        expected_candidate_shape=args.expected_candidate_shape,
        baseline_csv=args.baseline_csv,
        candidate_csv=args.candidate_csv,
        require_csv_match=not args.no_require_csv_match,
    )

    print("=== H5 VALIDATION REPORT ===")
    for line in result.messages:
        print(line)

    if result.ok:
        print("RESULT: PASS")
        return 0

    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
