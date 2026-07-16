#!/usr/bin/env python3
"""Compare Week-3 pair_level_summary.csv files with paired bootstrapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=path/to/pair_level_summary.csv")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Condition name is empty")
    return name, Path(path)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "pair_id", "same_yes_no_majority", "yes_yes_majority", "no_no_majority",
        "fwd_gold", "rev_gold", "fwd_majority_label", "rev_majority_label",
        "fwd_p_correct", "rev_p_correct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for col in ("same_yes_no_majority", "yes_yes_majority", "no_no_majority"):
        df[col] = as_bool(df[col])
    df["majority_accuracy"] = (
        (df["fwd_majority_label"] == df["fwd_gold"]).astype(float)
        + (df["rev_majority_label"] == df["rev_gold"]).astype(float)
    ) / 2
    df["joint_majority_correct"] = (
        (df["fwd_majority_label"] == df["fwd_gold"])
        & (df["rev_majority_label"] == df["rev_gold"])
    ).astype(float)
    df["response_accuracy"] = (df["fwd_p_correct"] + df["rev_p_correct"]) / 2
    return df


def summary(name: str, df: pd.DataFrame) -> dict:
    return {
        "condition": name,
        "n_pairs": int(len(df)),
        "iphr_rate": float(df["same_yes_no_majority"].mean()),
        "same_answer_pairs": int(df["same_yes_no_majority"].sum()),
        "yes_yes_pairs": int(df["yes_yes_majority"].sum()),
        "no_no_pairs": int(df["no_no_majority"].sum()),
        "response_accuracy": float(df["response_accuracy"].mean()),
        "question_majority_accuracy": float(df["majority_accuracy"].mean()),
        "joint_pair_accuracy": float(df["joint_majority_correct"].mean()),
        "mean_abs_p_yes_gap": float(df["abs_p_yes_gap"].mean()) if "abs_p_yes_gap" in df else None,
    }


def paired_bootstrap(
    baseline: pd.DataFrame,
    condition: pd.DataFrame,
    column: str,
    n_boot: int,
    seed: int,
) -> dict:
    merged = baseline[["pair_id", column]].merge(
        condition[["pair_id", column]], on="pair_id", suffixes=("_baseline", "_condition")
    )
    if len(merged) != len(baseline) or len(merged) != len(condition):
        raise ValueError("Conditions do not contain the same pair IDs")
    delta = (
        merged[f"{column}_condition"].astype(float).to_numpy()
        - merged[f"{column}_baseline"].astype(float).to_numpy()
    )
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = delta[rng.integers(0, n, n)].mean()
    return {
        "metric": column,
        "n_pairs": n,
        "condition_minus_baseline": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True,
                        help="Repeat NAME=path; one condition must be named baseline")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = dict(args.condition)
    if "baseline" not in paths:
        parser.error("One --condition must be named baseline")

    frames = {name: prepare(path) for name, path in paths.items()}
    summaries = [summary(name, df) for name, df in frames.items()]
    baseline = frames["baseline"]

    comparisons: list[dict] = []
    for name, df in frames.items():
        if name == "baseline":
            continue
        for metric in (
            "same_yes_no_majority", "response_accuracy",
            "majority_accuracy", "joint_majority_correct",
        ):
            result = paired_bootstrap(baseline, df, metric, args.bootstrap, args.seed)
            result["condition"] = name
            comparisons.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out_dir / "week3_condition_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.out_dir / "week3_paired_bootstrap.csv", index=False)
    (args.out_dir / "week3_comparison.json").write_text(
        json.dumps({"summaries": summaries, "comparisons": comparisons}, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
