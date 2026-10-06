#!/usr/bin/env python3
"""
Aggregate COCO metrics from many runs into one mean ± std table per model.

Walks the layout written by train.py -> validate.py -> coco_evaluate.py:

    <root>/<model>/<experiment>/validation/coco_metrics.csv

and summarises every model over its experiments (typically different seeds). With 23 test images a single
run moves by several AP points between seeds, so compare models on mean ± std over >= 3 seeds, never on one run.

Usage
-----
python scripts/aggregate_results.py --root output/dental_opg_2024
python scripts/aggregate_results.py --root output/dental_opg_2024 --baseline yolo26s --include "yolo26s|v16"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

METRICS = ["AP", "AP50", "AP75", "AR_100"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise coco_metrics.csv files as mean ± std per model.")
    parser.add_argument("--root", type=str, required=True, help="output/<dataset> directory (any parent of the runs).")
    parser.add_argument("--baseline", type=str, default=None, help="Model name to report deltas against.")
    parser.add_argument("--include", type=str, default=None, help="Regex; keep only models whose name matches.")
    parser.add_argument("--out", type=str, default=None, help="Summary CSV path (default: <root>/summary.csv).")
    return parser.parse_args()


def find_column(df, *needles):
    for col in df.columns:
        if all(n.lower() in col.lower() for n in needles):
            return col
    return None


def main():
    args = parse_args()
    root = Path(args.root)
    files = sorted(root.glob("**/validation/coco_metrics.csv"))
    if not files:
        sys.exit(f"No validation/coco_metrics.csv found under {root}")

    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if args.include:
        runs = runs[runs["Model"].str.contains(args.include, regex=True)]
    if runs.empty:
        sys.exit("No runs left after filtering.")

    metrics = [m for m in METRICS if m in runs.columns]
    params_col = find_column(runs, "param")
    gflops_col = find_column(runs, "gflop")

    print(f"\nRuns found: {len(runs)}  (models: {runs['Model'].nunique()})\n")
    print(runs[["Model", "Experiment", *metrics]].sort_values(["Model", "Experiment"]).to_string(index=False))

    rows = []
    for model, sub in runs.groupby("Model"):
        row = {"Model": model, "runs": len(sub)}
        if params_col:
            row["Params(M)"] = round(float(sub[params_col].iloc[0]) / 1e6, 2)
        if gflops_col:
            row["GFLOPs"] = round(float(sub[gflops_col].iloc[0]), 1)
        for m in metrics:
            row[f"{m}_mean"] = 100 * sub[m].mean()
            row[f"{m}_std"] = 100 * sub[m].std(ddof=1) if len(sub) > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("AP_mean" if "AP_mean" in rows[0] else "Model", ascending=False)

    base = None
    if args.baseline:
        hit = summary[summary["Model"] == args.baseline]
        if hit.empty:
            print(f"\nWARNING: baseline '{args.baseline}' not found; available: {sorted(summary['Model'])}")
        else:
            base = hit.iloc[0]

    # Pretty table
    print("\nSummary (mean ± std over runs, AP in %):\n")
    header = ["Model", "runs"] + (["Params(M)"] if params_col else []) + (["GFLOPs"] if gflops_col else []) + metrics
    if base is not None:
        header.append(f"ΔAP vs {args.baseline}")
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for _, r in summary.iterrows():
        cells = [r["Model"], str(int(r["runs"]))]
        if params_col:
            cells.append(f"{r['Params(M)']:.2f}")
        if gflops_col:
            cells.append(f"{r['GFLOPs']:.1f}")
        for m in metrics:
            std = r[f"{m}_std"]
            cells.append(f"{r[f'{m}_mean']:.1f}" + ("" if pd.isna(std) else f" ± {std:.1f}"))
        if base is not None:
            cells.append(f"{r['AP_mean'] - base['AP_mean']:+.1f}")
        print(" | ".join(cells))

    out = Path(args.out) if args.out else root / "summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved summary: {out}")
    if (summary["runs"] < 3).any():
        print("Note: models with < 3 runs cannot be compared reliably on this dataset size; add seeds.")


if __name__ == "__main__":
    main()
