"""Aggregate UCLA ablation metrics into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate ablation experiment metrics.")
    parser.add_argument(
        "--ablation_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Root directory of ablation experiments.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="ablation_final_summary.txt",
        help="Filename for final text summary.",
    )
    return parser.parse_args()


def discover_metrics(ablation_root: Path) -> List[Path]:
    """Discover all metrics.json files under ablation subdirectories."""
    candidates: List[Path] = []
    for metrics_path in ablation_root.glob("*/*/metrics.json"):
        if metrics_path.is_file():
            candidates.append(metrics_path)
    for metrics_path in ablation_root.glob("*/logs/*/metrics.json"):
        if metrics_path.is_file():
            candidates.append(metrics_path)
    unique_sorted = sorted(set(path.resolve() for path in candidates))
    return unique_sorted


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Parse one metrics.json file into a normalized record."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    summary = payload.get("summary", {})
    cfg = metadata.get("config", {})
    return {
        "experiment_name": metadata.get("experiment_name", metrics_path.parents[2].name),
        "seed": int(metadata.get("seed", -1)),
        "modality": cfg.get("modality", "unknown"),
        "pretrain_epochs": int(cfg.get("pretrain_epochs", 0)),
        "finetune_epochs": int(cfg.get("finetune_epochs", 0)),
        "best_epoch": int(summary.get("best_epoch", -1)),
        "best_val_acc": safe_float(summary.get("best_val_acc", 0.0)),
        "best_test_acc": safe_float(summary.get("best_test_acc", 0.0)),
        "metrics_path": str(metrics_path),
    }


def format_mean_std(values: List[float]) -> str:
    """Format value list as mean±std."""
    if not values:
        return "0.0000±0.0000"
    val_mean = mean(values)
    val_std = pstdev(values) if len(values) > 1 else 0.0
    return f"{val_mean:.4f}±{val_std:.4f}"


def format_mean_std_epoch(values: List[int]) -> str:
    """Format epoch list as mean±std."""
    if not values:
        return "0.00±0.00"
    val_mean = mean(values)
    val_std = pstdev(values) if len(values) > 1 else 0.0
    return f"{val_mean:.2f}±{val_std:.2f}"


def aggregate_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate records by experiment and modality."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = f"{record['experiment_name']}|{record['modality']}"
        if key not in grouped:
            grouped[key] = {
                "experiment_name": record["experiment_name"],
                "modality": record["modality"],
                "seeds": [],
                "pretrain_epochs": [],
                "finetune_epochs": [],
                "best_epochs": [],
                "best_val_accs": [],
                "best_test_accs": [],
            }
        grouped[key]["seeds"].append(record["seed"])
        grouped[key]["pretrain_epochs"].append(record["pretrain_epochs"])
        grouped[key]["finetune_epochs"].append(record["finetune_epochs"])
        grouped[key]["best_epochs"].append(record["best_epoch"])
        grouped[key]["best_val_accs"].append(record["best_val_acc"])
        grouped[key]["best_test_accs"].append(record["best_test_acc"])

    rows: List[Dict[str, Any]] = []
    for group in grouped.values():
        rows.append(
            {
                "experiment_name": group["experiment_name"],
                "modality": group["modality"],
                "num_runs": len(group["seeds"]),
                "seeds": ",".join(str(seed) for seed in sorted(group["seeds"])),
                "pretrain_epochs": format_mean_std_epoch(group["pretrain_epochs"]),
                "finetune_epochs": format_mean_std_epoch(group["finetune_epochs"]),
                "best_epoch_mean_std": format_mean_std_epoch(group["best_epochs"]),
                "best_val_acc_mean_std": format_mean_std(group["best_val_accs"]),
                "best_test_acc_mean_std": format_mean_std(group["best_test_accs"]),
            }
        )
    rows.sort(key=lambda item: (item["experiment_name"], item["modality"]))
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write aggregate rows to CSV."""
    fieldnames = [
        "experiment_name",
        "modality",
        "num_runs",
        "seeds",
        "pretrain_epochs",
        "finetune_epochs",
        "best_epoch_mean_std",
        "best_val_acc_mean_std",
        "best_test_acc_mean_std",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write aggregate rows to LaTeX table."""
    lines: List[str] = [
        r"\begin{tabular}{llrllllll}",
        r"\hline",
        (
            r"Experiment & Modality & Runs & Seeds & Pretrain Epochs (mean$\pm$std) & "
            r"Finetune Epochs (mean$\pm$std) & Best Epoch (mean$\pm$std) & "
            r"Val (mean$\pm$std) & Test (mean$\pm$std) \\"
        ),
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['experiment_name']} & {row['modality']} & {row['num_runs']} & "
            f"{row['seeds']} & {row['pretrain_epochs']} & {row['finetune_epochs']} & "
            f"{row['best_epoch_mean_std']} & {row['best_val_acc_mean_std']} & "
            f"{row['best_test_acc_mean_std']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_txt(
    records: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write concise final summary without intermediate training logs."""
    sorted_records = sorted(
        records,
        key=lambda item: (item["experiment_name"], item["seed"], item["modality"]),
    )
    lines: List[str] = ["Ablation Final Summary", "=====================", ""]
    lines.append("Per-Run Final Results")
    lines.append("---------------------")
    for record in sorted_records:
        lines.append(
            f"{record['experiment_name']} | seed={record['seed']} | modality={record['modality']} | "
            f"pretrain_epochs={record['pretrain_epochs']} | finetune_epochs={record['finetune_epochs']} | "
            f"best_epoch={record['best_epoch']} | best_val_acc={record['best_val_acc']:.4f} | "
            f"best_test_acc={record['best_test_acc']:.4f}"
        )
    lines.append("")
    lines.append("Aggregated Results (Mean±Std)")
    lines.append("-----------------------------")
    for row in rows:
        lines.append(
            f"{row['experiment_name']} | modality={row['modality']} | runs={row['num_runs']} | "
            f"seeds={row['seeds']} | pretrain_epochs={row['pretrain_epochs']} | "
            f"finetune_epochs={row['finetune_epochs']} | best_epoch={row['best_epoch_mean_std']} | "
            f"best_val_acc={row['best_val_acc_mean_std']} | best_test_acc={row['best_test_acc_mean_std']}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Aggregate all experiment results and print summary statistics."""
    args = parse_args()
    ablation_root = Path(args.ablation_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ablation_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = discover_metrics(ablation_root)
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json found under: {ablation_root}")

    records = [parse_metrics(path) for path in metrics_files]
    rows = aggregate_rows(records)

    csv_path = output_dir / "ablation_comparison.csv"
    tex_path = output_dir / "ablation_comparison.tex"
    summary_path = output_dir / args.summary_file
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    write_summary_txt(records, rows, summary_path)

    print(f"Discovered {len(metrics_files)} metrics files.")
    print(f"CSV table written to: {csv_path}")
    print(f"LaTeX table written to: {tex_path}")
    print(f"Summary text written to: {summary_path}")
    print("Mean±std summary:")
    for row in rows:
        print(
            f"{row['experiment_name']} ({row['modality']}): "
            f"best_epoch={row['best_epoch_mean_std']}, "
            f"val={row['best_val_acc_mean_std']}, test={row['best_test_acc_mean_std']}"
        )


if __name__ == "__main__":
    main()
