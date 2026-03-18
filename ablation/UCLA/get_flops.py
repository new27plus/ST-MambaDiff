from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from thop import profile

from ablation_core import ClassifierModel, load_simple_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Params and GFLOPs for UCLA ablations.")
    parser.add_argument(
        "--ablation_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Ablation root directory.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="conv_only,transformer_only,mamba_only,standard_ddpm,neutral_decoder",
        help="Comma-separated experiment names.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy input batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for FLOPs profiling.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first profiling error.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV path for results.",
    )
    return parser.parse_args()


def load_experiment_cfg(ablation_root: Path, experiment_name: str) -> Dict[str, Any]:
    config_path = ablation_root / experiment_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return load_simple_yaml(config_path)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --device cuda is requested.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_one(cfg: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[str, Any]:
    model = ClassifierModel(cfg).to(device)
    model.eval()
    dummy_input = torch.randn(
        batch_size,
        int(cfg["in_channels"]),
        int(cfg["max_frames"]),
        int(cfg["num_joints"]),
        device=device,
    )
    params = sum(param.numel() for param in model.parameters())
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return {
        "params_m": float(params) / 1e6,
        "gflops": float(macs) * 2.0 / 1e9,
    }


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "experiment",
        "temporal_operator",
        "decoder_time_condition",
        "device",
        "status",
        "params_m",
        "gflops",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ablation_root = Path(args.ablation_root).resolve()
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []
    device = resolve_device(args.device)

    print("=" * 132)
    print("Experiment                 | Temporal     | DecoderT | Device | Status  | Params(M) | GFLOPs")
    print("-" * 132)
    for experiment_name in experiments:
        cfg = load_experiment_cfg(ablation_root, experiment_name)
        temporal_operator = str(cfg.get("temporal_operator", "unknown"))
        decoder_time_condition = str(cfg.get("decoder_time_condition", False))
        row: Dict[str, Any] = {
            "experiment": experiment_name,
            "temporal_operator": temporal_operator,
            "decoder_time_condition": decoder_time_condition,
            "device": str(device),
            "status": "ok",
            "params_m": "",
            "gflops": "",
            "error": "",
        }
        try:
            stats = compute_one(cfg, args.batch_size, device)
            row["params_m"] = round(stats["params_m"], 4)
            row["gflops"] = round(stats["gflops"], 4)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            if args.strict:
                raise
        rows.append(row)
        print(
            f"{experiment_name:<25} | "
            f"{row['temporal_operator']:<12} | "
            f"{row['decoder_time_condition']:<8} | "
            f"{row['device']:<6} | "
            f"{row['status']:<7} | "
            f"{str(row['params_m']):>8} | "
            f"{str(row['gflops']):>6}"
        )
    print("=" * 132)

    if args.output_csv:
        output_path = Path(args.output_csv).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output_path)
        print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
