"""Train script for transformer_only ablation experiment."""

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
AB_ROOT = CURRENT_DIR.parent
if str(AB_ROOT) not in sys.path:
    sys.path.insert(0, str(AB_ROOT))

from ablation_core import run_experiment


def main() -> None:
    """Run transformer_only ablation."""
    run_experiment("transformer_only", CURRENT_DIR / "config.yaml")


if __name__ == "__main__":
    main()
