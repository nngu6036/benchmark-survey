from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.reporting.figures import make_default_metric_plots
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark plots from aggregated CSV.")
    parser.add_argument("--input", type=str, default="outputs/tables/aggregated_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()
    paths = make_default_metric_plots(args.input, args.output_dir)
    logger.info("Saved %d figure(s): %s", len(paths), [str(p) for p in paths])


if __name__ == "__main__":
    main()
