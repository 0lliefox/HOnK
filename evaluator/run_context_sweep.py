#!/usr/bin/env python3
"""run_context_sweep.py — equal-context ablation over the retrieval benchmark.

Runs the full evaluation at several ``max_context_triples`` budgets and emits
the cross-budget comparison table (``equal_context.tex``, tab:equalctx).

Extraction is budget-independent (the budget is applied in format_context,
after the extraction cache), so the per-case extraction cache is reused across
budgets and only the embedding and LLM phases re-run. The budget equal to the
config's ``max_context_triples`` reuses the main run's CSVs when present
instead of paying its LLM cost again.

Each sweep run redirects its summary/paper-table output into ``sweep/`` so the
main run's ``paper_tables/`` are never overwritten; the sweep's product is the
per-budget CSVs plus the equal-context table.

Usage (LLM-bound; from the evaluator/ directory):
    PYTHONPATH=.. ../.venv/bin/python run_context_sweep.py [--budgets 50 100 200]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

try:
    from honk_evaluator import HonkEvaluator
    from reporter import write_equal_context_table
except ImportError:  # run from the repo root
    from evaluator.honk_evaluator import HonkEvaluator
    from evaluator.reporter import write_equal_context_table

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Equal-context sweep runner")
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200])
    ap.add_argument("--force", action="store_true",
                    help="re-run a budget even if its CSV already exists")
    args = ap.parse_args()

    for candidate in ["config.yaml", "../config.yaml"]:
        if Path(candidate).is_file():
            cfg = candidate
            break
    else:
        logger.error("config.yaml not found. Run from the project root or evaluator/.")
        return 1

    ev = HonkEvaluator(cfg)  # loads both stores and all embedders once
    main_budget = ev.max_context_triples
    main_csv = Path(ev.output_csv)
    paper_tables_dir = Path(ev.summary_table_output).parent / "paper_tables"
    sweep_dir = Path("sweep")
    sweep_dir.mkdir(exist_ok=True)

    for budget in args.budgets:
        budget_csv = Path(f"evaluation_results_ctx{budget}.csv")
        if budget_csv.is_file() and not args.force:
            logger.info("Budget %d: %s exists, skipping (--force to re-run).", budget, budget_csv)
            continue

        if budget == main_budget and main_csv.is_file() and not args.force:
            logger.info("Budget %d equals the main run's budget: reusing %s.", budget, main_csv)
            shutil.copy(main_csv, budget_csv)
            by_model = main_csv.with_name(main_csv.stem + "_llm_by_model.csv")
            if by_model.is_file():
                shutil.copy(by_model, budget_csv.with_name(budget_csv.stem + "_llm_by_model.csv"))
            continue

        logger.info("=== Sweep run: max_context_triples=%d ===", budget)
        ev.max_context_triples = budget
        ev.output_csv = str(budget_csv)
        ev.summary_table_output = str(sweep_dir / f"summary_table_ctx{budget}.txt")
        ev.run_evaluation()

    write_equal_context_table(
        sorted(args.budgets),
        csv_dir=".",
        out_path=str(paper_tables_dir / "equal_context.tex"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
