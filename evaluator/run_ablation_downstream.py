#!/usr/bin/env python3
"""run_ablation_downstream.py — downstream evaluation per method-versus-data arm.

The method-versus-data ablation (configurations_ablation.json) reports intrinsic
graph statistics per arm. This runner answers the complementary question: how do
the *downstream* retrieval metrics move across the same arms?

Each arm is evaluated against the same fixed baseline, arm (a) (ConceptNet as
ingested, without HOnK's normalisation), over the same 100 benchmark sentences,
embedders, and judges as the main run, so the arms are mutually comparable.

Scope: the ablation family excludes GeoNames, so arm (d) is NOT the released
HOnK graph (12.7M vs 88.45M triples). These numbers are internally comparable
across arms but are not comparable to the headline HOnK-vs-ConceptNet figures.

The baseline store and the embedders load once and are reused; only the HOnK-side
store swaps per arm. Summary/paper-table output is redirected into
``ablation_downstream/`` so the main run's canonical ``paper_tables/`` (which the
paper splices) are never overwritten.

Usage (LLM-bound, hours; from the evaluator/ directory):
    PYTHONPATH=.. ../.venv/bin/python run_ablation_downstream.py [--arms b1 b2 c d]
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import yaml

try:
    from honk_evaluator import HonkEvaluator
    from store_loader import get_file_hash, load_graph
except ImportError:  # run from the repo root
    from evaluator.honk_evaluator import HonkEvaluator
    from evaluator.store_loader import get_file_hash, load_graph

logger = logging.getLogger(__name__)

# arm key -> (ontology filename, human label for the paper table)
ARMS = {
    "b1": ("ablation_b1_union_raw.nt", "(b1) +WK+WN, raw"),
    "b2": ("ablation_b2_union_canonicalise.nt", "(b2) +WK+WN, canonicalised"),
    "c": ("ablation_c_full_unclustered.nt", "(c) full, normalised (unclustered)"),
    "d": ("ablation_d_full_clustered.nt", "(d) full, normalised (clustered)"),
}

ONTOLOGY_DIR = "../ontologies"
OUT_DIR = Path("ablation_downstream")


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-arm downstream ablation runner")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--config", default=None,
                    help="evaluator config (default: eval_config.yaml, then config.yaml)")
    ap.add_argument("--force", action="store_true",
                    help="re-run an arm even if its CSV already exists")
    args = ap.parse_args()

    candidates = [args.config] if args.config else [
        "eval_config.yaml", "../eval_config.yaml", "config.yaml", "../config.yaml"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            cfg = candidate
            break
    else:
        logger.error("No evaluator config found (tried %s).", candidates)
        return 1
    logger.info("Using config: %s", cfg)

    OUT_DIR.mkdir(exist_ok=True)

    # HonkEvaluator.__init__ eagerly loads whatever `ontologies.honk` names, so
    # point it at the first arm we need. Otherwise it would spend minutes (and
    # gigabytes) loading the released 11 GB graph only for us to discard it.
    first_arm_path = str(Path(ONTOLOGY_DIR) / ARMS[args.arms[0]][0])
    patched = yaml.safe_load(Path(cfg).read_text())
    patched.setdefault("evaluator", {}).setdefault("ontologies", {})["honk"] = first_arm_path
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(patched, fh, sort_keys=False)
        patched_cfg = fh.name

    ev = HonkEvaluator(patched_cfg)  # baseline store + embedders load once and are reused

    for arm in args.arms:
        filename, label = ARMS[arm]
        arm_path = str(Path(ONTOLOGY_DIR) / filename)
        if not Path(arm_path).is_file():
            logger.error("Arm %s: ontology not found: %s", arm, arm_path)
            return 1

        arm_csv = OUT_DIR / f"evaluation_results_arm_{arm}.csv"
        if arm_csv.is_file() and not args.force:
            logger.info("Arm %s: %s exists, skipping (--force to re-run).", arm, arm_csv)
            continue

        logger.info("=== Arm %s: %s ===", arm, label)
        # Swap only the HOnK side; the extraction cache is keyed on the store
        # hash, so each arm gets its own cache entries. The first arm is already
        # loaded by __init__ above, so skip a redundant re-load.
        arm_hash = get_file_hash(arm_path)
        if ev._honk_hash != arm_hash:
            ev.honk_store = load_graph(arm_path, ev.base_uri, ev.cache_dir)
            ev._honk_hash = arm_hash
        ev.output_csv = str(arm_csv)
        # Redirect summary + paper tables away from the canonical output dir.
        ev.summary_table_output = str(OUT_DIR / f"summary_table_arm_{arm}.txt")
        ev.run_evaluation()
        logger.info("Arm %s complete -> %s", arm, arm_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
