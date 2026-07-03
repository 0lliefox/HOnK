#!/usr/bin/env python3
"""regenerate_tables.py — rebuild the paper tables from existing result CSVs.

Reconstructs the reporter's in-memory structures (processed_cases,
sentence_sim, llm_scores) from ``evaluation_results.csv`` and
``evaluation_results_llm_by_model.csv`` plus the provenance/label metadata in
``config.yaml``, then calls ``reporter.export_paper_tables``. This regenerates
every table (including the provenance breakdown) without re-running the
expensive embedding and LLM phases — use it after adding a reporter emitter or
editing case metadata.

Usage (from the evaluator/ directory):
    PYTHONPATH=.. ../.venv/bin/python regenerate_tables.py
"""

import csv
import sys
from pathlib import Path

import yaml

try:
    import reporter
except ImportError:
    from evaluator import reporter


def _num(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def main() -> int:
    for candidate in ["config.yaml", "../config.yaml"]:
        if Path(candidate).is_file():
            cfg = yaml.safe_load(open(candidate, encoding="utf-8"))
            break
    else:
        print("config.yaml not found", file=sys.stderr)
        return 1

    ev = cfg["evaluator"]
    meta = {c["sentence"]: c for c in ev["test_cases"]}
    results_csv = ev.get("output_csv", "evaluation_results.csv")
    by_model_csv = str(Path(results_csv).with_name(Path(results_csv).stem + "_llm_by_model.csv"))

    det: dict = {}
    sentence_sim: dict = {}
    order: list = []
    with open(results_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = row["Test Sentence"]
            if s not in det:
                order.append(s)
                det[s] = {
                    "b_vol": int(_num(row, "Baseline Vol (Triples)")),
                    "h_vol": int(_num(row, "HOnK Vol (Triples)")),
                    "b_hit": _num(row, "Baseline Hit Rate (%)"),
                    "h_hit": _num(row, "HOnK Hit Rate (%)"),
                    "b_div": int(_num(row, "Baseline Relation Diversity")),
                    "h_div": int(_num(row, "HOnK Relation Diversity")),
                    "b_conn": _num(row, "Baseline Keyword Connectivity (%)"),
                    "h_conn": _num(row, "HOnK Keyword Connectivity (%)"),
                    "b_path": _num(row, "Baseline Avg Path Length"),
                    "h_path": _num(row, "HOnK Avg Path Length"),
                    "b_br": int(_num(row, "Baseline Bridging Triples")),
                    "h_br": int(_num(row, "HOnK Bridging Triples")),
                    "diff": int(_num(row, "Differential Triples (HOnK extras)")),
                }
            model = row.get("Embedding Model", "")
            if model:
                sentence_sim[(s, model)] = {
                    "baseline": _num(row, "Baseline Similarity"),
                    "honk": _num(row, "HOnK Similarity"),
                }

    llm_scores: dict = {}
    if Path(by_model_csv).is_file():
        with open(by_model_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                llm_scores[(row["Sentence"], row["LLM Model"])] = {
                    "b_score": _num(row, "Baseline Score"),
                    "h_score": _num(row, "HOnK Score"),
                }

    processed_cases = []
    for s in order:
        d = det[s]
        m = meta.get(s, {})
        processed_cases.append({
            "sentence": s,
            "label": m.get("label", ""),
            "provenance": m.get("provenance", ""),
            "keywords": m.get("keywords", []),
            # reporter reads len(...) on these; reconstruct placeholder lists of the right size
            "b_bridges": [None] * d["b_br"],
            "h_bridges": [None] * d["h_br"],
            "diff_set": [None] * d["diff"],
            "b_det": {
                "subgraph_volume": d["b_vol"], "hit_rate": d["b_hit"],
                "relation_diversity": d["b_div"], "keyword_connectivity": d["b_conn"],
                "avg_path_length": d["b_path"], "bridging_triple_count": d["b_br"],
            },
            "h_det": {
                "subgraph_volume": d["h_vol"], "hit_rate": d["h_hit"],
                "relation_diversity": d["h_div"], "keyword_connectivity": d["h_conn"],
                "avg_path_length": d["h_path"], "bridging_triple_count": d["h_br"],
            },
        })

    out_dir = Path(ev.get("summary_table_output", "summary_table.txt")).parent / "paper_tables"
    reporter.export_paper_tables(processed_cases, sentence_sim, str(out_dir), llm_scores)
    print(f"Regenerated tables in {out_dir} from {results_csv} "
          f"({len(processed_cases)} cases, {len(llm_scores)} judge rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
