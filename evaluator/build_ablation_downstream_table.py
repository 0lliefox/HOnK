#!/usr/bin/env python3
"""build_ablation_downstream_table.py — per-arm downstream table (tab:ablation-downstream).

Aggregates the per-arm CSVs written by run_ablation_downstream.py into one
LaTeX table showing how the downstream retrieval metrics move across the
method-versus-data ablation arms.

Every arm is evaluated against the same fixed baseline, arm (a), so each run's
``Baseline *`` columns describe arm (a) and its ``HOnK *`` columns describe that
arm. Arm (a)'s own row is therefore read from any run's baseline columns (they
are identical across runs by construction, which this script asserts).

Means are computed over the unrounded per-row values, matching the convention
used for the main paper tables.

Scope: the ablation family excludes GeoNames, so arm (d) is not the released
HOnK graph. These numbers are comparable across arms, not to the headline
HOnK-vs-ConceptNet figures.

Usage (from the evaluator/ directory):
    PYTHONPATH=.. ../.venv/bin/python build_ablation_downstream_table.py
"""

import csv
import statistics
import sys
from pathlib import Path

IN_DIR = Path("ablation_downstream")
OUT_TEX = IN_DIR / "paper_tables" / "ablation_downstream.tex"

ARMS = [
    ("b1", r"(b1) $+$WK$+$WN, raw"),
    ("b2", r"(b2) $+$WK$+$WN, canonicalised"),
    ("c", r"(c) full, normalised (unclustered)"),
    ("d", r"(d) full, normalised (clustered)"),
]
ARM_A_LABEL = r"(a) ConceptNet only, raw"

# Arm (e) is the released graph. The main evaluation run already compares exactly
# arm (a) against it (see eval_config.yaml: baseline=ablation_a_conceptnet_raw.nt,
# honk=HOnK-v1-5-0.nt), so its CSV is reused rather than recomputed.
ARM_E_CSV = Path("evaluation_results.csv")
ARM_E_LABEL = r"(e) $+$GeoNames $=$ released \gls{onto}"


def _mean(rows, col):
    return statistics.fmean(float(r[col]) for r in rows)


def _load(arm):
    path = IN_DIR / f"evaluation_results_arm_{arm}.csv"
    if not path.is_file():
        print(f"missing: {path}", file=sys.stderr)
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    data = {arm: _load(arm) for arm, _ in ARMS}
    if any(v is None for v in data.values()):
        print("Not all arms present; run run_ablation_downstream.py first.", file=sys.stderr)
        return 1

    # Arm (a) is the shared baseline: verify every run agrees on it before using
    # any single run's baseline columns as arm (a)'s row.
    base = {}
    for col in ("Baseline Vol (Triples)", "Baseline Bridging Triples",
                "Baseline Keyword Connectivity (%)", "Baseline Similarity",
                "Avg Baseline LLM Score"):
        vals = {arm: round(_mean(rows, col), 6) for arm, rows in data.items()}
        if len(set(vals.values())) != 1:
            print(f"WARNING: baseline column '{col}' differs across arms: {vals}",
                  file=sys.stderr)
        base[col] = next(iter(vals.values()))

    def row(label, vol, bridge, conn, sim, llm, sim_delta=None, llm_delta=None):
        d_sim = "--" if sim_delta is None else f"{sim_delta:+.2f}\\%"
        d_llm = "--" if llm_delta is None else f"{llm_delta:+.2f}\\%"
        return (f"{label} & {vol:,.0f} & {bridge:.2f} & {conn:.2f} & "
                f"{sim:.3f} & {d_sim} & {llm:.2f} & {d_llm} \\\\")

    a_sim = base["Baseline Similarity"]
    a_llm = base["Avg Baseline LLM Score"]
    lines = [row(ARM_A_LABEL, base["Baseline Vol (Triples)"],
                 base["Baseline Bridging Triples"],
                 base["Baseline Keyword Connectivity (%)"], a_sim, a_llm)]

    for arm, label in ARMS:
        rows = data[arm]
        sim = _mean(rows, "HOnK Similarity")
        llm = _mean(rows, "Avg HOnK LLM Score")
        lines.append(row(
            label,
            _mean(rows, "HOnK Vol (Triples)"),
            _mean(rows, "HOnK Bridging Triples"),
            _mean(rows, "HOnK Keyword Connectivity (%)"),
            sim, llm,
            (sim - a_sim) / a_sim * 100 if a_sim else None,
            (llm - a_llm) / a_llm * 100 if a_llm else None,
        ))

    # Arm (e): the released graph, from the main run. Its own baseline columns
    # describe the same arm (a) graph, so its deltas are computed against them
    # rather than against this session's baseline (the two agree to ~0.01).
    if ARM_E_CSV.is_file():
        with ARM_E_CSV.open() as f:
            rows = list(csv.DictReader(f))
        e_sim, e_llm = _mean(rows, "HOnK Similarity"), _mean(rows, "Avg HOnK LLM Score")
        ea_sim, ea_llm = _mean(rows, "Baseline Similarity"), _mean(rows, "Avg Baseline LLM Score")
        lines.append(r"\midrule")
        lines.append(row(
            ARM_E_LABEL,
            _mean(rows, "HOnK Vol (Triples)"),
            _mean(rows, "HOnK Bridging Triples"),
            _mean(rows, "HOnK Keyword Connectivity (%)"),
            e_sim, e_llm,
            (e_sim - ea_sim) / ea_sim * 100, (e_llm - ea_llm) / ea_llm * 100,
        ))
    else:
        print(f"NOTE: {ARM_E_CSV} not found; released-graph row omitted.", file=sys.stderr)

    caption = (
        "Downstream retrieval metrics across the method-versus-data ablation arms, "
        "over the same 100 benchmark sentences. Every arm is evaluated against the "
        "same fixed baseline, arm (a); $\\Delta$ is relative to that arm. Volume, "
        "bridging, and keyword connectivity are means over the 100 cases; "
        "similarity is the mean top-5 cosine across the three embedding models and "
        "\\gls{llm} the mean score across the three judges. Arms (a)--(d) follow "
        "Table~\\ref{table:method-vs-data} and exclude GeoNames; arm (e) adds it, "
        "recovering the released graph, and is the main run of "
        "Tables~\\ref{tab:emb} and~\\ref{tab:llm} (its rows come from that run, "
        "whose arm-(a) baseline agrees with this one to within $0.01$)."
    )
    tex = "\n".join([
        r"\begin{table*}[tb]", r"\centering", f"\\caption{{{caption}}}",
        r"\label{tab:ablation-downstream}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{} X r r r r r r r @{}}", r"\toprule",
        r"\textbf{Configuration} & \textbf{Volume} & \textbf{Bridging} & "
        r"\textbf{Conn.\ (\%)} & \textbf{Sim.} & \textbf{$\Delta$} & "
        r"\textbf{\gls{llm}} & \textbf{$\Delta$} \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabularx}", r"\end{table*}", "",
    ])
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(tex)
    print(f"\nwritten: {OUT_TEX}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
