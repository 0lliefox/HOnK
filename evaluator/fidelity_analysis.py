#!/usr/bin/env python3
"""fidelity_analysis.py — analyse the manual fidelity labels.

Turns the filled ``label`` column of ``fidelity_audit_sample.csv`` into
per-stratum and overall fidelity estimates with 95% Wilson intervals, the
false-merge rate (the ``url_cluster`` stratum's incorrect fraction, which
gates the confidence-aware clustering work), and, when the test-retest file
is labelled, intra-rater agreement (percent agreement + Cohen's kappa).

``--make-retest`` writes ``fidelity_audit_retest.csv``: a deterministic
stratified 30-row subset (8/8/7/7, seed 42) with blank labels, to be
labelled again at least two weeks after the main pass. See
FIDELITY_LABELLING.md for the rubric.

Usage (from the repo root):
    ./.venv/bin/python evaluator/fidelity_analysis.py [--make-retest]
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "fidelity_audit_sample.csv")
RETEST = os.path.join(HERE, "fidelity_audit_retest.csv")

VALID = {"correct", "partial", "incorrect"}
RETEST_QUOTA = {"pos_mapping": 8, "edge_mapping": 8,
                "url_cluster": 7, "enriched_cluster": 7}
RETEST_SEED = 42


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _row_key(row):
    return (row["stratum"], row["item"], row["detail"])


def make_retest():
    if os.path.exists(RETEST):
        print(f"{RETEST} already exists; not overwriting.")
        return 0
    rows = load(SAMPLE)
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)
    rng = random.Random(RETEST_SEED)
    picked = []
    for stratum, quota in RETEST_QUOTA.items():
        pool = sorted(by_stratum[stratum], key=_row_key)
        picked.extend(rng.sample(pool, quota))
    with open(RETEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in picked:
            out = dict(row)
            out["label"] = ""
            out["notes"] = ""
            writer.writerow(out)
    print(f"Wrote {len(picked)} retest rows to {RETEST} "
          f"(stratified {'/'.join(str(v) for v in RETEST_QUOTA.values())}, "
          f"seed {RETEST_SEED}).")
    return 0


def analyse():
    rows = load(SAMPLE)
    labelled = [r for r in rows if r["label"].strip()]
    bad = [r for r in labelled if r["label"].strip().lower() not in VALID]
    if bad:
        print(f"ERROR: {len(bad)} row(s) with labels outside {sorted(VALID)}; "
              f"first: {bad[0]['label']!r} ({bad[0]['item']!r})")
        return 1
    if not labelled:
        print(f"No labels filled in {SAMPLE} yet — nothing to analyse. "
              f"See FIDELITY_LABELLING.md.")
        return 0
    if len(labelled) < len(rows):
        print(f"WARNING: only {len(labelled)}/{len(rows)} rows labelled; "
              f"estimates below cover the labelled subset only.\n")

    def report(name, subset):
        n = len(subset)
        counts = Counter(r["label"].strip().lower() for r in subset)
        strict, lo_s, hi_s = wilson(counts["correct"], n)
        lenient, lo_l, hi_l = wilson(counts["correct"] + counts["partial"], n)
        print(f"{name:18s} n={n:3d}  "
              f"correct={counts['correct']:3d} partial={counts['partial']:3d} "
              f"incorrect={counts['incorrect']:3d}  "
              f"strict={strict*100:5.1f}% [{lo_s*100:.1f}, {hi_s*100:.1f}]  "
              f"lenient={lenient*100:5.1f}% [{lo_l*100:.1f}, {hi_l*100:.1f}]")

    by_stratum = defaultdict(list)
    for row in labelled:
        by_stratum[row["stratum"]].append(row)
    print("Per-stratum fidelity (strict = correct; lenient = correct+partial; "
          "95% Wilson):")
    for stratum in ("pos_mapping", "edge_mapping", "url_cluster", "enriched_cluster"):
        if by_stratum.get(stratum):
            report(stratum, by_stratum[stratum])
    report("OVERALL", labelled)

    merges = by_stratum.get("url_cluster", [])
    if merges:
        k = sum(1 for r in merges if r["label"].strip().lower() == "incorrect")
        p, lo, hi = wilson(k, len(merges))
        print(f"\nFalse-merge rate (url_cluster incorrect): "
              f"{k}/{len(merges)} = {p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] "
              f"(gates the confidence-aware clustering item)")

    if os.path.exists(RETEST):
        retest = [r for r in load(RETEST) if r["label"].strip()]
        if retest:
            first = {_row_key(r): r["label"].strip().lower() for r in labelled}
            pairs = [(first[_row_key(r)], r["label"].strip().lower())
                     for r in retest if _row_key(r) in first]
            if pairs:
                agree = sum(1 for a, b in pairs if a == b) / len(pairs)
                cats = sorted({c for p in pairs for c in p})
                p_e = sum(
                    (sum(1 for a, _ in pairs if a == c) / len(pairs))
                    * (sum(1 for _, b in pairs if b == c) / len(pairs))
                    for c in cats)
                kappa = (agree - p_e) / (1 - p_e) if p_e < 1 else float("nan")
                print(f"\nTest-retest (n={len(pairs)}): "
                      f"agreement={agree*100:.1f}%  Cohen's kappa={kappa:.3f}")
        else:
            print("\nTest-retest file present but unlabelled — "
                  "label it >=2 weeks after the main pass.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fidelity label analysis.")
    ap.add_argument("--make-retest", action="store_true",
                    help="write the deterministic 30-row test-retest subset")
    args = ap.parse_args()
    return make_retest() if args.make_retest else analyse()


if __name__ == "__main__":
    sys.exit(main())
