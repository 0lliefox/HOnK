#!/usr/bin/env python3
"""score_matchers.py — unified scoring of every automated matcher against
HOnK's manual mappings (the "match stage" gold), so BERTMap, BERTMap-Lt and the
similarity baselines are reported on the same footing.

For each matcher we report, over the non-identity gold (the correspondences that
require real semantic bridging, not a trivial string identity):

  coverage  = fraction of source labels the matcher maps at all
  precision = correct / mapped
  recall    = correct / all gold  (the % of the manual mappings recovered)
  F1        = harmonic mean

plus breakdowns for the hard curator cases (POS, argument-swap, negation).

Predictions files (source_local -> predicted_local):
  baseline_predictions.tsv              (cols: matcher, source, gold, predicted, correct)
  bertmap_predictions.<flavour>.tsv     (cols: source, predicted, score)

Usage (main .venv, from tools/boomer):
    python score_matchers.py
"""

import csv
import glob
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.join(HERE, "match")


def loc(iri: str) -> str:
    return iri.split("#")[-1].split("/")[-1]


def load_reference():
    gold, flags = {}, {}
    with open(os.path.join(MATCH, "reference.tsv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            s = loc(row["source_iri"])
            gold[s] = loc(row["canonical_iri"])
            flags[s] = row
    return gold, flags


def load_predictions():
    """matcher_name -> {source_local: predicted_local}."""
    preds = defaultdict(dict)
    base = os.path.join(MATCH, "baseline_predictions.tsv")
    if os.path.isfile(base):
        with open(base, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                preds[row["matcher"]][row["source"]] = row["predicted"]
    for path in glob.glob(os.path.join(MATCH, "bertmap_predictions.*.tsv")):
        name = os.path.basename(path).split(".")[1]           # bertmap / bertmaplt
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("predicted"):
                    preds[name][row["source"]] = row["predicted"]
    return preds


def metrics(pred, gold, flags, subset_pred):
    """subset_pred(source)->bool restricts the gold considered."""
    gold_keys = [s for s in gold if subset_pred(flags[s])]
    total = len(gold_keys)
    mapped = [s for s in gold_keys if s in pred]
    correct = [s for s in mapped if pred[s] == gold[s]]
    cov = len(mapped) / total if total else 0.0
    prec = len(correct) / len(mapped) if mapped else 0.0
    rec = len(correct) / total if total else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return cov, prec, rec, f1, total


def main() -> int:
    gold, flags = load_reference()
    preds = load_predictions()

    order = ["lexical", "embedding", "bertmaplt", "bertmap", "bertmap_bio", "bertmap_base"]
    names = [n for n in order if n in preds] + [n for n in preds if n not in order]

    nonid = lambda f: f["class"] == "nonidentity"
    print(f"{'matcher':11s} | {'cover':>6} {'prec':>6} {'rec':>6} {'F1':>6}  "
          f"(non-identity gold; recall = % of manual mappings recovered)")
    print("-" * 78)
    for n in names:
        cov, prec, rec, f1, tot = metrics(preds[n], gold, flags, nonid)
        print(f"{n:11s} | {cov*100:5.1f}% {prec*100:5.1f}% {rec*100:5.1f}% {f1*100:5.1f}%  (gold={tot})")

    print("\nRecovery (recall) on hard sub-cases:")
    subs = [("POS", lambda f: f["class"] == "nonidentity" and f["kind"] == "pos"),
            ("edge-relation", lambda f: f["class"] == "nonidentity" and f["kind"] == "edge"),
            ("argument-swap", lambda f: bool(f["swap"])),
            ("negation", lambda f: bool(f["neg"]))]
    hdr = "  ".join(f"{n:>13}" for n, _ in subs)
    print(f"{'matcher':11s} | {hdr}")
    print("-" * 78)
    for n in names:
        cells = []
        for _, fn in subs:
            _, _, rec, _, tot = metrics(preds[n], gold, flags, fn)
            cells.append(f"{rec*100:5.1f}% ({tot})")
        print(f"{n:11s} | " + "  ".join(f"{c:>13}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
