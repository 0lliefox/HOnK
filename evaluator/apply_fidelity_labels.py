#!/usr/bin/env python3
"""Apply the semantic fidelity labels to fidelity_audit_sample.csv.

Labels follow evaluator/FIDELITY_LABELLING.md. Each non-`correct` row carries a
one-line justification in the notes column. Rows are addressed by their 0-based
position in the file (stable: the sample is not reshuffled).
"""
import csv
import sys

CSV = "evaluator/fidelity_audit_sample.csv"

# Rows carrying a non-default label or an explanatory note are enumerated;
# everything else defaults to ("correct", ""). Post-fix v1.5.0 audit:
# 146 correct / 4 partial / 0 incorrect. The pre-fix labels (which included
# the magenta false merge and the Alabama self-loop, both since fixed by
# sense-aware identity, the type-coherence guard, and the self-loop filter)
# are preserved in git history.
OVERRIDES = {
    # pos_mapping (0-39): all correct.

    # edge_mapping (40-79)
    52: ("partial",
         "relatedTo broadens EtymologicallyRelatedTo (no dedicated etymological-relation class); meaning not contradicted."),
    61: ("correct",
         "Fix E: InstanceOf -> instanceOf (instance/subclass distinction preserved)."),
    66: ("correct",
         "negation preserved via is_negated triple on the reified relation (verified in graph + clustering signature)."),
    67: ("correct",
         "negation preserved via is_negated triple on the reified relation (verified in graph + clustering signature)."),
    68: ("correct",
         "negation preserved via is_negated triple on the reified relation (verified in graph + clustering signature)."),
    72: ("partial",
         "relatedTo broadens SimilarTo (no similarTo class in ontology_classes); meaning not contradicted."),

    # url_cluster (80-114)
    80: ("partial",
         "same-supersense polysemy: CN 'disappointment' (common feeling sense) blurred with the merged act synset; guard cannot split same-supersense senses."),
    89: ("partial",
         "same-supersense polysemy: surface term 'recession' (economic sense) blurred with the merged 'ceding back' act synset."),
    91: ("correct",
         "FIX VALIDATED: act-sense-only merge (battle); the town Austerlitz is now a separate sense node."),
    92: ("correct",
         "FIX VALIDATED: act-sense-only merge (battle); the colour magenta is now a separate sense node (magenta--noun-attribute)."),

    # enriched_cluster (115-149)
    119: ("correct",
          "reflexive self-eq (cluster to itself); holds trivially; candidate for extending the self-loop filter to eq."),
    135: ("correct",
          "reflexive self-eq (cluster to itself); holds trivially; candidate for extending the self-loop filter to eq."),
}


def main():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys()
    for i, r in enumerate(rows):
        label, notes = OVERRIDES.get(i, ("correct", ""))
        r["label"] = label
        r["notes"] = notes
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print(f"wrote {len(rows)} labels: {dict(c)}")


if __name__ == "__main__":
    sys.exit(main())
