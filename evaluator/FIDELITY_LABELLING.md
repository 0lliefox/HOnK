# Fidelity audit: manual semantic labelling instrument

The automatic provenance audit (`provenance_audit_adjudicator.py`) verified
that all 150 sampled outputs in `fidelity_audit_sample.csv` are *consistent
with the source records they are attributed to*. This instrument covers the
question automation cannot answer: whether each output is *semantically
correct*. Fill the blank `label` column with exactly one of:

- `correct` — the output is semantically right.
- `partial` — defensible but imperfect (see the per-stratum rubric).
- `incorrect` — semantically wrong.

Use the `notes` column for anything non-obvious (one line; especially for
`partial`/`incorrect`). Label every row; do not skip.

## Per-stratum rubric

**`pos_mapping`** (a concept's part-of-speech class, produced by a
source-to-HOnK mapping). Judge whether the assigned class fits the term as it
is used in the originating source.
- correct: the class is right for the source sense.
- partial: defensible but too coarse or too fine (e.g. `Noun` where a
  specific noun subclass exists), or right for only one of several source
  senses.
- incorrect: the wrong part of speech.

**`edge_mapping`** (a source relation type mapped to a canonical HOnK
relation). Judge whether the canonical relation preserves what the source
relation asserted between the two concepts.
- correct: meaning preserved (including deliberate design choices such as
  argument swaps declared in the mapping).
- partial: meaning broadened or narrowed but not contradicted.
- incorrect: meaning changed, reversed, or lost.

**`url_cluster`** (two or more concepts merged because they share an external
URL). Judge whether the merged concepts denote the same entity or sense.
- correct: same entity/sense (spelling variants, synonyms, cross-source
  duplicates).
- partial: closely related but not identical (e.g. two senses of one lemma,
  or an entity and its type).
- incorrect: distinct entities merged. **This is a false merge** — the
  quantity that gates the confidence-aware clustering work.

**`enriched_cluster`** (a cluster-level relation added by enrichment,
projected from a member-level edge). Judge whether the relation actually
holds between the clusters' referents.
- correct: holds.
- partial: holds only under some member senses.
- incorrect: does not hold (an enrichment error).

## Procedure

1. Label all 150 rows of `fidelity_audit_sample.csv` in one or few sittings;
   consult the `provenance`/`detail` columns and, where needed, the sources.
2. **Test-retest**: `fidelity_audit_retest.csv` holds a fixed 30-row subset
   (stratified 8/8/7/7, seed 42). At least two weeks after the main pass,
   label it again *without* looking at your first answers.
3. Run the analysis:

```bash
# regenerate the retest subset (idempotent; only if the file is missing)
./.venv/bin/python evaluator/fidelity_analysis.py --make-retest
# per-stratum fidelity, Wilson CIs, false-merge rate, retest agreement
./.venv/bin/python evaluator/fidelity_analysis.py
```

The analysis reports, per stratum and overall: strict fidelity
(`correct / n`) and lenient fidelity (`(correct+partial) / n`) with 95%
Wilson intervals; the false-merge rate from `url_cluster`; and, when the
retest file is labelled, percent agreement and Cohen's kappa between the two
passes.
