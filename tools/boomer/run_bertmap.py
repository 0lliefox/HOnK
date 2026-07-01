#!/usr/bin/env python3
"""run_bertmap.py — run BERTMap / BERTMap-Lt (DeepOnto) to align the source
vocabulary to the canonical HOnK vocabulary, as the automated "match stage"
baseline against HOnK's manual mappings (reviewers R1-W2/R2-W5).

Must run in the isolated bertmap venv (Python 3.11 + deeponto). The JVM memory
prompt is answered on stdin, e.g.:

    echo "4g" | .../bertmap-venv/bin/python run_bertmap.py --flavour bertmaplt

Writes the predicted mappings to match/bertmap_predictions.<flavour>.tsv
(source_local, predicted_local, score) — one best target per source.
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.join(HERE, "match")
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


def local(iri: str) -> str:
    return iri.rstrip("<>").split("#")[-1].split("/")[-1]


def run(flavour: str) -> int:
    from deeponto.onto import Ontology
    from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE

    out_dir = os.path.join(MATCH, f"bertmap_out_{flavour}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    src = Ontology(os.path.join(MATCH, "source.owl"))
    tgt = Ontology(os.path.join(MATCH, "canonical.owl"))
    print(f"[{flavour}] source classes={len(src.owl_classes)} "
          f"canonical classes={len(tgt.owl_classes)}", flush=True)

    config = BERTMapPipeline.load_bertmap_config(DEFAULT_CONFIG_FILE)
    config.model = flavour                      # "bertmap" or "bertmaplt"
    config.output_path = out_dir
    config.annotation_property_iris = [RDFS_LABEL, RDFS_COMMENT]
    config.global_matching.enabled = True
    # keep any BERT fine-tuning cheap on this tiny vocabulary
    try:
        config.bert.resume_training = False
        config.bert.max_length_for_input = 128
        config.bert.num_epochs_for_training = 1
        config.bert.batch_size_for_training = 8
    except Exception:
        pass

    try:
        BERTMapPipeline(src, tgt, config)
    except Exception as e:
        print(f"[{flavour}] pipeline raised: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # collect predicted mappings from whatever tsv the pipeline produced
    candidates = glob.glob(os.path.join(out_dir, "**", "*.tsv"), recursive=True)
    print(f"[{flavour}] output tsvs: {[os.path.relpath(c, out_dir) for c in candidates]}", flush=True)
    best = {}   # src_local -> (score, tgt_local)
    for path in candidates:
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split("\t")
            try:
                si = header.index("SrcEntity"); ti = header.index("TgtEntity")
                ci = header.index("Score") if "Score" in header else None
            except ValueError:
                continue
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= max(si, ti):
                    continue
                s, t = local(cols[si]), local(cols[ti])
                sc = float(cols[ci]) if ci is not None and cols[ci] else 1.0
                if s not in best or sc > best[s][0]:
                    best[s] = (sc, t)

    out_tsv = os.path.join(MATCH, f"bertmap_predictions.{flavour}.tsv")
    with open(out_tsv, "w", encoding="utf-8") as fh:
        fh.write("source\tpredicted\tscore\n")
        for s, (sc, t) in sorted(best.items()):
            fh.write(f"{s}\t{t}\t{sc:.4f}\n")
    print(f"[{flavour}] wrote {len(best)} predictions -> {out_tsv}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flavour", choices=["bertmap", "bertmaplt"], default="bertmaplt")
    args = ap.parse_args()
    return run(args.flavour)


if __name__ == "__main__":
    sys.exit(main())
