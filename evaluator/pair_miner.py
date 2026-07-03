#!/usr/bin/env python3
"""pair_miner.py — mine candidate keyword pairs for the retrieval benchmark.

Replaces the SPARQL scoring loop of sentence_suggester.py, which issues two
CONTAINS full scans per candidate pair (~60-90 s/pair on the full graph and
therefore infeasible beyond a handful of pairs). This tool streams each
N-Triples graph once and counts direct subject-object label pairs, which is
exactly the evaluator's bridging notion (one keyword matching the subject
label and the other the object label).

Outputs a YAML file with two strata of candidates:
  differential  pairs ranked by (honk_bridges - baseline_bridges), the cases
                where integration plausibly adds retrievable structure;
  neutral       a seeded random sample of pairs the BASELINE already bridges,
                selected without differential ranking, as a selection-bias
                control stratum.

The HOnK pass bounds memory by keeping a deterministic 1/K hash-sample of the
pair space (exact baseline counts are always kept); this is a candidate
generator, not a census, and the sample is unbiased across pairs. Labels are
gated to short alphabetic terms and pairs involving extreme hub labels are
dropped.

Usage (from the repo root; ~5-10 min, single pass per graph):
    ./.venv/bin/python evaluator/pair_miner.py \
        --baseline ontologies/ablation_a_conceptnet_raw.nt \
        --honk ontologies/HOnK-v1-5-0.nt \
        --out evaluator/mined_pairs.yaml
"""

import argparse
import hashlib
import logging
import re
import sys
import urllib.parse
from collections import Counter

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_LABEL_RE = re.compile(r"^[a-z][a-z '\-]{2,24}$")
_SKIP_PREDICATES = ("rdf-syntax-ns#type", "rdf-schema#subClassOf",
                    "owl#", "rdf-schema#label")


def _label(uri: str) -> str:
    local = uri.rstrip(">").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return urllib.parse.unquote(local).replace("_", " ").lower()


def _ok(label: str) -> bool:
    return bool(_LABEL_RE.match(label))


def _pair_sampled(a: str, b: str, k: int) -> bool:
    if k <= 1:
        return True
    digest = hashlib.md5(f"{a}|{b}".encode()).digest()
    return digest[0] % k == 0


def mine(path: str, sample_k: int = 1, keep_pairs=None):
    """One streaming pass: (pair -> bridge count, label -> degree)."""
    pairs: Counter = Counter()
    degrees: Counter = Counter()
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            if n % 10_000_000 == 0:
                logger.info("  %s: %dM lines, %d pairs so far",
                            path.rsplit('/', 1)[-1], n // 1_000_000, len(pairs))
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2].rstrip(" .\n")
            if not (s.startswith("<") and o.startswith("<")):
                continue
            if any(sk in p for sk in _SKIP_PREDICATES):
                continue
            la, lb = _label(s), _label(o)
            if la == lb or not (_ok(la) and _ok(lb)):
                continue
            degrees[la] += 1
            degrees[lb] += 1
            key = (la, lb) if la < lb else (lb, la)
            if keep_pairs is not None and key not in keep_pairs:
                if not _pair_sampled(key[0], key[1], sample_k):
                    continue
            elif keep_pairs is None and not _pair_sampled(key[0], key[1], sample_k):
                continue
            pairs[key] += 1
    logger.info("%s: %d lines, %d gated pairs, %d labels",
                path, n, len(pairs), len(degrees))
    return pairs, degrees


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine benchmark keyword pairs.")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--honk", required=True)
    ap.add_argument("--out", default="evaluator/mined_pairs.yaml")
    ap.add_argument("--sample-k", type=int, default=8,
                    help="keep 1/K of the HOnK pair space (memory bound)")
    ap.add_argument("--top", type=int, default=400,
                    help="differential candidates to emit")
    ap.add_argument("--neutral", type=int, default=120,
                    help="neutral baseline-bridged candidates to emit")
    ap.add_argument("--max-degree", type=int, default=2000,
                    help="drop pairs where either label exceeds this degree")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b_pairs, b_deg = mine(args.baseline, sample_k=1)
    # Baseline pairs are always kept exactly in the HOnK pass so differentials
    # against them are never lost to sampling.
    h_pairs, h_deg = mine(args.honk, sample_k=args.sample_k,
                          keep_pairs=frozenset(b_pairs.keys()))

    def hubby(pair) -> bool:
        return (max(h_deg.get(pair[0], 0), b_deg.get(pair[0], 0)) > args.max_degree
                or max(h_deg.get(pair[1], 0), b_deg.get(pair[1], 0)) > args.max_degree)

    diffs = []
    for pair, h_count in h_pairs.items():
        if hubby(pair):
            continue
        d = h_count - b_pairs.get(pair, 0)
        if d > 0:
            diffs.append((d, h_count, b_pairs.get(pair, 0), pair))
    diffs.sort(reverse=True)
    differential = [
        {"keywords": list(pair), "honk_bridges": h, "baseline_bridges": b,
         "differential": d}
        for d, h, b, pair in diffs[:args.top]
    ]

    import random
    rng = random.Random(args.seed)
    neutral_pool = sorted(pair for pair, c in b_pairs.items()
                          if c >= 1 and not hubby(pair))
    neutral_picks = rng.sample(neutral_pool, min(args.neutral, len(neutral_pool)))
    neutral = [
        {"keywords": list(pair),
         "baseline_bridges": b_pairs[pair],
         "honk_bridges": h_pairs.get(pair, 0)}
        for pair in neutral_picks
    ]

    with open(args.out, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"differential": differential, "neutral": neutral},
                       fh, allow_unicode=True, sort_keys=False)
    logger.info("Wrote %d differential + %d neutral candidates to %s",
                len(differential), len(neutral), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
