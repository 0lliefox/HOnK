import re
import random
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_URL_ENCODED_NONASCII = re.compile(r'%[C-Fc-f][0-9A-Fa-f]')

_GEO_SUBJECTS = {
    "populated%20place", "populated_place",
    "stream", "river", "lake", "mountain", "hill", "island", "peninsula",
    "cape", "bay", "valley", "plateau", "desert", "forest", "wetland",
    "glacier", "reef", "atoll", "lagoon", "estuary", "fjord", "strait",
    "channel", "sea", "ocean", "continent", "country", "state", "province",
    "county", "district", "township", "municipality", "hamlet", "village",
    "town", "city", "airport", "port", "harbor", "harbour", "beach",
    "park", "reserve", "sanctuary", "monument", "memorial",
}

_NAMED_ENTITY_RE = re.compile(r'^[A-Z]|[0-9]')

PREDICATE_WEIGHTS: Dict[str, float] = {
    "#eq":        0.35,
    "#relatedTo": 0.30,
    "#mannerOf":  0.20,
    "#isA":       0.10,
    "#partOf":    0.05,
}

ALL_PREDICATES: List[str] = list(PREDICATE_WEIGHTS.keys())

_URI_RE = re.compile(r'<([^>]+)>')


def _local_name(uri: str, lower: bool = True) -> str:
    name = uri.rsplit("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    return name.lower() if lower else name


def _parse_ntriples(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    uris = _URI_RE.findall(line)
    if len(uris) < 3:
        return None, None, None
    return uris[0], uris[1], uris[2]


def _passes_quality_gates(line: str, subj_uri: str, obj_uri: str) -> bool:
    if _URL_ENCODED_NONASCII.search(line):
        return False
    if _local_name(subj_uri) in _GEO_SUBJECTS:
        return False
    if _NAMED_ENTITY_RE.search(_local_name(subj_uri, lower=False)):
        return False
    if _NAMED_ENTITY_RE.search(_local_name(obj_uri, lower=False)):
        return False
    return True


def stratified_reservoir_sample(
    input_path: str,
    output_path: str,
    sample_size: int = 250,
    predicate_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    weights = predicate_weights or PREDICATE_WEIGHTS
    predicates = list(weights.keys())

    quotas: Dict[str, int] = {
        pred: max(1, round(weights[pred] * sample_size))
        for pred in predicates
    }
    total_quota = sum(quotas.values())
    if total_quota != sample_size:
        largest = max(quotas, key=quotas.__getitem__)
        quotas[largest] += sample_size - total_quota

    reservoirs: Dict[str, List[str]] = {pred: [] for pred in predicates}
    counts: Dict[str, int] = defaultdict(int)

    in_file = Path(input_path)
    if not in_file.is_file():
        logger.error("Input file not found: %s", in_file)
        sys.exit(1)

    logger.info("Stratified reservoir sampling — target %d triples", sample_size)
    logger.info("Quota per predicate: %s", quotas)
    logger.info("Scanning: %s", in_file)

    try:
        with in_file.open("r", encoding="utf-8") as fh:
            for raw_idx, line in enumerate(fh):
                if (raw_idx + 1) % 5_000_000 == 0:
                    logger.info("  … scanned %d M lines", (raw_idx + 1) // 1_000_000)

                if not any(p in line for p in predicates):
                    continue

                subj_uri, pred_uri, obj_uri = _parse_ntriples(line)
                if subj_uri is None:
                    continue

                # Match predicate by URI suffix to avoid partial matches in subject/object URIs
                pred_match = next(
                    (p for p in predicates if pred_uri.endswith(p.lstrip("#"))), None
                )
                if pred_match is None:
                    continue

                if not _passes_quality_gates(line, subj_uri, obj_uri):
                    continue

                quota = quotas[pred_match]
                n = counts[pred_match]

                if n < quota:
                    reservoirs[pred_match].append(line)
                else:
                    idx = random.randint(0, n)
                    if idx < quota:
                        reservoirs[pred_match][idx] = line

                counts[pred_match] += 1

    except Exception as exc:
        logger.error("Error during sampling: %s", exc)
        sys.exit(1)

    for pred in predicates:
        found = counts[pred]
        quota = quotas[pred]
        taken = len(reservoirs[pred])
        logger.info(
            "  %-14s  found=%7d  quota=%d  sampled=%d%s",
            pred, found, quota, taken,
            "  *** under-quota ***" if taken < quota else "",
        )

    combined = []
    for pred in predicates:
        combined.extend(reservoirs[pred])
    random.shuffle(combined)

    out_file = Path(output_path)
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            fh.writelines(combined)
        logger.info("Written %d triples to %s", len(combined), out_file)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    stratified_reservoir_sample(
        input_path="../ontologies/unique_edges_HOnK.txt",
        output_path="sampled_250_triples.txt",
        sample_size=250,
        seed=42,
    )
