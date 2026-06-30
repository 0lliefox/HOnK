import argparse
import itertools
import logging
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyoxigraph
import yaml
from tqdm import tqdm

try:
    from .store_loader import load_graph
except ImportError:
    from store_loader import load_graph  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_TEMPLATES: Dict[Tuple[str, str], List[str]] = {
    ("geonames", "geonames"): [
        "The settlement of {a} is administratively linked to {b}.",
        "Both {a} and {b} are recorded under the same regional classification.",
        "The area around {a} borders the district known as {b}.",
    ],
    ("conceptnet", "geonames"): [
        "The place called {b} is commonly associated with {a}.",
        "Visitors to {b} often encounter situations that require knowledge of {a}.",
        "The concept of {a} is particularly relevant in the context of {b}.",
    ],
    ("conceptnet", "wordnet"): [
        "The word {a} is a type of {b} in common usage.",
        "Most people associate {a} with {b} in everyday language.",
        "Understanding {a} requires familiarity with the broader concept of {b}.",
    ],
    ("conceptnet", "wiktionary"): [
        "The term {a} is listed as a variant or synonym of {b} in lexical sources.",
        "In everyday speech, {a} and {b} are often used interchangeably.",
        "Dictionaries record {b} as an alternative form of {a}.",
    ],
    ("geonames", "wordnet"): [
        "The name {a} derives from the {b} tradition in that region.",
        "The place {a} is categorised under the class {b} in geographic taxonomies.",
    ],
    ("geonames", "wiktionary"): [
        "The toponym {a} shares an etymology with the word {b}.",
        "Local dialect uses {b} to refer to features found near {a}.",
    ],
    ("conceptnet", "parmenides"): [
        "Formally speaking, {a} entails or implies {b}.",
        "The logical structure of {a} is captured by the relation {b}.",
        "In structured knowledge, {a} is formally classified under {b}.",
    ],
    ("geonames", "parmenides"): [
        "The place {a} satisfies the formal predicate {b}.",
        "Geographic records classify {a} according to the property {b}.",
    ],
    ("parmenides", "wordnet"): [
        "The ontological concept {a} corresponds to the lexical entry {b}.",
        "Formal reasoning about {a} aligns with the word sense of {b}.",
    ],
    ("parmenides", "wiktionary"): [
        "The formal predicate {a} maps onto the lexical definition of {b}.",
    ],
    ("wordnet", "wiktionary"): [
        "The word {a} shares semantic overlap with {b} across lexical databases.",
        "Linguists classify {a} as a related form of {b}.",
        "Both {a} and {b} appear in the same synset neighbourhood.",
    ],
    ("parmenides", "parmenides"): [
        "The relation {a} formally subsumes {b} in the knowledge hierarchy.",
        "Logical inference connects {a} and {b} through shared axioms.",
    ],
    ("wordnet", "wordnet"): [
        "The word {a} is a hypernym or hyponym of {b}.",
        "WordNet places {a} and {b} in the same semantic cluster.",
    ],
    ("wiktionary", "wiktionary"): [
        "Wiktionary lists {a} and {b} under the same etymology entry.",
        "The definitions of {a} and {b} cross-reference each other.",
    ],
    ("conceptnet", "conceptnet"): [
        "People who know about {a} typically also understand {b}.",
        "The ideas of {a} and {b} are closely related in everyday thought.",
        "Common sense tells us that {a} and {b} are connected.",
    ],
}
_DEFAULT_TEMPLATES = [
    "The semantic relationship between {a} and {b} is worth examining.",
    "Understanding {a} requires knowledge of {b}.",
    "The concepts {a} and {b} are linked in structured knowledge bases.",
]


def _template_for(src_a: str, src_b: str) -> str:
    key = tuple(sorted([src_a, src_b]))
    return _TEMPLATES.get(key, _DEFAULT_TEMPLATES)[0]  # type: ignore[arg-type]


_SOURCE_PATTERNS = [
    ("geonames",   re.compile(r"geonames", re.I)),
    ("wordnet",    re.compile(r"wordnet|wn\.|/wn/", re.I)),
    ("wiktionary", re.compile(r"wiktionary|wkt", re.I)),
    ("parmenides", re.compile(r"parmenides", re.I)),
    ("conceptnet", re.compile(r"conceptnet|/c/en/|example\.org/ontology", re.I)),
]


def _detect_source(uri: str) -> str:
    for name, pattern in _SOURCE_PATTERNS:
        if pattern.search(uri):
            return name
    return "unknown"


def _is_meaningful(label: str) -> bool:
    if not label or len(label) < 3:
        return False
    if re.fullmatch(r'\d+', label):
        return False
    if re.fullmatch(r'[0-9a-f\-]{8,}', label, re.I):
        return False
    if label.count('_') > len(label) * 0.5:
        return False
    if not re.search(r'[a-zA-Z]', label):
        return False
    return True


def _bridge_filter(kw_a: str, kw_b: str) -> str:
    an = kw_a.lower().replace(" ", "_")
    bn = kw_b.lower().replace(" ", "_")
    as_ = kw_a.lower()
    bs = kw_b.lower()
    return (
        f'( (CONTAINS(LCASE(STR(?s)), "{an}") || CONTAINS(LCASE(STR(?s)), "{as_}"))'
        f' && (CONTAINS(LCASE(STR(?o)), "{bn}") || CONTAINS(LCASE(STR(?o)), "{bs}")) )'
        f' || '
        f'( (CONTAINS(LCASE(STR(?s)), "{bn}") || CONTAINS(LCASE(STR(?s)), "{bs}"))'
        f' && (CONTAINS(LCASE(STR(?o)), "{an}") || CONTAINS(LCASE(STR(?o)), "{as_}")) )'
    )


class SentenceSuggester:

    def __init__(
        self,
        config_path: str = "config.yaml",
        baseline_path: Optional[str] = None,
        honk_path: Optional[str] = None,
    ) -> None:
        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            logger.error("Config not found: '%s'", config_path)
            sys.exit(1)
        with cfg_path.open("r") as f:
            self.config = yaml.safe_load(f)

        ev = self.config.get("evaluator", {})
        self.base_uri = self.config.get("turtle_export", {}).get("base_uri", "http://example.org/ontology/")
        self.cache_dir = self.config.get("local_files", {}).get("cache", ".cache")
        self.bridging_limit = ev.get("bridging_limit", 100)

        ont = ev.get("ontologies", {})
        resolved_baseline = baseline_path or ont.get("baseline", "")
        resolved_honk = honk_path or ont.get("honk", "")

        logger.info("Loading baseline store from '%s'...", resolved_baseline)
        self.baseline = load_graph(resolved_baseline, self.base_uri, self.cache_dir)
        logger.info("Loading HOnK store from '%s'...", resolved_honk)
        self.honk = load_graph(resolved_honk, self.base_uri, self.cache_dir)

        # Guard the silent failure behind empty suggester output: load_graph()
        # returns an EMPTY store when a path is unset/unresolved, so every
        # differential is <= 0 and discover() yields nothing. Fail loudly.
        n_baseline, n_honk = len(self.baseline), len(self.honk)
        logger.info("Loaded baseline=%d triples, HOnK=%d triples.", n_baseline, n_honk)
        problems = []
        if not resolved_baseline or not resolved_honk:
            problems.append("evaluator.ontologies.baseline/.honk is unset "
                            "(pass --baseline/--honk or set them in config.yaml).")
        if n_baseline == 0:
            problems.append(f"baseline store is empty (path: '{resolved_baseline}').")
        if n_honk == 0:
            problems.append(f"HOnK store is empty (path: '{resolved_honk}').")
        if (resolved_baseline and resolved_honk
                and Path(resolved_baseline).resolve() == Path(resolved_honk).resolve()):
            problems.append("baseline and HOnK point at the SAME file; "
                            "all differentials will be zero.")
        if problems:
            raise ValueError("SentenceSuggester cannot run:\n  - " + "\n  - ".join(problems))

    def _sample_hub_concepts(self, store: pyoxigraph.Store, limit: int = 200) -> List[Dict]:
        query = (
            "SELECT ?s (COUNT(DISTINCT ?o) AS ?degree) WHERE { ?s ?p ?o } "
            "GROUP BY ?s ORDER BY DESC(?degree) LIMIT " + str(limit)
        )
        results = []
        try:
            for r in store.query(query):
                uri = r["s"].value
                label = uri.split("/")[-1].split("#")[-1]
                label = urllib.parse.unquote(label).replace("_", " ")
                if _is_meaningful(label):
                    results.append({
                        "uri": uri,
                        "label": label,
                        "keyword": label.lower(),
                        "source": _detect_source(uri),
                        "degree": int(r["degree"].value),
                    })
        except Exception as e:
            logger.error("Hub concept query failed: %s", e)
        return results

    def _count_bridges(self, store: pyoxigraph.Store, kw_a: str, kw_b: str) -> int:
        query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER( {_bridge_filter(kw_a, kw_b)} ) }} LIMIT {self.bridging_limit}"
        try:
            return sum(1 for _ in store.query(query))
        except Exception:
            return 0

    def _get_bridge_examples(
        self, store: pyoxigraph.Store, kw_a: str, kw_b: str, limit: int = 3
    ) -> List[Tuple[str, str, str]]:
        query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER( {_bridge_filter(kw_a, kw_b)} ) }} LIMIT {limit}"
        out = []
        try:
            for r in store.query(query):
                s = r["s"].value.split("/")[-1].split("#")[-1]
                p = r["p"].value.split("/")[-1].split("#")[-1]
                o = r["o"].value.split("/")[-1].split("#")[-1]
                out.append((s, p, o))
        except Exception:
            pass
        return out

    def discover(self, hub_sample: int = 50, top_k: int = 25) -> List[Dict]:
        logger.info("Sampling top %d hub concepts from HOnK...", hub_sample)
        concepts = self._sample_hub_concepts(self.honk, limit=hub_sample)
        logger.info("Found %d meaningful hub concepts.", len(concepts))

        pairs = list(itertools.combinations(concepts, 2))
        logger.info("Scoring %d candidate pairs...", len(pairs))

        scored = []
        with tqdm(pairs, desc="Scoring pairs", unit="pair", ncols=100) as pbar:
            for c_a, c_b in pbar:
                kw_a, kw_b = c_a["keyword"], c_b["keyword"]
                h_bridges = self._count_bridges(self.honk, kw_a, kw_b)
                b_bridges = self._count_bridges(self.baseline, kw_a, kw_b)
                differential = h_bridges - b_bridges
                if differential > 0:
                    examples = self._get_bridge_examples(self.honk, kw_a, kw_b)
                    sentence = _template_for(c_a["source"], c_b["source"]).format(
                        a=c_a["label"], b=c_b["label"]
                    )
                    scored.append({
                        "keyword_a": c_a["label"],
                        "keyword_b": c_b["label"],
                        "source_a": c_a["source"],
                        "source_b": c_b["source"],
                        "honk_bridges": h_bridges,
                        "baseline_bridges": b_bridges,
                        "differential": differential,
                        "sentence": sentence,
                        "examples": examples,
                    })
                pbar.set_postfix(candidates=len(scored))

        scored.sort(key=lambda x: x["differential"], reverse=True)
        return scored[:top_k]

    def cluster_into_groups(self, pairs: List[Dict], group_size: int = 3) -> List[Dict]:
        adjacency: Dict[str, List[str]] = defaultdict(list)
        pair_map: Dict[Tuple[str, str], Dict] = {}
        for p in pairs:
            a, b = p["keyword_a"], p["keyword_b"]
            adjacency[a].append(b)
            adjacency[b].append(a)
            pair_map[tuple(sorted([a, b]))] = p  # type: ignore[index]

        groups = []
        used: set = set()
        for p in pairs:
            a, b = p["keyword_a"], p["keyword_b"]
            if a in used or b in used:
                continue
            # grow group from the keyword with the most connections
            pivot = a if len(adjacency[a]) >= len(adjacency[b]) else b
            group_kws = [pivot]
            for neighbour in adjacency[pivot]:
                if neighbour not in group_kws:
                    group_kws.append(neighbour)
                if len(group_kws) >= group_size:
                    break

            if len(group_kws) < 2:
                continue

            total_diff = sum(
                pair_map.get(tuple(sorted([x, y])), {}).get("differential", 0)
                for x, y in itertools.combinations(group_kws, 2)
            )
            sentence = p["sentence"]
            extra = [k for k in group_kws if k not in sentence.lower()]
            if extra:
                sentence = sentence.rstrip(".") + f", involving {' and '.join(extra)}."

            groups.append({
                "keywords": group_kws[:group_size],
                "sentence": sentence,
                "sources": list({
                    pair_map.get(tuple(sorted([x, y])), {}).get("source_a", "unknown")
                    for x, y in itertools.combinations(group_kws[:group_size], 2)
                }),
                "total_differential": total_diff,
                "top_pair_differential": p["differential"],
                "examples": p["examples"],
            })
            used.update(group_kws[:group_size])

        groups.sort(key=lambda x: x["total_differential"], reverse=True)
        return groups

    def write_report(self, pairs: List[Dict], groups: List[Dict], output_path: str) -> None:
        lines = [
            "=" * 72,
            "HOnK Sentence Suggester — Discovery Report",
            "=" * 72,
            "",
            f"Top {len(pairs)} keyword pairs ranked by differential bridge count",
            "(HOnK bridges − ConceptNet baseline bridges)",
            "",
            "-" * 72,
        ]
        for i, p in enumerate(pairs, 1):
            lines += [
                f"{i:>2}. '{p['keyword_a']}' ↔ '{p['keyword_b']}'",
                f"    Sources : {p['source_a']} / {p['source_b']}",
                f"    Bridges : HOnK={p['honk_bridges']}  Baseline={p['baseline_bridges']}  "
                f"Differential=+{p['differential']}",
                f"    Sentence: {p['sentence']}",
            ]
            for s, pred, o in p["examples"]:
                lines.append(f"    Evidence: ({s} -> {pred} -> {o})")
            lines.append("")

        lines += [
            "=" * 72,
            f"Top {len(groups)} keyword groups (clustered, ready for config.yaml)",
            "=" * 72,
            "",
        ]
        for i, g in enumerate(groups, 1):
            lines += [
                f"{i:>2}. Keywords : {g['keywords']}",
                f"    Sentence: {g['sentence']}",
                f"    Total Δ  : +{g['total_differential']}",
                "",
            ]

        text = "\n".join(lines)
        Path(output_path).write_text(text, encoding="utf-8")
        logger.info("Report written to '%s'", output_path)
        print("\n" + text)

    def write_yaml_suggestions(self, groups: List[Dict], output_path: str) -> None:
        test_cases = [
            {"sentence": g["sentence"], "keywords": g["keywords"]}
            for g in groups
        ]
        snippet = yaml.dump(
            {"test_cases": test_cases},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        Path(output_path).write_text(snippet, encoding="utf-8")
        logger.info("YAML snippet written to '%s'", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover the best HOnK evaluator test cases from ontology stores."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (auto-detected if omitted).")
    parser.add_argument("--baseline", default=None, help="Path to the ConceptNet baseline ontology file.")
    parser.add_argument("--honk", default=None, help="Path to the HOnK ontology file.")
    parser.add_argument("--hub-sample", type=int, default=50, help="Hub concepts to sample from HOnK (default: 50).")
    parser.add_argument("--top-k", type=int, default=30, help="Top keyword pairs to return (default: 30).")
    parser.add_argument("--group-size", type=int, default=3, help="Keywords per test-case group (default: 3).")
    args = parser.parse_args()

    if args.config:
        cfg = args.config
        if not Path(cfg).is_file():
            logger.error("Config not found: '%s'", cfg)
            sys.exit(1)
    else:
        for candidate in ["config.yaml", "../config.yaml"]:
            if Path(candidate).is_file():
                cfg = candidate
                break
        else:
            logger.error("config.yaml not found. Run from project root or evaluator/.")
            sys.exit(1)

    suggester = SentenceSuggester(cfg, baseline_path=args.baseline, honk_path=args.honk)

    pairs  = suggester.discover(hub_sample=args.hub_sample, top_k=args.top_k)
    groups = suggester.cluster_into_groups(pairs, group_size=args.group_size)

    out_dir = Path("evaluator")
    out_dir.mkdir(exist_ok=True)

    suggester.write_report(pairs, groups, str(out_dir / "suggested_sentences.txt"))
    suggester.write_yaml_suggestions(groups, str(out_dir / "suggested_sentences.yaml"))

    logger.info("Done. Top suggestion: %s", groups[0]["sentence"] if groups else "no results")
