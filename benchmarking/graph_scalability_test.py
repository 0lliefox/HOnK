import gc
import logging
import os
import sys
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdflib import Graph, Namespace, URIRef, RDF, RDFS
from rdflib import Literal as RDFLiteral
from pyoxigraph import Store, NamedNode, Quad, DefaultGraph
from pyoxigraph import Literal as OxiLiteral

from benchmarking.benchmark import Benchmark
from clustering.cluster_graph_concepts_oxi import OxiConceptGraphClusterer
from clustering.cluster_graph_concepts_rdf import RDFConceptGraphClusterer
from graph.graph_funcs import GraphManager
from graph.graph_funcs_oxi import OxiNamespace
from tools.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

_OXI_RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
_OXI_RDFS_LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")


class MockGraphManager:
    """Stand-in for the real graph managers, exposing exactly what the graph clusterers
    use: the reified-relation base-name lookup, the structural relation set, and the
    same-label test behind the reflexive self-loop filter."""
    STRUCTURAL_IRREFLEXIVE_RELS = GraphManager.STRUCTURAL_IRREFLEXIVE_RELS
    _normalise_label = staticmethod(GraphManager._normalise_label)

    def __init__(self, graph, ns, backend):
        self.g = graph
        self.ns = ns
        self.backend = backend
        self.cross_pos_hints = []   # Required by OxiConceptGraphClusterer
        self.node_supersense = {}   # sampled subgraphs carry no sense discriminators
        self._rel_base_cache = {}
        self._label_index = None

    def rel_base_of(self, rel_uri):
        """Base relation name for a reified relation-instance URI, via its rdf:type."""
        key = rel_uri.value if self.backend == "oxi" else str(rel_uri)
        if key not in self._rel_base_cache:
            base = None
            if self.backend == "oxi":
                for q in self.g.quads_for_pattern(rel_uri, _OXI_RDF_TYPE, None):
                    base = q.object.value.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                    break
            else:
                t = self.g.value(rel_uri, RDF.type)
                if t is not None:
                    base = str(t).rsplit('#', 1)[-1].rsplit('/', 1)[-1]
            self._rel_base_cache[key] = base
        return self._rel_base_cache[key]

    def _ensure_label_index(self):
        if self._label_index is None:
            idx = {}
            if self.backend == "oxi":
                for q in self.g.quads_for_pattern(None, _OXI_RDFS_LABEL, None):
                    idx[q.subject.value] = self._normalise_label(q.object.value)
            else:
                for s, o in self.g.subject_objects(RDFS.label):
                    idx[str(s)] = self._normalise_label(str(o))
            self._label_index = idx
        return self._label_index

    def _same_label(self, a_value, b_value):
        idx = self._ensure_label_index()
        la = idx.get(a_value)
        return la is not None and la == idx.get(b_value)


class MockBuilder:
    def __init__(self, graph, ns, run_id, backend):
        self.graph_manager = MockGraphManager(graph, ns, backend)
        self.benchmarking = Benchmark(f"graph_{backend}_scalability")
        self.memory_benchmarking = Benchmark(f"graph_{backend}_memory")
        self.run_id = run_id


def rdflib_to_oxi(rdf_graph):
    store = Store()
    for s, p, o in rdf_graph:
        if not isinstance(s, URIRef):
            continue
        oxi_s = NamedNode(str(s))
        oxi_p = NamedNode(str(p))
        if isinstance(o, URIRef):
            oxi_o = NamedNode(str(o))
        elif isinstance(o, RDFLiteral):
            if o.datatype:
                oxi_o = OxiLiteral(str(o), datatype=NamedNode(str(o.datatype)))
            elif o.language:
                oxi_o = OxiLiteral(str(o), language=o.language)
            else:
                oxi_o = OxiLiteral(str(o))
        else:
            continue  # Skip BNode objects
        store.add(Quad(oxi_s, oxi_p, oxi_o, DefaultGraph()))
    return store


def create_subset_graph(full_graph, fraction):
    logging.info(f"--- Creating subset graph for {fraction * 100:.0f}% of data ---")

    all_subjects = list(set(full_graph.subjects()))
    if not all_subjects:
        return None, 0

    sample_size = int(len(all_subjects) * fraction)
    sampled_subjects = set(random.sample(all_subjects, sample_size))

    logging.info(f"Sampling {sample_size} out of {len(all_subjects)} subjects...")

    subset_graph = Graph()
    for prefix, uri in full_graph.namespaces():
        subset_graph.bind(prefix, uri)

    predicates_seen = set()
    for s in sampled_subjects:
        for p, o in full_graph.predicate_objects(s):
            subset_graph.add((s, p, o))
            predicates_seen.add(p)

    # Include predicate metadata (source_pos/target_pos) needed by the clusterer
    for p in predicates_seen:
        for attr, val in full_graph.predicate_objects(p):
            subset_graph.add((p, attr, val))

    logging.info(f"Created subset with {len(subset_graph)} triples.")
    return subset_graph, len(sampled_subjects)


def run_experiment(input_file, config, fractions):
    logging.info(f"Loading full graph from {input_file}...")
    full_graph = Graph()
    full_graph.parse(input_file, format='nt')

    base_uri = config['turtle_export']['base_uri']
    rdf_ns = Namespace(base_uri)
    oxi_ns = OxiNamespace(base_uri)

    failures = []

    for frac in fractions:
        rdf_subset, concept_count = create_subset_graph(full_graph, frac)
        if concept_count == 0:
            continue

        id_percentage = int(frac * 100)
        logging.info(f"=== Fraction {id_percentage}% ({concept_count} subjects) ===")

        # Convert to Oxi store *before* the RDF clusterer modifies rdf_subset in place,
        # so both backends start from identical data.
        oxi_store = rdflib_to_oxi(rdf_subset)

        logging.info(f"  Running RDF clusterer...")
        rdf_builder = MockBuilder(rdf_subset, rdf_ns, id_percentage, "rdf")
        rdf_builder.benchmarking.add_row(id_percentage, 'concept_count', concept_count)
        rdf_clusterer = RDFConceptGraphClusterer(rdf_builder, config)
        try:
            rdf_clusterer.run()
            rdf_builder.benchmarking.to_csv("graph_rdf_clustering_benchmark", data_length=False, append=True)
            rdf_builder.memory_benchmarking.to_csv("graph_rdf_memory_benchmark", data_length=False, append=True)
        except Exception:
            logging.exception(f"RDF clustering failed at {id_percentage}%")
            failures.append(f"rdf@{id_percentage}%")

        logging.info(f"  Running Oxi clusterer...")
        oxi_builder = MockBuilder(oxi_store, oxi_ns, id_percentage, "oxi")
        oxi_builder.benchmarking.add_row(id_percentage, 'concept_count', concept_count)
        oxi_clusterer = OxiConceptGraphClusterer(oxi_builder, config)
        try:
            oxi_clusterer.run()
            oxi_builder.benchmarking.to_csv("graph_oxi_clustering_benchmark", data_length=False, append=True)
            oxi_builder.memory_benchmarking.to_csv("graph_oxi_memory_benchmark", data_length=False, append=True)
        except Exception:
            logging.exception(f"Oxi clustering failed at {id_percentage}%")
            failures.append(f"oxi@{id_percentage}%")

        del rdf_subset, oxi_store
        gc.collect()

    if failures:
        # A run that produced no usable measurements must not exit successfully.
        raise RuntimeError(f"graph clustering failed for: {', '.join(failures)}")


def main():
    parser = argparse.ArgumentParser(
        description="Test scalability of graph clustering for both RDF and Oxigraph backends. "
                    "Input must be a pre-clustering .nt file with hasURL triples present "
                    "(set keep_url_triples: true in config.yaml before building the ontology)."
    )
    parser.add_argument("input_file", help="Path to the input .nt file")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per fraction (default: 10)")
    args = parser.parse_args()

    config = get_config()
    # Disable caching during scalability tests to avoid stale adj-list files
    config['general']['should_cache'] = False

    for i in range(args.runs):
        logging.info(f"=== Run {i + 1}/{args.runs} ===")
        run_experiment(args.input_file, config, TEST_FRACTIONS)


if __name__ == "__main__":
    main()
