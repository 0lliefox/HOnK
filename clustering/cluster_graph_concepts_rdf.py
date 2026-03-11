import logging
from collections import defaultdict
from tqdm import tqdm

from rdflib import RDF, URIRef
from tools.timer import timer
from clustering.cluster_graph_concepts import AbstractConceptGraphClusterer

class RDFConceptGraphClusterer(AbstractConceptGraphClusterer):

    @timer
    def build_adj_list(self):
        logging.info("  - Building adjacency list from Graph URLs")

        db = defaultdict(set)
        for s, o in tqdm(self.g.subject_objects(self.ns.hasURL), desc="Fetching URL triples", disable=not self.verbose):
            o, pos = o.split('==')
            s = f"{s}=={pos}"

            db[s].add(o)
            db[o].add(s)

        return {k: sorted(list(v)) for k, v in db.items()}

    @timer
    def add_unclustered_concepts(self, cluster_mappings, visited_nodes):
        logging.info("  - Fetching all concept IDs from graph (RDFLib)...")

        unvisited_candidates = set()
        for s, t in self.g.subject_objects(RDF.type):
            if self.base_uri in str(s):
                t_str = str(t)
                if self.base_uri in t_str:
                    pos = t_str.split('#')[-1] if '#' in t_str else t_str.split('/')[-1]
                    candidate = f"{s}=={pos}"
                    if candidate not in visited_nodes:
                        unvisited_candidates.add(candidate)

        cluster_id = 0
        if cluster_mappings:
            last_cluster_label = cluster_mappings[-1][1]
            try:
                cluster_id = int(last_cluster_label.replace('c', ''))
            except ValueError:
                cluster_id = 0

        for node in tqdm(unvisited_candidates, desc="Handling isolated nodes", disable=not self.verbose):
            cluster_id += 1
            cluster_mappings.append((node, f"c{cluster_id}"))

    @timer
    def coalesce_relationships(self, cluster_mappings):
        logging.info("Coalescing relationships between concepts based on clusters (RDFLib)...")

        raw_uri_to_cluster_info = defaultdict(list)
        cluster_to_raw_uris = defaultdict(set)

        for key, cluster_label in cluster_mappings:
            parts = key.split('==')
            raw_uri = parts[0]
            member_pos = parts[1] if len(parts) > 1 else None

            raw_uri_to_cluster_info[raw_uri].append((cluster_label, member_pos))
            cluster_to_raw_uris[cluster_label].add(raw_uri)

        cluster_relations = set()
        predicate_constraints = {}
        excluded_predicates = {self.ns.hasURL, RDF.type}

        logging.info(f"  - Scanning {len(raw_uri_to_cluster_info)} unique concepts for existing relations...")

        for s_raw in tqdm(raw_uri_to_cluster_info.keys(), desc="Scanning graph", disable=not self.verbose):
            for p, o in self.g.predicate_objects(URIRef(s_raw)):
                if self.base_uri not in str(p) or p in excluded_predicates:
                    continue

                if p not in predicate_constraints:
                    s_constraint = self.g.value(p, self.ns.source_pos)
                    t_constraint = self.g.value(p, self.ns.target_pos)
                    predicate_constraints[p] = (
                        str(s_constraint) if s_constraint else None,
                        str(t_constraint) if t_constraint else None
                    )

                s_constraint_pos, t_constraint_pos = predicate_constraints[p]
                o_str = str(o)

                if o_str in raw_uri_to_cluster_info:
                    s_clusters_info = raw_uri_to_cluster_info[s_raw]
                    o_clusters_info = raw_uri_to_cluster_info[o_str]

                    for c_s_label, s_member_pos in s_clusters_info:
                        if s_constraint_pos and s_member_pos and s_constraint_pos != s_member_pos:
                            continue

                        for c_o_label, o_member_pos in o_clusters_info:
                            if t_constraint_pos and o_member_pos and t_constraint_pos != o_member_pos:
                                continue

                            cluster_relations.add((c_s_label, p, c_o_label))

        logging.info(f"  - Found {len(cluster_relations)} unique cluster-level relationships.")

        # Recover cluster relations for same-term cross-POS pairs that were silently dropped
        # in add_relation_to_graph (start_uri == end_uri)
        # These are added in DB mode but collapse to a self-loop in graph mode
        cross_pos_hints = getattr(self.graph_manager, 'cross_pos_hints', [])
        if cross_pos_hints:
            logging.info(f"  - Processing {len(cross_pos_hints)} cross-POS same-URI hints...")
            recovered = 0
            for uri, rel_uri, s_pos, t_pos in cross_pos_hints:
                entries = raw_uri_to_cluster_info.get(str(uri), [])
                pos_to_cluster = {pos: label for label, pos in entries}
                c_s = pos_to_cluster.get(s_pos)
                c_t = pos_to_cluster.get(t_pos)
                if c_s and c_t and c_s != c_t:
                    cluster_relations.add((c_s, rel_uri, c_t))
                    recovered += 1
            logging.info(f"  - Recovered {recovered} cross-POS cluster relations.")

        logging.info("  - Optimizing and propagating relationships to all equivalent concepts...")

        cluster_to_urirefs = {
            label: [URIRef(uri) for uri in uris]
            for label, uris in cluster_to_raw_uris.items()
        }

        grouped_targets = defaultdict(set)
        for c_start, p, c_end in tqdm(cluster_relations, desc="Grouping relations", disable=not self.verbose):
            grouped_targets[(c_start, p)].update(cluster_to_urirefs[c_end])

        def triples_generator():
            for (c_start, p), target_uris in tqdm(grouped_targets.items(), desc="Materialising triples", disable=not self.verbose):
                start_uris = cluster_to_urirefs[c_start]
                for s in start_uris:
                    for o in target_uris:
                        if s != o:
                            yield (s, p, o, self.g)

        self.g.addN(triples_generator())
        logging.info("Relationship coalescing complete.")

    @timer
    def cleanup_graph(self):
        logging.info("Cleaning up intermediate graph data (RDFLib)...")
        self.g.remove((None, self.ns.hasURL, None))
        self.g.remove((None, self.ns.source_pos, None))
        self.g.remove((None, self.ns.target_pos, None))

        logging.info("Merging duplicate relationship predicates...")
        weights = {s: str(o) for s, o in self.g.subject_objects(self.ns.weight)}
        if not weights: return

        negations = {}
        for s, o in self.g.subject_objects(self.ns.is_negated):
            if s in weights:
                negations[s] = str(o)

        props_map = defaultdict(list)
        for rel_uri, weight_val in tqdm(weights.items(), desc="Analyzing predicates", disable=not self.verbose):
            is_negated_val = negations.get(rel_uri)
            if is_negated_val is not None:
                base_type = self.g.value(rel_uri, RDF.type)
                if base_type:
                    sig = (str(base_type), is_negated_val, weight_val)
                    props_map[sig].append(rel_uri)

        merged_count = 0
        for sig, uris in tqdm(props_map.items(), desc="Merging predicates", disable=not self.verbose):
            if len(uris) > 1:
                uris.sort(key=lambda u: str(u))
                canonical = uris[0]
                duplicates = uris[1:]
                for dup in duplicates:
                    triples_to_move = list(self.g.triples((None, dup, None)))
                    if triples_to_move:
                        for s, _, o in triples_to_move:
                            self.g.add((s, canonical, o))
                        self.g.remove((None, dup, None))
                    self.g.remove((dup, None, None))
                    merged_count += 1

        logging.info(f"Cleanup complete: removed hasURL/POS annotations and merged {merged_count} duplicate predicates.")