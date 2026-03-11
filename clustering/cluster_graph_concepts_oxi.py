import logging
from collections import defaultdict
from tqdm import tqdm

from pyoxigraph import NamedNode, Quad, DefaultGraph
from tools.timer import timer
from clustering.cluster_graph_concepts import AbstractConceptGraphClusterer

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")


class OxiConceptGraphClusterer(AbstractConceptGraphClusterer):

    @timer
    def build_adj_list(self):
        logging.info("  - Building adjacency list from Graph URLs")

        db = defaultdict(set)
        for quad in tqdm(self.g.quads_for_pattern(None, self.ns.hasURL, None), desc="Fetching URL triples",
                         disable=not self.verbose):
            s_uri = quad.subject.value
            o_val = quad.object.value

            o, pos = o_val.split('==')
            s = f"{s_uri}=={pos}"

            db[s].add(o)
            db[o].add(s)

        return {k: sorted(list(v)) for k, v in db.items()}

    @timer
    def add_unclustered_concepts(self, cluster_mappings, visited_nodes):
        logging.info("  - Fetching all concept IDs from graph (Oxigraph)...")
        unvisited_candidates = set()
        for quad in self.g.quads_for_pattern(None, RDF_TYPE, None):
            s_val = quad.subject.value
            if self.base_uri in s_val:
                t_val = quad.object.value
                if self.base_uri in t_val:
                    pos = t_val.split('#')[-1] if '#' in t_val else t_val.split('/')[-1]
                    candidate = f"{s_val}=={pos}"
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
        logging.info("Coalescing relationships between concepts based on clusters (Oxigraph)...")

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
        excluded_predicates = {self.ns.hasURL.value, RDF_TYPE.value}

        logging.info(f"  - Scanning {len(raw_uri_to_cluster_info)} unique concepts for existing relations...")

        for s_raw in tqdm(raw_uri_to_cluster_info.keys(), desc="Scanning graph", disable=not self.verbose):
            s_node = NamedNode(s_raw)

            for quad in self.g.quads_for_pattern(s_node, None, None):
                p_node = quad.predicate
                p_val = p_node.value

                if self.base_uri not in p_val or p_val in excluded_predicates:
                    continue

                if p_node not in predicate_constraints:
                    s_constraints = {q.object.value for q in self.g.quads_for_pattern(p_node, self.ns.source_pos, None)}
                    t_constraints = {q.object.value for q in self.g.quads_for_pattern(p_node, self.ns.target_pos, None)}
                    predicate_constraints[p_node] = (s_constraints, t_constraints)

                s_constraints, t_constraints = predicate_constraints[p_node]
                o_str = quad.object.value

                if o_str in raw_uri_to_cluster_info:
                    s_clusters_info = raw_uri_to_cluster_info[s_raw]
                    o_clusters_info = raw_uri_to_cluster_info[o_str]

                    # Only enforce a POS constraint if at least one cluster entry for this
                    # URI actually satisfies it. When no entry matches, the constraint is a
                    # cross-source artifact and should not silently drop the relation.
                    s_has_match = not s_constraints or any(
                        p in s_constraints for _, p in s_clusters_info if p is not None
                    )
                    o_has_match = not t_constraints or any(
                        p in t_constraints for _, p in o_clusters_info if p is not None
                    )

                    for c_s_label, s_member_pos in s_clusters_info:
                        if s_has_match and s_constraints and s_member_pos and s_member_pos not in s_constraints:
                            continue

                        for c_o_label, o_member_pos in o_clusters_info:
                            if o_has_match and t_constraints and o_member_pos and o_member_pos not in t_constraints:
                                continue

                            cluster_relations.add((c_s_label, p_node, c_o_label))

        logging.info(f"  - Found {len(cluster_relations)} unique cluster-level relationships.")

        # Recover cluster relations for same-term cross-POS pairs that were silently dropped
        # in add_relation_to_graph (start_uri == end_uri)
        # These are added in DB mode but collapse to a self-loop in graph mode
        # Example: come_up (StativeVerb, c110230) isA come_up (MotionVerb, c107946) produces
        # bob_up isA come_up via clustering
        cross_pos_hints = getattr(self.graph_manager, 'cross_pos_hints', [])
        if cross_pos_hints:
            logging.info(f"  - Processing {len(cross_pos_hints)} cross-POS same-URI hints...")
            recovered = 0
            for uri, rel_uri, s_pos, t_pos in cross_pos_hints:
                entries = raw_uri_to_cluster_info.get(uri.value, [])
                pos_to_cluster = {pos: label for label, pos in entries}
                c_s = pos_to_cluster.get(s_pos)
                c_t = pos_to_cluster.get(t_pos)
                if c_s and c_t:
                    # Include same-cluster hints (c_s == c_t): intra-cluster propagation in the
                    # quads_generator will emit all s≠o URI pairs, matching DB mode behaviour
                    # where these are separate concept IDs that coalesce to a self-cluster relation.
                    cluster_relations.add((c_s, rel_uri, c_t))
                    recovered += 1
            logging.info(f"  - Recovered {recovered} cross-POS cluster relations.")

        logging.info("  - Optimizing and propagating relationships to all equivalent concepts...")

        cluster_to_urirefs = {
            label: [NamedNode(uri) for uri in uris]
            for label, uris in cluster_to_raw_uris.items()
        }

        grouped_targets = defaultdict(set)
        for c_start, p_node, c_end in tqdm(cluster_relations, desc="Grouping relations", disable=not self.verbose):
            grouped_targets[(c_start, p_node)].update(cluster_to_urirefs[c_end])

        def quads_generator():
            for (c_start, p_node), target_nodes in tqdm(grouped_targets.items(), desc="Materialising triples",
                                                        disable=not self.verbose):
                start_nodes = cluster_to_urirefs[c_start]
                for s in start_nodes:
                    for o in target_nodes:
                        if s.value != o.value:
                            yield Quad(s, p_node, o, DefaultGraph())

        self.g.extend(quads_generator())
        logging.info("Relationship coalescing complete.")

    @timer
    def cleanup_graph(self):
        logging.info("Cleaning up intermediate graph data (Oxigraph)...")

        def remove_pattern(s, p, o):
            to_remove = list(self.g.quads_for_pattern(s, p, o))
            for q in to_remove:
                self.g.remove(q)

        remove_pattern(None, self.ns.hasURL, None)
        remove_pattern(None, self.ns.source_pos, None)
        remove_pattern(None, self.ns.target_pos, None)

        logging.info("Merging duplicate relationship predicates...")

        weights = {q.subject: q.object.value for q in self.g.quads_for_pattern(None, self.ns.weight, None)}
        if not weights: return

        negations = {}
        for q in self.g.quads_for_pattern(None, self.ns.is_negated, None):
            if q.subject in weights:
                negations[q.subject] = q.object.value

        props_map = defaultdict(list)
        for rel_node, weight_val in tqdm(weights.items(), desc="Analyzing predicates", disable=not self.verbose):
            is_negated_val = negations.get(rel_node)
            if is_negated_val is not None:
                base_type_quad = next(self.g.quads_for_pattern(rel_node, RDF_TYPE, None), None)
                if base_type_quad:
                    sig = (base_type_quad.object.value, is_negated_val, weight_val)
                    props_map[sig].append(rel_node)

        merged_count = 0
        for sig, uris in tqdm(props_map.items(), desc="Merging predicates", disable=not self.verbose):
            if len(uris) > 1:
                uris.sort(key=lambda u: u.value)
                canonical = uris[0]
                duplicates = uris[1:]

                for dup in duplicates:
                    triples_to_move = list(self.g.quads_for_pattern(None, dup, None))
                    if triples_to_move:
                        for q in triples_to_move:
                            self.g.add(Quad(q.subject, canonical, q.object, DefaultGraph()))
                        remove_pattern(None, dup, None)
                    remove_pattern(dup, None, None)
                    merged_count += 1

        logging.info(
            f"Cleanup complete: removed hasURL/POS annotations and merged {merged_count} duplicate predicates.")