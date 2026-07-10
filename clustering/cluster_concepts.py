import json
import logging
import os
from collections import defaultdict

import psycopg2
import psycopg2.extras
from psycopg2._psycopg import AsIs
from tqdm import tqdm



from knowledge_bases.abstract_loader import timer
from tools.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConceptClusterer:
    def __init__(self, builder, config, table_names=None):
        self.builder = builder
        self.conn = self.builder.conn
        self.config = config
        self.verbose = self.config['general']['verbose']

        if table_names is None:
            self.tables = {
                "concepts": "concepts",
                "urls": "urls",
                "relations": "relations",
                "clusters": "clusters",
                "cluster_relations": "cluster_relations"
            }
        else:
            self.tables = table_names

    @timer
    def run(self):
        logging.info("Starting clustering...")
        self.setup_database()
        cluster_mappings = self.find_and_store_clusters()
        self.store_clusters_in_db(cluster_mappings)
        self.coalesce_relationships()
        logging.info("Concept clustering finished")

    def setup_database(self):
        logging.info("Setting up database tables for clustering...")
        with self.conn.cursor() as cursor:
            for table in [self.tables['clusters'], self.tables['cluster_relations']]:
                cursor.execute("DROP TABLE IF EXISTS %s CASCADE", [AsIs(table)])

            cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {AsIs(self.tables['clusters'])}
                        (
                            concept_id INTEGER,
                            cluster_id VARCHAR(10) NOT NULL,
                            FOREIGN KEY (concept_id) REFERENCES {AsIs(self.tables['concepts'])} (id) ON DELETE CASCADE
                            );
                        """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON {AsIs(self.tables['clusters'])}(cluster_id);")

            cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {AsIs(self.tables['cluster_relations'])}
                        (
                            start_cluster_id VARCHAR(10) NOT NULL,
                            end_cluster_id VARCHAR(10) NOT NULL,
                            relation_type TEXT NOT NULL,
                            weight REAL,
                            source TEXT
                            );
                        """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_cluster_relations_start ON {AsIs(self.tables['cluster_relations'])}(start_cluster_id);")
            self.conn.commit()
        logging.info("Clustering tables are set up.")

    def find_and_store_clusters(self):
        cache_path = f"{self.config['local_files']['cache']}/adj_list.json"
        with self.conn.cursor() as cursor:
            # Adjacency list
            db = self.build_adj_list(cursor)

            if self.config['general']['should_cache']:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(db, f, ensure_ascii=False, indent=4)

            # Transitive closure
            self.transitive_closure(db)

            # Build clusters
            cluster_mappings, visited_nodes = self.build_clusters_from_adj(db)

            # Create clusters for unclustered concepts
            self.add_unclustered_concepts(cluster_mappings, cursor, visited_nodes)
        logging.info("Cluster identification and storage complete.")
        return cluster_mappings

    @timer
    def build_adj_list(self, cursor):
        logging.info("  - Building adjacency list from URLs")
        cursor.execute(f"SELECT t1.concept_id, t1.external_url FROM {AsIs(self.tables['urls'])} AS t1")
        edges = cursor.fetchall()

        db = defaultdict(set)
        for id1, id2 in tqdm(edges, desc="Processing edges from URLs database", disable=not self.verbose):
            db[id1].add(id2)
            db[id2].add(id1)

        return {k: sorted(list(v)) for k, v in db.items()}

    @timer
    def add_unclustered_concepts(self, cluster_mappings, cursor, visited_nodes):
        logging.info("  - Fetching all concept IDs from database...")
        cursor.execute(f"SELECT id FROM {AsIs(self.tables['concepts'])};")
        all_concept_ids = {row[0] for row in cursor.fetchall()}
        unvisited_nodes = list(all_concept_ids - visited_nodes)
        cluster_id = int(cluster_mappings[-1][1].split('c')[-1])
        for node in tqdm(unvisited_nodes, desc="Handling isolated nodes", disable=not self.verbose):
            cluster_id += 1
            cluster_mappings.append((node, f"c{cluster_id}"))

    @timer
    def build_clusters_from_adj(self, db):
        logging.info("  - Building clusters from adjacency list...")
        cluster_mappings = []
        visited_nodes = set()

        # Fix B (type-coherence guard): concept_id -> WordNet supersense, derived from
        # the sense column via the same function graph mode uses, so both modes
        # partition disjoint-supersense URL-clusters into identical memberships.
        supersense_of = self.builder.graph_manager.supersense_of_sense
        concept_supersense = {}
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id, sense FROM {AsIs(self.tables['concepts'])} "
                        f"WHERE sense IS NOT NULL AND sense != ''")
            for cid, sense in cur.fetchall():
                ss = supersense_of(sense)
                if ss is not None:
                    concept_supersense[cid] = ss

        visited_clusters = set()
        cluster_id = 0
        split_count = 0
        for key_node, adjacency_list in tqdm(db.items(), desc="Building clusters", disable=not self.verbose):
            cluster_nodes = set(adjacency_list)
            cluster_nodes.add(key_node)

            c_key = tuple(sorted(tuple(map(str, cluster_nodes))))
            if c_key in visited_clusters:
                continue
            visited_clusters.add(c_key)

            groups = defaultdict(list)
            for node in cluster_nodes:
                visited_nodes.add(node)
                if isinstance(node, int):
                    groups[concept_supersense.get(node)].append(node)

            if sum(1 for g in groups if g is not None) > 1:
                split_count += 1

            for ss in sorted(groups.keys(), key=lambda x: (x is not None, x or '')):
                label = f"c{cluster_id}"
                cluster_id += 1
                for c_id in groups[ss]:
                    cluster_mappings.append((c_id, label))

        if split_count:
            logging.info(f"  - Fix B: type-coherence guard split {split_count} disjoint-supersense clusters.")
        return cluster_mappings, visited_nodes

    @timer
    def transitive_closure(self, adjacency_db):
        logging.info("  - Calculating transitive closure on adjacency list...")
        for i in tqdm(adjacency_db.keys(), desc="Transitive closure", disable=not self.verbose):
            adjacency_list = adjacency_db[i]
            seen = set(adjacency_list)
            j_idx = 0
            while j_idx < len(adjacency_list):
                j = adjacency_list[j_idx]
                adjacency_list_j = adjacency_db[j]
                for k in adjacency_list_j:
                    if i != k and k not in seen:
                        seen.add(k)
                        adjacency_list.append(k)
                j_idx += 1
            adjacency_db[i] = adjacency_list

    @timer
    def store_clusters_in_db(self, cluster_mappings):
        with self.conn.cursor() as cursor:
            logging.info(f"  - Storing {len(cluster_mappings)} concept to cluster mappings...")
            psycopg2.extras.execute_values(cursor, f"INSERT INTO {AsIs(self.tables['clusters'])} (concept_id, cluster_id) VALUES %s",
                                           cluster_mappings)
            self.conn.commit()

    @timer
    def coalesce_relationships(self):
        # Join clusters table from cluster ID to start_concept_id and end_concept_id from relations table
        logging.info("Coalescing relationships between clusters...")
        with self.conn.cursor() as cursor:
            # The projection hash-joins all relations against the clusters table twice; at the
            # default work_mem the hash spills to temp across hundreds of batches (the I/O cost
            # reviewers flagged, R1-D3). Raising it for this transaction cuts the spill by two
            # orders of magnitude (512 batches -> 8 on the full graph); a single in-memory
            # partition would need ~1GB, which is unsafe per-connection under concurrent builds.
            cursor.execute("SET LOCAL work_mem = '256MB'")
            cursor.execute(f"TRUNCATE TABLE {AsIs(self.tables['cluster_relations'])};")
            cursor.execute(f"""
                        INSERT INTO {AsIs(self.tables['cluster_relations'])} (start_cluster_id, end_cluster_id, relation_type, weight, source)
                        SELECT DISTINCT c1.cluster_id,
                                        c2.cluster_id,
                                        r.relation_type,
                                        r.weight,
                                        r.source
                        FROM {AsIs(self.tables['relations'])} AS r
                                 JOIN
                             {AsIs(self.tables['clusters'])} AS c1 ON r.start_concept_id = c1.concept_id
                                 JOIN
                             {AsIs(self.tables['clusters'])} AS c2 ON r.end_concept_id = c2.concept_id;
                        """)
            # Build the concept_id index AFTER the bulk projection so the planner keeps the
            # faster hash join for the projection itself (with the index present it switches to
            # a slower nested loop). The index then speeds downstream point-lookups: graph
            # serialisation, word_relations, and the fidelity audit.
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_clusters_concept_id ON {AsIs(self.tables['clusters'])}(concept_id);")
            self.conn.commit()
            logging.info(f"  - {cursor.rowcount} new cluster relationships were created.")
        logging.info("Relationship coalescing complete.")

def main():
    os.chdir('../')
    config = get_config()

    from build_ontology import OntologyBuilder
    from benchmarking.benchmark import Benchmark
    builder = OntologyBuilder(config, 0, Benchmark("scalability"))
    clusterer = ConceptClusterer(builder, config)
    clusterer.run()


if __name__ == "__main__":
    main()
