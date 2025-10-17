import json
import logging
import time
from collections import defaultdict

import psycopg2
import psycopg2.extras
import yaml
from psycopg2._psycopg import AsIs
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConceptClusterer:
    def __init__(self, builder, config):
        self.builder = builder
        self.conn = self.builder.conn
        self.config = config
        psycopg2.extras.register_uuid() # Used to convert Python UUID to PostgreSQL UUID

    def run(self):
        start = time.time()

        logging.info("Starting clustering...")
        self._setup_database()
        self._find_and_store_clusters()
        self._coalesce_relationships()
        logging.info("Concept clustering finished")

        end = time.time()
        self.builder.benchmarking.add_row(self.builder.run_id, "Clustering", end - start)
        logging.info(f"Clustering took {end - start:.2f} seconds.")

    def _setup_database(self):
        logging.info("Setting up database tables for clustering...")
        with self.conn.cursor() as cursor:
            for table in ['clusters', 'cluster_relations']:
                cursor.execute("DROP TABLE IF EXISTS %s CASCADE", [AsIs(table)])

            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS clusters
                        (
                            concept_id INTEGER,
                            cluster_id VARCHAR(10) NOT NULL,
                            FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
                            );
                        """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON clusters(cluster_id);")

            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS cluster_relations
                        (
                            start_cluster_id VARCHAR(10) NOT NULL,
                            end_cluster_id VARCHAR(10) NOT NULL,
                            relation_type TEXT NOT NULL,
                            weight REAL,
                            source TEXT,
                            PRIMARY KEY (start_cluster_id, end_cluster_id, relation_type)
                            );
                        """)
            self.conn.commit()
        logging.info("Clustering tables are set up.")

    def _find_and_store_clusters(self):
        cache_path = f"{self.config['local_files']['cache']}/adj_list.json"
        with self.conn.cursor() as cursor:
            # Adjacency list
            logging.info("  - Building adjacency list from URLs")

            # if not os.path.exists(cache_path):
                # cursor.execute("SELECT t1.concept_id, t2.concept_id "
                #                "FROM urls AS t1 JOIN urls AS t2 ON t1.external_url = t2.external_url "
                #                "WHERE t1.concept_id != t2.concept_id;")
            cursor.execute("SELECT t1.concept_id, t1.external_url FROM urls AS t1")
            edges = cursor.fetchall()

            db = defaultdict(set)
            for id1, id2 in tqdm(edges, desc="Processing edges from URLs database"):
                db[id1].add(id2)
                db[id2].add(id1)

            db = {k: sorted(list(v)) for k, v in db.items()}

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=4)
            # else:
            #     db = json.load(open(cache_path))

            # Transitive closure (Floyd Warshall)
            self.floyd_warshall(db)

            # Build clusters
            logging.info("  - Building clusters from adjacency list...")
            cluster_mappings = []
            visited_nodes = set()

            visited_clusters = dict()
            for key_node, adjacency_list in tqdm(db.items(), desc="Building clusters"):
                cluster_nodes = set(adjacency_list)
                cluster_nodes.add(key_node)

                c_key = tuple(sorted(tuple(map(str, cluster_nodes))))
                if not (c_key in visited_clusters.keys()):
                    cluster_id = len(visited_clusters)
                    visited_clusters[c_key] = cluster_id

                    c_ids = {c_id for c_id in cluster_nodes if isinstance(c_id, int)}
                    for c_id in c_ids:
                        cluster_mappings.append((c_id, f"c{cluster_id}"))
                    for node in cluster_nodes:
                        visited_nodes.add(node)

            logging.info("  - Fetching all concept IDs from database...")
            cursor.execute("SELECT id FROM concepts;")
            all_concept_ids = {row[0] for row in cursor.fetchall()}
            unvisited_nodes = list(all_concept_ids - visited_nodes)
            cluster_id = int(cluster_mappings[-1][1].split('c')[-1])
            for node in tqdm(unvisited_nodes, desc="Handling isolated nodes"):
                cluster_id += 1
                cluster_mappings.append((node, f"c{cluster_id}"))

            logging.info(f"  - Storing {len(cluster_mappings)} concept to cluster mappings...")
            psycopg2.extras.execute_values(cursor, "INSERT INTO clusters (concept_id, cluster_id) VALUES %s", cluster_mappings)
            self.conn.commit()
        logging.info("Cluster identification and storage complete.")

    def floyd_warshall(self, adjacency_db):
        logging.info("  - Calculating transitive closure on adjacency list...")
        for i in tqdm(adjacency_db.keys(), desc="Floyd Warshall"):
            adjacency_list = adjacency_db[i]
            j_idx = 0
            while j_idx < len(adjacency_list):
                j = adjacency_list[j_idx]
                adjacency_list_j = adjacency_db[j]
                k_idx = 0
                while k_idx < len(adjacency_list_j):
                    k = adjacency_list_j[k_idx]
                    if i != k and k not in adjacency_list:
                        adjacency_list.append(k)
                    k_idx += 1
                j_idx += 1
                adjacency_db[j] = adjacency_list_j
            adjacency_db[i] = adjacency_list

    def _coalesce_relationships(self):
        # Join clusters table from cluster ID to start_concept_id and end_concept_id from relations table
        logging.info("Coalescing relationships between clusters...")
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE cluster_relations;")
            cursor.execute("""
                        INSERT INTO cluster_relations (start_cluster_id, end_cluster_id, relation_type, weight, source)
                        SELECT DISTINCT c1.cluster_id,
                                        c2.cluster_id,
                                        r.relation_type,
                                        r.weight,
                                        r.source
                        FROM relations AS r
                                 JOIN
                             clusters AS c1 ON r.start_concept_id = c1.concept_id
                                 JOIN
                             clusters AS c2 ON r.end_concept_id = c2.concept_id
                        WHERE c1.cluster_id != c2.cluster_id
                        ON CONFLICT (start_cluster_id, end_cluster_id, relation_type) DO NOTHING;
                        """)
            self.conn.commit()
            logging.info(f"  - {cursor.rowcount} new cluster relationships were created.")
        logging.info("Relationship coalescing complete.")

def main():
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    db = config.get('database')

    conn = None
    try:
        conn = psycopg2.connect(**db)
        clusterer = ConceptClusterer(conn, config)
        clusterer.run()
    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    main()

