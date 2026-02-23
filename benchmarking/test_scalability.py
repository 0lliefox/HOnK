import logging
import os
import sys
import argparse

# Add the project root to sys.path to allow imports from 'tools' and 'clustering'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
import yaml
from psycopg2._psycopg import AsIs

from benchmarking.benchmark import Benchmark
from benchmarking.plot_results import ResultsPlotter
from clustering.cluster_concepts import ConceptClusterer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SOURCE_TABLES = {
    "concepts": "concepts",
    "urls": "urls",
    "relations": "relations",
    "clusters": "clusters",
}
TEST_TABLES = {
    "concepts": "concepts_test",
    "urls": "urls_test",
    "relations": "relations_test",
    "clusters": "clusters_test",
    "cluster_relations": "cluster_relations_test"
}

class MockBuilder:
    def __init__(self, conn, run_id):
        self.conn = conn
        self.benchmarking = Benchmark("scalability")
        self.memory_benchmarking = Benchmark("memory")
        self.run_id = run_id

def create_subset_tables(conn, fraction):
    logging.info(f"--- Creating subset tables for {fraction * 100:.0f}% of data (sampled by cluster) ---")

    with conn.cursor() as cursor:
        try:
            # Drop old test tables
            for table in TEST_TABLES.values():
                cursor.execute(f"DROP TABLE IF EXISTS {AsIs(table)} CASCADE;")

            # Get total distinct clusters
            cursor.execute(f"SELECT COUNT(DISTINCT cluster_id) FROM {AsIs(SOURCE_TABLES['clusters'])}")
            total_distinct_clusters = cursor.fetchone()[0]

            sample_count = int(total_distinct_clusters * fraction)
            logging.info(f"Sampling {sample_count} out of {total_distinct_clusters} unique cluster IDs...")

            # Sample data
            cursor.execute(f"""
                CREATE TEMPORARY TABLE _sampled_cluster_ids AS
                SELECT cluster_id 
                FROM (
                    SELECT DISTINCT cluster_id FROM {AsIs(SOURCE_TABLES['clusters'])}
                ) AS unique_clusters
                ORDER BY RANDOM()
                LIMIT %s;

                CREATE INDEX ON _sampled_cluster_ids (cluster_id);

                CREATE TABLE {AsIs(TEST_TABLES['concepts'])} AS
                SELECT c_main.*
                FROM {AsIs(SOURCE_TABLES['concepts'])} AS c_main
                JOIN {AsIs(SOURCE_TABLES['clusters'])} AS cl_main ON c_main.id = cl_main.concept_id
                JOIN _sampled_cluster_ids AS s ON cl_main.cluster_id = s.cluster_id;

                ALTER TABLE {AsIs(TEST_TABLES['concepts'])} ADD PRIMARY KEY (id);
                DROP TABLE _sampled_cluster_ids;
            """, (sample_count,))

            cursor.execute(f"SELECT COUNT(*) FROM {AsIs(TEST_TABLES['concepts'])}")
            concept_count = cursor.fetchone()[0]

            logging.info("Populating subset of urls...")
            cursor.execute(f"""
                CREATE TABLE {AsIs(TEST_TABLES['urls'])} AS
                SELECT t1.*
                FROM {AsIs(SOURCE_TABLES['urls'])} AS t1
                JOIN {AsIs(TEST_TABLES['concepts'])} AS t2 ON t1.concept_id = t2.id;
            """)
            cursor.execute(f"SELECT COUNT(*) FROM {AsIs(TEST_TABLES['urls'])}")
            url_count = cursor.fetchone()[0]

            logging.info("Populating subset of relations...")
            cursor.execute(f"""
                CREATE TABLE {AsIs(TEST_TABLES['relations'])} AS
                SELECT r.*
                FROM {AsIs(SOURCE_TABLES['relations'])} AS r
                WHERE EXISTS (
                    SELECT 1 FROM {AsIs(TEST_TABLES['concepts'])} AS c1
                    WHERE r.start_concept_id = c1.id
                )
                AND EXISTS (
                    SELECT 1 FROM {AsIs(TEST_TABLES['concepts'])} AS c2
                    WHERE r.end_concept_id = c2.id
                );
            """)
            cursor.execute(f"SELECT COUNT(*) FROM {AsIs(TEST_TABLES['relations'])}")
            relation_count = cursor.fetchone()[0]

            conn.commit()
            logging.info("Subset tables created successfully.")
            return concept_count, url_count, relation_count

        except psycopg2.Error as e:
            logging.error(f"Error creating subset tables: {e}")
            conn.rollback()
            return 0, 0, 0


def drop_subset_tables(conn):
    logging.info("Dropping subset tables")
    try:
        with conn.cursor() as cursor:
            for table in TEST_TABLES.values():
                cursor.execute(f"DROP TABLE IF EXISTS {AsIs(table)} CASCADE;")
            conn.commit()
        logging.info("Subset tables dropped.")
    except psycopg2.Error as e:
        logging.error(f"Error dropping subset tables: {e}")
        conn.rollback()


def run_experiment(config, fractions):
    db = config.get('database')

    conn = None
    try:
        conn = psycopg2.connect(**db)

        for frac in fractions:
            counts = create_subset_tables(conn, frac)
            concept_count, url_count, relation_count = counts

            if concept_count == 0:
                continue

            id_percentage = int(frac * 100)
            mock_builder = MockBuilder(conn, id_percentage)
            mock_builder.benchmarking.add_row(id_percentage, 'concept_count', concept_count)
            clusterer = ConceptClusterer(mock_builder, config, table_names=TEST_TABLES)

            logging.info(f"Running test for {id_percentage}% ({concept_count} concepts)")

            clusterer.run()
            mock_builder.benchmarking.to_csv("clustering_benchmark", data_length=False, append=True)

            drop_subset_tables(conn)
    except psycopg2.Error as e:
        logging.error(f"Database connection error: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


def main(num_runs=1):
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.chdir('../')
    
    for i in range(num_runs):
        run_experiment(config, TEST_FRACTIONS)

    os.chdir('benchmarking')
    file_path = 'results/clustering_benchmark.csv'
    plotter = ResultsPlotter(file_path)
    plotter.run()


if __name__ == "__main__":
    main(10)