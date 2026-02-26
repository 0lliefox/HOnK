import argparse
import logging
import os
import subprocess
import time
from collections import defaultdict

import psycopg2
from psycopg2._psycopg import AsIs
from tqdm import tqdm

from benchmarking.benchmark import Benchmark
from clustering.cluster_concepts import ConceptClusterer
from clustering.cluster_graph_concepts import ConceptGraphClusterer
from knowledge_bases import ConceptNetLoader
from tools.config import get_config
from tools.graph_funcs import GraphManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from knowledge_bases.geonames_loader import GeoNamesLoader
from knowledge_bases.parmenides_loader import ParmenidesLoader
from knowledge_bases.wordnet_loader import WordNetLoader
from knowledge_bases.wiktionary_loader import WiktionaryLoader


class OntologyBuilder:
    def __init__(self, config, run_id = 0, benchmarking = None):
        self.run_id = run_id
        self.config = config
        self.mode = self.config['general']['mode']  # db / graph
        self.should_cache = self.config['general']['should_cache']  # Should the pipeline pickle/load from pickles at certain stages
        self.sources = self.config['general']['sources']
        self.verbose = self.config['general']['verbose']

        # Initialise database
        if self.mode == 'db':
            self.db_params = config['database']
            self.conn = None
            self._connect_db()
            self._setup_database()

        # Serialisation
        self.graph_manager = GraphManager(self, config)
        self.g = self.graph_manager.g
        self.convert = self.config['turtle_export']['convert']  # Should the pipeline convert from .nt to .ttl

        self.should_cluster = self.config['clustering']['enabled']

        if benchmarking is None:
            self.benchmarking = Benchmark("timing")
        else:
            self.benchmarking = benchmarking
        self.memory_benchmarking = Benchmark("memory")

    def get_all_keys(self, nested_dict):
        keys = []
        for key, value in nested_dict.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(self.get_all_keys(value))
        return keys

    def _connect_db(self):
        try:
            self.conn = psycopg2.connect(**self.db_params)
            logging.info(f"Connected to PostgreSQL database '{self.db_params['dbname']}'")
        except psycopg2.OperationalError as e:
            logging.error(f"Database connection error: {e}")
            raise

    def _setup_database(self):
        with self.conn.cursor() as cursor:
            try:
                if self.config['general']['clear_db_on_start'] and self.mode == 'db':
                    confirm_clear_db = self.config['general'].get('confirm_clear_db', True)
                    if confirm_clear_db:
                        user_input = input("Are you sure you want to clear the database? [y/n] ")
                        if user_input.lower() != 'y':
                            logging.info("Database clear aborted by user.")
                            return
                    
                    tables_to_delete = self.config['general']['tables_to_delete']
                    sources_to_delete = self.config['general']['source_to_delete']

                    if not sources_to_delete:
                        logging.info(f"Clearing existing tables ({', '.join(tables_to_delete)}) for all sources")
                        for table in tables_to_delete:
                            cursor.execute("DROP TABLE IF EXISTS %s CASCADE", [AsIs(table)])
                    else:
                        logging.info(f"Clearing existing tables ({', '.join(tables_to_delete)}) for sources: {sources_to_delete}")
                        for table in tables_to_delete:
                            for source in sources_to_delete:
                                cursor.execute("DELETE FROM %s WHERE source = %s", [AsIs(table), source])

                # Create tables
                if not self.config['general']['unique_source']:
                    cursor.execute('''
                       CREATE TABLE IF NOT EXISTS concepts
                       (
                           id             SERIAL PRIMARY KEY,
                           term           TEXT NOT NULL,
                           part_of_speech TEXT NOT NULL,
                           source         TEXT,
                           UNIQUE (term, part_of_speech)
                       );
                       CREATE TABLE IF NOT EXISTS relations
                       (
                           id               SERIAL PRIMARY KEY,
                           start_concept_id INTEGER NOT NULL REFERENCES concepts (id),
                           end_concept_id   INTEGER NOT NULL REFERENCES concepts (id),
                           relation_type    TEXT    NOT NULL,
                           weight           REAL,
                           source           TEXT,
                           UNIQUE (start_concept_id, end_concept_id, relation_type, weight)
                       );
                       CREATE TABLE IF NOT EXISTS properties
                       (
                           id         SERIAL PRIMARY KEY,
                           concept_id INTEGER NOT NULL REFERENCES concepts (id),
                           type       TEXT,
                           value      TEXT,
                           source     TEXT,
                           UNIQUE (concept_id, type, value)
                       );
                       CREATE TABLE IF NOT EXISTS urls
                       (
                           id           SERIAL PRIMARY KEY,
                           concept_id   INTEGER NOT NULL REFERENCES concepts (id),
                           external_url TEXT    NOT NULL,
                           source       TEXT,
                           UNIQUE (concept_id, external_url)
                       )
                    ''')
                else:
                    cursor.execute('''
                       CREATE TABLE IF NOT EXISTS concepts
                       (
                           id             SERIAL PRIMARY KEY,
                           term           TEXT NOT NULL,
                           part_of_speech TEXT NOT NULL,
                           source         TEXT,
                           UNIQUE (term, part_of_speech, source)
                       );
                       CREATE TABLE IF NOT EXISTS relations
                       (
                           id               SERIAL PRIMARY KEY,
                           start_concept_id INTEGER NOT NULL REFERENCES concepts (id),
                           end_concept_id   INTEGER NOT NULL REFERENCES concepts (id),
                           relation_type    TEXT    NOT NULL,
                           weight           REAL,
                           source           TEXT,
                           UNIQUE (start_concept_id, end_concept_id, relation_type, source)
                       );
                       CREATE TABLE IF NOT EXISTS properties
                       (
                           id         SERIAL PRIMARY KEY,
                           concept_id INTEGER NOT NULL REFERENCES concepts (id),
                           type       TEXT,
                           value      TEXT,
                           source     TEXT,
                           UNIQUE (concept_id, type, value, source)
                       );
                       CREATE TABLE IF NOT EXISTS urls
                       (
                           id           SERIAL PRIMARY KEY,
                           concept_id   INTEGER NOT NULL REFERENCES concepts (id),
                           external_url TEXT    NOT NULL,
                           source       TEXT,
                           UNIQUE (concept_id, external_url, source)
                       )
                    ''')
                self.conn.commit()
                logging.info("Database tables are set up")
            except psycopg2.Error as e:
                logging.error(f"Error setting up tables: {e}")
                self.conn.rollback()
                raise e

    def build(self):
        if 'parmenides' in self.sources:
            ParmenidesLoader(self).load_data()
        if 'conceptnet' in self.sources:
            ConceptNetLoader(self).load_data()
        if 'wiktionary' in self.sources:
            WiktionaryLoader(self).load_data()
        if 'wordnet' in self.sources:
            WordNetLoader(self).load_data()
        if 'geonames' in self.sources:
            GeoNamesLoader(self).load_data()
        # DBpediaLoader(self).load_data()
        logging.info("Ontology build process finished")

    def build_graph_from_db(self, file_path):
        start = time.time()

        if not self.conn: logging.error("No DB connection for Turtle dump"); return
        logging.info(f"Building graph for Turtle file: {file_path}")

        with self.conn.cursor() as count_cursor:
            count_cursor.execute("SELECT COUNT(*) FROM concepts")
            total_concepts = count_cursor.fetchone()[0]

        with self.conn.cursor(name='concepts') as cursor:
            cursor.execute("SELECT id, term, part_of_speech, source FROM concepts")
            for cid, term, pos, source in tqdm(cursor, total=total_concepts, desc="Processing concepts", disable=not self.verbose):
                self.graph_manager.add_concept_to_graph(term, pos, cid)

        if not self.should_cluster:
            with self.conn.cursor(name='relations') as cursor:
                cursor.execute("SELECT start_concept_id, end_concept_id, relation_type, weight, source FROM relations")
                for start_id, end_id, rel_type, weight, source in tqdm(cursor, desc="Processing relations", disable=not self.verbose):
                    self.graph_manager.add_relation_to_graph(start_id, end_id, rel_type, weight)
        else:
            cluster_to_concepts_map = defaultdict(list)
            chunk_size = 10000

            with self.conn.cursor(name='fetch_clusters') as cluster_cursor:
                cluster_cursor.execute("SELECT cluster_id, concept_id FROM clusters")
                pbar = tqdm(desc="Fetching cluster data", unit=" mappings", disable=not self.verbose)
                while True:
                    rows = cluster_cursor.fetchmany(size=chunk_size)
                    if not rows:
                        break
                    for cluster_id, concept_id in rows:
                        cluster_to_concepts_map[cluster_id].append(concept_id)
                    pbar.update(len(rows))
                pbar.close()

            logging.info(f"Loaded mappings for {len(cluster_to_concepts_map)} clusters.")

            with self.conn.cursor() as count_cursor:
                count_cursor.execute("SELECT COUNT(*) FROM cluster_relations")
                total_relations = count_cursor.fetchone()[0]

            with self.conn.cursor(name='cluster_relations') as cursor:
                cursor.execute("SELECT start_cluster_id, end_cluster_id, relation_type, weight FROM cluster_relations")
                for start_cluster_id, end_cluster_id, rel_type, weight in tqdm(cursor, total=total_relations, desc="Processing clustered relations", disable=not self.verbose):
                    start_ids = cluster_to_concepts_map.get(start_cluster_id, [])
                    end_ids = cluster_to_concepts_map.get(end_cluster_id, [])

                    if not start_ids or not end_ids:
                        continue

                    for start_id in start_ids:
                        for end_id in end_ids:
                            self.graph_manager.add_relation_to_graph(start_id, end_id, rel_type, weight)

        with self.conn.cursor(name='properties') as cursor:
            cursor.execute("SELECT concept_id, type, value, source FROM properties")
            for cid, prop_type, value, source in tqdm(cursor, desc="Processing properties", disable=not self.verbose):
                self.graph_manager.add_property_to_graph(prop_type, value, cid=cid)

        self.graph_manager.add_pos_tag_classes(self.g)

        end = time.time()
        self.benchmarking.add_row(self.run_id, f"Graph building", end - start)
        self.serialise_graph(file_path, file_path.split('.')[-1], self.g)

    def serialise_graph(self, file_path, ont_format, g):
        try:
            start = time.time()
            file_path = f"ontologies/{file_path}"

            if not os.path.isdir('ontologies'):
                os.mkdir('ontologies')

            logging.info("Serialising ontology")
            g.serialize(destination=file_path, format=ont_format)
            logging.info(f"Successfully saved ontology to '{file_path}'")

            if self.config['general'].get('show_stats', False):
                logging.info(f"Final graph contains {len(g)} triples.")

            end = time.time()
            self.benchmarking.add_row(self.run_id, f"Graph dumping", end - start)

            if ont_format == 'nt' and self.convert:
                logging.info(f"Converting '{file_path}' to .ttl")
                result = subprocess.run(['rapper', '-i', "ntriples", "-o", 'turtle', file_path], capture_output=True,
                                        text=True, check=True)
                output_file_path = file_path.replace('.nt', '.ttl')
                with open(output_file_path, 'w') as f:
                    f.write(result.stdout)
                logging.info(f"Successfully saved ontology to '{output_file_path}'")
        except Exception as e:
            logging.error(f"Failed to write Turtle file: {e}")

    def close(self):
        if self.conn: self.conn.close(); logging.info("Database connection closed")

def run_process(config, run_id, benchmarking):
    builder = OntologyBuilder(config, run_id, benchmarking)
    builder.build()

    output_file = config['turtle_export']['output_file']
    if builder.mode == 'db':
        if builder.should_cluster:
            clusterer = ConceptClusterer(builder, config)
            clusterer.run()

        builder.build_graph_from_db(output_file)
    elif builder.mode == 'graph':
        final_g = builder.graph_manager.g

        if builder.should_cluster:
            clusterer = ConceptGraphClusterer(builder, config)
            final_g = clusterer.run()

        builder.graph_manager.add_pos_tag_classes(final_g)
        builder.serialise_graph(output_file, output_file.split('.')[-1], final_g)

    benchmarking.to_csv(filename=f'benchmark_{output_file.split(".")[0]}', data_length=False, append=True)
    builder.memory_benchmarking.to_csv(filename=f'memory_benchmark_{output_file.split(".")[0]}', data_length=False, append=True)

    if builder.mode == 'db':
        builder.close()

def main(config_file='config.yaml', run_id=0, benchmarking=None):
    config = get_config(config_file)
    benchmarking = Benchmark("timing") if benchmarking is None else benchmarking

    run_process(config, run_id, benchmarking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the ontology.')
    parser.add_argument('--config', dest='config_file', default='config.yaml', help='The configuration file to use.')
    parser.add_argument('--run-id', type=int, default=None, help='Run ID for benchmarking.')
    args = parser.parse_args()
    main(args.config_file, args.run_id)
