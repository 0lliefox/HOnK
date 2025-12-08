import json
import logging
import re
import subprocess
from functools import lru_cache
from urllib.parse import quote
import argparse

import psycopg2
import yaml
from psycopg2._psycopg import AsIs
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from tqdm import tqdm
from collections import defaultdict
import time

from benchmarking.benchmark import Benchmark
from clustering.cluster_concepts import ConceptClusterer
from knowledge_bases import ConceptNetLoader
from knowledge_bases.dbpedia_loader import DBpediaLoader
from tools.config import get_config
from tools.database_utilities import dump_database, restore_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from knowledge_bases.geonames_loader import GeoNamesLoader
from knowledge_bases.parmenides_loader import ParmenidesLoader
from knowledge_bases.wordnet_loader import WordNetLoader
from knowledge_bases.wiktionary_loader import WiktionaryLoader


class OntologyBuilder:
    def __init__(self, config, run_id = 0, benchmarking = None):
        self.run_id = run_id
        self.config = config
        self.mode = self.config['general']['mode']
        self.should_cache = self.config['general']['should_cache']
        self.db_params = config['database']
        self.conn = None
        self.ns = Namespace(self.config['turtle_export']['base_uri'])
        self.normalise_pos = config['turtle_export']['normalise_pos']

        sources = self.config['general']['sources']
        mapping_types = ["edge", "pos"]

        self.full_mappings = {
            k.lower(): v
            for source in sources
            for m_type in mapping_types
            for k, v in self.load_mappings(config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
        }

        self.lang_code_map = {'en': 'eng'}
        self.wordnet_lang = self.lang_code_map.get(config['general']['language'], 'eng')
        self._connect_db()
        self._setup_database()

        # Class mappings
        with open(self.config['local_files']['ontology_classes'], 'r') as f:
            self.ontology_classes = json.load(f)
            self.lemma_mappings = self.ontology_classes['MetaGrammaticalFunction']['classes']['GrammaticalFunction']['classes']

        # POS tag mappings
        with open(self.config['local_files']['pos_tag_classes'], 'r') as f:
            self.pos_tag_mappings = json.load(f)

        self.equivalent_classes = {}  # Map of classes that should be added to a concept as equivalent classes
        self.annotation_property_list = {}
        self.prop_id = 1

        self.should_cluster = self.config['clustering']['enabled']

        # Rejected classes
        with open(self.config['local_files']['rejected_classes'], 'r') as f:
            self.rejected_classes = json.load(f)

        if benchmarking is None:
            self.benchmarking = Benchmark("timing")
        else:
            self.benchmarking = benchmarking

    def get_all_keys(self, nested_dict):
        keys = []
        for key, value in nested_dict.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(self.get_all_keys(value))
        return keys

    def load_mappings(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {k.lower(): v for k, v in data.items()}

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
                if self.config['general']['clear_db_on_start']:
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
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS concepts
                               (
                                   id SERIAL PRIMARY KEY,
                                   term TEXT NOT NULL,
                                   part_of_speech TEXT NOT NULL,
                                   source TEXT,
                                   UNIQUE (term, part_of_speech)
                               );
                               CREATE TABLE IF NOT EXISTS relations
                               (
                                   id SERIAL PRIMARY KEY,
                                   start_concept_id INTEGER NOT NULL REFERENCES concepts (id),
                                   end_concept_id INTEGER NOT NULL REFERENCES concepts (id),
                                   relation_type TEXT NOT NULL,
                                   weight REAL,
                                   source TEXT,
                                   UNIQUE (start_concept_id, end_concept_id, relation_type)
                               );
                               CREATE TABLE IF NOT EXISTS properties
                               (
                                   id SERIAL PRIMARY KEY,
                                   concept_id INTEGER NOT NULL REFERENCES concepts (id),
                                   type TEXT,
                                   value TEXT,
                                   source TEXT,
                                   UNIQUE (concept_id, type, value)
                               );
                               CREATE TABLE IF NOT EXISTS urls
                               (
                                   id SERIAL PRIMARY KEY,
                                   concept_id INTEGER NOT NULL REFERENCES concepts (id),
                                   external_url TEXT NOT NULL,
                                   source TEXT,
                                   UNIQUE (concept_id, external_url)
                               )
                               ''')
                self.conn.commit()
                logging.info("Database tables are set up")
            except psycopg2.Error as e:
                logging.error(f"Error setting up tables: {e}")
                self.conn.rollback()
                raise e

    def build(self):
        ParmenidesLoader(self).load_data_with_timer()
        GeoNamesLoader(self).load_data_with_timer()
        # dump_database(self.db_params, '.cache/ontology_geonames_backup.sql')
        WordNetLoader(self).load_data_with_timer()
        ConceptNetLoader(self).load_data_with_timer()
        WiktionaryLoader(self).load_data_with_timer()
        # dump_database(self.db_params, '.cache/ontology_all_minus_dbpedia_backup.sql')
        # DBpediaLoader(self).load_data_with_timer()
        logging.info("Ontology build process finished")

    def get_safe_uri(self, term):
        return URIRef(self.ns + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else self.ns[term]

    def build_graph(self, file_path):
        start = time.time()

        if not self.conn: logging.error("No DB connection for Turtle dump"); return
        logging.info(f"Dumping database to Turtle file: {file_path}")

        # g = load_from_pickle('graph.pkl')
        # if g is None:
        g = Graph()
        g.bind(self.config['turtle_export']['base_prefix'], self.ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        # Setup classes from JSON
        self.create_classes(g)

        # Add logical functions
        # ParmenidesLoader.add_logical_functions(self.config, g)

        all_properties = set()

        # self.id_to_uri = load_from_pickle('id_to_uri.pkl')
        # if self.id_to_uri is None:
        self.id_to_uri = {}
        with self.conn.cursor() as count_cursor:
            count_cursor.execute("SELECT COUNT(*) FROM concepts")
            total_concepts = count_cursor.fetchone()[0]

        with self.conn.cursor(name='concepts') as cursor:
            cursor.execute("SELECT id, term, part_of_speech, source FROM concepts")
            for cid, term, pos, source in tqdm(cursor, total=total_concepts, desc="Processing Concepts"):
                if pos.lower() in self.rejected_classes:
                    continue

                uri = self.get_safe_uri(term)
                self.id_to_uri[cid] = uri

                for i_pos in self.equivalent_classes.get(pos, [pos]):
                    g.add((uri, RDF.type, self.ns[i_pos]))
                g.add((uri, RDFS.label, Literal(term, datatype=XSD.string)))

            # save_to_pickle('graph.pkl', g)
            # save_to_pickle('id_to_uri.pkl', self.id_to_uri)

        if not self.should_cluster:
            declared_base_properties = set()
            with self.conn.cursor(name='relations') as cursor:
                cursor.execute("SELECT start_concept_id, end_concept_id, relation_type, weight, source FROM relations")
                for start_id, end_id, rel_type, weight, source in tqdm(cursor, desc="Processing Relations"):
                    self.add_relation_to_graph(g, start_id, end_id, rel_type, weight, declared_base_properties)
        else:
            cluster_to_concepts_map = defaultdict(list)
            chunk_size = 10000

            with self.conn.cursor(name='fetch_clusters') as cluster_cursor:
                cluster_cursor.execute("SELECT cluster_id, concept_id FROM clusters")
                pbar = tqdm(desc="Fetching Cluster Data", unit=" mappings")
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

            declared_base_properties = set()
            with self.conn.cursor(name='cluster_relations') as cursor:
                cursor.execute("SELECT start_cluster_id, end_cluster_id, relation_type, weight FROM cluster_relations")
                for start_cluster_id, end_cluster_id, rel_type, weight in tqdm(cursor, total=total_relations, desc="Processing Clustered Relations"):
                    start_ids = cluster_to_concepts_map.get(start_cluster_id, [])
                    end_ids = cluster_to_concepts_map.get(end_cluster_id, [])

                    if not start_ids or not end_ids:
                        continue

                    for start_id in start_ids:
                        for end_id in end_ids:
                            self.add_relation_to_graph(g, start_id, end_id, rel_type, weight, declared_base_properties)

        with self.conn.cursor(name='properties') as cursor:
            cursor.execute("SELECT concept_id, type, value, source FROM properties")
            for concept_id, prop_type, value, source in tqdm(cursor, desc="Processing Properties"):
                if concept_id in self.id_to_uri:
                    concept_uri = self.id_to_uri[concept_id]
                    prop_uri = self.get_safe_uri(prop_type)
                    g.add((concept_uri, prop_uri, Literal(value)))
                    all_properties.add((prop_uri, OWL.DatatypeProperty))

        for prop_uri, prop_type in tqdm(all_properties, desc="Adding Properties"):
            g.add((prop_uri, RDF.type, prop_type))

        self.add_pos_tag_classes(g)

        end = time.time()
        self.benchmarking.add_row(self.run_id, f"Graph building", end - start)
        self.serialise_graph(file_path, file_path.split('.')[-1], g)

    def serialise_graph(self, file_path, ont_format, g):
        try:
            start = time.time()
            logging.info("Serialising ontology")
            g.serialize(destination=file_path, format=ont_format)
            logging.info(f"Successfully saved ontology to '{file_path}'")

            end = time.time()
            self.benchmarking.add_row(self.run_id, f"Graph dumping", end - start)

            if ont_format == 'nt':
                logging.info(f"Converting '{file_path}' to .ttl")
                result = subprocess.run(['rapper', '-i', "ntriples", "-o", 'turtle', file_path], capture_output=True,
                                        text=True, check=True)
                output_file_path = file_path.replace('.nt', '.ttl')
                with open(output_file_path, 'w') as f:
                    f.write(result.stdout)
                logging.info(f"Successfully saved ontology to '{output_file_path}'")
        except Exception as e:
            logging.error(f"Failed to write Turtle file: {e}")

    def add_relation_to_graph(self, g, start_id, end_id, rel_type, weight, declared_base_properties):
        try:
            start_uri, end_uri = self.id_to_uri[start_id], self.id_to_uri[end_id]

            if self.normalise_pos:
                mapping = self.full_mappings.get(rel_type.lower(), {'rel': rel_type, 'relNegated': False, 'swap': False})
            else:
                mapping = {'rel': rel_type, 'relNegated': False, 'swap': False}

            rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

            sub_property_list = {'rel': rel, 'is_negated': Literal(is_negated), 'weight': Literal(weight)}
            sub_list_key = frozenset(sub_property_list.items())

            if sub_list_key in self.annotation_property_list:
                rel_uri = self.annotation_property_list[sub_list_key]
                new_annotation_instance = False
            else:
                rel_uri = self.get_safe_uri(f"{rel} {self.prop_id}")
                self.annotation_property_list[sub_list_key] = rel_uri
                self.prop_id += 1
                new_annotation_instance = True

            # g.add((self.ns[rel], RDF.type, OWL.AnnotationProperty))
            # g.add((rel_uri, RDF.type, self.ns[rel]))

            # for sub_prop_key, sub_prop_value in sub_property_list.items():
            #     if sub_prop_key == 'rel': continue
            #     g.add((rel_uri, self.ns[sub_prop_key], sub_prop_value))

            base_prop_uri = self.ns[rel]
            if base_prop_uri not in declared_base_properties:
                g.add((base_prop_uri, RDF.type, OWL.AnnotationProperty))
                declared_base_properties.add(base_prop_uri)

            if new_annotation_instance:
                g.add((rel_uri, RDF.type, base_prop_uri))
                g.add((rel_uri, self.ns['is_negated'], Literal(is_negated)))
                g.add((rel_uri, self.ns['weight'], Literal(float(weight))))

            s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
            g.add((s, rel_uri, t))
        except KeyError:
            pass

    def create_classes(self, g):
        logging.info("Creating ontology classes")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = self.ns[parent]
            g.add((parent_uri, RDF.type, OWL.Class))
            self.create_sub_classes(parent, children, g, parent_uri)

    def create_sub_classes(self, parent, children, g: Graph, parent_uri: URIRef):
        ignored_keys = ['classes', 'sameAs']
        for child in children:
            child_uri = self.ns[child]
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.subClassOf, parent_uri))
            if 'classes' in children[child]:
                self.create_sub_classes(parent, children[child]['classes'], g, child_uri)
            elif child not in ignored_keys and len(children[child]) > 0:
                self.create_sub_classes(parent, children[child], g, child_uri)

            if len(children[child]) > 0 and 'sameAs' in children[child]:
                self.equivalent_classes[child] = [child, children[child]['sameAs']]

    def add_pos_tag_classes(self, g):
        logging.info("Adding POS tag classes")
        g.add((self.ns['POSTag'], RDF.type, OWL.Class))

        new_triples = set()
        for pos_tag, mapping in tqdm(self.pos_tag_mappings.items(), desc="Processing POS tagging"):
            pos_uri = self.get_safe_uri(pos_tag)
            g.add((pos_uri, RDFS.subClassOf, self.ns['POSTag']))

            if "classes" not in mapping:
                continue

            for class_name, class_details in mapping["classes"].items():
                class_uri = self.ns[class_name]

                required_property_uris = [
                    (self.get_safe_uri(prop), Literal(True))
                    for prop in class_details.get("properties", [])
                ]

                candidate_subjects = g.subjects(RDF.type, class_uri)
                for subject_uri in candidate_subjects:
                    has_all_properties = all(
                        (subject_uri, prop_uri, prop_val) in g
                        for prop_uri, prop_val in required_property_uris
                    )

                    if has_all_properties:
                        new_triples.add((subject_uri, RDF.type, pos_uri))

        logging.info(f"Identified {len(new_triples)} new POS tag classifications to add.")
        for triple in new_triples:
            if triple not in g:
                g.add(triple)

    @lru_cache(maxsize=1024)
    def get_concept_from_cluster(self, cluster_id):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT concept_id, cluster_id FROM clusters WHERE cluster_id = %s", [cluster_id])
            concept_ids = [row[0] for row in cursor.fetchall()]

        return concept_ids

    def close(self):
        if self.conn: self.conn.close(); logging.info("Database connection closed")

def main(config_file='config.yaml', iterations=1):
    config = get_config(config_file)

    builder = None
    try:
        benchmarking = Benchmark("timing")
        for i in range(iterations):
            builder = OntologyBuilder(config, i, benchmarking)
            # restore_database(config['database'], '.cache/ontology_clusters_backup.sql')
            builder.build()

            if builder.mode == 'db':
                if builder.should_cluster:
                    clusterer = ConceptClusterer(builder, config)
                    clusterer.run()

                builder.build_graph(config['turtle_export']['output_file'])
        benchmarking.to_csv()
    except (psycopg2.Error, ConnectionRefusedError) as e:
        print(f"\nA database error occurred: {e}")
    finally:
        if builder:
            builder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the ontology.')
    parser.add_argument('--config', dest='config_file', default='config.yaml',
                        help='The configuration file to use.')
    args = parser.parse_args()
    main(args.config_file)
