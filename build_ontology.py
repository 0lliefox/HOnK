import dataclasses
import json
import logging
import re
from urllib.parse import quote

import psycopg2
import yaml
from oxrdflib import OxigraphStore
from psycopg2._psycopg import AsIs
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from tqdm import tqdm

from knowledge_bases import ConceptNetLoader
from supporting_files.parmenides.classes import SentenceStructure
from supporting_files.parmenides.classes.ParmenidesBuild import ParmenidesBuild

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from knowledge_bases.geonames_loader import GeoNamesLoader
from knowledge_bases.parmenides_loader import ParmenidesLoader
from knowledge_bases.wordnet_loader import WordNetLoader
from knowledge_bases.wiktionary_loader import WiktionaryLoader


class OntologyBuilder:
    def __init__(self, config):
        self.config = config
        self.db_params = config['database']
        self.conn = None

        sources = self.config['general']['sources']
        mapping_types = ["edge", "pos"]

        self.full_mappings = {
            k.lower(): v
            for source in sources
            for m_type in mapping_types
            for k, v in self._load_mappings(config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
        }

        self.lang_code_map = {'en': 'eng'}
        self.wordnet_lang = self.lang_code_map.get(config['general']['language'], 'eng')
        self._connect_db()
        self._setup_database()

        # Class mappings
        with open(self.config['local_files']['ontology_classes'], 'r') as f:
            self.ontology_classes = json.load(f)
            self.lemma_mappings = self.ontology_classes['MetaGrammaticalFunction']['classes']['GrammaticalFunction']['classes']

        self.equivalent_classes = {}

        # Rejected classes
        with open(self.config['local_files']['rejected_classes'], 'r') as f:
            self.rejected_classes = json.load(f)

    def get_all_keys(self, nested_dict):
        keys = []
        for key, value in nested_dict.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(self.get_all_keys(value))
        return keys

    def _load_mappings(self, filepath):
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
                    tables_to_delete = self.config['general']['tables_to_delete']
                    sources_to_delete = self.config['general']['source_to_delete']

                    if len(self.config['general']['source_to_delete']) == 0:
                        logging.info(f"Clearing existing tables ({', '.join(tables_to_delete)}) for all sources")
                        for table in tables_to_delete:
                            cursor.execute("DROP TABLE IF EXISTS %s CASCADE", [AsIs(table)])
                    else:
                        logging.info(f"Clearing existing tables ({', '.join(tables_to_delete)}) for sources: {self.config['general']['source_to_delete']}")
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
                                   prop_type TEXT NOT NULL,
                                   source TEXT,
                                   UNIQUE (concept_id, prop_type)
                               );
                               CREATE TABLE IF NOT EXISTS undirected
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
        ParmenidesLoader(self).load_data()
        GeoNamesLoader(self).load_data()
        WordNetLoader(self).load_data()
        ConceptNetLoader(self).load_data()
        WiktionaryLoader(self).load_data()
        logging.info("Ontology build process finished")

    def _get_safe_uri(self, term, namespace):
        return URIRef(namespace + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else namespace[term]

    def dump_to_turtle(self, file_path):
        if not self.conn: logging.error("No DB connection for Turtle dump"); return
        logging.info(f"Dumping database to Turtle file: {file_path}")

        g = Graph()
        ns = Namespace(self.config['turtle_export']['base_uri'])
        g.bind(self.config['turtle_export']['base_prefix'], ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        # Setup classes from JSON
        self.create_classes(g, ns)

        # Add logical functions
        ParmenidesLoader.add_logical_functions(g)

        all_properties = set()
        with self.conn.cursor(name='concepts') as cursor:
            cursor.execute("SELECT id, term, part_of_speech, source FROM concepts")
            concepts_data = cursor.fetchall()
            id_to_uri = {cid: self._get_safe_uri(term, ns) for cid, term, _, _ in concepts_data}

            for cid, term, pos, source in tqdm(concepts_data, desc="Processing Concepts"):
                if pos.lower() in self.rejected_classes or 'GeoNames' in source: continue

                uri = id_to_uri[cid]

                for i_pos in self.equivalent_classes.get(pos, [pos]):
                    g.add((uri, RDF.type, ns[i_pos]))
                g.add((uri, RDFS.label, Literal(term, datatype=XSD.string)))

                # for prop in ['alwaysOmit', 'alwaysTakeInfinitive', 'canOmit', 'comparative', 'superlative']:
                #     g.add((uri, ns[prop], Literal(False)))
                #     all_properties.add((ns[prop], OWL.DatatypeProperty))

        with self.conn.cursor(name='undirected') as cursor:
            cursor.execute("SELECT concept_id, type, value, source FROM undirected")
            for cid, ctype, value, source in tqdm(cursor, desc="Processing Undirected (Definitions, alternatives)"):
                uri = id_to_uri[cid]
                g.add((uri, ns[ctype], Literal(value)))

        with self.conn.cursor(name='relations') as cursor:
            prop_id = 1
            annotation_property_list = {}

            cursor.execute("SELECT start_concept_id, end_concept_id, relation_type, weight, source FROM relations")
            for start_id, end_id, rel_type, weight, source in tqdm(cursor, desc="Processing Relations"):
                if 'GeoNames' in source: continue

                if start_id in id_to_uri and end_id in id_to_uri:
                    start_uri, end_uri = id_to_uri[start_id], id_to_uri[end_id]

                    mapping = self.full_mappings.get(rel_type.lower(), {'rel': rel_type, 'relNegated': False, 'swap': False})

                    rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

                    sub_property_list = {'rel': rel, 'is_negated': Literal(is_negated), 'weight': Literal(weight)}
                    sub_list_key = frozenset(sub_property_list.items())

                    if sub_list_key in annotation_property_list:
                        rel_uri = annotation_property_list[sub_list_key]
                    else:
                        rel_uri = self._get_safe_uri(f"{rel} {prop_id}", ns)
                        annotation_property_list[sub_list_key] = rel_uri
                        prop_id += 1

                    g.add((ns[rel], RDF.type, OWL.AnnotationProperty))
                    g.add((rel_uri, RDF.type, ns[rel]))

                    for sub_prop_key, sub_prop_value in sub_property_list.items():
                        if sub_prop_key == 'rel': continue
                        g.add((rel_uri, ns[sub_prop_key], sub_prop_value))

                    # p_uri = self._get_safe_uri(rel, ns)
                    # all_properties.add((rel_uri, OWL.ObjectProperty))

                    s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
                    g.add((s, rel_uri, t))

        with self.conn.cursor(name='properties') as cursor:
            cursor.execute("SELECT concept_id, prop_type FROM properties")
            for concept_id, prop_type in tqdm(cursor, desc="Processing Properties"):
                if concept_id in id_to_uri:
                    concept_id = id_to_uri[concept_id]
                    g.add((concept_id, ns[prop_type], Literal(True)))
                    all_properties.add((ns[prop_type], OWL.DatatypeProperty))

        for prop_uri, prop_type in tqdm(all_properties, desc="Adding Properties"):
            g.add((prop_uri, RDF.type, prop_type))

        try:
            logging.info("Serialising ontology")
            g.serialize(destination=file_path, format="ttl")
            logging.info(f"Successfully saved ontology to {file_path}")
        except Exception as e:
            logging.error(f"Failed to write Turtle file: {e}")

    def create_classes(self, g: Graph, ns: Namespace):
        logging.info("Creating ontology classes")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = ns[parent]
            g.add((parent_uri, RDF.type, OWL.Class))
            self.create_sub_classes(parent, children, g, ns, parent_uri)

    def create_sub_classes(self, parent, children, g: Graph, ns: Namespace, parent_uri: URIRef):
        ignored_keys = ['classes', 'sameAs']
        for child in children:
            child_uri = ns[child]
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.subClassOf, parent_uri))
            if 'classes' in children[child]:
                self.create_sub_classes(parent, children[child]['classes'], g, ns, child_uri)
            elif child not in ignored_keys and len(children[child]) > 0:
                self.create_sub_classes(parent, children[child], g, ns, child_uri)

            if len(children[child]) > 0 and 'sameAs' in children[child]:
                self.equivalent_classes[child] = [child, children[child]['sameAs']]

    def close(self):
        if self.conn: self.conn.close(); logging.info("Database connection closed")

def main():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file 'config.yaml' not found")
        exit(1)

    builder = None
    try:
        builder = OntologyBuilder(config)
        builder.build()
        builder.dump_to_turtle(config['turtle_export']['output_file'])
    except (psycopg2.Error, ConnectionRefusedError) as e:
        print(f"\nA database error occurred: {e}")
    finally:
        if builder:
            builder.close()


if __name__ == '__main__':
    main()