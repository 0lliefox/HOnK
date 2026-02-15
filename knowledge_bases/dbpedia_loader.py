import csv
import logging
import os
import re
from functools import lru_cache

from tqdm import tqdm
from rdflib import Graph

from .abstract_loader import AbstractLoader
from collections import defaultdict

class DBpediaLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.lang = self.config['general']['language']

        self.labels_file = self.config['local_files']['dbpedia_labels']
        self.types_file = self.config['local_files']['dbpedia_types']
        self.properties_file = self.config['local_files']['dbpedia_properties']
        with open(self.config['local_files']['dbpedia_ignored_uris'], 'r') as f:
            self.ignored_uris = f.readlines()
        self.wikidata_property_labels = self.config['local_files']['wikidata_property_labels']
        self.wikidata_labels_file = self.config['local_files']['wikidata_labels']

        self.wikidata_prefix = "http://www.wikidata.org/entity/"
        self.resource_prefix = "http://dbpedia.org/resource/"

        with open(self.config['local_files']['dbpedia_ignored_uris'], 'r') as f:
            self.allowed_ids = f.readlines()


    def parse_data(self):
        wikidata_id_to_label = self.get_wikidata_ids()
        concept_data = self.get_concepts()
        concept_types = self.get_types(concept_data)
        # concept_data = self.get_properties(concept_data)
        return wikidata_id_to_label, concept_data, concept_types

    def store_data(self, data):
        wikidata_id_to_label, concept_data, concept_types = data
        self.insert_concepts_to_db(concept_data, concept_types, wikidata_id_to_label)
        logging.info("Finished loading DBpedia")

    def get_wikidata_ids(self):
        wikidata_label_pattern = re.compile(
            r'^\s*<(' + re.escape(self.wikidata_prefix) + r'Q[0-9]+)>\s+' +  # Subject (Wikidata QID)
            r'<http://www.w3.org/2000/01/rdf-schema#label>\s+' +  # Predicate (rdfs:label)
            r'"((?:[^"\\]|\\.)*)"@' + self.lang + r'\s*\.\s*$'  # Object (literal label) + lang
        )

        # Stream parse Wikidata labels file
        wikidata_id_to_label = self.pickle_manager.load('wikidata_labels.pkl')
        if not wikidata_id_to_label:
            wikidata_id_to_label = {}
            logging.info(f"Streaming Wikidata labels file to build lookup map...")
            with open(self.wikidata_labels_file, 'r', encoding='utf-8') as f_wd:
                for line in tqdm(f_wd, desc="Streaming Wikidata Labels", unit=" lines"):
                    match = wikidata_label_pattern.match(line)
                    if match:
                        wd_uri = match.group(1)
                        wd_label = match.group(2).replace('\\"', '"').replace('\\\\', '\\')
                        wikidata_id_to_label[wd_uri] = wd_label
            logging.info(f"Built lookup map with {len(wikidata_id_to_label)} Wikidata labels.")
            self.pickle_manager.save('wikidata_labels.pkl', wikidata_id_to_label)
        return wikidata_id_to_label

    def get_concepts(self):
        g = self.pickle_manager.load('dbpedia_labels_graph.pkl')
        if not g:
            g = Graph()
            logging.info(f"Parsing DBpedia labels file: '{self.labels_file}' (may take time)...")
            g.parse(self.labels_file, format='turtle')
            logging.info(f"Finished parsing DBpedia labels. Total triples: {len(g)}")
            self.pickle_manager.save('dbpedia_labels_graph.pkl', g)

        # Extract labels
        concept_data = self.pickle_manager.load('dbpedia_concepts.pkl')
        if not concept_data:
            query = f"""
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
                    SELECT ?concept ?label
                    WHERE {{
                        ?concept rdfs:label ?label .
                        FILTER(langMatches(lang(?label), "{self.lang}")) .
                        FILTER(STRSTARTS(STR(?concept), "{self.resource_prefix}")) .
                    }}
                """
            logging.info("Querying DBpedia graph for labels...")
            results = g.query(query)
            logging.info(f"Found {len(results)} DBpedia concepts")

            concept_data = {}
            for row in tqdm(results, desc="Aggregating DBpedia Concepts"):
                concept_uri = str(row.concept)
                concept_data[concept_uri] = {
                    'label': str(row.label),
                    'properties': {}
                }
            self.pickle_manager.save('dbpedia_concepts.pkl', concept_data)
        return concept_data

    def get_types(self, concept_data):
        type_pattern = re.compile(
            r'^\s*<(' + re.escape(self.resource_prefix) + r'[^>]+)>\s+' +
            r'<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\s+' +
            r'<([^>]+)>\s*\.\s*$'
        )

        # Types file must be streamed due to large file size cannot load graph into memory
        concept_types = self.pickle_manager.load('dbpedia_types.pkl')
        if not concept_types:
            concept_types = defaultdict(set)  # Store {concept_uri: {type_uri1, type_uri2,...}}
            logging.info(f"Streaming DBpedia types file...")
            with open(self.types_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Streaming DBpedia Types", unit=" lines"):
                    match = type_pattern.match(line)
                    if match:
                        concept_uri = match.group(1)
                        type_uri = match.group(2)
                        if concept_uri in concept_data:  # Only store types for concepts that have labels
                            if type_uri not in self.ignored_uris:
                                concept_types[concept_uri].add(type_uri)
            logging.info(f"Extracted types for {len(concept_types)} concepts.")
            self.pickle_manager.save('dbpedia_types.pkl', concept_types)
        return concept_types

    def get_properties(self, concept_data):
        # Properties
        properties_map = self.pickle_manager.load('wikidata_props.pkl')
        if not properties_map:
            properties_map = {}
            with open(self.wikidata_property_labels, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                for line in tqdm(reader, desc="Processing Wikidata Properties"):
                    p_id, label, _, _, _ = line  # ID, label, description, Datatype[1], Counts[2]
                    if p_id in self.allowed_ids:
                        properties_map[f"http://www.wikidata.org/entity/{p_id}"] = label

        logging.info(f"Streaming DBpedia properties file...")
        with open(self.properties_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Streaming DBpedia properties", unit=" lines"):
                concept_uri, property_uri, target_uri = [match[0] or match[1] for match in re.findall(r'<(.*?)>|"(.*?)"', line)][:3]
                if concept_uri in concept_data and property_uri in properties_map:  # Only store types for concepts that have labels
                    concept_data[concept_uri]['properties'][self.format_term(properties_map[property_uri])] = target_uri

        return concept_data

    def insert_concepts_to_db(self, concept_data, concept_types, wikidata_id_to_label):
        # Insert concepts
        with self.conn.cursor() as cursor:
            logging.info("Inserting concepts and type relationships...")
            for concept_uri, data in tqdm(concept_data.items(), desc="Inserting DBpedia Data"):
                term = data['label']
                pos = 'Concept'
                concept_id = self.get_or_create_concept(term, pos, 'DBpedia', cursor)
                if concept_id:
                    self.add_url(concept_id, concept_uri, cursor, pos)

                    # if 'properties' in data:
                    #     for prop, values in data['properties'].items():
                    #         for value in values:
                    #             if 'http' in value and value in concept_data:
                    #                 end_id = self.get_or_create_concept(concept_data[value], pos, 'DBpedia', cursor)
                    #             else:
                    #                 end_id = self.get_or_create_concept(value, pos, 'DBpedia', cursor)
                    #             self.add_relation(concept_id, end_id, prop, 1.0, 'DBpedia', cursor)

                    if concept_uri in concept_types:
                        for type_uri in concept_types[concept_uri]:
                            type_name = None
                            is_wikidata_uri = type_uri.startswith("http://www.wikidata.org/entity/")

                            if is_wikidata_uri:
                                type_name = wikidata_id_to_label.get(type_uri)
                                if not type_name:  # If we cannot get a type from the Wikidata ID, skip it (truthy db still doesn't contain *all* IDs...)
                                    continue

                            if not type_name:
                                type_name = type_uri.split('/')[-1]
                                if '#' in type_name:
                                    type_name = self.format_term(type_name.split('#')[-1])
                                else:
                                    type_name = self.format_term(type_name)

                            type_concept_id = self.get_or_create_concept(type_name, 'Concept', 'DBpedia', cursor)
                            if type_concept_id:
                                self.add_relation(concept_id, type_concept_id, 'isA', 1.0, 'DBpedia', cursor)

            self.conn.commit()

    @lru_cache(maxsize=1024)
    def format_term(self, term):
        term = ' '.join(re.split('(?<=.)(?=[A-Z])', term)) if "_" not in term else term.replace('_', ' ')
        return term.replace("Category:", "").replace('"', '').replace(" .", '')