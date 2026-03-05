import logging
import os
import ssl
from collections import defaultdict

import nltk
from nltk import word_tokenize
import pyoxigraph
from tqdm import tqdm

from .abstract_loader import AbstractLoader


class WordNetLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.source = "WordNet"
        self.mappings = self.get_mappings(["edge", "pos"])
        self.verbose = self.config['general']['verbose']
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk_packages = [
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng',
            'punkt_tab'
        ]
        for nltk_package in nltk_packages:
            try:
                if nltk_package == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                else:
                    nltk.data.find(f'taggers/{nltk_package}')
            except LookupError:
                if os.environ.get('SLURM_JOB_ID'):
                    logging.error(
                        f"NLTK resource '{nltk_package}' missing on compute node. "
                        "Please run 'python -c \"import nltk; nltk.download('" + nltk_package + "')\"' "
                                                                                                "on the login node before submitting."
                    )
                else:
                    logging.info(f"Downloading NLTK's '{nltk_package}'...")
                    nltk.download(nltk_package)

    def parse_data(self):
        filepath = self.config['local_files']['wordnet']
        try:
            synset_data = self.pickle_manager.load('synset_data')

            if not synset_data:
                logging.info(f"Loading WordNet data from file: '{filepath}'...")

                g = pyoxigraph.Store()
                logging.info("Parsing WordNet file (this may take a few seconds)...")

                file_extension = os.path.splitext(filepath)[1].lower()
                if file_extension in ['.ttl', '.turtle']:
                    rdf_format = pyoxigraph.RdfFormat.TURTLE
                elif file_extension == '.nt':
                    rdf_format = pyoxigraph.RdfFormat.N_TRIPLES
                else:
                    logging.warning(
                        f"Unknown WordNet file extension '{file_extension}'. Attempting to parse as Turtle.")
                    rdf_format = pyoxigraph.RdfFormat.TURTLE

                with open(filepath, 'rb') as f:
                    g.load(f, format=rdf_format)

                logging.info("Finished parsing WordNet file.")

                # Process the data
                synset_data = self.get_synsets(g)
                self.get_components(g, synset_data)
                self.get_relationships(g, synset_data)

                logging.info("Finished loading WordNet data from file.")
            return synset_data
        except FileNotFoundError:
            logging.error(f"WordNet file not found at '{filepath}'")

    def store_data(self, synset_data):
        if not synset_data:
            return

        def iterate_over_file(cursor=None):
            logging.info("Processing and inserting WordNet concepts...")

            # Map synset_uris directly to their normalised terms/poses (instead of DB IDs)
            synset_uri_to_term_pos = defaultdict(list)

            # --- PHASE 1: Concepts and Internal Relations ---
            for synset_uri, data in tqdm(synset_data.items(), desc="Inserting WordNet Concepts", disable=not self.verbose):
                if '#Component' in synset_uri:
                    component_split = synset_uri.split('#Component-')
                    component_index = int(component_split[1]) - 1
                    component_parts = component_split[0][:-2].split('/')[-1].split('+')
                    component_to_relate = component_parts[int(component_split[1]) - 1]
                    full_component_label = ' '.join(component_parts)

                    # Queue component phrase
                    norm_full_comp, norm_full_pos = self.queue_concept(full_component_label, 'Phrase')
                    synset_uri_to_term_pos[synset_uri].append((norm_full_comp, norm_full_pos))

                    tagged_phrase = nltk.pos_tag(word_tokenize(full_component_label))
                    if tagged_phrase[component_index][0].lower() == component_to_relate.lower():
                        nltk_pos = tagged_phrase[component_index][1]
                    else:
                        for word, tag in tagged_phrase:
                            if word.lower() == component_to_relate.lower():
                                nltk_pos = tag
                                break

                    pos_classes = self.graph_manager.pos_tag_mappings[nltk_pos]['classes'] if 'classes' in self.graph_manager.pos_tag_mappings.get(nltk_pos, {}) else [nltk_pos]

                    for pos_class in pos_classes:
                        norm_comp, norm_comp_pos = self.queue_concept(component_to_relate, pos_class)
                        self.queue_relation(norm_full_comp, norm_full_pos, norm_comp, norm_comp_pos, 'compositeFormWith', 1.0)

                        if isinstance(pos_classes, dict) and 'properties' in pos_classes.get(pos_class, {}):
                            for prop in pos_classes[pos_class]['properties']:
                                self.queue_property(norm_comp, norm_comp_pos, prop, True)
                else:
                    if data['pos'] == 'phrase':
                        pos = self.get_mapped_pos(data['phrase_type']) if data['phrase_type'] else 'Phrase'
                    elif data['lexical_domain'] == '':
                        pos = self.get_mapped_pos(data['pos'])
                    else:
                        pos = self.get_mapped_pos(data['lexical_domain'])

                    if (pos == '' or pos.lower() == data['pos']) and data['lexical_domain'] != '':
                        lexical_pos, lexical_domain = data['lexical_domain'].split('.')
                        pos = self.get_mapped_pos(lexical_pos)
                        norm_lex_domain, norm_lex_pos = self.queue_concept(lexical_domain, pos)
                    else:
                        lexical_domain = None

                    lemma_term_pos_list = []
                    for term, uri in data['lemmas'].items():
                        norm_term, norm_pos = self.queue_concept(term, pos)
                        lemma_term_pos_list.append((norm_term, norm_pos))
                        synset_uri_to_term_pos[synset_uri].append((norm_term, norm_pos))

                        self.queue_url(norm_term, norm_pos, data['lemmas'][term])

                        if lexical_domain:
                            self.queue_relation(norm_term, norm_pos, norm_lex_domain, norm_lex_pos, 'relatedTo', 1.0)

                    # Link synonymous lemmas
                    if len(lemma_term_pos_list) > 1:
                        for i in range(len(lemma_term_pos_list)):
                            for j in range(i + 1, len(lemma_term_pos_list)):
                                self.queue_relation(
                                    lemma_term_pos_list[i][0], lemma_term_pos_list[i][1],
                                    lemma_term_pos_list[j][0], lemma_term_pos_list[j][1],
                                    'eq', 1.0)

                # Flush Phase 1 batches
                if len(self.batch_concepts) >= self.batch_size:
                    self.flush_batch(cursor)

            self.flush_batch(cursor)  # Final flush for Phase 1

            # Cross-Synset Relationships
            logging.info("Adding mapped semantic relationships...")
            for synset_uri, data in tqdm(synset_data.items(), desc="Adding WordNet Relations", disable=not self.verbose):
                for rel_fragment, related_uri in data['relations']:
                    start_nodes = synset_uri_to_term_pos.get(synset_uri, [])
                    end_nodes = synset_uri_to_term_pos.get(related_uri, [])

                    if not start_nodes or not end_nodes: continue

                    mapping = self.mappings.get(rel_fragment.lower())
                    if not mapping: continue

                    rel, swap = mapping.get('rel'), mapping.get('swap', False)
                    if not rel: continue

                    for s_term, s_pos in start_nodes:
                        for e_term, e_pos in end_nodes:
                            # We must queue the concepts again here so `flush_batch`
                            # knows to fetch their DB IDs to construct the relations.
                            self.queue_concept(s_term, s_pos)
                            self.queue_concept(e_term, e_pos)

                            if swap:
                                self.queue_relation(e_term, e_pos, s_term, s_pos, rel, 1.0)
                            else:
                                self.queue_relation(s_term, s_pos, e_term, e_pos, rel, 1.0)

                # Flush Phase 2 batches
                if len(self.batch_concepts) >= self.batch_size:
                    self.flush_batch(cursor)

            self.flush_batch(cursor)  # Final flush for Phase 2

        if self.mode == 'db':
            with self.conn.cursor() as cursor:
                iterate_over_file(cursor)
                self.conn.commit()
        else:
            iterate_over_file()

    def get_relationships(self, g, synset_data):
        logging.info("Querying for WordNet relationships using targeted queries...")
        relations_to_query = list(self.mappings.keys())
        total_relations_found = 0

        for rel_fragment in tqdm(relations_to_query, desc="Querying Relation Types", disable=not self.verbose):
            relation_query = f"""
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX wnp: <http://wordnet-rdf.princeton.edu/ontology#>
                        SELECT ?synset ?related_synset
                        WHERE {{
                            ?synset wnp:{rel_fragment} ?related_synset .
                            ?synset rdf:type wnp:Synset .
                        }}
                    """
            try:
                relation_results = list(g.query(relation_query))
                total_relations_found += len(relation_results)
                for row in relation_results:
                    synset_uri = row['synset'].value
                    if synset_uri in synset_data:
                        related_synset_uri = row['related_synset'].value
                        synset_data[synset_uri]['relations'].add((rel_fragment, related_synset_uri))
            except Exception as e:
                logging.warning(f"Could not query for relation '{rel_fragment}': {e}")

        logging.info(f"Found {total_relations_found} total relationship rows across all types.")
        logging.info(f"Aggregated data for {len(synset_data)} unique synsets.")

        self.pickle_manager.save('synset_data', synset_data)

    def get_components(self, g, synset_data):
        component_query = """
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX lemon: <http://lemon-model.net/lemon#>

                    SELECT ?component ?lemma
                    WHERE {
                        ?component rdf:type lemon:Component .
                        ?component rdfs:label ?lemma .
                    }
                """
        logging.info("Querying for WordNet Components...")
        component_results = list(g.query(component_query))
        logging.info(f"Found {len(component_results)} Component rows.")

        for row in tqdm(component_results, desc="Aggregating Component Data", disable=not self.verbose):
            component_uri = row['component'].value
            if component_uri not in synset_data:
                synset_data[component_uri] = {
                    'lemmas': dict(),
                    'pos': "",
                    'lexical_domain': "",
                    'phrase_type': "",
                    'relations': set()
                }
            synset_data[component_uri]['lemmas'][row['lemma'].value] = component_uri

    def get_synsets(self, g):
        core_data_query = """
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX wnp: <http://wordnet-rdf.princeton.edu/ontology#>

                    SELECT ?synset ?lemma ?pos ?lexical_domain ?synset_member ?phrase_type
                    WHERE {
                        ?synset rdf:type wnp:Synset .

                        ?synset rdfs:label ?lemma . 

                        ?synset wnp:part_of_speech ?pos_uri .
                        ?synset wnp:synset_member ?synset_member .

                        BIND(STRAFTER(STR(?pos_uri), "http://wordnet-rdf.princeton.edu/ontology#") AS ?pos)

                        OPTIONAL {
                            ?synset wnp:lexical_domain ?raw_lexical_domain .
                            BIND(STRAFTER(STR(?raw_lexical_domain), "http://wordnet-rdf.princeton.edu/ontology#") AS ?lexical_domain)
                        }

                        OPTIONAL {
                            ?synset wnp:phrase_type ?raw_phrase_type .
                            BIND(STRAFTER(STR(?raw_phrase_type), "http://wordnet-rdf.princeton.edu/ontology#") AS ?phrase_type)
                        }
                    }
                """
        logging.info("Querying for core WordNet concepts...")
        core_results = list(g.query(core_data_query))
        logging.info(f"Found {len(core_results)} core concept rows.")

        synset_data = {}
        for row in tqdm(core_results, desc="Aggregating Core Data", disable=not self.verbose):
            synset_uri = row['synset'].value
            lemma_val = row['lemma'].value
            synset_member_val = row['synset_member'].value

            if synset_uri not in synset_data:
                synset_data[synset_uri] = {
                    'lemmas': dict(),
                    'pos': row['pos'].value,
                    'lexical_domain': row['lexical_domain'].value if row['lexical_domain'] is not None else '',
                    'phrase_type': row['phrase_type'].value if row['phrase_type'] is not None else '',
                    'relations': set()
                }

            if lemma_val.replace(" ", "+") == synset_member_val.split('/')[-1][:-2]:
                synset_data[synset_uri]['lemmas'][lemma_val] = synset_member_val
            else:
                if len(synset_data[synset_uri]['lemmas']) == 0:
                    synset_data[synset_uri]['lemmas'][lemma_val] = synset_uri

        return synset_data