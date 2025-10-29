import logging
import os
import ssl
from collections import defaultdict

import nltk
from nltk import word_tokenize
from rdflib import Graph
from tqdm import tqdm

from tools.pickling import load_from_pickle, save_to_pickle
from .abstract_loader import AbstractLoader


class WordNetLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk_packages = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
        for nltk_package in nltk_packages:
            try:
                nltk.data.find(nltk_package)
            except LookupError:
                logging.info(f"Downloading NLTK's '{nltk_package}'...")
                nltk.download(nltk_package)

    def _load_data_implementation(self):
        filepath = self.config['local_files']['wordnet']
        try:
            synset_data = load_from_pickle('synset_data')
            if not synset_data:
                logging.info(f"Loading WordNet data from file: '{filepath}'...")
                g = load_from_pickle(filepath)
                if not g:
                    g = Graph()
                    logging.info("Parsing WordNet file (this may take a few minutes)...")

                    file_extension = os.path.splitext(filepath)[1].lower()
                    if file_extension == '.ttl':
                        file_format = 'turtle'
                    elif file_extension == '.nt':
                        file_format = 'nt'
                    else:
                        logging.warning(
                            f"Unknown WordNet file extension '{file_extension}'. Attempting to parse as Turtle.")
                        file_format = 'turtle'

                    g.parse(filepath, format=file_format)

                    logging.info(f"Finished parsing WordNet file. Found {len(g)} triples.")
                    save_to_pickle(filepath, g)

                synset_data = self.get_synsets(g)
                self.get_components(g, synset_data)
                self.get_relationships(g, synset_data)

            synset_item_to_db_id = {}
            with self.conn.cursor() as cursor:
                logging.info("Processing and inserting WordNet concepts...")
                for synset_uri, data in tqdm(synset_data.items(), desc="Inserting WordNet Concepts"):
                    if '#Component' in synset_uri:
                        component_split = synset_uri.split('#Component-')
                        component_index = int(component_split[1]) - 1
                        component_parts = component_split[0][:-2].split('/')[-1].split('+')
                        component_to_relate = component_parts[int(component_split[1]) - 1]
                        full_component_label = ' '.join(component_parts)
                        full_component_db_id = self.get_or_create_concept(full_component_label, 'Phrase', "WordNet", cursor)

                        tagged_phrase = nltk.pos_tag(word_tokenize(full_component_label))
                        if tagged_phrase[component_index][0].lower() == component_to_relate.lower():
                            nltk_pos = tagged_phrase[component_index][1]
                        else:
                            # Search for the word incase index doesn't match
                            for word, tag in tagged_phrase:
                                if word.lower() == component_to_relate.lower():
                                    nltk_pos = tag
                                    break

                        if 'classes' in self.builder.pos_tag_mappings[nltk_pos]:
                            pos_classes = self.builder.pos_tag_mappings[nltk_pos]['classes']
                        else:
                            pos_classes = [nltk_pos]

                        for pos_class in pos_classes:
                            component_to_relate_db_id = self.get_or_create_concept(component_to_relate, pos_class, "WordNet", cursor)
                            self.add_relation(full_component_db_id, component_to_relate_db_id, 'compositeFormWith', 1.0, 'WordNet', cursor)
                            if isinstance(pos_classes, dict) and 'properties' in pos_classes[pos_class]:
                                for prop in pos_classes[pos_class]['properties']:
                                    self.add_property(component_to_relate_db_id, prop, True, "WordNet", cursor)
                    else:
                        if data['pos'] == 'phrase':
                            pos = 'Phrase'
                            if data['phrase_type']:
                                pos = self._get_mapped_pos(data['phrase_type'])
                        elif data['lexical_domain'] == '':
                            pos = self._get_mapped_pos(data['pos'])
                        else:
                            pos = self._get_mapped_pos(data['lexical_domain'])

                        if (pos == '' or pos.lower() == data['pos']) and data['lexical_domain'] != '':
                            lexical_pos, lexical_domain = data['lexical_domain'].split('.')
                            pos = self._get_mapped_pos(lexical_pos)
                            lexical_domain_db_id = self.get_or_create_concept(lexical_domain, pos, "WordNet", cursor)
                        else:
                            if data['phrase_type'] == '':
                                pos = data['pos']
                            lexical_domain = None

                        lemma_db_ids = [self.get_or_create_concept(term, pos, "WordNet", cursor) for term, uri in data['lemmas'].items()]
                        for idx, db_id in enumerate(lemma_db_ids):
                            synset_item_to_db_id[synset_uri, list(data['lemmas'])[idx]] = db_id  # A synset_uri might have multiple db_ids (?)
                            self.add_url(db_id, data['lemmas'][list(data['lemmas'])[idx]], 'WordNet', cursor)
                            # self.add_undirected(db_id, 'definition', data['definition'], 'WordNet', cursor)

                            if lexical_domain:
                                self.add_relation(db_id, lexical_domain_db_id, 'relatedTo', 1.0, 'WordNet', cursor)

                        if len(lemma_db_ids) > 1:
                            for i in range(len(lemma_db_ids)):
                                for j in range(i + 1, len(lemma_db_ids)):
                                    self.add_relation(lemma_db_ids[i], lemma_db_ids[j], 'eq', 1.0, 'WordNet', cursor)

                synset_uri_to_db_ids = defaultdict(list)
                for (synset_uri, lemma), db_id in synset_item_to_db_id.items():
                    synset_uri_to_db_ids[synset_uri].append(db_id)

                logging.info("Adding mapped semantic relationships...")
                for synset_uri, data in tqdm(synset_data.items(), desc="Adding WordNet Relations"):
                    for lemma, uri in data['lemmas'].items():
                        l_key = synset_uri, lemma
                        if l_key not in synset_item_to_db_id: continue
                        for rel_fragment, related_uri in data['relations']:
                            related_db_ids = synset_uri_to_db_ids.get(related_uri, [])
                            if len(related_db_ids) == 0: continue

                            mapping = self.mappings.get(rel_fragment.lower())
                            if not mapping: continue

                            rel, negated, swap = mapping.get('rel'), mapping.get('isNegated', False), mapping.get('swap', False)
                            if not rel: continue

                            for related_id in related_db_ids:
                                start_id, end_id = synset_item_to_db_id[l_key], related_id

                                if swap:
                                    self.add_relation(end_id, start_id, rel, 1.0, 'WordNet', cursor)
                                else:
                                    self.add_relation(start_id, end_id, rel, 1.0, 'WordNet', cursor)

                self.conn.commit()
            logging.info("Finished loading WordNet data from file.")
        except FileNotFoundError:
            logging.error(f"WordNet file not found at '{filepath}'")

    def get_relationships(self, g, synset_data):
        logging.info("Querying for WordNet relationships using targeted queries...")
        relations_to_query = list(self.mappings.keys())  # TODO: Restrict to just WordNet mappings
        total_relations_found = 0

        for rel_fragment in tqdm(relations_to_query, desc="Querying Relation Types"):
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
                relation_results = g.query(relation_query)
                total_relations_found += len(relation_results)
                for row in relation_results:
                    synset_uri = str(row.synset)
                    if synset_uri in synset_data:
                        related_synset_uri = str(row.related_synset)
                        synset_data[synset_uri]['relations'].add((rel_fragment, related_synset_uri))
            except Exception as e:
                logging.warning(f"Could not query for relation '{rel_fragment}': {e}")

        logging.info(f"Found {total_relations_found} total relationship rows across all types.")
        logging.info(f"Aggregated data for {len(synset_data)} unique synsets.")

        save_to_pickle('synset_data', synset_data)

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
        component_results = g.query(component_query)
        logging.info(f"Found {len(component_results)} Component rows.")

        for row in tqdm(component_results, desc="Aggregating Component Data"):
            component_uri = str(row.component)
            if component_uri not in synset_data:
                synset_data[component_uri] = {
                    'lemmas': dict(),
                    'definition': "",
                    'pos': "",
                    'lexical_domain': "",
                    'phrase_type': "",
                    'relations': set()
                }
            synset_data[component_uri]['lemmas'][str(row.lemma)] = component_uri

    def get_synsets(self, g):
        core_data_query = """
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX wnp: <http://wordnet-rdf.princeton.edu/ontology#>

                    SELECT ?synset ?lemma ?definition ?pos ?lexical_domain ?synset_member ?phrase_type
                    WHERE {
                        ?synset rdf:type wnp:Synset .
                        ?synset rdfs:label ?lemma . FILTER(langMatches(lang(?lemma), "eng")) .
                        ?synset wnp:gloss ?definition . FILTER(langMatches(lang(?definition), "eng")) .
                        ?synset wnp:part_of_speech ?pos_uri .
                        ?synset wnp:synset_member ?synset_member .
                        BIND(STRAFTER(STR(?pos_uri), STR(wnp:)) AS ?pos) .

                        OPTIONAL {
                            ?synset wnp:lexical_domain ?lexical_domain .
                            BIND(STRAFTER(STR(?lexical_domain), STR(wnp:)) AS ?lexical_domain) .
                        }
                        
                        OPTIONAL {
                            ?synset wnp:phrase_type ?phrase_type .
                            BIND(STRAFTER(STR(?phrase_type), STR(wnp:)) AS ?phrase_type) .
                        }

                        # BIND(IF(BOUND(?lexical_domain) && ?lexical_domain != 'unlabeled', ?lexical_domain, ?pos_fallback) AS ?pos) .
                    }
                """
        logging.info("Querying for core WordNet concepts...")
        core_results = g.query(core_data_query)
        logging.info(f"Found {len(core_results)} core concept rows.")

        synset_data = {}
        for row in tqdm(core_results, desc="Aggregating Core Data"):
            synset_uri = str(row.synset)
            if synset_uri not in synset_data:
                synset_data[synset_uri] = {
                    'lemmas': dict(),
                    'definition': str(row.definition),
                    'pos': str(row.pos),
                    'lexical_domain': str(row.get('lexical_domain', '')),
                    'phrase_type': str(row.get('phrase_type', '')),
                    'relations': set()
                }
            if str(row.lemma).replace(" ", "+") == str(row.synset_member).split('/')[-1][:-2]:
                synset_data[synset_uri]['lemmas'][str(row.lemma)] = str(row.synset_member)
            else:
                if len(synset_data[synset_uri]['lemmas']) == 0:
                    synset_data[synset_uri]['lemmas'][str(row.lemma)] = synset_uri
        return synset_data