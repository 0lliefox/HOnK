import logging
import os
import ssl
from collections import defaultdict

import nltk
from nltk import word_tokenize
from rdflib import Graph
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
                # Check if we are on a login node
                # or compute node
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
                g = self.pickle_manager.load(filepath)
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
                    self.pickle_manager.save(filepath, g)

                synset_data = self.get_synsets(g)
                self.get_components(g, synset_data)
                self.get_relationships(g, synset_data)

                logging.info("Finished loading WordNet data from file.")
            return synset_data
        except FileNotFoundError:
            logging.error(f"WordNet file not found at '{filepath}'")

    def store_data(self, synset_data):
        synset_item_to_db_id = {}

        def iterate_over_file(cursor=None):
            logging.info("Processing and inserting WordNet concepts...")
            for synset_uri, data in tqdm(synset_data.items(), desc="Inserting WordNet Concepts", disable=not self.verbose):
                if '#Component' in synset_uri:
                    component_split = synset_uri.split('#Component-')
                    component_index = int(component_split[1]) - 1
                    component_parts = component_split[0][:-2].split('/')[-1].split('+')
                    component_to_relate = component_parts[int(component_split[1]) - 1]
                    full_component_label = ' '.join(component_parts)
                    full_component_db_id = self.get_or_create_concept(full_component_label, 'Phrase', cursor)

                    tagged_phrase = nltk.pos_tag(word_tokenize(full_component_label))
                    if tagged_phrase[component_index][0].lower() == component_to_relate.lower():
                        nltk_pos = tagged_phrase[component_index][1]
                    else:
                        # Search for the word incase index doesn't match
                        for word, tag in tagged_phrase:
                            if word.lower() == component_to_relate.lower():
                                nltk_pos = tag
                                break

                    if 'classes' in self.graph_manager.pos_tag_mappings[nltk_pos]:
                        pos_classes = self.graph_manager.pos_tag_mappings[nltk_pos]['classes']
                    else:
                        pos_classes = [nltk_pos]

                    for pos_class in pos_classes:
                        component_to_relate_db_id = self.get_or_create_concept(component_to_relate, pos_class, cursor)

                        self.add_relation({
                            'id': full_component_db_id,
                            'term': full_component_label,
                            'pos': 'Phrase'
                        },
                            {
                                'id': component_to_relate_db_id,
                                'term': component_to_relate,
                                'pos': pos_class
                            },
                            'compositeFormWith', 1.0, cursor)
                        if isinstance(pos_classes, dict) and 'properties' in pos_classes[pos_class]:
                            for prop in pos_classes[pos_class]['properties']:
                                self.add_property(
                                    {
                                        'id': component_to_relate_db_id,
                                        'term': component_to_relate
                                    },
                                    prop, True, cursor)
                else:
                    if data['pos'] == 'phrase':
                        pos = 'Phrase'
                        if data['phrase_type']:
                            pos = self.get_mapped_pos(data['phrase_type'])
                    elif data['lexical_domain'] == '':
                        pos = self.get_mapped_pos(data['pos'])
                    else:
                        pos = self.get_mapped_pos(data['lexical_domain'])

                    if (pos == '' or pos.lower() == data['pos']) and data['lexical_domain'] != '':
                        lexical_pos, lexical_domain = data['lexical_domain'].split('.')
                        pos = self.get_mapped_pos(lexical_pos)
                        lexical_domain_db_id = self.get_or_create_concept(lexical_domain, pos, cursor)
                    else:
                        if data['phrase_type'] == '':
                            pos = data['pos']
                        lexical_domain = None

                    lemma_db_ids = [[self.get_or_create_concept(term, pos, cursor), term] for term, uri in
                                    data['lemmas'].items()]
                    for idx, db_info in enumerate(lemma_db_ids):
                        db_id, term = db_info
                        synset_item_to_db_id[synset_uri, list(data['lemmas'])[
                            idx]] = [db_id, term, pos]  # A synset_uri might have multiple db_ids (?)
                        self.add_url({'id': db_id, 'term': term, 'pos': pos}, data['lemmas'][list(data['lemmas'])[idx]],
                                     cursor)
                        # self.add_undirected(db_id, 'definition', data['definition'], cursor)

                        if lexical_domain:
                            self.add_relation(
                                {
                                    'id': db_id,
                                    'term': term,
                                    'pos': pos
                                },
                                {
                                    'id': lexical_domain_db_id,
                                    'term': lexical_domain,
                                    'pos': pos
                                },
                                'relatedTo', 1.0, cursor)

                    if len(lemma_db_ids) > 1:
                        for i in range(len(lemma_db_ids)):
                            for j in range(i + 1, len(lemma_db_ids)):
                                self.add_relation({
                                    'id': lemma_db_ids[i][0],
                                    'term': lemma_db_ids[i][1],
                                    'pos': pos
                                },
                                    {
                                        'id': lemma_db_ids[j][0],
                                        'term': lemma_db_ids[j][1],
                                        'pos': pos
                                    }, 'eq', 1.0, cursor)

            synset_uri_to_db_ids = defaultdict(list)
            for (synset_uri, lemma), db_id in synset_item_to_db_id.items():
                synset_uri_to_db_ids[synset_uri].append(db_id)

            logging.info("Adding mapped semantic relationships...")
            for synset_uri, data in tqdm(synset_data.items(), desc="Adding WordNet Relations", disable=not self.verbose):
                for lemma, uri in data['lemmas'].items():
                    l_key = synset_uri, lemma
                    if l_key not in synset_item_to_db_id: continue
                    for rel_fragment, related_uri in data['relations']:
                        related_db_ids = synset_uri_to_db_ids.get(related_uri, [])
                        if len(related_db_ids) == 0: continue

                        mapping = self.mappings.get(rel_fragment.lower())
                        if not mapping: continue

                        rel, negated, swap = mapping.get('rel'), mapping.get('isNegated', False), mapping.get('swap',
                                                                                                              False)
                        if not rel: continue

                        for related_id in related_db_ids:
                            start_id, end_id = synset_item_to_db_id[l_key], related_id

                            if swap:
                                self.add_relation(
                                    {
                                        'id': end_id[0],
                                        'term': end_id[1],
                                        'pos': end_id[2]
                                    },
                                    {
                                        'id': start_id[0],
                                        'term': start_id[1],
                                        'pos': start_id[2]
                                    },
                                    rel, 1.0, cursor)
                            else:
                                self.add_relation(
                                    {
                                        'id': start_id[0],
                                        'term': start_id[1],
                                        'pos': start_id[2]
                                    },
                                    {
                                        'id': end_id[0],
                                        'term': end_id[1],
                                        'pos': end_id[2]
                                    },
                                    rel, 1.0, cursor)

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
        component_results = g.query(component_query)
        logging.info(f"Found {len(component_results)} Component rows.")

        for row in tqdm(component_results, desc="Aggregating Component Data", disable=not self.verbose):
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
        for row in tqdm(core_results, desc="Aggregating Core Data", disable=not self.verbose):
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
