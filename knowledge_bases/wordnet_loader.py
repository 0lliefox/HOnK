import logging
import re
import os
import pickle
from tqdm import tqdm
from rdflib import Graph
# from nltk.corpus import wordnet as wn

from .abstract_loader import AbstractLoader


class WordNetLoader(AbstractLoader):
    def load_data(self):
        filepath = self.config['local_files']['wordnet']
        try:
            synset_data = self.load_from_pickle('synset_data')
            if not synset_data:
                logging.info(f"Loading WordNet data from file: '{filepath}'...")
                g = self.load_from_pickle(filepath)
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
                    self.save_to_pickle(filepath, g)

                core_data_query = """
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX wnp: <http://wordnet-rdf.princeton.edu/ontology#>
    
                    SELECT ?synset ?lemma ?definition ?pos
                    WHERE {
                        ?synset rdf:type wnp:Synset .
                        ?synset rdfs:label ?lemma . FILTER(langMatches(lang(?lemma), "eng")) .
                        ?synset wnp:gloss ?definition . FILTER(langMatches(lang(?definition), "eng")) .
                        ?synset wnp:part_of_speech ?pos_uri .
                        BIND(STRAFTER(STR(?pos_uri), STR(wnp:)) AS ?pos_fallback) .
                        
                        OPTIONAL {
                            ?synset wnp:lexical_domain ?lexical_domain .
                            BIND(STRAFTER(STR(?lexical_domain), STR(wnp:)) AS ?lexical_domain) .
                        }
                
                        BIND(IF(BOUND(?lexical_domain) && ?lexical_domain != 'unlabeled', ?lexical_domain, ?pos_fallback) AS ?pos) .
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
                            'lemmas': set(),
                            'definition': str(row.definition),
                            'pos': str(row.pos),
                            'lexical_domain': str(row.lexical_domain),
                            'relations': set()
                        }
                    synset_data[synset_uri]['lemmas'].add(str(row.lemma))

                logging.info("Querying for WordNet relationships using targeted queries...")
                relations_to_query = list(self.mappings.keys())
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

                self.save_to_pickle('synset_data', synset_data)

            synset_to_primary_db_id = {}
            with self.conn.cursor() as cursor:
                logging.info("Processing and inserting WordNet concepts...")
                for synset_uri, data in tqdm(synset_data.items(), desc="Inserting WordNet Concepts"):
                    pos = self._get_mapped_pos(data['pos'])

                    if (pos == '' or pos == data['pos']) and len(data['pos'].split('.')) > 1:
                        lexical_pos, lexical_domain = data['pos'].split('.')
                        pos = lexical_pos.capitalize()
                        lexical_domain_db_id = self.get_or_create_concept(lexical_domain, pos, "WordNet", cursor)
                    else:
                        pos = data['pos']
                        lexical_domain = None

                    lemma_db_ids = [self.get_or_create_concept(term, pos, "WordNet", cursor) for term in data['lemmas']]
                    lemma_db_ids = [db_id for db_id in lemma_db_ids if db_id is not None]

                    if not lemma_db_ids: continue
                    synset_to_primary_db_id[synset_uri] = lemma_db_ids[0]
                    for db_id in lemma_db_ids:
                        self.add_undirected(db_id, 'definition', data['definition'], 'WordNet', cursor)
                        # cursor.execute("UPDATE definitions SET wordnet_definition = %s WHERE id = %s", (data['definition'], db_id))

                        if lexical_domain:
                            self.add_relation(db_id, lexical_domain_db_id, 'relatedTo', 1.0, 'WordNet', cursor)

                    if len(lemma_db_ids) > 1:
                        for i in range(len(lemma_db_ids)):
                            for j in range(i + 1, len(lemma_db_ids)):
                                self.add_relation(lemma_db_ids[i], lemma_db_ids[j], 'eq', 1.0, 'WordNet', cursor)

                logging.info("Adding mapped semantic relationships...")
                for synset_uri, data in tqdm(synset_data.items(), desc="Adding WordNet Relations"):
                    if synset_uri not in synset_to_primary_db_id: continue
                    for rel_fragment, related_uri in data['relations']:
                        if related_uri not in synset_to_primary_db_id: continue

                        mapping = self.mappings.get(rel_fragment.lower())
                        if not mapping: continue

                        rel, negated, swap = mapping.get('rel'), mapping.get('isNegated', False), mapping.get('swap', False)
                        if not rel: continue

                        start_id, end_id = synset_to_primary_db_id[synset_uri], synset_to_primary_db_id[related_uri]

                        if swap:
                            self.add_relation(end_id, start_id, rel, 1.0, 'WordNet', cursor)
                        else:
                            self.add_relation(start_id, end_id, rel, 1.0, 'WordNet', cursor)

                self.conn.commit()
            logging.info("Finished loading WordNet data from file.")
        except FileNotFoundError:
            logging.error(f"WordNet file not found at '{filepath}'")

