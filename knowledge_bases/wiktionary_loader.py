import itertools
import json
import logging
import re

from tqdm import tqdm

from .abstract_loader import AbstractLoader


class WiktionaryLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.source = "Wiktionary"
        self.edge_mappings = self.get_mappings(["edge"])

    def load_data(self):
        filepath = self.config['local_files']['wiktionary']
        logging.info(f"Loading Wiktionary data: '{filepath}'")

        entries = self.pickle_manager.load(filepath)
        if not entries:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            self.pickle_manager.save(filepath, entries)

        word_to_pos = {
            key: {self.get_mapped_pos(item['pos']) for item in group}
            for key, group in itertools.groupby(sorted(entries, key=lambda x: x['word']), key=lambda x: x['word'])
        }
        with self.conn.cursor() as cursor:
            for data in tqdm(entries, desc="Processing Wiktionary Entries"):
                term = data.get('word')
                lang = data.get('lang_code')

                if not term or not self.does_term_matches_language(lang):
                    continue

                found_poses = set()
                found_props = set()
                pos = data.get('pos')
                standardised_pos = self.get_mapped_pos(pos)
                searchable_classes = {}

                if standardised_pos in list(self.builder.lemma_mappings.keys()):
                    lemmas = self.builder.lemma_mappings[standardised_pos]
                    all_class_names = self.builder.get_all_keys(lemmas['classes'])
                    all_properties = lemmas['properties']

                    searchable_classes = {
                        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', class_name).lower(): class_name
                        for class_name in all_class_names
                    }

                    categories = data.get('categories', [])
                    self.extract_classes(categories, found_poses, searchable_classes)

                    tags = data.get('tags', [])
                    self.extract_properties(categories + tags, found_props, all_properties)
                else:
                    found_poses.add(standardised_pos)

                definitions = []
                senses = data.get('senses', [])
                for sense in senses:
                    # glosses = sense.get('glosses')
                    # if glosses and isinstance(glosses, list) and glosses[0]:
                    #     definitions.append(glosses[0])

                    # Search through glosses to find extra POS tags
                    if searchable_classes:
                        self.extract_classes(sense.get('categories', []), found_poses, searchable_classes)

                if len(found_poses) == 0:
                    found_poses.add(standardised_pos)

                if not found_poses:
                    logging.warning(f"No POS tags found for '{term}'")
                    continue

                for pos in found_poses:
                    main_concept_id = self.get_or_create_concept(term, pos, cursor)
                    if not main_concept_id and self.mode == 'db':
                        continue

                    # if definitions:
                    #     full_definition = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(definitions))
                    #     self.add_undirected(main_concept_id, 'definition', full_definition, cursor)

                    for class_type in list(self.edge_mappings.keys()):
                        for c_item in [s_obj.get('word') for s_obj in data.get(class_type, []) if s_obj.get('word')]:
                            if class_type in {'related', 'derived'}:
                                c_item_pos = word_to_pos.get(c_item, [])
                            else:
                                c_item_pos = [pos]

                            for c_pos in c_item_pos:
                                c_id = self.get_or_create_concept(c_item, c_pos, cursor)
                                if c_id:
                                    self.add_relation(
                                        {
                                            'id': main_concept_id,
                                            'term': term
                                        },
                                        {
                                            'id': c_id,
                                            'term': c_item
                                        },
                                        class_type, 1.0, cursor)

                    for prop in found_props:
                        self.add_property({'id': main_concept_id, 'term': term}, prop, True, cursor)

        self.conn.commit()
        logging.info("Finished loading Wiktionary data from file.")

    def extract_classes(self, categories, found_poses, searchable_classes):
        for category in categories:
            for search_term, original_class_name in searchable_classes.items():
                if self.match_class_name(search_term, category):
                    found_poses.add(original_class_name)

    def extract_properties(self, categories, found_props, searchable_properties):
        if searchable_properties:
            for category in categories:
                for search_prop in searchable_properties:
                    if self.match_class_name(search_prop, category):
                        found_props.add(search_prop)
        if 'participle' in found_props:
            if 'present' in found_props:
                found_props.add('present participle')
                found_props.remove('present')
                found_props.remove('participle')
            elif 'past' in found_props:
                found_props.add('past participle')
                found_props.remove('past')
                found_props.remove('participle')

    def match_class_name(self, search_term, category_string):
        safe_search_term = re.escape(search_term)
        pattern = r'\b' + safe_search_term
        return re.search(pattern, category_string, re.IGNORECASE) is not None
