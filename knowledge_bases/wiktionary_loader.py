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
        self.cached_lemma_data = {} # Cache for pre-compiled regex patterns and class names
        self.verbose = self.config['general']['verbose']

    def parse_data(self):
        filepath = self.config['local_files']['wiktionary']
        logging.info(f"Loading Wiktionary data: '{filepath}'")

        entries = self.pickle_manager.load(filepath)
        if not entries:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            self.pickle_manager.save(filepath, entries)

        return entries

    def store_data(self, entries):
        word_to_pos = {
            key: {self.get_mapped_pos(item['pos']) for item in group}
            for key, group in itertools.groupby(sorted(entries, key=lambda x: x['word']), key=lambda x: x['word'])
        }

        def iterate_over_file(cursor=None):
            for data in tqdm(entries, desc="Processing Wiktionary Entries", disable=not self.verbose):
                term = data.get('word')
                lang = data.get('lang_code')

                if not term or not self.does_term_matches_language(lang):
                    continue

                found_poses = set()
                found_props = set()
                pos = data.get('pos')
                
                # Use normalise_data to get standardised POS, but we only need the POS here
                _, standardised_pos = self.normalise_data(term, pos)

                current_lemma_data = self.cached_lemma_data.get(standardised_pos)

                if not current_lemma_data:
                    if standardised_pos in self.graph_manager.lemma_mappings:
                        lemmas = self.graph_manager.lemma_mappings[standardised_pos]
                        all_class_names = self.builder.get_all_keys(lemmas['classes'])
                        all_properties = lemmas['properties']

                        searchable_classes = []
                        for class_name in all_class_names:
                            search_term = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', class_name).lower()
                            pattern = r'\b' + re.escape(search_term)
                            searchable_classes.append((re.compile(pattern, re.IGNORECASE), class_name))

                        searchable_properties = []
                        for prop_name in all_properties:
                            pattern = r'\b' + re.escape(prop_name)
                            searchable_properties.append(re.compile(pattern, re.IGNORECASE))

                        current_lemma_data = {
                            'searchable_classes': searchable_classes,
                            'searchable_properties': searchable_properties,
                            'all_properties': all_properties  # Keep original for post-processing
                        }
                        self.cached_lemma_data[standardised_pos] = current_lemma_data
                    else:
                        # If standardised_pos is not in lemma_mappings, we still need to cache this fact
                        current_lemma_data = {
                            'searchable_classes': [],
                            'searchable_properties': [],
                            'all_properties': []
                        }
                        self.cached_lemma_data[standardised_pos] = current_lemma_data

                searchable_classes = current_lemma_data['searchable_classes']
                searchable_properties = current_lemma_data['searchable_properties']
                all_properties = current_lemma_data['all_properties']

                if searchable_classes:  # Only extract if there are classes to search for
                    categories = data.get('categories', [])
                    self.extract_classes(categories, found_poses, searchable_classes)

                    tags = data.get('tags', [])
                    self.extract_properties(categories + tags, found_props, searchable_properties, all_properties)
                else:
                    found_poses.add(standardised_pos)

                senses = data.get('senses', [])
                for sense in senses:

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

                    for class_type in list(self.edge_mappings.keys()):
                        for s_obj in data.get(class_type, []):
                            c_item = s_obj.get('word')
                            if not c_item:
                                continue

                            sense = s_obj.get('sense', '')
                            rel_type = 'neqTo' if 'unrelated' in sense.lower() else class_type

                            c_item_pos = self.resolve_target_pos(c_item, pos, class_type, word_to_pos)

                            for c_pos in c_item_pos:
                                c_id = self.get_or_create_concept(c_item, c_pos, cursor)
                                if (c_id and self.mode == 'db') or self.mode == 'graph':
                                    self.add_relation(
                                        {
                                            'id': main_concept_id,
                                            'term': term,
                                            'pos': pos
                                        },
                                        {
                                            'id': c_id,
                                            'term': c_item,
                                            'pos': c_pos
                                        },
                                        rel_type, 1.0, cursor)

                    for prop in found_props:
                        self.add_property({'id': main_concept_id, 'term': term}, prop, True, cursor)

        if self.mode == 'db':
            with self.conn.cursor() as cursor:
                iterate_over_file(cursor)

                self.conn.commit()
        else:
            iterate_over_file()

        logging.info("Finished loading Wiktionary data from file.")

    def resolve_target_pos(self, c_item, source_pos, class_type, word_to_pos):
        """Part(s) of speech to use for a relation *target* word.

        Previously every relation except ``related``/``derived`` forced the
        *source* word's POS onto the target, producing wrong-POS edges and
        phantom nodes (e.g. ``nonsense`` (noun) --synonym--> ``unreasoning``
        typed as a noun though it is only an adjective).

        - ``related``/``derived``: use the target's own POS(es); derivation is
          routinely cross-POS (``happy`` -> ``happiness``). Unknown target: drop.
        - other relations (synonym, antonym, hypernym, meronym, troponym, ...):
          POS-preserving when the target actually has the source POS (the common
          case, and avoids fanning out to every sense of a multi-POS target);
          otherwise use the target's real POS(es); if the target is unknown, fall
          back to the source POS as a best effort rather than dropping the edge.
        """
        target_poses = word_to_pos.get(c_item)
        if class_type in {'related', 'derived'}:
            return list(target_poses) if target_poses else []
        if target_poses and source_pos in target_poses:
            return [source_pos]
        if target_poses:
            return list(target_poses)
        return [source_pos]

    def extract_classes(self, categories, found_poses, searchable_classes):
        for category in categories:
            for pattern, original_class_name in searchable_classes:
                if pattern.search(category):
                    found_poses.add(original_class_name)

    def extract_properties(self, categories, found_props, searchable_properties, all_properties):
        if searchable_properties:
            for category in categories:
                for idx, pattern in enumerate(searchable_properties):
                    if pattern.search(category):
                        found_props.add(all_properties[idx])

        if 'participle' in found_props:
            if 'present' in found_props:
                found_props.add('present participle')
                found_props.remove('present')
                found_props.remove('participle')
            elif 'past' in found_props:
                found_props.add('past participle')
                found_props.remove('past')
                found_props.remove('participle')
