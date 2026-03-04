import csv
import json
import logging

from tqdm import tqdm

from knowledge_bases.abstract_loader import AbstractLoader


class GeoNamesLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.source = "GeoNames"
        self.verbose = self.config['general']['verbose']

        # Check if GeoNames data exists in the database
        if self.mode == 'db':
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM concepts WHERE source = 'GeoNames' LIMIT 1;")
                self.geonames_data_exists = cursor.fetchone() is not None
        else:
            self.geonames_data_exists = False

        self.map_cache_filepath = f"{self.config['local_files']['cache']}/geonames_map.pkl"
        self.id_term_map = {}
        if self.mode == 'db' and self.geonames_data_exists:
             self.id_term_map = self.pickle_manager.load(self.map_cache_filepath)
        
        with open(self.config['local_files']['geonames_alternates'], 'r') as f:
            self.alternate_names = json.load(f)  # map of alternate ID to geoname ID
        with open(self.config['local_files']['geonames_ignore'], 'r') as f:
            self.ignore_names = f.readlines()
        with open(self.config['local_files']['geonames_feature_codes'], 'r') as f:
            self.feature_codes = json.load(f)
        with open(self.config['local_files']['geonames_links'], 'r') as f:
            self.links = json.load(f)

    def parse_data(self):
        filepath = self.config['local_files']['geonames']
        hierarchy_filepath = self.config['local_files']['geonames_hierarchy']
        with open(hierarchy_filepath, 'r') as f:
            hierarchy = json.load(f)

        needed_ids = set()
        if self.mode == 'graph' or not self.geonames_data_exists:
            self.id_term_map = {}

            if hierarchy:
                def collect_chain(start_id):
                    curr = str(start_id)
                    chain_seen = set()
                    while curr:
                        needed_ids.add(curr)
                        chain_seen.add(curr)
                        if curr in self.alternate_names:
                            curr = self.alternate_names[curr]
                            if curr in chain_seen:
                                break
                        else:
                            break
                for parent, children in tqdm(hierarchy.items(), desc="Analyzing hierarchy for required IDs"):
                    collect_chain(parent)
                    for child in children:
                        collect_chain(child)

            f = open(filepath, 'r')
            reader = csv.reader(f, delimiter='\t')
            return (f, reader, hierarchy, needed_ids)

        return (None, None, hierarchy, needed_ids)

    def store_data(self, data):
        if data is None:
            return

        file_handle, reader, hierarchy, needed_ids = data

        try:
            def iterate_over_file(cursor=None):
                if reader:
                    for line in tqdm(reader, desc="Processing GeoNames", disable=not self.verbose):
                        n_id, name, _, translations, _, _, feature_class, feature_code = line[:8]

                        if name in self.ignore_names:
                            continue

                        if feature_class == 'A':
                            pos = 'GPE'
                        else:
                            pos = 'LOC'

                        # Queue concept
                        norm_name, norm_pos = self.queue_concept(name, pos)

                        if n_id in needed_ids:
                            self.id_term_map[n_id] = [norm_name, norm_pos]

                        if n_id in self.links:
                            self.queue_url(norm_name, norm_pos, self.links[n_id])

                        if feature_code != '':
                            feature_instance = self.feature_codes.get(f"{feature_class}.{feature_code}")
                            if feature_instance:
                                norm_feat, norm_feat_pos = self.queue_concept(feature_instance, "Noun")
                                self.queue_relation(norm_name, norm_pos, norm_feat, norm_feat_pos, "instanceOf", 1)

                        if len(translations) > 0:
                            for translation in translations.split(','):
                                if name != translation and translation != '':
                                    self.queue_property(norm_name, norm_pos, 'alternativeOf', translation)

                        # Flush to the database if the batch limit is reached
                        if len(self.batch_concepts) >= self.batch_size:
                            self.flush_batch(cursor)

                    self.flush_batch(cursor)  # Flush any remaining items from Phase 1
                    self.pickle_manager.save(self.map_cache_filepath, self.id_term_map)

                for parent, children in tqdm(hierarchy.items(), desc="Processing GeoNames hierarchy",
                                             total=len(hierarchy)):
                    parent = self.check_id(str(parent))

                    if parent and parent in self.id_term_map:
                        parent_term, parent_pos = self.id_term_map[parent]
                        p_term, p_pos = self.queue_concept(parent_term, parent_pos)

                        for child in children:
                            child = self.check_id(str(child))
                            if child and child in self.id_term_map:
                                child_term, child_pos = self.id_term_map[child]
                                c_term, c_pos = self.queue_concept(child_term, child_pos)

                                self.queue_relation(c_term, c_pos, p_term, p_pos, "partOf", 1)

                        # Flush to the database if the batch limit is reached
                        if len(self.batch_concepts) >= self.batch_size:
                            self.flush_batch(cursor)

                self.flush_batch(cursor)  # Flush any remaining items from Phase 2

            if self.mode == 'db':
                with self.conn.cursor() as cursor:
                    iterate_over_file(cursor)
                    self.conn.commit()
            else:
                iterate_over_file()
        finally:
            if file_handle:
                file_handle.close()

    def check_id(self, c_id):
        # while loop as first alternate to geoname ID might not be in GeoNames table
        while c_id not in self.id_term_map:
            if c_id not in self.alternate_names:
                # logging.error(f"Could not find alternative name {c_id}")
                return None
            else:
                c_id = self.alternate_names[c_id]
        return c_id
