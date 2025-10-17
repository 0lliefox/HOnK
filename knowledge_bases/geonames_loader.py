import csv
import json
import logging

from tqdm import tqdm

from knowledge_bases import AbstractLoader


class GeoNamesLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.map_cache_filepath = f"{self.config['local_files']['cache']}/geonames_map.pkl"
        self.id_term_map = self.load_from_pickle(self.map_cache_filepath)
        with open(self.config['local_files']['geonames_alternates'], 'r') as f:
            self.alternate_names = json.load(f)  # map of alternate ID to geoname ID
        with open(self.config['local_files']['geonames_ignore'], 'r') as f:
            self.ignore_names = f.readlines()

    def _load_data_implementation(self):
        filepath = self.config['local_files']['geonames']
        hierarchy_filepath = self.config['local_files']['geonames_hierarchy']
        with open(hierarchy_filepath, 'r') as f:
            hierarchy = json.load(f)

        with self.conn.cursor() as cursor:
            if not self.id_term_map:
                self.id_term_map = {}
                with open(filepath, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for line in tqdm(reader, desc="Processing GeoNames"):
                        n_id, name, _, translations = line[:4] # https://download.geonames.org/export/dump/readme.txt

                        if name in self.ignore_names:
                            continue

                        self.id_term_map[n_id] = name
                        current_id = self.get_or_create_concept(name, "GPE", "GeoNames", cursor)

                        if len(translations) > 0:
                            translations = translations.split(',')
                            for translation in translations:
                                if name != translation and translation != '':
                                    self.add_property(current_id, 'alternativeOf', translation, "GeoNames", cursor)
                self.conn.commit()
                self.save_to_pickle(self.map_cache_filepath, self.id_term_map)
            else:
                logging.info(f"Loading ID map from pickle file {filepath}")

            for parent, children in tqdm(hierarchy.items(), desc="Processing GeoNames hierarchy", total=len(hierarchy)):
                parent = self.check_id(parent)
                parent_db_id = self.get_or_create_concept(self.id_term_map[parent], "GPE", "GeoNames", cursor)
                for child in children:
                    child = self.check_id(child)
                    child_db_id = self.get_or_create_concept(self.id_term_map[child], "GPE", "GeoNames", cursor)
                    self.add_relation(child_db_id, parent_db_id, "partOf", 1, "GeoNames", cursor)

            self.conn.commit()

    def check_id(self, c_id):
        # while loop as first alternate to geoname ID might not be in GeoName table
        while c_id not in self.id_term_map:
            if c_id not in self.alternate_names:
                logging.error(f"Could not find alternative name {c_id}")
            else:
                c_id = self.alternate_names[c_id]
        return c_id
