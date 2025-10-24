import csv
import json
import logging

from tqdm import tqdm

from knowledge_bases import AbstractLoader
from tools.pickling import load_from_pickle, save_to_pickle


class GeoNamesLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.map_cache_filepath = f"{self.config['local_files']['cache']}/geonames_map.pkl"
        self.id_term_map = load_from_pickle(self.map_cache_filepath)
        with open(self.config['local_files']['geonames_alternates'], 'r') as f:
            self.alternate_names = json.load(f)  # map of alternate ID to geoname ID
        with open(self.config['local_files']['geonames_ignore'], 'r') as f:
            self.ignore_names = f.readlines()
        with open(self.config['local_files']['geonames_feature_codes'], 'r') as f:
            self.feature_codes = json.load(f)

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
                        n_id, name, _, translations, _, _, feature_class, feature_code  = line[:8] # https://download.geonames.org/export/dump/readme.txt

                        if name in self.ignore_names:
                            continue

                        # 'A' is country, state, region: http://www.geonames.org/export/codes.html
                        if feature_class == 'A':
                            pos = 'GPE'
                        else:
                            pos = 'LOC'

                        self.id_term_map[n_id] = [name, pos]
                        current_id = self.get_or_create_concept(name, pos, "GeoNames", cursor)

                        # Feature code might be empty, feature class is too general for instanceOf relationship (?)
                        if feature_code != '':
                            feature_instance = self.feature_codes[f"{feature_class}.{feature_code}"]
                            feature_db_id = self.get_or_create_concept(feature_instance, "Noun", "GeoNames", cursor)
                            self.add_relation(current_id, feature_db_id, "instanceOf", 1, "GeoNames", cursor)

                        if len(translations) > 0:
                            translations = translations.split(',')
                            for translation in translations:
                                if name != translation and translation != '':
                                    self.add_property(current_id, 'alternativeOf', translation, "GeoNames", cursor)
                self.conn.commit()
                save_to_pickle(self.map_cache_filepath, self.id_term_map)
            else:
                logging.info(f"Loading ID map from pickle file {filepath}")

            for parent, children in tqdm(hierarchy.items(), desc="Processing GeoNames hierarchy", total=len(hierarchy)):
                parent = self.check_id(parent)
                parent_db_id = self.get_or_create_concept(self.id_term_map[parent][0], self.id_term_map[parent][1], "GeoNames", cursor)
                for child in children:
                    child = self.check_id(child)
                    child_db_id = self.get_or_create_concept(self.id_term_map[child][0], self.id_term_map[child][1], "GeoNames", cursor)
                    self.add_relation(child_db_id, parent_db_id, "partOf", 1, "GeoNames", cursor)

            self.conn.commit()

    def check_id(self, c_id):
        # while loop as first alternate to geoname ID might not be in GeoName table
        while c_id not in self.id_term_map:
            if c_id not in self.alternate_names:
                logging.error(f"Could not find alternative name {c_id}")
                return None
            else:
                c_id = self.alternate_names[c_id]
        return c_id
