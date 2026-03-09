import dataclasses
import json
import logging
from os import listdir
from os.path import isfile, join

import pyoxigraph

from knowledge_bases import AbstractLoader
from supporting_files.parmenides.classes import SentenceStructure
from supporting_files.parmenides.classes.ParmenidesBuild import ParmenidesBuild


class ParmenidesLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.source = "Parmenides"

    def parse_data(self):
        filepath = self.config['local_files']['parmenides']
        logging.info(f"Loading Parmenides from '{filepath}'")

        data = {
            'pronouns': self.list_files('/pronouns'),
            'verbs': self.list_files('/verbs'),
            'concepts': self.list_files('/concepts'),
            'prepositions': self.list_files('/prepositions'),
            'measures': self.list_files('/measures'),
            'wh': self.list_files('/wh'),
            'predeterminers': self.list_files('/predeterminers')
        }
        return data

    def store_data(self, data):
        def iterate_over_files(cursor=None):
            self.create_concepts(data['pronouns'], cursor, True)
            self.create_concepts(data['verbs'], cursor, True)
            self.create_concepts(data['concepts'], cursor, False)
            self.create_concepts(data['prepositions'], cursor, True)
            self.create_concepts(data['measures'], cursor, False)
            self.create_concepts(data['wh'], cursor, True)
            self.create_concepts(data['predeterminers'], cursor, True)

        if self.mode == 'db':
            with self.conn.cursor() as cursor:
                iterate_over_files(cursor)

                self.conn.commit()
        else:
            iterate_over_files()
        logging.info(f"Finished processing Parmenides")

    def create_concepts(self, file_paths, cursor, trim):
        for path in file_paths:
            with open(path, "r") as dep:
                pos = self.get_class_name(path, trim)
                if path.endswith('.txt'):
                    for line in dep:
                        line = line.strip()
                        self.get_or_create_concept(line, pos, cursor)
                    dep.close()
                elif path.endswith('.json'):
                    lines = json.load(dep)
                    for term, v in lines.items():
                        if not term.startswith("__"):
                            c_id = self.get_or_create_concept(term, pos, cursor)
                            for prop in v:
                                self.add_property({'id': c_id, 'term': term}, prop, v[prop], cursor)
                    dep.close()

    def list_files(self, folder) -> list[str]:
        new_folder = f"{self.config['local_files']['parmenides']}{folder}"
        return [f"{new_folder}/{f}" for f in listdir(new_folder) if isfile(join(new_folder, f))]

    def get_class_name(self, file_name, trim):
        name = ''.join([n.upper() if n.upper() == 'WH' else n.capitalize() for n in
                        file_name.split('/')[-1].split('.')[:1][0].split('_')])
        return name[:-1] if trim else name

    @staticmethod
    def add_logical_functions(config, g):
        logging.info("Adding logical functions")

        p = ParmenidesBuild(g)
        log_defs, log_rewr_rules = SentenceStructure.load_logical_analysis(
            f"{config['local_files']['parmenides']}/logical_analysis/logical_analysis.json")
        for name, v in log_defs.items():
            for x in v.specs:
                d = dataclasses.asdict(x)
                if "property" in d and d["property"] is None:
                    d.pop("property")
                else:
                    d["logicalConstructProperty"] = d.pop("property")
                d["logicalConstructName"] = name
                entity_name = f"log/{name}/{d['logicalConstructProperty']}" if "logicalConstructProperty" in d else f"log/{name}"
                p.create_entity(entity_name, "LogicalFunction", entity_name, **d)
        ruleid = 1
        for rule in log_rewr_rules:
            for result in rule.classification:
                dres = dataclasses.asdict(result)
                if "property" in dres and dres["property"] is None:
                    dres.pop("property")
                else:
                    dres["logicalConstructProperty"] = dres.pop("property")
                dres["logicalConstructName"] = dres.pop("type")
                dres["rule_order"] = ruleid
                dres.update(rule.premise)
                p.create_entity(f"logrule/{ruleid}", "LogicalRewritingRule", **dres)
                ruleid += 1