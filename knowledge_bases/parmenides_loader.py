import dataclasses
import logging
from os import listdir
from os.path import isfile, join

from rdflib import Graph

from knowledge_bases import AbstractLoader
from supporting_files.parmenides.classes import SentenceStructure
from supporting_files.parmenides.classes.ParmenidesBuild import ParmenidesBuild


class ParmenidesLoader(AbstractLoader):
    def load_data(self):
        filepath = self.config['local_files']['parmenides']

        with self.conn.cursor() as cursor:
            logging.info(f"Loading Parmenides from '{filepath}'")

            all_pronouns = self.list_files('/pronouns')
            self.create_concepts(all_pronouns, cursor, True)

            all_verbs = self.list_files('/verbs')
            self.create_concepts(all_verbs, cursor, True)

            all_concepts = self.list_files('/concepts')
            self.create_concepts(all_concepts, cursor, False)

            all_prepositions = self.list_files('/prepositions')
            self.create_concepts(all_prepositions, cursor, True)

            all_measures = self.list_files('/measures')
            self.create_concepts(all_measures, cursor, False)

            self.conn.commit()
            logging.info(f"Finished processing Parmenides")

    def create_concepts(self, file_paths, cursor, trim):
        for path in file_paths:
            with open(path, "r") as dep:
                pos = self.get_class_name(path, trim)
                for line in dep:
                    line = line.strip()
                    self.get_or_create_concept(line, pos, "Parmenides", cursor)

    def list_files(self, folder) -> list[str]:
        new_folder = f"{self.config['local_files']['parmenides']}{folder}"
        return [f"{new_folder}/{f}" for f in listdir(new_folder) if isfile(join(new_folder, f))]

    def get_class_name(self, file_name, trim):
        name = ''.join([n.capitalize() for n in file_name.split('/')[-1].split('.')[:1][0].split('_')])
        return name[:-1] if trim else name

    @staticmethod
    def add_logical_functions(config, g: Graph):
        logging.info("Adding logical functions")

        p = ParmenidesBuild(g)
        log_defs, log_rewr_rules = SentenceStructure.load_logical_analysis(f"{config['local_files']['parmenides']}/logical_analysis/logical_analysis.json")
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
