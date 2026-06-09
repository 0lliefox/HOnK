import dataclasses
import json
import logging
import re
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
            'nouns': self.list_files('/nouns'),
            'pronouns': self.list_files('/pronouns'),
            'verbs': self.list_files('/verbs'),
            'concepts': self.list_files('/concepts'),
            'prepositions': self.list_files('/prepositions'),
            'measures': self.list_files('/measures'),
            'wh': self.list_files('/wh'),
            'predeterminers': self.list_files('/predeterminers'),
            'dependency': self.list_files('/dependency'),
            'adjectives': self.list_files('/adjectives'),
            'locations': self.list_files('/locations'),
            'relations': self.list_files('/relations'),
        }
        return data

    def store_data(self, data):
        def iterate_over_files(cursor=None):
            self.create_concepts(data['nouns'], cursor, True)
            self.create_concepts(data['pronouns'], cursor, True)
            self.create_concepts(data['verbs'], cursor, True)
            self.create_concepts(data['concepts'], cursor, False)
            self.create_concepts(data['dependency'], cursor, False)
            self.create_concepts(data['prepositions'], cursor, True)
            self.create_concepts(data['measures'], cursor, False)
            self.create_concepts(data['wh'], cursor, True)
            self.create_concepts(data['predeterminers'], cursor, True)
            weather_adj_files = [p for p in data['adjectives'] if 'weather_condition' in p]
            progress_hedge_adj_files = [p for p in data['adjectives'] if 'progress_hedge' in p]
            other_adj_files = [p for p in data['adjectives'] if 'weather_condition' not in p and 'progress_hedge' not in p]
            self.create_concepts_with_pos(weather_adj_files, ['Adjective', 'JJ', 'WeatherConditionAdjective'], cursor)
            self.create_concepts_with_pos(progress_hedge_adj_files, ['Adjective', 'JJ', 'ProgressHedgeAdjective'], cursor)
            self.create_concepts_with_pos(other_adj_files, ['Adjective', 'JJ'], cursor)
            gpe_files = [p for p in data['locations'] if 'gpes' in p]
            loc_files = [p for p in data['locations'] if 'gpes' not in p]
            self.create_concepts_with_pos(gpe_files, ['GPE'], cursor)
            self.create_concepts_with_pos(loc_files, ['LOC'], cursor)
            metro_station_db_id = self.get_or_create_concept("metro station", "Noun", cursor)
            for path in [p for p in data['locations'] if 'metro' in p]:
                with open(path, 'r') as f:
                    for line in f:
                        station = line.strip()
                        if station:
                            station_db_id = self.get_or_create_concept(station, "LOC", cursor)
                            self.add_relation(
                                {'id': station_db_id, 'term': station, 'pos': 'LOC'},
                                {'id': metro_station_db_id, 'term': 'metro station', 'pos': 'Noun'},
                                "isA", 1, cursor)
            if 'relations' in data:
                self.create_relations(data['relations'], cursor)

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

    def create_relations(self, file_paths, cursor):
        relation_pattern = re.compile(r'\((.*?)\^\((.*?)\)\)-\[(.*?)\]->\((.*?)\^\((.*?)\)\)')
        for path in file_paths:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = relation_pattern.match(line)
                    if match:
                        term1, pos1, rel, term2, pos2 = match.groups()
                        id1 = self.get_or_create_concept(term1, pos1, cursor)
                        id2 = self.get_or_create_concept(term2, pos2, cursor)
                        self.add_relation(
                            {'id': id1, 'term': term1, 'pos': pos1},
                            {'id': id2, 'term': term2, 'pos': pos2},
                            rel, 1, cursor
                        )

    def create_concepts_with_pos(self, file_paths, pos_tags, cursor):
        for path in file_paths:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        for pos in pos_tags:
                            self.get_or_create_concept(line, pos, cursor)

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

        extra_path = f"{config['local_files']['parmenides']}/extra_entities.json"
        try:
            with open(extra_path, 'r') as f:
                extra = json.load(f)
                for conc in extra.get('concepts', []):
                    kwargs = {k: v for k, v in conc.items() if k not in ['full_name', 'type_val', 'entity_name', 'hasAdjective', 'entryPoint', 'subject', 'd_object', 'composite_with', 'comment']}
                    p.create_concept(
                        full_name=conc.get('full_name'),
                        type_val=conc.get('type_val'),
                        entity_name=conc.get('entity_name'),
                        hasAdjective=conc.get('hasAdjective'),
                        entryPoint=conc.get('entryPoint'),
                        subject=conc.get('subject'),
                        d_object=conc.get('d_object'),
                        composite_with=conc.get('composite_with'),
                        comment=conc.get('comment'),
                        **kwargs
                    )
                for rel in extra.get('relationships', []):
                    p.create_relationship_instance(
                        src=rel.get('src'),
                        rel=rel.get('rel'),
                        dst=rel.get('dst'),
                        refl=rel.get('refl', False)
                    )
        except Exception as e:
            logging.error(f"Could not load extra entities: {e}")