import logging
import csv
import json
from tqdm import tqdm
from typing import List

from .abstract_loader import AbstractLoader


class Relation:
    relation_id: str
    start_id: str
    end_id: str
    weight: str
    rel: str
    surfaceStart: str
    surfaceEnd: str
    langStart: str
    langEnd: str
    startPOS: str
    endPOS: str

    def __init__(self, line: List[str], headers: List[str] = None):
        if headers is None:
            # headers = ["id", "uri", "relation_id", "start_id", "end_id", "weight", "data"]
            headers = ["uri", "rel", "start", "end", "data"]
        d = dict(zip(headers, line))
        # try:
        #     d["data"] = json.loads(d["data"])
        # except json.JSONDecodeError:
        #     d["data"] = json.loads(d["data"].replace('\\\\"', '\\"'))

        # self.relation_id = d["relation_id"]
        self.rel = d["rel"].replace("/r/", "").replace("dbpedia/", "")
        self.start = d["start"]
        self.end = d["end"]
        self.data = json.loads(d['data'])
        self.weight = self.data['weight']

        self.surfaceStart = self.data.get("surfaceStart")
        self.surfaceEnd = self.data.get("surfaceEnd")

        self.startR = self.start.replace("_", " ")
        self.startRS = self.startR.split('/')
        self.langStart = self.startRS[2] if len(self.startRS) > 2 else None
        self.startPOS = self.startRS[4] if len(self.startRS) > 4 else self.startRS[1]
        if self.surfaceStart is None and len(self.startRS) > 3:
            self.surfaceStart = self.startRS[3]

        self.lang = self.langStart

        if not self.rel == "ExternalURL":
            # self.endR = self.end
            # self.endRS = self.end.split('//')[1].split('.')
            # if len(self.endRS[0]) > 3:
            #     self.langEnd = self.data['dataset'].split('/')[-1]
            # else:
            #     self.langEnd = self.endRS[0]
            # self.endPOS = self.startPOS
            self.endR = self.end.replace("_", " ")
            self.endRS = self.endR.split('/')
            self.langEnd = self.endRS[2] if len(self.endRS) > 2 else None
            self.endPOS = self.endRS[4] if len(self.endRS) > 4 else self.endRS[1]
            if self.surfaceEnd is None and len(self.endRS) > 3:
                self.surfaceEnd = self.endRS[3]


class ConceptNetLoader(AbstractLoader):
    def __init__(self, builder):
        super().__init__(builder)
        self.source = "ConceptNet"

    def load_data(self):
        filepath = self.config['local_files']['conceptnet']
        lang = self.config['general']['language']
        logging.info(f"Loading ConceptNet data from local file: '{filepath}'...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')

                with self.conn.cursor() as cursor:
                    for row in tqdm(reader, desc="Processing ConceptNet Edges"):
                        relation = Relation(row)

                        is_url = self.is_url(relation)
                        if (not is_url and (relation.langStart != lang or relation.langEnd != lang)) or (is_url and relation.lang != lang) or (is_url and '#' in relation.end) or (relation.surfaceStart == ''):
                            continue

                        # Use the POS extracted by the Relation class
                        start_pos = self._get_mapped_pos(relation.startPOS)
                        start_concept_id = self.get_or_create_concept(relation.surfaceStart, start_pos, cursor)

                        if self.is_url(relation):
                            self.add_url(
                                start_concept_id,
                                relation.end,
                                cursor
                            )
                        else:
                            end_pos = self._get_mapped_pos(relation.endPOS)
                            end_concept_id = self.get_or_create_concept(relation.surfaceEnd, end_pos, cursor)

                            if (self.mode == 'db' and start_concept_id and end_concept_id) or self.mode == 'graph':
                                self.add_relation(
                                    {
                                        'id': start_concept_id,
                                        'term': relation.surfaceStart
                                    },
                                    {
                                        'id': end_concept_id,
                                        'term': relation.surfaceEnd
                                     },
                                    relation.rel,
                                    float(relation.weight),
                                    cursor
                                )
                    self.conn.commit()
            logging.info("Finished loading ConceptNet data from file.")
        except FileNotFoundError:
            logging.error(f"ConceptNet file not found at '{filepath}'")
        # except Exception as e:
        #     logging.error(f"An error occurred: {e}")

    def is_url(self, relation):
        return relation.rel == 'ExternalURL'