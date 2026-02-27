from functools import lru_cache
from tools.timer import timer

class DBManager:
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.conn = builder.conn
        self.source = None

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_or_get_concept_from_db(self, term, pos, cursor=None):
        standalone = cursor is None
        if standalone: cursor = self.conn.cursor()
        try:
            if not self.config['general']['unique_source']:
                cursor.execute("""
                   INSERT INTO concepts (term, part_of_speech, source)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (term, part_of_speech) DO NOTHING
                   RETURNING id;
                   """,
                   (term, pos, self.source)
               )
            else:
                cursor.execute("""
                   INSERT INTO concepts (term, part_of_speech, source)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (term, part_of_speech, source) DO NOTHING
                   RETURNING id;
                   """,
                   (term, pos, self.source)
                )
            result = cursor.fetchone()
            if result:
                # if standalone: self.conn.commit()
                return result[0]
            else:
                # If the insert did nothing (due to conflict), fetch the existing ID
                cursor.execute("SELECT id FROM concepts WHERE term = %s AND part_of_speech = %s", (term, pos))
                return cursor.fetchone()[0]
        finally:
            if standalone: cursor.close()

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_relation_to_db(self, start_id, end_id, rel_type, weight, cursor):
        if not self.config['general']['unique_source']:
            cursor.execute(
                "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (start_concept_id, end_concept_id, relation_type, weight) DO NOTHING",
                (start_id, end_id, rel_type, weight, self.source)
            )
        else:
            cursor.execute(
                "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (start_concept_id, end_concept_id, relation_type, source) DO NOTHING",
                (start_id, end_id, rel_type, weight, self.source)
            )

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_property_to_db(self, c_id, c_type, c_value, cursor):
        if not self.config['general']['unique_source']:
            cursor.execute(
                "INSERT INTO properties (concept_id, type, value, source) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (concept_id, type, value) DO NOTHING",
                (c_id, c_type, c_value, self.source)
            )
        else:
            cursor.execute(
                "INSERT INTO properties (concept_id, type, value, source) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (concept_id, type, value, source) DO NOTHING",
                (c_id, c_type, c_value, self.source)
            )

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_url_to_db(self, concept_id, e_url, cursor):
        if not self.config['general']['unique_source']:
            cursor.execute(
                "INSERT INTO urls (concept_id, external_url, source) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (concept_id, external_url) DO NOTHING",
                (concept_id, e_url, self.source)
            )
        else:
            cursor.execute(
                "INSERT INTO urls (concept_id, external_url, source) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (concept_id, external_url, source) DO NOTHING",
                (concept_id, e_url, self.source)
            )
