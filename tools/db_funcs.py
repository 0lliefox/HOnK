from functools import lru_cache
from psycopg2.extras import execute_values

from tools.timer import timer


class DBManager:
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.conn = builder.conn
        self.source = None

        # DB Configuration
        db_config = self.config.get('db_config', {})
        self.unique_source = db_config.get('unique_source', False)
        self.use_bulk = db_config.get('bulk_insert', False)
        self.batch_size = db_config.get('batch_size', 10000)

        # Buffers for transparent bulk insertion
        self.relation_buffer = set()
        self.property_buffer = set()
        self.url_buffer = set()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def flush_all(self, cursor=None):
        # Forces all remaining items in the buffers to be inserted into the database
        standalone = cursor is None
        if standalone: cursor = self.conn.cursor()

        self.flush_relations(cursor)
        self.flush_properties(cursor)
        self.flush_urls(cursor)

        if standalone:
            self.conn.commit()
            cursor.close()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def flush_relations(self, cursor):
        if self.relation_buffer:
            self.add_relations_bulk(list(self.relation_buffer), cursor)
            self.relation_buffer.clear()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def flush_properties(self, cursor):
        if self.property_buffer:
            self.add_properties_bulk(list(self.property_buffer), cursor)
            self.property_buffer.clear()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def flush_urls(self, cursor):
        if self.url_buffer:
            self.add_urls_bulk(list(self.url_buffer), cursor)
            self.url_buffer.clear()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_or_get_concept_from_db(self, term, pos, cursor=None):
        # This remains synchronous as it must return the specific database ID immediately
        standalone = cursor is None
        if standalone: cursor = self.conn.cursor()
        try:
            if not self.unique_source:
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
                return result[0]
            else:
                cursor.execute(
                    "SELECT id FROM concepts WHERE term = %s AND part_of_speech = %s",
                    (term, pos)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        finally:
            if standalone:
                self.conn.commit()
                cursor.close()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relation_to_db(self, start_id, end_id, rel_type, weight, cursor):
        if self.use_bulk:
            self.relation_buffer.add((start_id, end_id, rel_type, weight, self.source))
            if len(self.relation_buffer) >= self.batch_size:
                self.flush_relations(cursor)
            return

        if not self.unique_source:
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

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_property_to_db(self, c_id, c_type, c_value, cursor):
        if self.use_bulk:
            self.property_buffer.add((c_id, c_type, c_value, self.source))
            if len(self.property_buffer) >= self.batch_size:
                self.flush_properties(cursor)
            return

        if not self.unique_source:
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

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_url_to_db(self, concept_id, e_url, cursor):
        if self.use_bulk:
            self.url_buffer.add((concept_id, e_url, self.source))
            if len(self.url_buffer) >= self.batch_size:
                self.flush_urls(cursor)
            return

        if not self.unique_source:
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

    # Bulk insertion methods
    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relations_bulk(self, relations_list, cursor):
        if not relations_list:
            return

        relations_list = sorted(relations_list)

        if not self.unique_source:
            conflict_target = "(start_concept_id, end_concept_id, relation_type, weight)"
        else:
            conflict_target = "(start_concept_id, end_concept_id, relation_type, source)"

        query = f"""
                INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        execute_values(cursor, query, relations_list, page_size=self.batch_size)

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_properties_bulk(self, properties_list, cursor):
        if not properties_list:
            return

        properties_list = sorted(properties_list)

        if not self.unique_source:
            conflict_target = "(concept_id, type, value)"
        else:
            conflict_target = "(concept_id, type, value, source)"

        query = f"""
                INSERT INTO properties (concept_id, type, value, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        execute_values(cursor, query, properties_list, page_size=self.batch_size)

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_urls_bulk(self, urls_list, cursor):
        if not urls_list:
            return

        urls_list = sorted(urls_list)

        if not self.unique_source:
            conflict_target = "(concept_id, external_url)"
        else:
            conflict_target = "(concept_id, external_url, source)"

        query = f"""
                INSERT INTO urls (concept_id, external_url, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        execute_values(cursor, query, urls_list, page_size=self.batch_size)