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

    # Normal insert methods
    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_or_get_concept_from_db(self, term, pos, cursor=None):
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
        except Exception as e:
            if standalone: self.conn.rollback()
            raise e
        finally:
            if standalone: cursor.close()

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relation_to_db(self, start_id, end_id, rel_type, weight, cursor):
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
    def get_or_create_concepts_bulk(self, concepts_set, cursor):
        if not concepts_set:
            return {}

        # Sort the records to guarantee consistent locking order and prevent deadlocks
        records = sorted([(term, pos, self.source) for term, pos in concepts_set])

        if not self.unique_source:
            conflict_target = "(term, part_of_speech)"
        else:
            conflict_target = "(term, part_of_speech, source)"

        insert_query = f"""
                INSERT INTO concepts (term, part_of_speech, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING
                RETURNING id, term, part_of_speech;
            """

        inserted_rows = execute_values(cursor, insert_query, records, page_size=self.batch_size, fetch=True)
        concept_mapping = {(row[1], row[2]): row[0] for row in inserted_rows}
        missing_concepts = [c for c in concepts_set if c not in concept_mapping]

        if missing_concepts:
            missing_concepts = sorted(missing_concepts)
            select_query = """
                           SELECT c.id, c.term, c.part_of_speech
                           FROM concepts c
                                    JOIN (VALUES %s) AS t(term, part_of_speech)
                                         ON c.term = t.term AND c.part_of_speech = t.part_of_speech; \
                           """
            existing_rows = execute_values(cursor, select_query, missing_concepts, page_size=self.batch_size,
                                           fetch=True)

            for row in existing_rows:
                concept_mapping[(row[1], row[2])] = row[0]

        return concept_mapping

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
