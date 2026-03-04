from psycopg2.extras import execute_values

from tools.timer import timer

class DBManager:
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.conn = builder.conn
        self.source = None

    @timer(log=False, threaded=False, independent=False, memory=False)
    def get_or_create_concepts_bulk(self, concepts_set, cursor):
        if not concepts_set:
            return {}

        records = [(term, pos, self.source) for term, pos in concepts_set]

        if not self.config['general']['unique_source']:
            conflict_target = "(term, part_of_speech)"
        else:
            conflict_target = "(term, part_of_speech, source)"

        insert_query = f"""
                INSERT INTO concepts (term, part_of_speech, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING
                RETURNING id, term, part_of_speech;
            """

        inserted_rows = execute_values(cursor, insert_query, records, fetch=True)
        concept_mapping = {(row[1], row[2]): row[0] for row in inserted_rows}
        missing_concepts = [c for c in concepts_set if c not in concept_mapping]

        if missing_concepts:
            select_query = """
                           SELECT c.id, c.term, c.part_of_speech
                           FROM concepts c
                                    JOIN (VALUES %s) AS t(term, part_of_speech)
                                         ON c.term = t.term AND c.part_of_speech = t.part_of_speech; \
                           """
            existing_rows = execute_values(cursor, select_query, missing_concepts, fetch=True)

            for row in existing_rows:
                concept_mapping[(row[1], row[2])] = row[0]

        return concept_mapping

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relations_bulk(self, relations_list, cursor):
        if not relations_list:
            return

        if not self.config['general']['unique_source']:
            conflict_target = "(start_concept_id, end_concept_id, relation_type, weight)"
        else:
            conflict_target = "(start_concept_id, end_concept_id, relation_type, source)"

        query = f"""
                INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        from psycopg2.extras import execute_values
        execute_values(cursor, query, relations_list)

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_properties_bulk(self, properties_list, cursor):
        if not properties_list:
            return

        if not self.config['general']['unique_source']:
            conflict_target = "(concept_id, type, value)"
        else:
            conflict_target = "(concept_id, type, value, source)"

        query = f"""
                INSERT INTO properties (concept_id, type, value, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        from psycopg2.extras import execute_values
        execute_values(cursor, query, properties_list)

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_urls_bulk(self, urls_list, cursor):
        if not urls_list:
            return

        if not self.config['general']['unique_source']:
            conflict_target = "(concept_id, external_url)"
        else:
            conflict_target = "(concept_id, external_url, source)"

        query = f"""
                INSERT INTO urls (concept_id, external_url, source)
                VALUES %s
                ON CONFLICT {conflict_target} DO NOTHING;
            """
        from psycopg2.extras import execute_values
        execute_values(cursor, query, urls_list)