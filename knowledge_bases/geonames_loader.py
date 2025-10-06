from tqdm import tqdm

from knowledge_bases import AbstractLoader


class GeoNamesLoader(AbstractLoader):
    def load_data(self):
        filepath = self.config['local_files']['geonames']

        with self.conn.cursor() as cursor:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                lines = lines[1:]
                for idx, line in tqdm(enumerate(lines), desc="Processing GeoNames", total=len(lines)):
                    n_id, name, translation = line.strip().split('\t')  # id, name, translation

                    current_id = self.get_or_create_concept(name, "GPE", "GeoNames", cursor)
                    if name != translation:
                        self.add_undirected(current_id, 'alternative', translation, "GeoNames", cursor)

            self.conn.commit()
