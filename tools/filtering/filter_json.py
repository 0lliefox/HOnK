import json
from tqdm import tqdm


# Used to filter Wiktionary file to English only to decrease size of initial dataset
def filter_json(input_file_path, output_file_path):
    english_objects = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, total=total_lines, desc="Filtering JSONL"):
            try:
                obj = json.loads(line)
                if obj.get('lang_code') == 'en':
                    top_level_keys = ('translations', 'etymology_templates', 'etymology_text', 'sounds',
                                      'examples', 'hyphenations', 'etymology_number' 'raw_glosses', 'glosses', 'head_templates', 'wikipedia')
                    sense_level_keys = ('glosses', 'raw_glosses', 'examples', 'attestations')
                    for key in top_level_keys:
                        obj.pop(key, None)
                    for sense in obj.get('senses', []):
                        for key in sense_level_keys:
                            sense.pop(key, None)
                    english_objects.append(obj)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                continue

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(english_objects, outfile, indent=4)

    print(f"Saved to {output_file_path}")


if __name__ == '__main__':
    input_file = '../../supporting_files/kb/raw-wiktextract-data.jsonl'
    output_file = 'wiktionary_en.json'

    filter_json(input_file, output_file)
