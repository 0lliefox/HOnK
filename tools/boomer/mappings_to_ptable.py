import json
import csv
import os

input_filenames = [
    '../../supporting_files/mappings/cn_edge_mappings.json',
    '../../supporting_files/mappings/cn_pos_mappings.json',
    '../../supporting_files/mappings/wk_edge_mappings.json',
    '../../supporting_files/mappings/wk_pos_mappings.json',
    '../../supporting_files/mappings/wn_edge_mappings.json',
    '../../supporting_files/mappings/wn_pos_mappings.json'
]

# Output file
output_file = 'mappings.tsv'


def get_prefix_from_filename(filename):
    base = os.path.basename(filename)
    # if 'wn_' in base:
    #     return 'wn'
    return 'parmenides'


rows_to_write = []

for filename in input_filenames:
    try:
        with open(filename, 'r') as f:
            mappings = json.load(f)

        source_prefix = get_prefix_from_filename(filename)

        for key, val in mappings.items():
            clean_key = key.strip()

            if isinstance(val, str):
                source = f"{source_prefix}:{clean_key}"
                target = f"parmenides:{val.strip()}"
                rows_to_write.append([source, target, "1", "1", "1", "1"])

            elif isinstance(val, dict):
                if "swap" not in val:
                    val["swap"] = False

                if not val.get('relNegated') and not val.get('swap'):
                    source = f"{source_prefix}:{clean_key}"
                    target = f"parmenides:{val['rel'].strip()}"
                    rows_to_write.append([source, target, "1", "1", "1", "1"])

    except FileNotFoundError:
        print(f"Warning: Could not find file {filename}")

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerows(rows_to_write)

print(f"Successfully generated {output_file} with {len(rows_to_write)} mappings.")