import logging
import os
import re

from tqdm import tqdm

INPUT_FILE = '/Volumes/T7/latest-truthy-en.nt'
OUTPUT_FILE = '/Volumes/T7/latest-truthy-en-labels-only.nt'
LANGUAGE_TAG = '@en .'

PREDICATE_URIS = [
    '<http://www.w3.org/2000/01/rdf-schema#label>'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_ttl_by_language(input_path, output_path, language_tag, predicate_uris=None):
    logging.info(f"Starting filtering process for file: {input_path}")
    logging.info(f"Filtering for lines ending with: '{language_tag}'")
    logging.info(f"Output will be written to: {output_path}")

    lines_written = 0
    lines_read = 0
    filter_by_predicate = bool(predicate_uris)  # Check if the list is non-empty
    predicate_regex = re.compile(r'^\s*<[^>]+>\s+(<[^>]+>)\s+.*') if filter_by_predicate else None

    # total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    # logging.info(f"Input file contains approximately {total_lines} lines.")

    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, desc="Filtering Lines", unit=" lines"):
            lines_read += 1
            stripped_line = line.rstrip()

            if stripped_line.endswith(language_tag):
                passes_predicate_filter = True  # Assume it passes unless filtering by predicate

                # If filtering by predicate, check the predicate
                if filter_by_predicate:
                    match = predicate_regex.match(stripped_line)
                    if match:
                        extracted_predicate = match.group(1)
                        if extracted_predicate not in predicate_uris:
                            passes_predicate_filter = False
                    else:
                        # If line isn't a standard triple, skip predicate check (for now?)
                        passes_predicate_filter = False

                if passes_predicate_filter:
                    outfile.write(line)
                    lines_written += 1

    logging.info("Filtering process complete.")
    logging.info(f"Read {lines_read} lines.")
    logging.info(f"Wrote {lines_written} lines ending with '{language_tag}' to {output_path}")

if __name__ == "__main__":
    filter_ttl_by_language(INPUT_FILE, OUTPUT_FILE, LANGUAGE_TAG, PREDICATE_URIS)
