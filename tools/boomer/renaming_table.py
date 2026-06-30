import json
import csv

# 1. Load Boomer's output JSON (The Cliques)
#    This file contains groups of equivalent terms.
boomer_json_file = "boomer_output/singletons.json"  # Adjust path
output_mapping_file = "renaming_map.tsv"

with open(boomer_json_file, 'r') as f:
    data = json.load(f)

rows = []
# 2. Iterate through each clique
for clique in data['cliques']:
    members = clique['members']
    if len(members) > 1:
        # Strategy: Pick the "best" term to keep.
        # Priority: Pick 'parmenides:isA' over 'parmenides:Hypernym'
        # Simple logic: Sort alphabetically, or prefer specific prefixes.

        # Example: Prefer terms NOT starting with "Hypernym" if possible
        # Or simply pick the first one as the 'Leader'
        leader = members[0]

        # Custom Logic: If 'parmenides:isA' is in the list, force it as leader
        for m in members:
            if "isA" in m:
                leader = m
                break

        # Create mapping rows: "Old Term" -> "Leader"
        for m in members:
            if m != leader:
                # Robot Rename format: "Old IRI" (tab) "New IRI"
                rows.append([m, leader])

# 3. Save as TSV
with open(output_mapping_file, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["Old", "New"])  # Header for ROBOT (Optional but good)
    writer.writerows(rows)

print(f"Generated {len(rows)} renaming rules in {output_mapping_file}")