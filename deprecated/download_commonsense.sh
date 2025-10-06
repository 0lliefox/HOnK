#!/bin/bash
echo "Downloading ConceptNet..."
curl https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz | gunzip > conceptnet.csv
echo "ConceptNet downloaded (conceptnet.csv)"

echo "Downloading Wiktionary data... (File is large and may take some time)"
curl https://kaikki.org/dictionary/raw-wiktextract-data.jsonl --output supporting_files/wiktionary.json
echo "Retrieving only English terms (wiktionary_en.json)"
jq 'select(.lang_code == "en")' supporting_files/wiktionary.json > supporting_files/wiktionary_en.json
echo "Wiktionary downloaded (wiktionary_en.json)"

echo "Downloading WordNet data..."
curl "http://ldf.fi/wordnet/data?graph=http://ldf.fi/wordnet/wn31" --output supporting_files/wordnet.ttl
echo "WordNet downloaded (wordnet.rdf)"

echo "All downloads are complete."
