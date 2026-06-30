export JAVA_OPTS="-Xmx16G"
boomer-0.2/bin/boomer \
  --ptable mappings.tsv \
  --ontology combined_graph_unnormalised.ttl \
  --window-count 4 \
  --runs 10 \
  --prefixes prefixes.yaml \
  --output boomer_output \
  --exhaustive-search-limit 6 \
  --output-internal-axioms true