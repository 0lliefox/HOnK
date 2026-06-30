robot convert --input boomer_output.ofn --format ttl --output solutions.ttl

robot merge \
  --input combined_graph_unnormalised.ttl \
  --input solutions.ttl \
  --output final_complete_ontology.ttl