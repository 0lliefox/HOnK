from rdflib import Graph

def get_triples(file_path, file_format):
    g = Graph()
    g.parse(file_path, format=file_format)
    return set(g)

if __name__ == '__main__':
    # 1. Load files (Ensure you converted .ofn -> .ttl first!)
    file_a = "../../ontologies/ontology_final.nt"
    file_b = "../../ontologies/combined_CWWG_graph_norm.nt"

    print("Loading graphs...")
    triples_a = get_triples(file_a, "nt")
    triples_b = get_triples(file_b, "nt")

    # 2. Calculate Intersection and Union
    intersection = triples_a.intersection(triples_b)
    union = triples_a.union(triples_b)

    # 3. Calculate metrics
    overlap_count = len(intersection)
    jaccard_index = overlap_count / len(union) * 100 if len(union) > 0 else 0

    # Precision/Recall relative to File B (Reference)
    # How much of Reference (B) did Boomer (A) capture?
    recall = overlap_count / len(triples_b) * 100 if len(triples_b) > 0 else 0

    print(f"--- Alignment Stats ---")
    print(f"Triples in Boomer Output: {len(triples_a)}")
    print(f"Triples in Reference:     {len(triples_b)}")
    print(f"Shared Triples:           {overlap_count}")
    print(f"Jaccard Similarity:       {jaccard_index:.2f}%")
    print(f"Coverage of Reference:    {recall:.2f}% (Recall)")