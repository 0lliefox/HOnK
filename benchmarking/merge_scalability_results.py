import glob
import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'scalability/results')

CSV_NAMES = [
    'clustering_benchmark',
    'graph_rdf_clustering_benchmark',
    'graph_oxi_clustering_benchmark',
    'graph_rdf_memory_benchmark',
    'graph_oxi_memory_benchmark',
]

for name in CSV_NAMES:
    pattern = os.path.join(RESULTS_DIR, 'task_*', 'benchmarking', 'results', f'{name}.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found for {name}, skipping.")
        continue

    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, f'{name}.csv')
    combined.to_csv(out_path, index=False)
    print(f"Merged {len(files)} files -> {out_path} ({len(combined)} rows)")
