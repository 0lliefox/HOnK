import argparse
import logging
import math
import os
import psutil
import gc
import shutil
import tempfile
import csv
import subprocess
import re
import json
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GraphComparator:
    def __init__(self, graph1_path, graph2_path, format1='turtle', format2='turtle', name1=None, name2=None):
        self.graph1_path = graph1_path
        self.graph2_path = graph2_path
        self.format1 = format1
        self.format2 = format2

        self.g1_name = name1 if name1 else os.path.basename(graph1_path)
        self.g2_name = name2 if name2 else os.path.basename(graph2_path)

        # Determine output directory based on the first graph's location
        self.output_dir = os.path.dirname(os.path.abspath(graph1_path))

        self.g1_mem = 0
        self.g2_mem = 0

        self.stats = {
            'g1': {},
            'g2': {}
        }
        self.plot_data = {
            'g1_name': self.g1_name,
            'g2_name': self.g2_name,
            'comparison_name': f"{self.g1_name} vs {self.g2_name}"
        }
        self.unique_samples = {
            'g1': [],
            'g2': []
        }
        self.num_of_unique_samples = 20
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB

    def _sort_file(self, path, unique=False):
        cmd = ["sort", "-o", path, path]
        if unique:
            cmd.append("-u")
        # Use LC_ALL=C for byte-wise sorting
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        subprocess.run(cmd, check=True, env=env)

    def _process_graph(self, path, fmt, key, graph_num):
        logging.info(f"Loading graph {key} from {path}...")
        mem_before = self._get_memory_usage()

        nodes_path = os.path.join(self.temp_dir, f"{key}_nodes.txt")
        edges_path = os.path.join(self.temp_dir, f"{key}_edges.txt")
        pairs_path = os.path.join(self.temp_dir, f"{key}_pairs.txt")
        sop_path = os.path.join(self.temp_dir, f"{key}_sop.txt")

        relation_counts = Counter()
        pos_counts = Counter()
        degree_counts = Counter()

        triple_count = 0

        xsd_string_suffix = "^^<http://www.w3.org/2001/XMLSchema#string>"

        with open(path, 'r', encoding='utf-8', errors='replace') as f_in, \
                open(nodes_path, 'w', encoding='utf-8') as f_nodes, \
                open(edges_path, 'w', encoding='utf-8') as f_edges, \
                open(pairs_path, 'w', encoding='utf-8') as f_pairs, \
                open(sop_path, 'w', encoding='utf-8') as f_sop:

            for line in f_in:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    idx1 = line.find(' ')
                    if idx1 == -1: continue
                    s = line[:idx1]

                    idx2 = line.find(' ', idx1 + 1)
                    if idx2 == -1: continue
                    p = line[idx1 + 1:idx2]

                    # Normalise predicate to handle equivalent IDs (e.g. isA230 == isA)
                    p = re.sub(r'\d+>$', '>', p)

                    idx3 = line.rfind(' .')
                    if idx3 == -1:
                        # Fallback if no space before dot
                        if line.endswith('.'):
                            idx3 = len(line) - 1
                        else:
                            continue
                    o = line[idx2 + 1:idx3].strip()

                    # Normalise subject equivalence
                    if s.startswith('<') and s.endswith('>'):
                        s = re.sub(r'\d+>$', '>', s)

                    # Normalise XSD string datatypes to treat them as basic literals
                    if o.endswith(xsd_string_suffix):
                        o = o[:-len(xsd_string_suffix)]

                    # Treat 'eq' relationships symmetrically by alphabetically sorting subject and object
                    if p in ['eq', '<eq>'] or p.endswith('/eq>') or p.endswith('#eq>') or 'sameAs' in p.lower():
                        if s > o:
                            s, o = o, s

                    triple_count += 1

                    # Stats
                    relation_counts[p] += 1
                    if p == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                        if not o.startswith("<http://www.w3.org/2002/07/owl#") and \
                                not o.startswith("<http://www.w3.org/2000/01/rdf-schema#"):
                            pos_counts[o] += 1

                    degree_counts[s] += 1
                    degree_counts[o] += 1

                    # Write to files
                    f_nodes.write(f"{s}\n")
                    f_nodes.write(f"{o}\n")
                    f_edges.write(f"{s}\t{p}\t{o}\n")
                    f_pairs.write(f"{s}\t{o}\n")
                    f_sop.write(f"{s}\t{o}\t{p}\n")

                except Exception:
                    continue

        mem_after = self._get_memory_usage()
        mem_used = mem_after - mem_before
        logging.info(f"Loaded {triple_count} triples. Memory used: {mem_used:.2f} MB")

        if graph_num == 1:
            self.g1_mem = mem_used
        else:
            self.g2_mem = mem_used

        # Sort files
        logging.info(f"Sorting files for {key}...")
        self._sort_file(nodes_path, unique=True)
        self._sort_file(edges_path, unique=False)
        self._sort_file(pairs_path, unique=True)
        self._sort_file(sop_path, unique=False)

        # Count nodes from sorted unique file
        with open(nodes_path, 'r') as f:
            node_count = sum(1 for _ in f)

        self.stats[key]['nodes'] = node_count
        self.stats[key]['triples'] = triple_count
        self.stats[key]['relations'] = len(relation_counts)
        self.stats[key]['pos_tags'] = len(pos_counts)
        self.stats[key]['relation_counts'] = relation_counts
        self.stats[key]['pos_counts'] = pos_counts

        if key == 'g1':
            self.plot_data['g1_relations'] = relation_counts
            self.plot_data['g1_pos'] = pos_counts
        else:
            self.plot_data['g2_pos'] = pos_counts

        # Density & Degree
        num_nodes = node_count
        num_edges = triple_count
        if num_nodes > 1:
            density = num_edges / (num_nodes * (num_nodes - 1))
        else:
            density = 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        self.stats[key]['density'] = density
        self.stats[key]['degree'] = avg_degree

        # Entropy
        total_degrees = sum(degree_counts.values())
        entropy = 0.0
        if total_degrees > 0:
            probs = np.array(list(degree_counts.values())) / total_degrees
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
        self.stats[key]['entropy'] = entropy

        # Structure placeholders
        self.stats[key]['structure'] = {'cc': 0, 'lcc': 0, 'asp': 0, 'diam': 0}

    def compare(self):
        print(f"\n--- Comparison: {self.g1_name} vs {self.g2_name} ---\n")

        self._process_graph(self.graph1_path, self.format1, 'g1', 1)
        gc.collect()

        self._process_graph(self.graph2_path, self.format2, 'g2', 2)
        gc.collect()

        # Find unique samples
        self._find_unique_samples()

        # Calculate overlap relations for plot data
        self._calculate_overlap_relations()

        self._print_report()
        self._print_latex_table()
        self._save_results_to_csv()
        self._save_plot_data_json()
        self._plot_results()

    def _calculate_overlap_relations(self):
        logging.info("Calculating relation overlap...")
        f1_path = os.path.join(self.temp_dir, "g1_edges.txt")
        f2_path = os.path.join(self.temp_dir, "g2_edges.txt")

        overlap_counts = Counter()

        with open(f1_path, 'r') as f1, open(f2_path, 'r') as f2:
            line1 = f1.readline()
            line2 = f2.readline()

            while line1 and line2:
                if line1 == line2:
                    # Overlap found
                    parts = line1.strip().split('\t')
                    if len(parts) >= 2:
                        p = parts[1]
                        overlap_counts[p] += 1
                    line1 = f1.readline()
                    line2 = f2.readline()
                elif line1 < line2:
                    line1 = f1.readline()
                else:
                    line2 = f2.readline()

        self.plot_data['overlap_relations'] = overlap_counts

    def _find_unique_samples(self):
        logging.info("Finding unique triples and saving to files...")

        f1_path = os.path.join(self.temp_dir, "g1_edges.txt")
        f2_path = os.path.join(self.temp_dir, "g2_edges.txt")

        out1_path = os.path.join(self.output_dir, f"unique_to_{self.g1_name}.txt")
        out2_path = os.path.join(self.output_dir, f"unique_to_{self.g2_name}.txt")

        def is_bnode_triple(line):
            # Check if subject or object is a blank node (_:...)
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                s, p, o = parts[0], parts[1], parts[2]
                return s.startswith("_:") or o.startswith("_:")
            return False

        with open(f1_path, 'r') as f1, open(f2_path, 'r') as f2, \
                open(out1_path, 'w') as out1, open(out2_path, 'w') as out2:

            line1 = f1.readline()
            line2 = f2.readline()

            while line1 or line2:
                if line1 and line2:
                    if line1 == line2:
                        line1 = f1.readline()
                        line2 = f2.readline()
                    elif line1 < line2:
                        # Unique to g1
                        if not is_bnode_triple(line1):
                            clean_line = line1.strip().replace('\t', ' ')
                            if len(self.unique_samples['g1']) < self.num_of_unique_samples:
                                self.unique_samples['g1'].append(clean_line)
                            out1.write(clean_line + '\n')
                        line1 = f1.readline()
                    else:
                        # Unique to g2
                        if not is_bnode_triple(line2):
                            clean_line = line2.strip().replace('\t', ' ')
                            if len(self.unique_samples['g2']) < self.num_of_unique_samples:
                                self.unique_samples['g2'].append(clean_line)
                            out2.write(clean_line + '\n')
                        line2 = f2.readline()
                elif line1:
                    # Remaining in g1
                    if not is_bnode_triple(line1):
                        clean_line = line1.strip().replace('\t', ' ')
                        if len(self.unique_samples['g1']) < self.num_of_unique_samples:
                            self.unique_samples['g1'].append(clean_line)
                        out1.write(clean_line + '\n')
                    line1 = f1.readline()
                elif line2:
                    # Remaining in g2
                    if not is_bnode_triple(line2):
                        clean_line = line2.strip().replace('\t', ' ')
                        if len(self.unique_samples['g2']) < self.num_of_unique_samples:
                            self.unique_samples['g2'].append(clean_line)
                        out2.write(clean_line + '\n')
                    line2 = f2.readline()

        logging.info(f"Unique triples written to {out1_path} and {out2_path}")

    def _print_report(self):
        print("1. Basic Statistics:")
        print(f"   Nodes: {self.g1_name}={self.stats['g1']['nodes']}, {self.g2_name}={self.stats['g2']['nodes']}")
        print(f"   Edges: {self.g1_name}={self.stats['g1']['triples']}, {self.g2_name}={self.stats['g2']['triples']}")
        print("")

        print("2. Memory Usage (Approximate):")
        print(f"   {self.g1_name}: {self.g1_mem:.2f} MB")
        print(f"   {self.g2_name}: {self.g2_mem:.2f} MB")
        print("")

        print("3. Graph Density & Degree:")
        print(
            f"   Density: {self.g1_name}={self.stats['g1']['density']:.12f}, {self.g2_name}={self.stats['g2']['density']:.12f}")
        print(
            f"   Avg Degree: {self.g1_name}={self.stats['g1']['degree']:.2f}, {self.g2_name}={self.stats['g2']['degree']:.2f}")
        print("")

        print("4. Node Entropy (Degree Distribution):")
        print(
            f"   Entropy: {self.g1_name}={self.stats['g1']['entropy']:.4f}, {self.g2_name}={self.stats['g2']['entropy']:.4f}")
        print("")

        print(f"5. Sample Triples Unique to {self.g1_name}:")
        if self.unique_samples['g1']:
            for t in self.unique_samples['g1']:
                print(f"     {t}")
        else:
            print("     (None)")
        print("")

        print(f"   Sample Triples Unique to {self.g2_name}:")
        if self.unique_samples['g2']:
            for t in self.unique_samples['g2']:
                print(f"     {t}")
        else:
            print("     (None)")
        print("")

        # 7. Relation Distribution
        print("7. Relation Type Distribution (Top 10):")
        print(
            f"   Unique Relation Types: {self.g1_name}={self.stats['g1']['relations']}, {self.g2_name}={self.stats['g2']['relations']}")
        print(f"   {'Relation':<50} | {self.g1_name:<15} | {self.g2_name:<15}")
        print("   " + "-" * 86)

        dist1 = self.stats['g1']['relation_counts']
        dist2 = self.stats['g2']['relation_counts']
        all_rels = set(dist1.keys()).union(set(dist2.keys()))
        sorted_rels = sorted(all_rels, key=lambda r: dist1[r] + dist2[r], reverse=True)

        for rel in sorted_rels[:10]:
            rel_name = str(rel).split('/')[-1].split('#')[-1]
            print(f"   {rel_name:<50} | {dist1[rel]:<15} | {dist2[rel]:<15}")
        print("")

        # 8. POS Coverage
        print("8. Part-of-Speech / Type Coverage (Top 10):")
        print(
            f"   Unique POS Tags Total: {self.g1_name}={self.stats['g1']['pos_tags']}, {self.g2_name}={self.stats['g2']['pos_tags']}")
        print(f"   {'Type':<50} | {self.g1_name:<15} | {self.g2_name:<15}")
        print("   " + "-" * 86)

        pos1 = self.stats['g1']['pos_counts']
        pos2 = self.stats['g2']['pos_counts']
        all_pos = set(pos1.keys()).union(set(pos2.keys()))
        sorted_pos = sorted(all_pos, key=lambda p: pos1[p] + pos2[p], reverse=True)

        for pos in sorted_pos[:10]:
            pos_name = str(pos).split('/')[-1].split('#')[-1]
            print(f"   {pos_name:<50} | {pos1[pos]:<15} | {pos2[pos]:<15}")
        print("")

        # 9. Unique POS Tags
        print("9. Unique Part-of-Speech / Type Tags (Not Overlapping):")
        unique_pos_g1 = set(pos1.keys()) - set(pos2.keys())
        unique_pos_g2 = set(pos2.keys()) - set(pos1.keys())

        print(f"   Unique to {self.g1_name} ({len(unique_pos_g1)} tags):")
        if unique_pos_g1:
            for p in sorted(unique_pos_g1):
                print(f"     - {p}")
        else:
            print("     (None)")
        print("")

        print(f"   Unique to {self.g2_name} ({len(unique_pos_g2)} tags):")
        if unique_pos_g2:
            for p in sorted(unique_pos_g2):
                print(f"     - {p}")
        else:
            print("     (None)")
        print("")

    def _print_latex_table(self):
        print("\n--- LaTeX Table Output ---\n")

        def fmt(num):
            if isinstance(num, int):
                return f"{num:,}"
            elif isinstance(num, float):
                return f"{num:.4f}"
            return str(num)

        def fmt_density(num):
            if isinstance(num, float):
                if num == 0:
                    return "0"
                exponent = int(math.floor(math.log10(abs(num))))
                mantissa = num / (10 ** exponent)
                return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
            return str(num)

        g1_name_latex = self.g1_name.replace('_', '\\_')
        g2_name_latex = self.g2_name.replace('_', '\\_')

        s1 = self.stats['g1']
        s2 = self.stats['g2']

        latex_code = f"""
\\begin{{table*}}[h!]
    \\centering
    \\caption{{Comparison of Dataset Overlaps}}
    \\label{{table:dataset-comparison}}
    \\small % Optional: slightly reduce font for narrow columns
    % X column automatically wraps text; r is for right-aligned numbers
    \\begin{{tabularx}}{{\\linewidth}}{{@{{}} X r r r r r r r @{{}}}}
        \\toprule
        \\textbf{{Dataset}} & \\textbf{{\\#Triples}} & \\textbf{{\\#Nodes}} & \\textbf{{\\#Relations}} & \\textbf{{\\#\\gls{{pos}}}} & \\textbf{{Density}} & \\textbf{{Degree}} & \\textbf{{Entropy}}  \\\\
        \\midrule
        {g1_name_latex} & {fmt(s1['triples'])} & {fmt(s1['nodes'])} & {fmt(s1['relations'])} & {fmt(s1['pos_tags'])} & {fmt_density(s1['density'])} & {fmt(s1['degree'])} & {fmt(s1['entropy'])} \\\\
        % Add some vertical space to separate the rows clearly
        \\addlinespace[0.5em] 
        {g2_name_latex} & {fmt(s2['triples'])} & {fmt(s2['nodes'])} & {fmt(s2['relations'])} & {fmt(s2['pos_tags'])} & {fmt_density(s2['density'])} & {fmt(s2['degree'])} & {fmt(s2['entropy'])} \\\\
        \\bottomrule
    \\end{{tabularx}}
\\end{{table*}}
"""
        print(latex_code)

    def _save_results_to_csv(self):
        filename = os.path.join(self.output_dir, f"comparison_{self.g1_name}_{self.g2_name}.csv")
        logging.info(f"Saving results to {filename}...")

        s1 = self.stats['g1']
        s2 = self.stats['g2']

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Transposed format
            writer.writerow(["Metric", self.g1_name, self.g2_name])
            writer.writerow(["Nodes", s1['nodes'], s2['nodes']])
            writer.writerow(["Triples", s1['triples'], s2['triples']])
            writer.writerow(["Relations", s1['relations'], s2['relations']])
            writer.writerow(["POS Tags", s1['pos_tags'], s2['pos_tags']])
            writer.writerow(["Density", s1['density'], s2['density']])
            writer.writerow(["Degree", s1['degree'], s2['degree']])
            writer.writerow(["Entropy", s1['entropy'], s2['entropy']])

    def _save_plot_data_json(self):
        filename = os.path.join(self.output_dir, f"plot_data_{self.g1_name}_{self.g2_name}.json")
        logging.info(f"Saving plot data to {filename}...")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.plot_data, f, indent=4, default=str)

    def _plot_results(self):
        logging.info("Generating plots...")

        pos1 = self.plot_data['g1_pos']
        pos2 = self.plot_data['g2_pos']

        all_pos = set(pos1.keys()).union(set(pos2.keys()))
        sorted_pos = sorted(all_pos, key=lambda p: pos1[p] + pos2[p], reverse=True)[:10]

        labels = [str(p).split('/')[-1].split('#')[-1] for p in sorted_pos]

        total1 = sum(pos1.values())
        total2 = sum(pos2.values())

        freq1 = [pos1[p] / total1 if total1 > 0 else 0 for p in sorted_pos]
        freq2 = [pos2[p] / total2 if total2 > 0 else 0 for p in sorted_pos]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, freq1, width, label=self.g1_name)
        rects2 = ax.bar(x + width / 2, freq2, width, label=self.g2_name)

        ax.set_ylabel('Relative Frequency')
        ax.set_title('Top 10 Syntactic Tag Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

        # Add gap between bars
        plt.subplots_adjust(bottom=0.2)

        fig.tight_layout()

        plot_path = os.path.join(self.output_dir, 'pos_distribution.png')
        plt.savefig(plot_path)
        logging.info(f"Saved '{plot_path}'")


def main():
    parser = argparse.ArgumentParser(description="Compare two RDF graphs.")
    parser.add_argument("graph1", help="Path to the first graph file")
    parser.add_argument("graph2", help="Path to the second graph file")
    parser.add_argument("--format1", default="turtle", help="Format of the first graph (default: turtle)")
    parser.add_argument("--format2", default="turtle", help="Format of the second graph (default: turtle)")
    parser.add_argument("--name1", help="Name for the first graph (for display)")
    parser.add_argument("--name2", help="Name for the second graph (for display)")

    args = parser.parse_args()

    comparator = GraphComparator(args.graph1, args.graph2, args.format1, args.format2, args.name1, args.name2)
    comparator.compare()


if __name__ == "__main__":
    main()