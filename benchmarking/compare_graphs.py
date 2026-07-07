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
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
from plotnine import (
    ggplot, aes, geom_bar, geom_point, geom_text,
    scale_fill_manual, labs, theme_minimal, theme, element_text, position_dodge
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D4AC0D", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
# Safe delimiter for temporary files to prevent tab collisions in RDF literals
SEPARATOR = "|||"


class GraphComparator:
    def __init__(self, graph1_path, graph2_path, format1='turtle', format2='turtle', name1=None, name2=None,
                 ignore_bnodes=True, ontology_path=None, light=False):
        self.graph1_path = graph1_path
        self.graph2_path = graph2_path
        self.format1 = format1
        self.format2 = format2
        # Now defaults to True
        self.ignore_bnodes = ignore_bnodes
        # Light mode: skip the per-node degree/navigability stats (which hold
        # 100M+-entry dicts in memory and OOM on the geonameid-scale ontology).
        # The equivalence assertion (triples/nodes/relations counts + unique
        # triples) is unaffected; only the density/degree/entropy/navigability
        # paper-table stats are omitted. Use for the equivalence gate on large builds.
        self.light = light

        self.g1_name = name1 if name1 else os.path.basename(graph1_path)
        self.g2_name = name2 if name2 else os.path.basename(graph2_path)

        self.output_dir = os.path.dirname(os.path.abspath(graph1_path))

        self.g1_mem = 0
        self.g2_mem = 0

        self.stats = {'g1': {}, 'g2': {}}
        self.plot_data = {
            'g1_name': self.g1_name,
            'g2_name': self.g2_name,
            'comparison_name': f"{self.g1_name} vs {self.g2_name}"
        }
        self.unique_samples = {'g1': [], 'g2': []}
        self.unique_node_samples = {'g1': [], 'g2': []}

        self.num_of_unique_samples = 20
        self.num_of_unique_node_samples = 20

        self.temp_dir = tempfile.mkdtemp()

        # Load Ontology Properties for conditional subject normalisation
        self.valid_properties = set()
        if ontology_path and os.path.exists(ontology_path):
            logging.info(f"Loading ontology properties from {ontology_path}")
            try:
                with open(ontology_path, 'r', encoding='utf-8') as f:
                    ontology_classes = json.load(f)
                    self.valid_properties = set(ontology_classes.get("Property", {}).get("classes", {}).keys())
            except Exception as e:
                logging.error(f"Failed to load ontology file: {e}")

        # Font setup
        try:
            font_path_medium = 'fonts/Satoshi-Medium.ttf' if os.path.exists(
                'fonts/Satoshi-Medium.ttf') else '../fonts/Satoshi-Medium.ttf'
            font_path_bold = 'fonts/Satoshi-Bold.ttf' if os.path.exists(
                'fonts/Satoshi-Bold.ttf') else '../fonts/Satoshi-Bold.ttf'
            self.font = fm.FontProperties(fname=font_path_medium, size=14)
            self.bold_font = fm.FontProperties(fname=font_path_bold, size=14)
            self.title_font = fm.FontProperties(fname=font_path_bold, size=18)
        except:
            self.font = fm.FontProperties(size=14)
            self.bold_font = fm.FontProperties(weight='bold', size=14)
            self.title_font = fm.FontProperties(weight='bold', size=18)

    def __del__(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _sort_file(self, path, unique=False):
        cmd = ["sort", "-o", path, path]
        if unique:
            cmd.append("-u")
        # Force byte-wise sorting to match Python's string comparison perfectly
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

        # Regex to capture Subject, Predicate, and Object safely, ignoring trailing punctuation like ., ;, or ,
        triple_pattern = re.compile(r'^([^\s]+)\s+([^\s]+)\s+(.+?)\s*[.;,]*$')

        with open(path, 'r', encoding='utf-8', errors='replace') as f_in, \
                open(nodes_path, 'w', encoding='utf-8') as f_nodes, \
                open(edges_path, 'w', encoding='utf-8') as f_edges, \
                open(pairs_path, 'w', encoding='utf-8') as f_pairs, \
                open(sop_path, 'w', encoding='utf-8') as f_sop:

            for line in f_in:
                line = line.strip()
                # Skip empty lines, comments, and prefix declarations
                if not line or line.startswith('#') or line.startswith('@prefix'):
                    continue

                match = triple_pattern.match(line)
                if not match:
                    continue

                try:
                    s, p, o = match.groups()

                    # Blank node anonymisation to prevent false positives (Now Default)
                    if self.ignore_bnodes:
                        if s.startswith('_:'): s = '_:bnode'
                        if o.startswith('_:'): o = '_:bnode'

                    # Conditional Subject Normalisation based on Ontology Properties
                    if s.startswith('<') and s.endswith('>'):
                        s_candidate = re.sub(r'\d+>$', '>', s)
                        if s_candidate != s:
                            # Extract the entity name to check against ontology
                            entity_name = re.split(r'[/#]', s_candidate[:-1])[-1]
                            if entity_name in self.valid_properties:
                                s = s_candidate

                    # Normalise PREDICATE ONLY (Unconditional)
                    p = re.sub(r'\d+>$', '>', p)

                    # Normalise numeric double/float representations (e.g., "-1.0" == "-1")
                    if o.endswith("^^<http://www.w3.org/2001/XMLSchema#double>") or o.endswith(
                            "^^<http://www.w3.org/2001/XMLSchema#float>"):
                        num_match = re.match(r'^"([^"]+)"\^\^', o)
                        if num_match:
                            try:
                                num_val = float(num_match.group(1))
                                # Convert 1.0 back to 1 to match the integer-style representation
                                norm_str = str(int(num_val)) if num_val.is_integer() else str(num_val)
                                o = f'"{norm_str}"^^' + o.split('^^')[1]
                            except ValueError:
                                pass

                    # Normalise XSD string datatypes
                    if o.endswith(xsd_string_suffix):
                        o = o[:-len(xsd_string_suffix)]

                    # Symmetrical eq relationships
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

                    if not self.light:
                        degree_counts[s] += 1
                        degree_counts[o] += 1

                    # Write to files using the safe SEPARATOR
                    f_nodes.write(f"{s}\n")
                    f_nodes.write(f"{o}\n")
                    f_edges.write(f"{s}{SEPARATOR}{p}{SEPARATOR}{o}\n")
                    f_pairs.write(f"{s}{SEPARATOR}{o}\n")
                    f_sop.write(f"{s}{SEPARATOR}{o}{SEPARATOR}{p}\n")

                except Exception:
                    continue

        mem_after = self._get_memory_usage()
        mem_used = mem_after - mem_before
        logging.info(f"Loaded {triple_count} triples. Memory used: {mem_used:.2f} MB")

        if graph_num == 1:
            self.g1_mem = mem_used
        else:
            self.g2_mem = mem_used

        logging.info(f"Sorting files for {key}...")
        self._sort_file(nodes_path, unique=True)
        self._sort_file(edges_path, unique=False)
        self._sort_file(pairs_path, unique=True)
        self._sort_file(sop_path, unique=False)

        with open(nodes_path, 'r', encoding='utf-8') as f:
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
            self.plot_data['g2_relations'] = relation_counts
            self.plot_data['g2_pos'] = pos_counts

        num_nodes = node_count
        num_edges = triple_count

        if self.light:
            # Skip the memory-heavy per-node stats; keep only what the equivalence
            # assertion needs (triples/nodes/relations already recorded above).
            for stat in ('density', 'degree', 'entropy', 'lcc_fraction', 'reachability', 'components'):
                self.stats[key][stat] = 0
            return

        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        self.stats[key]['density'] = density
        self.stats[key]['degree'] = avg_degree

        total_degrees = sum(degree_counts.values())
        entropy = 0.0
        if total_degrees > 0:
            probs = np.array(list(degree_counts.values())) / total_degrees
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
        self.stats[key]['entropy'] = entropy

        nav = self._compute_navigability(pairs_path, node_count)
        self.stats[key]['lcc_fraction'] = nav['lcc_fraction']
        self.stats[key]['reachability'] = nav['reachability']
        self.stats[key]['components'] = nav['components']

    def _compute_navigability(self, pairs_path, num_nodes):
        """Largest-connected-component fraction and mean reachability via a
        memory-light union-find over the (sorted, unique) node-pair file.
        Answers reviewer R1-D5 (is the enriched graph still navigable?) without
        building an in-memory graph, matching the streaming design for large KGs.

        reachability = expected fraction of nodes reachable from a uniformly
        random node = sum_c (size_c / N)^2, with isolated nodes as singletons.
        """
        parent = {}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                parts = line.split(SEPARATOR)
                if len(parts) != 2:
                    continue
                a, b = parts
                if a not in parent:
                    parent[a] = a
                if b not in parent:
                    parent[b] = b
                union(a, b)

        if num_nodes <= 0:
            return {'lcc_fraction': 0.0, 'reachability': 0.0, 'components': 0}

        sizes = {}
        for node in list(parent.keys()):
            root = find(node)
            sizes[root] = sizes.get(root, 0) + 1
        comp_sizes = list(sizes.values())
        # Nodes with no edges never appear in the pairs file; count them as singletons.
        isolated = max(0, num_nodes - len(parent))
        largest = max(comp_sizes) if comp_sizes else 1
        lcc_fraction = largest / num_nodes
        reach_sq = sum(s * s for s in comp_sizes) + isolated  # singletons contribute 1 each
        reachability = reach_sq / (num_nodes * num_nodes)
        components = len(comp_sizes) + isolated
        return {'lcc_fraction': lcc_fraction, 'reachability': reachability, 'components': components}

    def compare(self):
        print(f"\n--- Comparison: {self.g1_name} vs {self.g2_name} ---\n")

        self._process_graph(self.graph1_path, self.format1, 'g1', 1)
        gc.collect()

        self._process_graph(self.graph2_path, self.format2, 'g2', 2)
        gc.collect()

        self._find_unique_samples()
        self._find_unique_nodes()
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

        with open(f1_path, 'r', encoding='utf-8') as f1, open(f2_path, 'r', encoding='utf-8') as f2:
            line1 = f1.readline()
            line2 = f2.readline()

            while line1 and line2:
                val1 = line1.strip() if line1 else None
                val2 = line2.strip() if line2 else None

                if val1 is not None and val2 is not None:
                    if val1 == val2:
                        parts = val1.split(SEPARATOR)
                        if len(parts) >= 2:
                            p = parts[1]
                            overlap_counts[p] += 1
                        line1 = f1.readline()
                        line2 = f2.readline()
                    elif val1 < val2:
                        line1 = f1.readline()
                    else:
                        line2 = f2.readline()
                elif val1 is not None:
                    line1 = f1.readline()
                elif val2 is not None:
                    line2 = f2.readline()

        self.plot_data['overlap_relations'] = overlap_counts

    def _find_unique_samples(self):
        logging.info("Finding unique triples and saving to files...")

        f1_path = os.path.join(self.temp_dir, "g1_edges.txt")
        f2_path = os.path.join(self.temp_dir, "g2_edges.txt")

        out1_path = os.path.join(self.output_dir, f"unique_edges_{self.g1_name}.txt")
        out2_path = os.path.join(self.output_dir, f"unique_edges_{self.g2_name}.txt")

        with open(f1_path, 'r', encoding='utf-8') as f1, open(f2_path, 'r', encoding='utf-8') as f2, \
                open(out1_path, 'w', encoding='utf-8') as out1, open(out2_path, 'w', encoding='utf-8') as out2:

            line1 = f1.readline()
            line2 = f2.readline()

            while line1 or line2:
                val1 = line1.strip() if line1 else None
                val2 = line2.strip() if line2 else None

                if val1 is not None and val2 is not None:
                    if val1 == val2:
                        line1 = f1.readline()
                        line2 = f2.readline()
                    elif val1 < val2:
                        clean_line = val1.replace(SEPARATOR, ' ')
                        if len(self.unique_samples['g1']) < self.num_of_unique_samples:
                            self.unique_samples['g1'].append(clean_line)
                        out1.write(clean_line + '\n')
                        line1 = f1.readline()
                    else:
                        clean_line = val2.replace(SEPARATOR, ' ')
                        if len(self.unique_samples['g2']) < self.num_of_unique_samples:
                            self.unique_samples['g2'].append(clean_line)
                        out2.write(clean_line + '\n')
                        line2 = f2.readline()
                elif val1 is not None:
                    clean_line = val1.replace(SEPARATOR, ' ')
                    if len(self.unique_samples['g1']) < self.num_of_unique_samples:
                        self.unique_samples['g1'].append(clean_line)
                    out1.write(clean_line + '\n')
                    line1 = f1.readline()
                elif val2 is not None:
                    clean_line = val2.replace(SEPARATOR, ' ')
                    if len(self.unique_samples['g2']) < self.num_of_unique_samples:
                        self.unique_samples['g2'].append(clean_line)
                    out2.write(clean_line + '\n')
                    line2 = f2.readline()

        logging.info(f"Unique triples written to {out1_path} and {out2_path}")

    def _find_unique_nodes(self):
        logging.info("Finding unique nodes and saving to files...")

        f1_path = os.path.join(self.temp_dir, "g1_nodes.txt")
        f2_path = os.path.join(self.temp_dir, "g2_nodes.txt")

        out1_path = os.path.join(self.output_dir, f"unique_nodes_{self.g1_name}.txt")
        out2_path = os.path.join(self.output_dir, f"unique_nodes_{self.g2_name}.txt")

        with open(f1_path, 'r', encoding='utf-8') as f1, open(f2_path, 'r', encoding='utf-8') as f2, \
                open(out1_path, 'w', encoding='utf-8') as out1, open(out2_path, 'w', encoding='utf-8') as out2:

            line1 = f1.readline()
            line2 = f2.readline()

            while line1 or line2:
                val1 = line1.strip() if line1 else None
                val2 = line2.strip() if line2 else None

                if val1 is not None and val2 is not None:
                    if val1 == val2:
                        line1 = f1.readline()
                        line2 = f2.readline()
                    elif val1 < val2:
                        if len(self.unique_node_samples['g1']) < self.num_of_unique_node_samples:
                            self.unique_node_samples['g1'].append(val1)
                        out1.write(val1 + '\n')
                        line1 = f1.readline()
                    else:
                        if len(self.unique_node_samples['g2']) < self.num_of_unique_node_samples:
                            self.unique_node_samples['g2'].append(val2)
                        out2.write(val2 + '\n')
                        line2 = f2.readline()
                elif val1 is not None:
                    if len(self.unique_node_samples['g1']) < self.num_of_unique_node_samples:
                        self.unique_node_samples['g1'].append(val1)
                    out1.write(val1 + '\n')
                    line1 = f1.readline()
                elif val2 is not None:
                    if len(self.unique_node_samples['g2']) < self.num_of_unique_node_samples:
                        self.unique_node_samples['g2'].append(val2)
                    out2.write(val2 + '\n')
                    line2 = f2.readline()

        logging.info(f"Unique nodes written to {out1_path} and {out2_path}")

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
        print(
            f"   LCC Fraction: {self.g1_name}={self.stats['g1']['lcc_fraction']:.4f}, {self.g2_name}={self.stats['g2']['lcc_fraction']:.4f}")
        print(
            f"   Reachability: {self.g1_name}={self.stats['g1']['reachability']:.4f}, {self.g2_name}={self.stats['g2']['reachability']:.4f}")
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

        print(f"6. Sample Nodes Unique to {self.g1_name}:")
        if self.unique_node_samples['g1']:
            for n in self.unique_node_samples['g1']:
                print(f"     {n}")
        else:
            print("     (None)")
        print("")

        print(f"   Sample Nodes Unique to {self.g2_name}:")
        if self.unique_node_samples['g2']:
            for n in self.unique_node_samples['g2']:
                print(f"     {n}")
        else:
            print("     (None)")
        print("")

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

        latex_code = f"""\\begin{{table}}[h!]
    \\centering
    \\caption{{Comparison of metrics between {g1_name_latex} and {g2_name_latex}.}}
    \\label{{table:dataset-comparison}}
    \\small 
    \\begin{{tabularx}}{{\\columnwidth}}{{@{{}} X r r @{{}}}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{{g1_name_latex}}} & \\textbf{{{g2_name_latex}}} \\\\
        \\midrule
        \\#Triples    & {fmt(s1['triples'])} & {fmt(s2['triples'])} \\\\
        \\#Nodes      & {fmt(s1['nodes'])} & {fmt(s2['nodes'])} \\\\
        \\#Relations  & {fmt(s1['relations'])} & {fmt(s2['relations'])} \\\\
        \\#\\gls{{pos}}  & {fmt(s1['pos_tags'])} & {fmt(s2['pos_tags'])} \\\\
        Density      & {fmt_density(s1['density'])} & {fmt_density(s2['density'])} \\\\
        Degree       & {fmt(s1['degree'])} & {fmt(s2['degree'])} \\\\
        LCC Fraction & {fmt(s1['lcc_fraction'])} & {fmt(s2['lcc_fraction'])} \\\\
        Reachability & {fmt(s1['reachability'])} & {fmt(s2['reachability'])} \\\\
        Entropy      & {fmt(s1['entropy'])} & {fmt(s2['entropy'])} \\\\
        \\bottomrule
    \\end{{tabularx}}
\\end{{table}}
"""
        print(latex_code)

    def _save_results_to_csv(self):
        filename = os.path.join(self.output_dir, f"comparison_{self.g1_name}_{self.g2_name}.csv")
        logging.info(f"Saving results to {filename}...")

        s1 = self.stats['g1']
        s2 = self.stats['g2']

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", self.g1_name, self.g2_name])
            writer.writerow(["Nodes", s1['nodes'], s2['nodes']])
            writer.writerow(["Triples", s1['triples'], s2['triples']])
            writer.writerow(["Relations", s1['relations'], s2['relations']])
            writer.writerow(["POS Tags", s1['pos_tags'], s2['pos_tags']])
            writer.writerow(["Density", s1['density'], s2['density']])
            writer.writerow(["Degree", s1['degree'], s2['degree']])
            writer.writerow(["LCC Fraction", s1['lcc_fraction'], s2['lcc_fraction']])
            writer.writerow(["Reachability", s1['reachability'], s2['reachability']])
            writer.writerow(["Entropy", s1['entropy'], s2['entropy']])

    def _save_plot_data_json(self):
        filename = os.path.join(self.output_dir, f"plot_data_{self.g1_name}_{self.g2_name}.json")
        logging.info(f"Saving plot data to {filename}...")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.plot_data, f, indent=4, default=str)

    def _plot_results(self):
        logging.info("Generating plots...")
        self._plot_pos_distribution()
        self._plot_relation_frequency()

    def _plot_pos_distribution(self):
        pos1 = self.plot_data['g1_pos']
        pos2 = self.plot_data['g2_pos']

        all_pos = set(pos1.keys()).union(set(pos2.keys()))
        sorted_pos = sorted(all_pos, key=lambda p: pos1.get(p, 0) + pos2.get(p, 0), reverse=True)[:10]

        plot_data = []
        for pos in sorted_pos:
            label = str(pos).split('/')[-1].split('#')[-1]
            total1 = sum(pos1.values())
            total2 = sum(pos2.values())
            freq1 = pos1.get(pos, 0) / total1 if total1 > 0 else 0
            freq2 = pos2.get(pos, 0) / total2 if total2 > 0 else 0
            plot_data.append({'Tag': label, 'Graph': self.g1_name, 'Frequency': freq1})
            plot_data.append({'Tag': label, 'Graph': self.g2_name, 'Frequency': freq2})

        df = pd.DataFrame(plot_data)
        df['Tag'] = pd.Categorical(df['Tag'], categories=[str(p).split('/')[-1].split('#')[-1] for p in sorted_pos],
                                   ordered=True)

        plot = (
                ggplot(df, aes(x='Tag', y='Frequency', fill='Graph'))
                + geom_bar(stat='identity', position=position_dodge())
                + scale_fill_manual(values=CB_PALETTE)
                + labs(
            title='Top 10 Syntactic Tag Distribution',
            x='Syntactic Tag',
            y='Relative Frequency'
        )
                + theme_minimal()
                + theme(
            plot_title=element_text(fontproperties=self.title_font, ha='center'),
            axis_title_x=element_text(fontproperties=self.bold_font),
            axis_title_y=element_text(fontproperties=self.bold_font),
            axis_text_x=element_text(fontproperties=self.font, angle=45, ha='right'),
            legend_title=element_text(fontproperties=self.bold_font),
            legend_text=element_text(fontproperties=self.font),
            legend_position='bottom',
            legend_direction='horizontal'
        )
        )

        plot_path = os.path.join(self.output_dir, f'pos_distribution_{self.g1_name}_{self.g2_name}.png')
        plot.save(plot_path, dpi=150, width=12, height=7, units='in', verbose=False)
        logging.info(f"Saved '{plot_path}'")

    def _plot_relation_frequency(self):
        rels1 = self.plot_data['g1_relations']
        overlap_rels = self.plot_data.get('overlap_relations', Counter())

        total1 = sum(rels1.values())

        plot_data = []
        for r, count in rels1.items():
            freq = count / total1 if total1 > 0 else 0
            recall = overlap_rels.get(r, 0) / count if count > 0 else 0
            label = str(r).split('/')[-1].split('#')[-1]
            plot_data.append({'Frequency': freq, 'Recall': recall, 'Label': label, 'Magnitude': freq + recall})

        if not plot_data:
            logging.warning("No relation data to plot for frequency vs recall.")
            return

        df = pd.DataFrame(plot_data)
        df['AnnotationLabel'] = df['Label']

        plot = (
                ggplot(df, aes(x='Frequency', y='Recall'))
                + geom_point(alpha=0.5, color=CB_PALETTE[4])
                + geom_text(aes(label='AnnotationLabel'), nudge_y=0.02, size=14)
                + labs(
            title='Relation Frequency vs. Recall',
            x=f'Relative Frequency in {self.g1_name}',
            y=f'Recall in {self.g2_name}'
        )
                + theme_minimal()
                + theme(
            plot_title=element_text(fontproperties=self.title_font, ha='center'),
            axis_title_x=element_text(fontproperties=self.bold_font),
            axis_title_y=element_text(fontproperties=self.bold_font),
            axis_text_x=element_text(fontproperties=self.font),
            axis_text_y=element_text(fontproperties=self.font),
            legend_position='bottom',
            legend_direction='horizontal'
        )
        )

        plot_path = os.path.join(self.output_dir, f'relation_frequency_recall_{self.g1_name}_{self.g2_name}.png')
        plot.save(plot_path, dpi=150, width=10, height=8, units='in', verbose=False)
        logging.info(f"Saved '{plot_path}'")


def main():
    parser = argparse.ArgumentParser(description="Compare two RDF graphs.")
    parser.add_argument("graph1", help="Path to the first graph file")
    parser.add_argument("graph2", help="Path to the second graph file")
    parser.add_argument("--format1", default="turtle", help="Format of the first graph (default: turtle)")
    parser.add_argument("--format2", default="turtle", help="Format of the second graph (default: turtle)")
    parser.add_argument("--name1", help="Name for the first graph (for display)")
    parser.add_argument("--name2", help="Name for the second graph (for display)")
    parser.add_argument("--keep-bnodes", action="store_true",
                        help="Keep original blank node IDs (they are anonymised by default to prevent structural mismatches).")
    parser.add_argument("--ontology", default='supporting_files/mappings/ontology_classes.json',
                        help="Path to the ontology_classes.json file (optional). Valid properties will be conditionally normalised.")

    args = parser.parse_args()

    comparator = GraphComparator(
        args.graph1,
        args.graph2,
        args.format1,
        args.format2,
        args.name1,
        args.name2,
        not args.keep_bnodes,
        args.ontology
    )
    comparator.compare()


if __name__ == "__main__":
    main()