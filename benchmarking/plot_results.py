import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_errorbar, geom_bar,
    scale_y_log10, scale_color_brewer, scale_shape_manual, scale_fill_brewer,
    labs, theme_minimal, theme, element_text, position_dodge
)
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultsPlotter:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.melted_data = None
        self.plot_object = None
        self.phase_columns = [
            'Build adjacency list',
            'Transitive closure',
            'Build clusters from adjacency list',
            'Add unclustered concepts',
            'Store clusters in DB',
            'Coalesce relationships'
        ]

        # Font setup - fallback if files don't exist
        try:
            self.font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=11)
            self.bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=11)
            self.title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=14)
        except:
            self.font = fm.FontProperties(size=11)
            self.bold_font = fm.FontProperties(weight='bold', size=11)
            self.title_font = fm.FontProperties(weight='bold', size=14)

    def load_and_process_data(self):
        self.data = pd.read_csv(self.file_path)
        print("Data Head:")
        print(self.data.head())
        print("\nData Info:")
        self.data.info()

        self.data = self.data.rename(columns={
            'id': 'Percentage',
            'build_adj_list': 'Build adjacency list',
            'build_clusters_from_adj': 'Build clusters from adjacency list',
            'transitive_closure': 'Transitive closure',
            'add_unclustered_concepts': 'Add unclustered concepts',
            'store_clusters_in_db': 'Store clusters in DB',
            'coalesce_relationships': 'Coalesce relationships'
        })
        self.data['Total'] = self.data[self.phase_columns].sum(axis=1)

        columns_to_process = self.phase_columns + ['Total']

        # Group by Percentage and calculate mean and std
        grouped_data = self.data.groupby('Percentage')
        mean_data = grouped_data[columns_to_process].mean().reset_index()
        std_data = grouped_data[columns_to_process].std().reset_index()

        # Melt data for plotting
        melted_mean = mean_data.melt(
            id_vars=['Percentage'],
            value_vars=columns_to_process,
            var_name='Phase',
            value_name='Time'
        )

        melted_std = std_data.melt(
            id_vars=['Percentage'],
            value_vars=columns_to_process,
            var_name='Phase',
            value_name='Std'
        )

        self.melted_data = pd.merge(melted_mean, melted_std, on=['Percentage', 'Phase'])

        # Replace NaN in Std with 0, which can happen for single-entry groups
        self.melted_data['Std'] = self.melted_data['Std'].fillna(0)

        self.melted_data['ymin'] = self.melted_data['Time'] - self.melted_data['Std']
        self.melted_data['ymax'] = self.melted_data['Time'] + self.melted_data['Std']
        
        # Clip ymin for log scale
        self.melted_data['ymin'] = self.melted_data['ymin'].clip(lower=1e-9)

        # For curve fitting, use the mean data
        self.data = mean_data
        self.x_data = self.data['Percentage'].values

    def analyze_curve_fit(self):
        """Analyzes and prints the top 3 best-fit curve types for each phase."""
        print("\nCurve Fit Analysis (Top 3 R² Fits)")

        if self.data is None:
            print("Data not loaded. Call load_and_process_data() first.")
            return

        phases = self.phase_columns + ['Total']
        x = self.x_data

        if len(x) < 3:
            print("Skipping analysis: not enough data points (need at least 3).")
            return

        epsilon = 1e-9

        def f_log(x_in, a, b):
            return a * np.log(x_in + epsilon) + b

        def f_lin(x_in, a, b):
            return a * x_in + b

        def f_lin_log(x_in, a, b):
            return a * x_in * np.log(x_in + epsilon) + b

        def f_quad(x_in, a, b, c):
            return a * x_in ** 2 + b * x_in + c

        models_to_test = [
            ("Logarithmic", f_log),
            ("Linear", f_lin),
            ("Linearithmic (n*log n)", f_lin_log),
            ("Quadratic (n^2)", f_quad)
        ]

        for phase in phases:
            y = self.data[phase].values

            valid_indices = np.isfinite(x) & np.isfinite(y)
            x_fit = x[valid_indices]
            y_fit_data = y[valid_indices]

            display_name = phase.replace('_', ' ').title()

            if len(x_fit) < 3:
                print(f"{display_name:<25} -> Skipping (Not enough valid data points)")
                continue

            ss_tot = np.sum((y_fit_data - np.mean(y_fit_data)) ** 2)
            if ss_tot == 0:
                print(f"{display_name:<25} -> Constant (All values are identical)")
                continue

            all_fits = []

            for model_name, func in models_to_test:
                try:
                    popt, _ = curve_fit(func, x_fit, y_fit_data, maxfev=10000)
                    y_predicted = func(x_fit, *popt)
                    ss_res = np.sum((y_fit_data - y_predicted) ** 2)
                    r2 = 1 - (ss_res / ss_tot)

                    if r2 > -np.inf and r2 < np.inf:  # Ensure R² is a real number
                        all_fits.append({"model": model_name, "r2": r2})
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue

            if not all_fits:
                print(f"{display_name:<25} -> No models could be successfully fitted.")
                continue

            # Sort by R² value, descending
            all_fits.sort(key=lambda f: f["r2"], reverse=True)

            print(f"{display_name:<25}")
            for i, fit in enumerate(all_fits[:3]):
                print(f"  {i + 1}. {fit['model']:<25} (R² = {fit['r2']:.4f})")

    def create_plot(self):
        markers = ['o', 's', 'v', '^', 'D', 'P', 'X']

        self.plot_object = (
                ggplot(self.melted_data, aes(x='Percentage', y='Time', color='Phase', shape='Phase'))
                + geom_errorbar(aes(ymin='ymin', ymax='ymax'), width=2, alpha=0.5)
                + geom_line()
                + geom_point(size=3)
                + scale_y_log10()
                + scale_color_brewer(type='qual', palette='Dark2')
                + scale_shape_manual(values=markers)
                + labs(
            title='Comparison of Phase Execution Times for Clustering Pipeline',
            x='Dataset Percentage (%)',
            y='Time (seconds, log scale)'
        )
                + theme_minimal()
                + theme(
            plot_title=element_text(fontproperties=self.title_font, ha='center'),
            axis_title_x=element_text(fontproperties=self.bold_font),
            axis_title_y=element_text(fontproperties=self.bold_font),
            legend_title=element_text(fontproperties=self.bold_font),
            legend_text=element_text(fontproperties=self.font),
            legend_position='bottom',
            legend_direction='horizontal'
        )
        )

    def save_plot(self, plot_filename: str = 'phase_performance_plot.png', dpi: int = 300):
        self.plot_object.save(
            plot_filename,
            dpi=dpi,
            width=10,
            height=6,
            units='in',
            verbose=False
        )

    def run(self, plot_filename: str = 'phase_performance_plot.png'):
        self.load_and_process_data()
        self.analyze_curve_fit()
        self.create_plot()
        self.save_plot(plot_filename)


class BenchmarkComparator:
    def __init__(self, file1: str, file2: str, output_dir: str = '.'):
        self.file1 = file1
        self.file2 = file2
        self.output_dir = output_dir
        self.phases = [
            'ParmenidesLoader',
            'ConceptNetLoader',
            'WiktionaryLoader',
            'WordNetLoader',
            'GeoNamesLoader',
            'Graph building',
            'Graph dumping'
        ]
        
        # Font setup
        try:
            self.font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=11)
            self.bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=11)
            self.title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=14)
        except:
            self.font = fm.FontProperties(size=11)
            self.bold_font = fm.FontProperties(weight='bold', size=11)
            self.title_font = fm.FontProperties(weight='bold', size=14)

    def run(self):
        logging.info(f"Comparing benchmarks: {self.file1} vs {self.file2}")
        
        df1 = pd.read_csv(self.file1)
        df2 = pd.read_csv(self.file2)
        
        # Filter for available phases
        available_phases = [p for p in self.phases if p in df1.columns and p in df2.columns]
        if not available_phases:
            logging.error("No matching phases found in both CSV files.")
            return

        # Calculate mean and std for each phase
        def get_stats(df, label):
            stats = []
            for phase in available_phases:
                mean_val = df[phase].mean()
                std_val = df[phase].std() if len(df) > 1 else 0
                stats.append({
                    'Phase': phase,
                    'Dataset': label,
                    'Time': mean_val,
                    'Std': std_val
                })
            return pd.DataFrame(stats)

        label1 = os.path.basename(self.file1).replace('.csv', '')
        label2 = os.path.basename(self.file2).replace('.csv', '')
        
        stats1 = get_stats(df1, label1)
        stats2 = get_stats(df2, label2)
        
        combined_data = pd.concat([stats1, stats2])
        
        # Calculate ymin/ymax for error bars
        combined_data['ymin'] = combined_data['Time'] - combined_data['Std']
        combined_data['ymax'] = combined_data['Time'] + combined_data['Std']
        combined_data['ymin'] = combined_data['ymin'].clip(lower=0)

        # Create plot
        plot = (
            ggplot(combined_data, aes(x='Phase', y='Time', fill='Dataset'))
            + geom_bar(stat='identity', position=position_dodge(width=0.9), width=0.8)
            + geom_errorbar(aes(ymin='ymin', ymax='ymax'), position=position_dodge(width=0.9), width=0.25)
            + scale_fill_brewer(type='qual', palette='Set1')
            + labs(
                title='Comparison of Build Phase Execution Times',
                x='Phase',
                y='Time (seconds)'
            )
            + theme_minimal()
            + theme(
                plot_title=element_text(fontproperties=self.title_font, ha='center'),
                axis_title_x=element_text(fontproperties=self.bold_font),
                axis_title_y=element_text(fontproperties=self.bold_font),
                axis_text_x=element_text(fontproperties=self.font, angle=45, ha='right'),
                legend_title=element_text(fontproperties=self.bold_font),
                legend_text=element_text(fontproperties=self.font),
                legend_position='bottom'
            )
        )
        
        output_path = os.path.join(self.output_dir, 'benchmark_comparison.png')
        plot.save(output_path, dpi=300, width=12, height=8, units='in', verbose=False)
        logging.info(f"Saved comparison plot to '{output_path}'")


# --- Graph Comparison Plotting Functions ---

def plot_pos_distribution(data, output_dir):
    logging.info("Generating POS distribution plot...")
    
    g1_name = data['g1_name']
    g2_name = data['g2_name']
    pos1 = data['g1_pos']
    pos2 = data['g2_pos']
    
    all_pos = set(pos1.keys()).union(set(pos2.keys()))
    # Sort by total frequency
    sorted_pos = sorted(all_pos, key=lambda p: pos1.get(p, 0) + pos2.get(p, 0), reverse=True)[:10]
    
    labels = [str(p).split('/')[-1].split('#')[-1] for p in sorted_pos]
    
    total1 = sum(pos1.values())
    total2 = sum(pos2.values())
    
    freq1 = [pos1.get(p, 0) / total1 if total1 > 0 else 0 for p in sorted_pos]
    freq2 = [pos2.get(p, 0) / total2 if total2 > 0 else 0 for p in sorted_pos]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, freq1, width, label=g1_name)
    rects2 = ax.bar(x + width/2, freq2, width, label=g2_name)
    
    ax.set_ylabel('Relative Frequency')
    ax.set_title('Top 10 Syntactic Tag Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.subplots_adjust(bottom=0.2)
    
    output_path = os.path.join(output_dir, 'pos_distribution.png')
    plt.savefig(output_path)
    logging.info(f"Saved '{output_path}'")

def plot_relation_correlation(data, output_dir):
    logging.info("Generating relation correlation plot...")
    
    g1_name = data['g1_name']
    g2_name = data['g2_name']
    g1_rels = data['g1_relations']
    overlap_rels = data['overlap_relations']
    
    total_g1_rels = sum(g1_rels.values())
    
    rel_freqs = []
    recalls = []
    labels = []
    
    for rel, count in g1_rels.items():
        if total_g1_rels > 0:
            freq = count / total_g1_rels
            # overlap_rels might not have all keys
            overlap_count = overlap_rels.get(rel, 0)
            recall = overlap_count / count if count > 0 else 0
            
            rel_freqs.append(freq)
            recalls.append(recall)
            labels.append(str(rel).split('/')[-1].split('#')[-1])
    
    if not rel_freqs:
        logging.warning("No relation data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(rel_freqs, recalls, alpha=0.5)
    
    # Annotate top 5 most frequent
    sorted_indices = np.argsort(rel_freqs)[::-1]
    for i in sorted_indices[:5]:
        if i < len(labels):
            ax.annotate(labels[i], (rel_freqs[i], recalls[i]))
        
    ax.set_xlabel(f'Relative Frequency in {g1_name}')
    ax.set_ylabel(f'Recall in Overlap ({g1_name} $\cap$ {g2_name})')
    ax.set_title('Relation Frequency vs. Recall in Overlap')
    ax.grid(True)
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'relation_correlation.png')
    plt.savefig(output_path)
    logging.info(f"Saved '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Plotting utility for benchmarks.")
    
    # Argument for graph comparison mode
    parser.add_argument("--comparison-data", help="Path to the JSON data file for graph comparison plots")
    parser.add_argument("--output-dir", default=".", help="Directory to save comparison plots")
    
    # Argument for comparing two benchmark CSVs
    parser.add_argument("--compare-csvs", nargs=2, help="Paths to two CSV files to compare (e.g., timing_A.csv timing_B.csv)")
    
    # Argument for scalability benchmark mode (default if no comparison data)
    parser.add_argument("--scalability-csv", default='results/clustering_benchmark.csv', help="Path to scalability benchmark CSV")

    args = parser.parse_args()

    if args.comparison_data:
        # Graph Comparison Mode
        if not os.path.exists(args.comparison_data):
            logging.error(f"Comparison data file not found: {args.comparison_data}")
            return
            
        with open(args.comparison_data, 'r') as f:
            data = json.load(f)
            
        plot_pos_distribution(data, args.output_dir)
        plot_relation_correlation(data, args.output_dir)
    elif args.compare_csvs:
        # Benchmark Comparison Mode
        file1, file2 = args.compare_csvs
        if not os.path.exists(file1) or not os.path.exists(file2):
            logging.error("One or both CSV files not found.")
            return
        
        comparator = BenchmarkComparator(file1, file2, args.output_dir)
        comparator.run()
    else:
        # Scalability Benchmark Mode
        if not os.path.exists(args.scalability_csv):
            logging.warning(f"Scalability CSV not found at {args.scalability_csv}. Skipping scalability plot.")
            return

        plotter = ResultsPlotter(args.scalability_csv)
        plotter.run()


if __name__ == "__main__":
    main()
