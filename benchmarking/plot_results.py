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
    scale_y_log10, scale_x_continuous, scale_color_manual, scale_fill_manual, scale_shape_manual,
    labs, theme_minimal, theme, element_text, position_dodge, scale_y_continuous
)
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D4AC0D", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


class ResultsPlotter:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.melted_data = None
        self.plot_object = None
        self.phase_columns = [
            'Build adjacency list',
            'Connected components',
            'Build clusters from adjacency list',
            'Add unclustered concepts',
            'Store clusters in DB',
            'Coalesce relationships',
            'Clean up graph'
        ]

        # Font setup - resolve Satoshi from the working dir or the repo-adjacent fonts
        # directory (../../fonts relative to this file), falling back to matplotlib's
        # default if neither is present. FontProperties(fname=...) fails lazily at render
        # time, so an existence check is needed rather than a try/except here.
        medium = 'fonts/Satoshi-Medium.ttf'
        bold = 'fonts/Satoshi-Bold.ttf'
        if not os.path.exists(medium):
            here = os.path.dirname(os.path.abspath(__file__))
            medium = os.path.join(here, '../../fonts/Satoshi-Medium.ttf')
            bold = os.path.join(here, '../../fonts/Satoshi-Bold.ttf')
        if os.path.exists(medium) and os.path.exists(bold):
            self.font = fm.FontProperties(fname=medium, size=14)
            self.bold_font = fm.FontProperties(fname=bold, size=14)
            self.title_font = fm.FontProperties(fname=bold, size=18)
        else:
            self.font = fm.FontProperties(size=14)
            self.bold_font = fm.FontProperties(weight='bold', size=14)
            self.title_font = fm.FontProperties(weight='bold', size=18)

    def analyze_curve_fit(self):
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

    def load_and_process_data(self):
        self.data = pd.read_csv(self.file_path)
        print("Data Head:")
        print(self.data.head())
        print("\nData Info:")
        self.data.info()

        suffix_mapping = {
            'build_adj_list': 'Build adjacency list',
            'build_clusters_from_adj': 'Build clusters from adjacency list',
            'transitive_closure': 'Connected components',
            'add_unclustered_concepts': 'Add unclustered concepts',
            'store_clusters_in_db': 'Store clusters in DB',
            'coalesce_relationships': 'Coalesce relationships',
            'cleanup_graph': 'Clean up graph'
        }

        rename_dict = {'id': 'Percentage'}
        for col in self.data.columns:
            for suffix, new_name in suffix_mapping.items():
                if col.endswith(suffix):
                    rename_dict[col] = new_name

        self.data = self.data.rename(columns=rename_dict)

        # Only keep phase columns that exist after renaming
        self.phase_columns = [col for col in self.phase_columns if col in self.data.columns]

        self.data['Total'] = self.data[self.phase_columns].sum(axis=1)

        columns_to_process = self.phase_columns + ['Total']

        # Group by percentage to filter out anomalous runs per dataset size
        filtered_groups = []
        for pct, group in self.data.groupby('Percentage'):
            valid_mask = pd.Series(True, index=group.index)
            for col in columns_to_process:
                Q1 = group[col].quantile(0.25)
                Q3 = group[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                valid_mask &= (group[col] >= lower_bound) & (group[col] <= upper_bound)

            filtered_groups.append(group[valid_mask])

        # Reconstruct the dataframe without outliers
        self.data = pd.concat(filtered_groups).reset_index(drop=True)

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

    def create_plot(self):
        markers = ['o', 's', 'v', '^', 'D', 'P', 'X']

        # Preserve the published figure's phase -> colour/marker mapping. Colours and
        # markers are assigned in the categorical order of Phase (alphabetical by
        # default). The clustering phase was previously labelled 'Transitive closure';
        # sorting it under that old key keeps every series' colour, marker, and legend
        # position identical to the released figure so only the label text changes.
        def _phase_order_key(name):
            return 'Transitive closure' if name == 'Connected components' else name
        phase_order = sorted(self.melted_data['Phase'].unique(), key=_phase_order_key)
        self.melted_data['Phase'] = pd.Categorical(
            self.melted_data['Phase'], categories=phase_order, ordered=True
        )

        # Breaks at every integer power of 10, with extra headroom so the top bound is visible
        def even_log_breaks(limits):
            try:
                min_val = max(1e-9, limits[0])
                max_val = max(1e-9, limits[1])
                return 10.0 ** np.arange(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)) + 1)
            except Exception:
                return [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

        # Format y-axis labels as 10^x
        def power_of_10_labels(breaks):
            labels = []
            for b in breaks:
                if b <= 0:
                    labels.append('')
                else:
                    exp = int(round(np.log10(b)))
                    labels.append(f'$10^{{{exp}}}$')
            return labels

        self.plot_object = (
                ggplot(self.melted_data, aes(x='Percentage', y='Time', color='Phase', shape='Phase'))
                + geom_errorbar(aes(ymin='ymin', ymax='ymax'), width=2, alpha=0.5)
                + geom_line()
                + geom_point(size=3)
                + scale_y_log10(breaks=even_log_breaks, labels=power_of_10_labels,
                                expand=(0.05, 0, 0.15, 0))
                + scale_x_continuous(breaks=list(range(10, 101, 10)))
                + scale_color_manual(values=CB_PALETTE)
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
            width=12,
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
    def __init__(self, file1: str, file2: str, name1: str = None, name2: str = None, output_dir: str = '.'):
        self.file1 = file1
        self.file2 = file2
        self.name1 = name1 if name1 else os.path.basename(file1).replace('.csv', '')
        self.name2 = name2 if name2 else os.path.basename(file2).replace('.csv', '')
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
            self.font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=14)
            self.bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=14)
            self.title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=18)
        except:
            self.font = fm.FontProperties(size=14)
            self.bold_font = fm.FontProperties(weight='bold', size=14)
            self.title_font = fm.FontProperties(weight='bold', size=18)

    def run(self):
        logging.info(f"Comparing benchmarks: {self.name1} vs {self.name2}")

        df1 = pd.read_csv(self.file1)
        df2 = pd.read_csv(self.file2)

        self._plot_time_comparison(df1, df2)
        self._plot_memory_comparison(df1, df2)

    def _plot_time_comparison(self, df1, df2):
        # Filter for available phases
        available_phases = [p for p in self.phases if p in df1.columns and p in df2.columns]
        if not available_phases:
            logging.error("No matching phases found in both CSV files for time comparison.")
            return

        # Calculate mean and std for each phase
        def get_stats(df, label):
            stats = []
            for phase in available_phases:
                mean_val = df[phase].mean()
                std_val = df[phase].std() if len(df) > 1 else 0
                display_phase = phase.replace('Loader', '')

                stats.append({
                    'Phase': display_phase,
                    'Dataset': label,
                    'Time': mean_val,
                    'Std': std_val
                })
            return pd.DataFrame(stats)

        stats1 = get_stats(df1, self.name1)
        stats2 = get_stats(df2, self.name2)

        combined_data = pd.concat([stats1, stats2])

        # Enforce phase order
        display_phases = [p.replace('Loader', '') for p in available_phases]
        combined_data['Phase'] = pd.Categorical(combined_data['Phase'], categories=display_phases, ordered=True)

        # Calculate ymin/ymax for error bars
        combined_data['ymin'] = combined_data['Time'] - combined_data['Std']
        combined_data['ymax'] = combined_data['Time'] + combined_data['Std']
        combined_data['ymin'] = combined_data['ymin'].clip(lower=0)

        # Create plot
        plot = (
                ggplot(combined_data, aes(x='Phase', y='Time', fill='Dataset'))
                + geom_bar(stat='identity', position=position_dodge(width=0.9), width=0.8)
                + geom_errorbar(aes(ymin='ymin', ymax='ymax'), position=position_dodge(width=0.9), width=0.25)
                + scale_fill_manual(values=[CB_PALETTE[5], CB_PALETTE[4]])  # Vermilion and Blue
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

    def _plot_memory_comparison(self, df1, df2):
        # Identify memory columns (ending with _memory_mb)
        mem_cols1 = [c for c in df1.columns if c.endswith('_memory_mb')]
        mem_cols2 = [c for c in df2.columns if c.endswith('_memory_mb')]

        # Find common memory columns
        common_cols = list(set(mem_cols1).intersection(mem_cols2))

        # Filter to keep only those related to our phases of interest
        relevant_mem_cols = []
        for phase in self.phases:
            mem_col = f"{phase}_memory_mb"
            if mem_col in common_cols:
                relevant_mem_cols.append(mem_col)

        if not relevant_mem_cols:
            logging.warning("No matching memory columns found for the specified phases.")
            return

        def get_mem_stats(df, label):
            stats = []
            for col in relevant_mem_cols:
                # Clean phase name for display
                phase_name = col.replace('_memory_mb', '').replace('Loader', '')
                mean_val = df[col].mean()
                std_val = df[col].std() if len(df) > 1 else 0
                stats.append({
                    'Phase': phase_name,
                    'Dataset': label,
                    'Memory': mean_val,
                    'Std': std_val
                })
            return pd.DataFrame(stats)

        stats1 = get_mem_stats(df1, self.name1)
        stats2 = get_mem_stats(df2, self.name2)

        combined_data = pd.concat([stats1, stats2])

        # Enforce phase order
        display_phases = [col.replace('_memory_mb', '').replace('Loader', '') for col in relevant_mem_cols]
        combined_data['Phase'] = pd.Categorical(combined_data['Phase'], categories=display_phases, ordered=True)

        combined_data['ymin'] = combined_data['Memory'] - combined_data['Std']
        combined_data['ymax'] = combined_data['Memory'] + combined_data['Std']
        combined_data['ymin'] = combined_data['ymin'].clip(lower=0)

        plot = (
                ggplot(combined_data, aes(x='Phase', y='Memory', fill='Dataset'))
                + geom_bar(stat='identity', position=position_dodge(width=0.9), width=0.8)
                + geom_errorbar(aes(ymin='ymin', ymax='ymax'), position=position_dodge(width=0.9), width=0.25)
                + scale_fill_manual(values=[CB_PALETTE[5], CB_PALETTE[4]])  # Vermilion and Blue
                + labs(
            title='Comparison of Memory Usage per Phase',
            x='Phase',
            y='Memory Usage (MB)'
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

        output_path = os.path.join(self.output_dir, 'memory_comparison.png')
        plot.save(output_path, dpi=300, width=12, height=8, units='in', verbose=False)
        logging.info(f"Saved memory comparison plot to '{output_path}'")


class PipelinePerformancePlotter:
    def __init__(self, file_path: str, output_dir: str = '.'):
        self.file_path = file_path
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
            self.font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=14)
            self.bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=14)
            self.title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=18)
        except:
            self.font = fm.FontProperties(size=14)
            self.bold_font = fm.FontProperties(weight='bold', size=14)
            self.title_font = fm.FontProperties(weight='bold', size=18)

    def run(self):
        logging.info(f"Analyzing pipeline performance: {self.file_path}")

        df = pd.read_csv(self.file_path)

        # Filter for available phases
        available_phases = [p for p in self.phases if p in df.columns]
        if not available_phases:
            logging.error("No matching phases found in CSV file.")
            return

        # Calculate mean and std for time and memory
        stats = []
        total_time = 0
        peak_memory = 0

        for phase in available_phases:
            # Time
            time_mean = df[phase].mean()
            total_time += time_mean

            # Memory
            mem_col = f"{phase}_memory_mb"
            if mem_col in df.columns:
                mem_mean = df[mem_col].mean()
                peak_memory = max(peak_memory, mem_mean)
            else:
                mem_mean = 0

            # Clean phase name
            display_phase = phase.replace('Loader', '')

            stats.append({
                'Phase': display_phase,
                'Time': time_mean,
                'Memory': mem_mean
            })

        stats_df = pd.DataFrame(stats)

        # Enforce phase order
        display_phases = [p.replace('Loader', '') for p in available_phases]
        stats_df['Phase'] = pd.Categorical(stats_df['Phase'], categories=display_phases, ordered=True)

        self._plot_time_stacked(stats_df)
        self._plot_memory(stats_df)
        self._print_summary_latex(total_time, peak_memory)

    def _plot_time_stacked(self, df):
        # Stacked bar chart for time
        # We need a dummy x-axis variable since it's a single pipeline
        df['Pipeline'] = 'Pipeline'

        plot = (
                ggplot(df, aes(x='Pipeline', y='Time', fill='Phase'))
                + geom_bar(stat='identity', width=0.5)
                + scale_fill_manual(values=CB_PALETTE)
                + labs(
            title='Pipeline Execution Time Breakdown',
            x='',
            y='Time (seconds)'
        )
                + theme_minimal()
                + theme(
            plot_title=element_text(fontproperties=self.title_font, ha='center'),
            axis_title_y=element_text(fontproperties=self.bold_font),
            axis_text_y=element_text(fontproperties=self.font),
            axis_text_x=element_text(size=0),  # Hide x-axis text
            legend_title=element_text(fontproperties=self.bold_font),
            legend_text=element_text(fontproperties=self.font),
            legend_position='right'
        )
        )

        output_path = os.path.join(self.output_dir, 'pipeline_time_stacked.png')
        plot.save(output_path, dpi=300, width=8, height=8, units='in', verbose=False)
        logging.info(f"Saved stacked time plot to '{output_path}'")

    def _plot_memory(self, df):
        # Bar chart for memory usage per phase
        plot = (
                ggplot(df, aes(x='Phase', y='Memory', fill='Phase'))
                + geom_bar(stat='identity', width=0.7)
                + scale_fill_manual(values=CB_PALETTE)
                + labs(
            title='Memory Usage per Phase',
            x='Phase',
            y='Memory Usage (MB)'
        )
                + theme_minimal()
                + theme(
            plot_title=element_text(fontproperties=self.title_font, ha='center'),
            axis_title_x=element_text(fontproperties=self.bold_font),
            axis_title_y=element_text(fontproperties=self.bold_font),
            axis_text_x=element_text(fontproperties=self.font, angle=45, ha='right'),
            legend_position='none'  # Legend redundant with x-axis
        )
        )

        output_path = os.path.join(self.output_dir, 'pipeline_memory.png')
        plot.save(output_path, dpi=300, width=10, height=6, units='in', verbose=False)
        logging.info(f"Saved memory plot to '{output_path}'")

    def _print_summary_latex(self, total_time, peak_memory):
        print("\n--- Pipeline Performance Summary (LaTeX) ---\n")

        latex_code = f"""
\\begin{{table}}[h!]
    \\centering
    \\caption{{Pipeline Performance Summary}}
    \\label{{table:pipeline-performance}}
    \\begin{{tabular}}{{lr}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\midrule
        Total Runtime & {total_time:.2f} s \\\\
        Peak Memory Usage & {peak_memory:.2f} MB \\\\
        Output Graph Nodes & \\TODO \\\\
        Output Graph Edges & \\TODO \\\\
        Output Graph Triples & \\TODO \\\\
        Throughput & \\TODO triples/sec \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""
        print(latex_code)


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
    rects1 = ax.bar(x - width / 2, freq1, width, label=g1_name, color=CB_PALETTE[5])
    rects2 = ax.bar(x + width / 2, freq2, width, label=g2_name, color=CB_PALETTE[4])

    ax.set_ylabel('Relative Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 Syntactic Tag Distribution', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)

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
    ax.scatter(rel_freqs, recalls, alpha=0.5, color=CB_PALETTE[4])

    # Annotate top 5 most frequent
    sorted_indices = np.argsort(rel_freqs)[::-1]
    for i in sorted_indices[:5]:
        if i < len(labels):
            ax.annotate(labels[i], (rel_freqs[i], recalls[i]), fontsize=12)

    ax.set_xlabel(f'Relative Frequency in {g1_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Recall in Overlap ({g1_name} $\cap$ {g2_name})', fontsize=14, fontweight='bold')
    ax.set_title('Relation Frequency vs. Recall in Overlap', fontsize=18, fontweight='bold')
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
    parser.add_argument("--compare-csvs", nargs=2,
                        help="Paths to two CSV files to compare (e.g., timing_A.csv timing_B.csv)")
    parser.add_argument("--name1", help="Name for the first benchmark dataset")
    parser.add_argument("--name2", help="Name for the second benchmark dataset")

    # Argument for pipeline performance mode
    parser.add_argument("--pipeline-performance",
                        help="Path to a single benchmark CSV to visualize pipeline performance")

    # Argument for scalability benchmark mode (default if no comparison data)
    parser.add_argument("--scalability-csv", default='results/clustering_benchmark.csv',
                        help="Path to scalability benchmark CSV")

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

        comparator = BenchmarkComparator(file1, file2, args.name1, args.name2, args.output_dir)
        comparator.run()
    elif args.pipeline_performance:
        # Pipeline Performance Mode
        if not os.path.exists(args.pipeline_performance):
            logging.error(f"Pipeline performance CSV not found: {args.pipeline_performance}")
            return

        plotter = PipelinePerformancePlotter(args.pipeline_performance, args.output_dir)
        plotter.run()
    else:
        # Scalability Benchmark Mode
        if not os.path.exists(args.scalability_csv):
            logging.warning(f"Scalability CSV not found at {args.scalability_csv}. Skipping scalability plot.")
            return

        plotter = ResultsPlotter(args.scalability_csv)
        plotter.run()


if __name__ == "__main__":
    main()