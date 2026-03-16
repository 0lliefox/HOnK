import argparse
import pandas as pd
import glob
import matplotlib.font_manager as fm
import os
import textwrap
from plotnine import (
    ggplot, aes, geom_bar, geom_errorbar, scale_fill_manual, labs, theme_minimal, theme, element_text, position_dodge,
    element_blank, guide_legend, guides
)

# Universal Colorblind-friendly palette (Okabe-Ito)
CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D4AC0D", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#003771"]
COLOR_MAP = {
    'Parsing/ingestion': CB_PALETTE[0],
    'Normalisation': CB_PALETTE[1],
    'Clustering': CB_PALETTE[2],
    'Database staging': CB_PALETTE[5],
    'Graph staging': CB_PALETTE[8],
    'Final graph construction': CB_PALETTE[4],
    'Serialisation': CB_PALETTE[7]
}

# Explicitly defining your exact required order to guarantee no sorting bugs
EXPECTED_LOADER_ORDER = [
    'ParmenidesLoader',
    'ConceptNetLoader',
    'WiktionaryLoader',
    'WordNetLoader',
    'GeoNamesLoader'
]


def order_columns(available_columns, expected_order):
    ordered = []
    # Add expected columns in exact order if they exist
    for col in expected_order:
        if col in available_columns:
            ordered.append(col)
    # Append any extra columns that weren't in the expected list
    for col in available_columns:
        if col not in ordered:
            ordered.append(col)
    return ordered


def save_table(df_mean, df_std, filename_base, output_dir, experiment_path, caption):
    # Transpose so Phases (categories) are rows and Experiments are columns
    df_mean_t = df_mean.T
    df_std_t = df_std.T

    out_folder = os.path.join(output_dir, experiment_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    csv_path = os.path.join(out_folder, f'{filename_base}.csv')
    latex_path = os.path.join(out_folder, f'{filename_base}.tex')

    # --- UPDATED CSV SAVING LOGIC ---
    csv_combined = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)
    for col in df_mean.columns:
        # Combined logic: if both mean and std are 0, use N/A, else format with commas
        csv_combined[col] = [
            f"{m:,.2f} ± {s:,.2f}" if not (m == 0 and s == 0) else "N/A"
            for m, s in zip(df_mean[col], df_std[col])
        ]

    if 'execution_times' in filename_base:
        total_mean = df_mean.sum(axis=1)
        total_std = (df_std ** 2).sum(axis=1) ** 0.5
        csv_combined['Total Runtime'] = [
            f"{m:,.2f} ± {s:,.2f}" if m > 0 else "N/A"
            for m, s in zip(total_mean, total_std)
        ]

    csv_combined.to_csv(csv_path)

    # --- LATEX GENERATION ---
    exp_headers = list(df_mean_t.columns)
    formatted_headers = [f"\\makecell[r]{{\\textbf{{{h}}}}}" for h in exp_headers]
    header_str = " & ".join(["\\textbf{Phase}"] + formatted_headers) + " \\\\"

    rows = []
    for phase in df_mean_t.index:
        row_vals = []
        row_mean_series = df_mean_t.loc[phase]

        # Determine min and max ONLY for non-zero values to avoid N/A being "blue"
        valid_means = row_mean_series[row_mean_series > 0]
        row_min = valid_means.min() if not valid_means.empty else None
        row_max = valid_means.max() if not valid_means.empty else None

        for exp in exp_headers:
            m = df_mean_t.loc[phase, exp]
            s = df_std_t.loc[phase, exp]

            if m == 0 and s == 0:
                cell_content = "N/A"
            else:
                cell_content = f"\\makecell[r]{{{m:,.2f} \\\\ \\scriptsize $\\pm$ {s:,.2f}}}"

                # Apply colors only if the value is not zero and fits min/max criteria
                if len(exp_headers) > 1 and row_min is not None:
                    if m == row_min:
                        cell_content = f"\\textcolor{{blue}}{{{cell_content}}}"
                    elif m == row_max:
                        cell_content = f"\\textcolor{{red}}{{{cell_content}}}"

            row_vals.append(cell_content)

        safe_phase = str(phase).replace('_', '\\_')
        rows.append(f"{safe_phase} & {' & '.join(row_vals)} \\\\")

    content = "\n        \\addlinespace[1em]\n        ".join(rows)
    col_spec = "@{} X " + "r " * len(exp_headers) + "@{}"

    latex_code = f"""
\\begin{{table}}[t]
    \\centering
    \\caption{{{caption}. The lowest value in each row is highlighted in blue, and the highest in red.}}
    \\label{{tab:{filename_base.replace('_', '-')}}}
    \\small 
    \\begin{{tabularx}}{{\\columnwidth}}{{{col_spec}}}
        \\toprule
        {header_str}
        \\midrule
        {content}
        \\bottomrule
    \\end{{tabularx}}
\\end{{table}}
"""
    with open(latex_path, 'w') as f:
        f.write(latex_code)


def plot_benchmarks(output_dir='.', rename_map=None, experiment_path='.', title=''):
    # Font setup
    try:
        font_path_medium = 'fonts/Satoshi-Medium.ttf'
        font_path_bold = 'fonts/Satoshi-Bold.ttf'

        if not os.path.exists(font_path_medium):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            font_path_medium = os.path.join(script_dir, '../../fonts/Satoshi-Medium.ttf')
            font_path_bold = os.path.join(script_dir, '../../fonts/Satoshi-Bold.ttf')

        font = fm.FontProperties(fname=font_path_medium, size=20)
        bold_font = fm.FontProperties(fname=font_path_bold, size=20)
        title_font = fm.FontProperties(fname=font_path_bold, size=20)
    except:
        font = fm.FontProperties(size=20)
        bold_font = fm.FontProperties(weight='bold', size=20)
        title_font = fm.FontProperties(weight='bold', size=20)

    # Filter out generated result files to avoid recursion
    bench_files = [f for f in sorted(glob.glob(f'{experiment_path}/benchmark_*.csv'))
                   if 'execution_times' not in f]
    mem_files = [f for f in sorted(glob.glob(f'{experiment_path}/memory_*.csv'))
                 if 'peak_usage' not in f]

    # Dictionaries to store Mean and STD dev
    bench_mean, bench_std = {}, {}
    dataset_bench_mean, dataset_bench_std = {}, {}

    for file in bench_files:
        name = os.path.basename(file).replace('benchmark_', '').replace('.csv', '')
        if rename_map and name in rename_map:
            name = rename_map[name]

        df = pd.read_csv(file)

        # Phase columns mappings
        parse_cols = [c for c in df.columns if 'parse_data' in c]
        norm_cols = [c for c in df.columns if 'normalise_data' in c]
        clust_cols = [c for c in df.columns if 'Clusterer' in c]
        db_cols = [c for c in df.columns if 'DBManager' in c or c.endswith('flush_batch')]
        graph_stage_cols = [c for c in df.columns if 'GraphManager' in c]
        graph_final_cols = [c for c in df.columns if 'OntologyBuilder.build_graph_from_db' in c]
        serial_cols = [c for c in df.columns if 'OntologyBuilder.serialise_graph' in c]

        row_sums = {
            'Parsing/ingestion': df[parse_cols].sum(axis=1) if parse_cols else pd.Series([0] * len(df)),
            'Normalisation': df[norm_cols].sum(axis=1) if norm_cols else pd.Series([0] * len(df)),
            'Clustering': df[clust_cols].sum(axis=1) if clust_cols else pd.Series([0] * len(df)),
            'Final graph construction': df[graph_final_cols].sum(axis=1) if graph_final_cols else pd.Series(
                [0] * len(df)),
            'Serialisation': df[serial_cols].sum(axis=1) if serial_cols else pd.Series(
                [0] * len(df)),
        }

        row_sums['Database staging'] = df[db_cols].sum(axis=1)
        row_sums['Graph staging'] = df[graph_stage_cols].sum(axis=1) if graph_stage_cols else pd.Series(
            [0] * len(df))

        # Average and standard deviation for overall phases
        df_sums = pd.DataFrame(row_sums)
        bench_mean[name] = df_sums.mean()
        bench_std[name] = df_sums.std().fillna(0)

        # Identify all loaders in this file and force the expected order
        file_loaders = set(c.split('.')[0] for c in df.columns if 'Loader' in c.split('.')[0])
        ordered_file_loaders = order_columns(list(file_loaders), EXPECTED_LOADER_ORDER)

        ds_sums = {}
        for l in ordered_file_loaders:
            ds_cols = [c for c in df.columns if c.startswith(l)]
            if ds_cols:
                ds_sums[l] = df[ds_cols].sum(axis=1)

        df_ds_sums = pd.DataFrame(ds_sums)
        dataset_bench_mean[name] = df_ds_sums.mean()
        dataset_bench_std[name] = df_ds_sums.std().fillna(0)

    # Helper to wrap labels
    def wrap_labels(labels, width=15):
        return [textwrap.fill(label, width) for label in labels]

    def create_plot(df_mean, df_std, title, ylabel, filename, stacked=False, color_map=None, palette=None):
        if df_mean.empty:
            return

        # Prepare data for plotnine
        df_mean_reset = df_mean.reset_index().rename(columns={'index': 'Experiment'})
        df_std_reset = df_std.reset_index().rename(columns={'index': 'Experiment'})

        melted_mean = df_mean_reset.melt(id_vars='Experiment', var_name='Category', value_name='Value')
        melted_std = df_std_reset.melt(id_vars='Experiment', var_name='Category', value_name='Std')

        plot_data = pd.merge(melted_mean, melted_std, on=['Experiment', 'Category'])

        # Wrap Category labels
        plot_data['Category'] = plot_data['Category'].apply(lambda x: textwrap.fill(x, 35))

        # Enforce category order if needed
        if color_map:
            wrapped_color_map = {textwrap.fill(k, 35): v for k, v in color_map.items()}
            categories = [c for c in wrapped_color_map.keys() if c in plot_data['Category'].unique()]
            plot_data['Category'] = pd.Categorical(plot_data['Category'], categories=categories, ordered=True)
            active_palette = wrapped_color_map
        elif palette:
            wrapped_expected = [textwrap.fill(c, 35) for c in EXPECTED_LOADER_ORDER]
            categories = [c for c in wrapped_expected if c in plot_data['Category'].unique()]
            others = [c for c in plot_data['Category'].unique() if c not in categories]
            categories.extend(others)
            plot_data['Category'] = pd.Categorical(plot_data['Category'], categories=categories, ordered=True)
            active_palette = palette

        # Wrap Experiment labels for X-axis
        plot_data['Experiment'] = plot_data['Experiment'].apply(lambda x: textwrap.fill(x, 12))

        # Calculate ymin/ymax for error bars
        if stacked:
            # Sort Category in reverse (False) to match Plotnine's bottom-to-top stacking order
            plot_data = plot_data.sort_values(by=['Experiment', 'Category'], ascending=[True, False])
            plot_data['cum_value'] = plot_data.groupby('Experiment')['Value'].cumsum()
            plot_data['ymin'] = plot_data['cum_value'] - plot_data['Std']
            plot_data['ymax'] = plot_data['cum_value'] + plot_data['Std']
        else:
            plot_data['ymin'] = plot_data['Value'] - plot_data['Std']
            plot_data['ymax'] = plot_data['Value'] + plot_data['Std']

        plot_data['ymin'] = plot_data['ymin'].clip(lower=0)

        plot = ggplot(plot_data, aes(x='Experiment', y='Value', fill='Category'))

        # Removed color='Category' from aes() and added color='black' directly to geom_errorbar
        if stacked:
            plot += geom_bar(stat='identity', position='stack', width=0.6)
            plot += geom_errorbar(aes(ymin='ymin', ymax='ymax'), position='identity', width=0.4, size=0.8,
                                  color='black')
        else:
            plot += geom_bar(stat='identity', position=position_dodge(width=0.9), width=0.8)
            plot += geom_errorbar(aes(ymin='ymin', ymax='ymax'), position=position_dodge(width=0.9), width=0.4,
                                  size=0.8, color='black')

        plot += scale_fill_manual(values=active_palette)

        plot += labs(
            title=textwrap.fill(title, width=55),
            x='Experiment',
            y=ylabel,
            fill=''
        )

        plot += theme_minimal()
        plot += theme(
            plot_title=element_text(fontproperties=title_font, ha='center'),
            axis_title_x=element_text(fontproperties=bold_font),
            axis_title_y=element_text(fontproperties=bold_font),
            # axis_text_x=element_text(fontproperties=font, angle=45, ha='right'),
            axis_text_x=element_text(fontproperties=font, ha='center'),
            axis_text_y=element_text(fontproperties=font),
            legend_text=element_text(fontproperties=font),
            legend_position='bottom',
            legend_direction='horizontal',
            legend_box_margin=10
        )

        # Programmatically dictate row count based on number of items
        # (1 row if it fits, 2 if it's too long)
        unique_cats = len(plot_data['Category'].unique())
        legend_rows = 2 if unique_cats > 3 else 1
        plot += guides(fill=guide_legend(nrow=legend_rows, byrow=True))

        output_path = os.path.join(output_dir, f'{experiment_path}/{filename}')
        plot.save(output_path, dpi=150, width=12, height=8, units='in', verbose=False)
        print(f"Saved plot to {output_path}")

    # 1. Plot Execution Time by Phase
    df_mean = pd.DataFrame(bench_mean).T.fillna(0)
    df_std = pd.DataFrame(bench_std).T.fillna(0)

    # Filter out phases that are all 0
    df_mean = df_mean.loc[:, (df_mean != 0).any(axis=0)]
    df_std = df_std.loc[:, df_mean.columns]

    create_plot(
        df_mean, df_std,
        f'Comparison of Execution Time by Phase for {title}',
        'Time (s)',
        'benchmark_execution_times.png',
        stacked=True,
        color_map=COLOR_MAP
    )
    save_table(df_mean, df_std, 'benchmark_execution_times', output_dir, experiment_path, 'Execution Time by Phase (s)')

    # 2. Plot Execution Time by Dataset
    df_ds_mean = pd.DataFrame(dataset_bench_mean).T.fillna(0)
    df_ds_std = pd.DataFrame(dataset_bench_std).T.fillna(0)

    create_plot(
        df_ds_mean, df_ds_std,
        f'Comparison of Execution Time by Dataset for {title}',
        'Time (s)',
        'dataset_execution_times.png',
        stacked=False,
        palette=CB_PALETTE
    )
    save_table(df_ds_mean, df_ds_std, 'dataset_execution_times', output_dir, experiment_path,
               'Execution Time by Dataset (s)')

    # Memory Arrays
    mem_mean, mem_std = {}, {}
    dataset_mem_mean, dataset_mem_std = {}, {}

    for file in mem_files:
        name = os.path.basename(file).replace('memory_', '').replace('benchmark_', '').replace('.csv', '')
        if rename_map and name in rename_map:
            name = rename_map[name]

        df = pd.read_csv(file)

        parse_cols = [c for c in df.columns if 'parse_data' in c]
        norm_cols = [c for c in df.columns if 'normalise_data' in c]
        clust_cols = [c for c in df.columns if 'Clusterer' in c]
        serial_cols = [c for c in df.columns if 'OntologyBuilder' in c]
        db_cols = [c for c in df.columns if 'DBManager' in c]
        graph_manager_cols = [c for c in df.columns if 'GraphManager' in c]
        graph_final_cols = [c for c in df.columns if c in ['Graph building', 'Graph dumping']]

        row_maxes = {
            'Parsing/ingestion': df[parse_cols].max(axis=1) if parse_cols else pd.Series([0] * len(df)),
            'Normalisation': df[norm_cols].max(axis=1) if norm_cols else pd.Series([0] * len(df)),
            'Clustering': df[clust_cols].max(axis=1) if clust_cols else pd.Series([0] * len(df)),
            'Serialisation': df[serial_cols].max(axis=1) if serial_cols else pd.Series([0] * len(df)),
        }

        if db_cols:
            row_maxes['Database/Graph staging'] = df[db_cols].max(axis=1)
            row_maxes['Final graph construction'] = df[graph_manager_cols].max(
                axis=1) if graph_manager_cols else pd.Series([0] * len(df))
        else:
            row_maxes['Database/Graph staging'] = df[graph_manager_cols].max(
                axis=1) if graph_manager_cols else pd.Series([0] * len(df))
            row_maxes['Final graph construction'] = df[graph_final_cols].max(axis=1) if graph_final_cols else pd.Series(
                [0] * len(df))

        df_maxes = pd.DataFrame(row_maxes)
        mem_mean[name] = df_maxes.mean()
        mem_std[name] = df_maxes.std().fillna(0)

        # Identify loaders and map dynamically while adhering to fixed expectation
        file_loaders = set(c.split('.')[0] for c in df.columns if 'Loader' in c.split('.')[0])
        ordered_file_loaders = order_columns(list(file_loaders), EXPECTED_LOADER_ORDER)

        ds_maxes = {}
        for l in ordered_file_loaders:
            ds_cols = [c for c in df.columns if c.startswith(l)]
            if ds_cols:
                ds_maxes[l] = df[ds_cols].max(axis=1)

        df_ds_maxes = pd.DataFrame(ds_maxes)
        dataset_mem_mean[name] = df_ds_maxes.mean()
        dataset_mem_std[name] = df_ds_maxes.std().fillna(0)

    # 3. Plot Peak Memory by Phase
    df_mem_mean = pd.DataFrame(mem_mean).T.fillna(0)
    df_mem_std = pd.DataFrame(mem_std).T.fillna(0)

    # Filter out phases that are all 0
    df_mem_mean = df_mem_mean.loc[:, (df_mem_mean != 0).any(axis=0)]
    df_mem_std = df_mem_std.loc[:, df_mem_mean.columns]

    create_plot(
        df_mem_mean, df_mem_std,
        f'Comparison of Peak Memory Usage by Phase for {title}',
        'Memory (MB)',
        'memory_peak_usage.png',
        stacked=False,
        color_map=COLOR_MAP
    )
    save_table(df_mem_mean, df_mem_std, 'memory_peak_usage', output_dir, experiment_path,
               'Peak Memory Usage by Phase (MB)')

    # 4. Plot Peak Memory by Dataset
    df_ds_mem_mean = pd.DataFrame(dataset_mem_mean).T.fillna(0)
    df_ds_mem_std = pd.DataFrame(dataset_mem_std).T.fillna(0)

    create_plot(
        df_ds_mem_mean, df_ds_mem_std,
        f'Comparison of Peak Memory Usage by Source Dataset for {title}',
        'Memory (MB)',
        'dataset_memory_peak_usage.png',
        stacked=False,
        palette=CB_PALETTE
    )
    save_table(df_ds_mem_mean, df_ds_mem_std, 'dataset_memory_peak_usage', output_dir, experiment_path,
               'Peak Memory Usage by Dataset (MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graphs for execution times and memory usage.")
    parser.add_argument("--experiment", help="Path to experiment folder")
    parser.add_argument("--title", help="Graph title")

    args = parser.parse_args()
    experiment_path = 'all'
    if args.experiment:
        experiment_path = args.experiment
    title = args.title if args.title else ''

    rename_map = {
        'bulk_db_5000_oxi_clustered_final': '5,000',
        'bulk_db_10000_oxi_clustered_final': '10,000',
        'bulk_db_25000_oxi_clustered_final': '25,000',
        'bulk_db_50000_oxi_clustered_final': '50,000',
        'bulk_db_100000_oxi_clustered_final': '100,000',
        'db_CN+WK+WN_oxi_clustered_final': 'CN+WK+WN (Oxigraph)',
        'db_CN+WK+WN_rdf_clustered_final': 'CN+WK+WN (RDFLib)',
        'db_CN_oxi_clustered_final': 'CN Only (Oxigraph)',
        'db_CN_rdf_clustered_final': 'CN Only (RDFLib)',
        'db_oxi_clustered_final': 'Database (Oxigraph)',
        'db_rdf_clustered_final': 'Database (RDFLib)',
        'graph_oxi_clustered_final': 'Graph (Oxigraph)',
        'graph_rdf_clustered_final': 'Graph (RDFLib)',
        'db_oxi_direct_to_turtle_clustered_final': 'Database (.ttl)'

    }

    plot_benchmarks(rename_map=rename_map, experiment_path=experiment_path, title=title)