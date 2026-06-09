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
CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D4AC0D", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
COLOR_MAP = {
    'Parsing/ingestion': CB_PALETTE[0],
    'Normalisation': CB_PALETTE[1],
    'Clustering': CB_PALETTE[2],
    'Database/Graph staging': CB_PALETTE[5],
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
    combined_df = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)
    for col in df_mean.columns:
        combined_df[col] = df_mean[col].apply(lambda x: f"{x:.2f}") + " ± " + df_std[col].apply(lambda x: f"{x:.2f}")

    out_folder = os.path.join(output_dir, experiment_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    csv_path = os.path.join(out_folder, f'{filename_base}.csv')
    latex_path = os.path.join(out_folder, f'{filename_base}.tex')

    combined_df.to_csv(csv_path)

    # Manual LaTeX generation to avoid Jinja2 dependency
    header_cols = ["Experiment"] + list(combined_df.columns)
    header_str = " & ".join([f"\\textbf{{{col}}}" for col in header_cols]) + " \\\\"

    rows = []
    for idx, row in combined_df.iterrows():
        # Escape special characters in index if necessary
        safe_idx = str(idx).replace('_', '\\_')
        row_vals = [str(val) for val in row]
        row_str = f"{safe_idx} & {' & '.join(row_vals)} \\\\"
        rows.append(row_str)

    content = "\n        ".join(rows)
    col_spec = "l" + "c" * len(combined_df.columns)

    latex_code = f"""
\\begin{{table}}[h!]
    \\centering
    \\caption{{{caption}}}
    \\label{{tab:{filename_base}}}
    \\resizebox{{\\textwidth}}{{!}}{{
    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        {header_str}
        \\midrule
        {content}
        \\bottomrule
    \\end{{tabular}}
    }}
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
        db_cols = [c for c in df.columns if 'DBManager' in c]
        graph_stage_cols = [c for c in df.columns if 'GraphManager' in c]
        graph_final_cols = [c for c in df.columns if c in ['Graph building', 'Graph dumping']]

        row_sums = {
            'Parsing/ingestion': df[parse_cols].sum(axis=1) if parse_cols else pd.Series([0] * len(df)),
            'Normalisation': df[norm_cols].sum(axis=1) if norm_cols else pd.Series([0] * len(df)),
            'Clustering': df[clust_cols].sum(axis=1) if clust_cols else pd.Series([0] * len(df)),
            'Final graph construction': df[graph_final_cols].sum(axis=1) if graph_final_cols else pd.Series(
                [0] * len(df)),
        }

        if db_cols:
            row_sums['Database/Graph staging'] = df[db_cols].sum(axis=1)
        else:
            row_sums['Database/Graph staging'] = df[graph_stage_cols].sum(axis=1) if graph_stage_cols else pd.Series(
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
        plot_data['Category'] = plot_data['Category'].apply(lambda x: textwrap.fill(x, 25))

        # Enforce category order if needed
        if color_map:
            wrapped_color_map = {textwrap.fill(k, 25): v for k, v in color_map.items()}
            categories = [c for c in wrapped_color_map.keys() if c in plot_data['Category'].unique()]
            plot_data['Category'] = pd.Categorical(plot_data['Category'], categories=categories, ordered=True)
            active_palette = wrapped_color_map
        elif palette:
            wrapped_expected = [textwrap.fill(c, 25) for c in EXPECTED_LOADER_ORDER]
            categories = [c for c in wrapped_expected if c in plot_data['Category'].unique()]
            others = [c for c in plot_data['Category'].unique() if c not in categories]
            categories.extend(others)
            plot_data['Category'] = pd.Categorical(plot_data['Category'], categories=categories, ordered=True)
            active_palette = palette

        # Wrap Experiment labels for X-axis
        plot_data['Experiment'] = plot_data['Experiment'].apply(lambda x: textwrap.fill(x, 15))

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
            title=title,
            x='Experiment',
            y=ylabel,
            fill=''
        )

        plot += theme_minimal()
        plot += theme(
            plot_title=element_text(fontproperties=title_font, ha='center'),
            axis_title_x=element_text(fontproperties=bold_font),
            axis_title_y=element_text(fontproperties=bold_font),
            axis_text_x=element_text(fontproperties=font, angle=0, ha='center'),
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
        'db_clustered_final': 'Database',
        # 'db_clustered_final': 'Non-Unique Source in DB Schema',
        # 'db_clustered_final': 'Export to NT',
        'db_unclustered_final': 'Final (Database, Unclustered)',
        'db_CN+WK+WN_clustered_final': 'ConceptNet + Wiktionary + WordNet',
        'db_CN+WK+WN_unclustered_final': 'ConceptNet + Wiktionary + WordNet (Database, Unclustered)',
        'db_CN_clustered_final': 'ConceptNet Only',
        'db_CN_unclustered_final': 'ConceptNet Only (Unclustered)',
        'db_direct_to_turtle_clustered_final': 'Export to Turtle',
        'db_turtle_converted_clustered_final': 'Export to NT converted to Turtle',
        'db_unique_source_clustered_final': 'Unique Source in DB Schema',
        'db_unnormalised_clustered_final': 'Unnormalised (Clustered)',
        'db_unnormalised_unclustered_final': 'Unnormalised (Unclustered)',
        'graph_clustered_final': 'Graph',
        'graph_unclustered_final': 'Final (Graph, Unclustered)'

    }

    plot_benchmarks(rename_map=rename_map, experiment_path=experiment_path, title=title)