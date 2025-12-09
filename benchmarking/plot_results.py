import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_line, geom_point,
    scale_y_log10, scale_color_brewer, scale_shape_manual,
    labs, theme_minimal, theme, element_text
)
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit


class ResultsPlotter:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.melted_data = None
        self.plot_object = None
        self.phase_columns = [
            'build_adj_list',
            'transitive_closure',
            'build_clusters_from_adj',
            'add_unclustered_concepts',
            'store_clusters_in_db',
            'coalesce_relationships'
        ]

        self.font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=11)
        self.bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=11)
        self.title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=14)

    def load_and_process_data(self):
        self.data = pd.read_csv(self.file_path)
        print("Data Head:")
        print(self.data.head())
        print("\nData Info:")
        self.data.info()

        self.data = self.data.rename(columns={'id': 'Percentage'})
        self.x_data = self.data['Percentage'].values
        self.data['Total'] = self.data[self.phase_columns].sum(axis=1)

        columns_to_melt = self.phase_columns + ['Total']
        self.melted_data = self.data.melt(
            id_vars=['Percentage'],
            value_vars=columns_to_melt,
            var_name='Phase',
            value_name='Time'
        )

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

            if len(x_fit) < 3:
                print(f"{phase:<25} -> Skipping (Not enough valid data points)")
                continue

            ss_tot = np.sum((y_fit_data - np.mean(y_fit_data)) ** 2)
            if ss_tot == 0:
                print(f"{phase:<25} -> Constant (All values are identical)")
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
                print(f"{phase:<25} -> No models could be successfully fitted.")
                continue

            # Sort by R² value, descending
            all_fits.sort(key=lambda f: f["r2"], reverse=True)

            print(f"{phase:<25}")
            for i, fit in enumerate(all_fits[:3]):
                print(f"  {i + 1}. {fit['model']:<25} (R² = {fit['r2']:.4f})")

    def create_plot(self):
        markers = ['o', 's', 'v', '^', 'D', 'P', 'X']

        self.plot_object = (
                ggplot(self.melted_data, aes(x='Percentage', y='Time', color='Phase', shape='Phase'))
                + geom_line()
                + geom_point(size=3)
                + scale_y_log10()
                + scale_color_brewer(type='qual', palette='Dark2')
                + scale_shape_manual(values=markers)
                + labs(
            title='Phase Execution Times vs. Dataset Percentage',
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


def main():
    file_path = 'results/clustering_benchmark.csv'

    plotter = ResultsPlotter(file_path)
    plotter.run()


if __name__ == "__main__":
    main()
