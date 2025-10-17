import csv
from typing import Dict, List

class Benchmark:
    _data: Dict[str, List[Dict[str, float]]] = {}
    _phase_names: Dict[str, List[str]] = {}

    def __init__(self, name: str = "default"):
        self.data = Benchmark._data
        self.phase_names = Benchmark._phase_names
        self.name = name
        if name not in Benchmark._data:
            Benchmark._data[name] = []
            Benchmark._phase_names[name] = []
        self.data = Benchmark._data[name]
        self.phase_names = Benchmark._phase_names[name]

    def add_row(self, id: Dict[int, str], phase_name: str, value: Dict[float, str], accumulate: bool=False) -> None:
        if self.name == "Metrics":
            value = f"{round(value, 2) if isinstance(value, float) else value:.2f}" if value != "N/A" else value

        if phase_name not in self.phase_names:
            self.phase_names.append(phase_name)

        for row in self.data:
            if row['id'] == id:
                if accumulate and phase_name in row:
                    row[phase_name] += value
                else:
                    row[phase_name] = value
                return

        new_row = {'id': id, phase_name: value}
        if self.data:
            previous_row = self.data[-1]
            for existing_phase in self.phase_names:
                if existing_phase not in new_row:
                    new_row[existing_phase] = 0.0
            for phase in new_row:
                if phase != 'id' and phase not in previous_row:
                    previous_row[phase] = 0.0
        self.data.append(new_row)

    def to_csv(self, filename: str = 'benchmark_results') -> None:
        if not self.data:
            print("No data to export.")
            return

        filename = f"benchmarking/results/{filename}_{len(self.data)}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                header = ['id'] + self.phase_names
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                for row in self.data:
                    full_row = {'id': row['id']}
                    for phase in self.phase_names:
                        full_row[phase] = row.get(phase, 0.0)
                    writer.writerow(full_row)
            print(f"Data successfully written to {filename}")
        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")