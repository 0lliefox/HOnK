import csv
import os
from typing import Dict, List, Set, Any


class Benchmark:
    _data: Dict[str, List[Dict[str, Any]]] = {}
    _phase_names: Dict[str, List[str]] = {}
    _phase_names_set: Dict[str, Set[str]] = {}
    _id_maps: Dict[str, Dict[Any, Dict[str, Any]]] = {}

    def __init__(self, name: str = "default"):
        self.name = name

        # Initialise shared storage for this benchmark name if not present
        if name not in Benchmark._data:
            Benchmark._data[name] = []
            Benchmark._phase_names[name] = []
            Benchmark._phase_names_set[name] = set()
            Benchmark._id_maps[name] = {}

        self.data = Benchmark._data[name]
        self.phase_names = Benchmark._phase_names[name]
        self.phase_names_set = Benchmark._phase_names_set[name]
        self.id_map = Benchmark._id_maps[name]

    def add_row(self, id: Any, phase_name: str, value: Any, accumulate: bool = False, mode: str = None) -> None:
        if self.name == "Metrics":
            value = f"{round(value, 2) if isinstance(value, float) else value:.2f}" if value != "N/A" else value

        if phase_name not in self.phase_names_set:
            self.phase_names.append(phase_name)
            self.phase_names_set.add(phase_name)

        if id in self.id_map:
            row = self.id_map[id]

            # If mode is max, only overwrite if the new value is greater
            if mode == 'max' and phase_name in row:
                row[phase_name] = max(row[phase_name], value)
            elif accumulate and phase_name in row:
                row[phase_name] += value
            else:
                row[phase_name] = value

        else:
            new_row = {'id': id, phase_name: value}

            self.data.append(new_row)
            self.id_map[id] = new_row

    def to_csv(self, filename='benchmark_results', data_length=True, append=True) -> None:
        if not self.data:
            print("No data to export.")
            return

        filename = f"benchmarking/results/{filename}{f'_{len(self.data)}' if data_length else ''}.csv"

        if not os.path.isdir('benchmarking/results'):
            os.mkdir('benchmarking/results')

        file_exists = os.path.isfile(filename)
        header = ['id'] + self.phase_names

        # Appending rows written in this run's phase order onto a file whose header
        # was written by a different run silently misaligns every column (and can
        # mix results from two different builds). If the schema has changed, keep
        # the old results under a .legacy name and start a fresh, well-formed file.
        if file_exists and append:
            with open(filename, newline='') as existing_file:
                existing_header = next(csv.reader(existing_file), None)
            if existing_header != header:
                stem = filename[:-4]
                legacy, n = f"{stem}.legacy.csv", 1
                while os.path.isfile(legacy):
                    legacy = f"{stem}.legacy{n}.csv"
                    n += 1
                os.rename(filename, legacy)
                print(f"WARNING: benchmark schema changed for {filename}; "
                      f"previous results moved to {legacy}")
                file_exists = False

        mode = 'a' if append else 'w'

        try:
            with open(filename, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not file_exists or not append:
                    writer.writeheader()
                for row in self.data:
                    full_row = {'id': row['id']}
                    for phase in self.phase_names:
                        full_row[phase] = row.get(phase, 0.0)
                    writer.writerow(full_row)
            print(f"Data successfully written to {filename}")
        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")
        finally:
            # Clear data after writing to avoid duplicate entries on subsequent calls
            self.data.clear()
            self.id_map.clear()