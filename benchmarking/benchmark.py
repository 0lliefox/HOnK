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
        
        # Initialize shared storage for this benchmark name if not present
        if name not in Benchmark._data:
            Benchmark._data[name] = []
            Benchmark._phase_names[name] = []
            Benchmark._phase_names_set[name] = set()
            Benchmark._id_maps[name] = {}
            
        self.data = Benchmark._data[name]
        self.phase_names = Benchmark._phase_names[name]
        self.phase_names_set = Benchmark._phase_names_set[name]
        self.id_map = Benchmark._id_maps[name]

    def add_row(self, id: Any, phase_name: str, value: Any, accumulate: bool=False) -> None:
        if self.name == "Metrics":
            value = f"{round(value, 2) if isinstance(value, float) else value:.2f}" if value != "N/A" else value

        # O(1) check for phase name
        if phase_name not in self.phase_names_set:
            self.phase_names.append(phase_name)
            self.phase_names_set.add(phase_name)

        # O(1) lookup for existing row
        if id in self.id_map:
            row = self.id_map[id]
            if accumulate and phase_name in row:
                row[phase_name] += value
            else:
                row[phase_name] = value
        else:
            # Create new row
            new_row = {'id': id, phase_name: value}
            
            # Backfill missing phases in the new row with 0.0 (optional, but consistent with previous logic)
            # Note: The previous logic also backfilled the *previous* row with 0.0 for new phases.
            # That part is a bit complex to maintain efficiently and might not be strictly necessary 
            # if we handle missing values during CSV export. 
            # However, to preserve exact behavior:
            
            # The previous code did:
            # if self.data:
            #    previous_row = self.data[-1]
            #    for existing_phase in self.phase_names:
            #        if existing_phase not in new_row: new_row[existing_phase] = 0.0
            #    for phase in new_row:
            #        if phase != 'id' and phase not in previous_row: previous_row[phase] = 0.0
            
            # This logic seems to try to keep rows "dense". 
            # But CSV export handles missing keys by using .get(phase, 0.0).
            # So we can skip the expensive backfilling logic here and rely on to_csv.
            
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
        mode = 'a' if append else 'w'

        try:
            with open(filename, mode, newline='') as csvfile:
                header = ['id'] + self.phase_names
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
            # Note: We don't clear phase_names because we might want to keep the columns consistent?
            # The original code didn't clear phase_names.
            # But if we clear data, we might want to start fresh?
            # The original code only cleared self.data.clear().
            # So I will stick to that, but also clear id_map.
