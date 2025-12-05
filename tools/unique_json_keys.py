import json

def get_unique_keys(data, keys_set, recursive=True):
    if isinstance(data, dict):
        for key, value in data.items():
            keys_set.add(key)
            if recursive:
                get_unique_keys(value, keys_set, recursive)

    elif isinstance(data, list):
        for item in data:
            get_unique_keys(item, keys_set, recursive)

if __name__ == '__main__':
    with open('../supporting_files/kb/Wiktionary.json', 'r') as f:
        data = json.load(f)

        unique_keys = set()
        get_unique_keys(data, unique_keys, False)

        sorted_keys = sorted(list(unique_keys))
        print(sorted_keys)