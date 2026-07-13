import json

# Used for debugging adjacency list coming from clustering
if __name__ == '__main__':
    with open('../.cache/adj_list_graph.json', 'r') as file:
        data = json.load(file)

    array_lengths = []
    for key, value in data.items():
        if isinstance(value, list):
            array_lengths.append((key, len(value)))

    sorted_arrays = sorted(array_lengths, key=lambda item: item[1], reverse=True)
    print(sorted_arrays[:10])
