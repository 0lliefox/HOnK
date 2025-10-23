import json

# From GeoNames hierarchy and alternate names table, create mappings to use when adding to ontology
if __name__ == '__main__':
    hierarchy = {}
    with open('geonames/hierarchy.txt') as f:
        lines = f.readlines()
        for line in lines:
            parent, child = line.strip().split('\t')[:2]
            if parent != child:
                if parent in hierarchy:
                    hierarchy[parent].append(child)
                else:
                    hierarchy[parent] = [child]

    with open('geonames/hierarchy.json', 'w') as f:
        f.write(json.dumps(hierarchy))

    alternates = {}
    with open('geonames/alternateNamesV2.csv') as f:
        lines = f.readlines()
        for line in lines:
            try:
                alt_id, geo_id, code = line.strip().split('\t')[:3]
                if code not in {'link', 'wkdt'}:
                    if alt_id not in alternates:
                        alternates[alt_id] = geo_id
                    else:
                        alternates[alt_id].append(geo_id)
            except Exception as e:
                print(e, line)

    with open('geonames/alternates_map.json', 'w') as f:
        f.write(json.dumps(alternates))

    features = {}
    with open('geonames/feature_codes.csv') as f:
        lines = f.readlines()
        for line in lines:
            try:
                feature_code, feature_instance = line.strip().split('\t')[:2]
                features[feature_code] = feature_instance
            except Exception as e:
                print(e, line)

    with open('geonames/feature_map.json', 'w') as f:
        f.write(json.dumps(features))