# HOnK (Hub Ontology for Knowledge)
Project for combining multiple knowledge bases into one, single ontology, to aid research in achieving better type resolution for work carried out in [LaSSI](https://github.com/LogDS/LaSSI).

## Setting Up
### 1.  Database (PostgreSQL)
PostgreSQL must be installed, along with a database and user:

#### Installation
*Linux (Ubuntu tested)*
```bash
sudo apt install postgresql -y
sudo -u postgres psql
```

*macOS*
```bash
brew install postgresql
psql postgres
```

#### Setup
```postgresql
create database ontology_db;
create user fox with encrypted password 'drowssap';
grant all privileges on database ontology_db to fox;
\c ontology_db fox
grant all on schema public to fox;
exit
```

### 2. Getting Supporting Files
For this particular research, the following Knowledge Bases (KBs) were used, this does not mean other KBs could not be incorporated to enhance the final ontology further.

The script below can be used, which will download and extract the files in the correct format: 
```bash
./download_knowledge.sh
```
Files used in our experiments can be found at OSF.io here: https://osf.io/8mqs4/?view_only=282c38027c8043d5abd76a98001c31fa, or can be retrieved manually using the methods below:
#### ConceptNet
[ConceptNet](https://github.com/commonsense/conceptnet5/) was downloaded where the English Wiktionary dump from 2025-09-20 was used in experiments.

#### Wiktionary
Pre-expanded data from [Wiktextract](https://github.com/tatuylonen/wiktextract?tab=readme-ov-file#pre-extracted-data), which was used for current experiments was from 2025-08-23, and was then filtered to only contain English terms.

#### WordNet
The WordNet dump file was originally found [here](https://doi.org/10.5281/zenodo.3739540), as an input for ConceptNet, therefore used here to ensure the best clustering possible.

#### GeoNames
Files for GeoNames can be found [here](https://download.geonames.org/export/dump/).

#### DBpedia
The artifacts found for [en-dbpedia-enriched-with-wikidata-dbpedia](https://databus.dbpedia.org/dbpedia-enterprise/en-dbpedia-enriched-with-wikidata-dbpedia) from 2025-08-21 were used, with [truthy dump](https://dumps.wikimedia.org/wikidatawiki/entities/) from 2025-10-24. (The implementation for DBpedia is not fully complete yet.)

### 3. Install Packages and Run
Setup a virtual Python environment (3.10.x and 3.11.x tested), and run:
```bash
pip install .
```
Then:
```bash
build-ontology
```

## Pipeline Overview
The ontology building process is orchestrated by [`build_ontology.py`](build_ontology.py) and can be configured via [`config.yaml`](config.yaml). The pipeline operates in two main modes: `db` and `graph`.

### 1. Data Ingestion
- **Loaders**: Data is ingested from various knowledge bases using loaders found in the [`knowledge_bases/`](knowledge_bases/) directory (e.g., `ConceptNetLoader`, `WiktionaryLoader`).
- **Mode-Specific Storage**:
    - In `db` mode, data is parsed and stored in a PostgreSQL database.
    - In `graph` mode, data is loaded directly into an in-memory graph (using `rdflib` or `pyoxigraph`).

### 2. Clustering
- **Purpose**: To identify and group concepts that refer to the same entity (e.g., concepts sharing the same external URL).
- **Process**:
    - An adjacency list is built based on shared URLs.
    - Transitive closure is computed on this list to find all connected concepts.
    - These connected components form the clusters.
- **Implementation**:
    - [`clustering/cluster_concepts.py`](clustering/cluster_concepts.py): Operates on the PostgreSQL database.
    - [`clustering/cluster_graph_concepts.py`](clustering/cluster_graph_concepts.py): Operates on the in-memory graph.

### 3. Graph Construction & Coalescing
- **DB Mode**:
    1. After clustering, [`clustering/cluster_concepts.py`](clustering/cluster_concepts.py) creates a `cluster_relations` table by joining the original `relations` with the new `clusters`.
    2. [`build_ontology.py`](build_ontology.py) then reads from the database tables (including `cluster_relations`) to construct the final RDF graph.
- **Graph Mode**:
    1. [`clustering/cluster_graph_concepts.py`](clustering/cluster_graph_concepts.py) directly modifies the graph.
    2. It identifies relationships between clusters and "coalesces" them, propagating relations to all member concepts within the clusters.

### 4. Benchmarking
- The pipeline is instrumented with a `@timer` decorator ([`tools/timer.py`](tools/timer.py)) to measure the performance of various stages.
- Results are saved to CSV files in `benchmarking/results/`.
