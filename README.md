# <img src="honk_logo.svg" style="height:80px; width: auto;" alt="Logo: Credits to Oliver Robert Fox (2024)" /> HOnK (Hub Ontology for Knowledge)

Project for combining multiple knowledge bases into one unified ontology, to aid research in achieving better type resolution for work carried out in [LaSSI](https://github.com/LogDS/LaSSI).

## Setting Up
### 1. Database (PostgreSQL)
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
The following knowledge bases are used. The download script fetches and extracts all files in the correct format:
```bash
./download_knowledge.sh
```
Files used in our experiments can also be found at OSF.io: https://osf.io/8mqs4/?view_only=282c38027c8043d5abd76a98001c31fa

#### Knowledge Bases

| KB | Description | Status |
|----|-------------|--------|
| [Parmenides](supporting_files/parmenides/) | Structured linguistic/logical resource providing nouns, verbs, adjectives, locations, dependency roles, and typed inter-concept relations for NLP type resolution | Active (default) |
| [ConceptNet](https://github.com/commonsense/conceptnet5/) | Commonsense relational knowledge; English Wiktionary dump from 2025-09-20 used in experiments | Active |
| [Wiktionary](https://github.com/tatuylonen/wiktextract) | Pre-expanded definitions, POS tags, lemmas, and derived/related terms; filtered to English, from 2026-03-03 | Active |
| [WordNet](https://doi.org/10.5281/zenodo.3739540) | Lexical relations and synsets; used as a ConceptNet input to improve clustering | Active |
| [GeoNames](https://download.geonames.org/export/dump/) | Geographic entities, feature hierarchy, and alternate names | Active |
| [DBpedia](https://databus.dbpedia.org/dbpedia-enterprise/en-dbpedia-enriched-with-wikidata-dbpedia) | Entity labels, types, and properties enriched with Wikidata (dump 2025-08-21; truthy dump 2025-10-24) | Work in progress |

### 3. Install Packages and Run
Set up a virtual Python environment (3.10.x and 3.11.x tested), then:
```bash
pip install .
build-ontology
```

With optional arguments:
```bash
build-ontology --config custom_config.yaml --run-id 0
```

## Pipeline Overview
The pipeline is orchestrated by [`build_ontology.py`](build_ontology.py) and configured via [`config.yaml`](config.yaml). It runs in two interchangeable modes controlled by `general.mode`:

| Mode | Storage | Use case |
|------|---------|----------|
| `db` | PostgreSQL | Default; scalable to large KB combinations |
| `graph` | In-memory RDF | Faster feedback; no database required |

The graph backend is selected via `graph.type`: `oxigraph` (Pyoxigraph, preferred) or `rdf` (RDFLib).

### Stages

#### 1. Ingestion
Loaders in [`knowledge_bases/`](knowledge_bases/) parse source files and write concepts, relations, properties, and external URLs. In `db` mode data goes to PostgreSQL; in `graph` mode it goes directly into the in-memory store. Results are pickled to `.cache/` when `should_cache: true` to allow incremental re-runs.

#### 2. Clustering
Groups concepts that refer to the same real-world entity using shared external URLs. An adjacency list is built from co-occurring URLs, then transitive closure identifies connected components. In `db` mode this produces `clusters` and `cluster_relations` tables ([`clustering/cluster_concepts.py`](clustering/cluster_concepts.py)); in `graph` mode the graph is modified in place ([`clustering/cluster_graph_concepts_oxi.py`](clustering/cluster_graph_concepts_oxi.py) / [`_rdf.py`](clustering/cluster_graph_concepts_rdf.py)).

#### 3. Graph Construction
In `db` mode, [`build_ontology.py`](build_ontology.py) reads the database tables and converts rows to RDF triples via the `GraphManager`. In `graph` mode the store is already populated after ingestion and clustering. POS tag classes are added at this stage.

#### 4. Serialisation
The graph is exported to `.nt` (N-Triples) or optionally converted to `.ttl` (Turtle) via `rapper`. The base URI and namespace prefix are set in the `turtle_export` section of `config.yaml`.

#### 5. Benchmarking
A `@timer` decorator ([`tools/timer.py`](tools/timer.py)) wraps each pipeline stage to capture wall time and peak memory. Results accumulate in a `Benchmark` instance and are exported to CSVs in [`benchmarking/results/`](benchmarking/results/).

### Configuration
Key sections of `config.yaml`:

| Key | Description |
|-----|-------------|
| `general.sources` | List of KBs to load (default: `['parmenides']`) |
| `general.mapping_sources` | KBs used for edge-type mappings (default: `['wordnet', 'conceptnet', 'wiktionary']`) |
| `general.mode` | `'db'` or `'graph'` |
| `graph.type` | `'oxigraph'` or `'rdf'` |
| `clustering.enabled` | Toggle clustering stage |
| `db_config.clear_db_on_start` | Drop and recreate tables on each run |
| `turtle_export.convert` | Convert output `.nt` to `.ttl` after serialisation |

### DB Schema (DB mode)
Tables are created automatically. `clear_db_on_start: true` drops and recreates them before each run.

| Table | Description |
|-------|-------------|
| `concepts` | `(id, term, part_of_speech, source)` |
| `relations` | `(start_concept_id, end_concept_id, relation_type, weight, source)` |
| `properties` | `(concept_id, type, value, source)` |
| `urls` | `(concept_id, external_url, source)` |
| `clusters` | `(concept_id, cluster_id)` — added when clustering is enabled |
| `cluster_relations` | `(start_cluster_id, end_cluster_id, relation_type, weight)` — added when clustering is enabled |

## Evaluation
The [`evaluator/`](evaluator/) pipeline assesses ontology quality at three tiers:

1. **Deterministic** — triple coverage and precision metrics against reference triples
2. **Semantic** — embedding similarity using Sentence Transformers to score how well the ontology captures meaning
3. **LLM-as-judge** — locally hosted models via Ollama judge triple coherence and relevance

Supporting utilities: `triple_sampler.py` samples triples for evaluation; `extractor.py` builds subgraph context around them; `reporter.py` generates CSV exports and LaTeX tables for papers.

Evaluation settings (models, test cases, thresholds) are configured under the `evaluator` section of `config.yaml`.

## Multi-run Experiments
[`run_experiments.py`](run_experiments.py) reads per-run config overrides from [`configurations.json`](configurations.json) and merges them with `config.yaml` to produce isolated temporary configs. [`run_iterations.sh`](run_iterations.sh) drives batch execution across multiple iterations.

```bash
python run_experiments.py --iteration 0 --config-index 0
./run_iterations.sh
```

## Testing
The primary correctness test runs the full pipeline in both `db` and `graph` modes and compares output triple-by-triple:
```bash
pytest tests/ -v
pytest tests/test_graph_db_equivalence.py -v
```

## Future Work
- **DBpedia** — the loader ([`knowledge_bases/dbpedia_loader.py`](knowledge_bases/dbpedia_loader.py)) can ingest entity labels and types, but property-relation extraction is incomplete. The loader is currently disabled in the pipeline pending this.
