import pytest
import yaml
import copy
import os
import sys
import subprocess
import logging

from benchmarking.compare_graphs import GraphComparator

# Repo root (this file lives in tests/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def config_setup():
    with open(os.path.join(REPO_ROOT, 'config.yaml'), 'r') as f:
        base_config = yaml.safe_load(f)

    temp_configs = []
    yield base_config, temp_configs

    # Cleanup temp configs written per build.
    for path in temp_configs:
        if os.path.exists(path):
            os.remove(path)


def _write_temp_config(base_config, mode, output_file):
    """Materialise a per-mode config with the equivalence-test overrides."""
    cfg = copy.deepcopy(base_config)
    cfg['general']['mode'] = mode
    cfg['general']['should_cache'] = True
    cfg['db_config']['confirm_clear_db'] = False

    cfg.setdefault('clustering', {})['enabled'] = True

    te = cfg.setdefault('turtle_export', {})
    te['convert'] = False
    te['normalise_pos'] = True
    te['output_file'] = output_file

    path = os.path.join(REPO_ROOT, f'temp_config_{mode}_test.yaml')
    with open(path, 'w') as f:
        yaml.dump(cfg, f)
    return path


def _build_in_subprocess(base_config, mode, output_file, run_id, temp_configs):
    """Run one build in its OWN process.

    The two modes used to run in a single pytest process (graph build then db
    build in a loop). On the full 5-source ontology that made peak memory the
    *sum* of both builds' in-memory graphs (~10 GB .nt each), which OOM-killed
    the process at db-mode serialisation with no traceback. Isolating each build
    in a subprocess caps peak memory at a single build's footprint, which is
    known to fit; it also turns a silent kill into a non-zero return code the
    test can assert on.
    """
    config_path = _write_temp_config(base_config, mode, output_file)
    temp_configs.append(config_path)
    logging.info("Building %s mode in subprocess -> %s", mode, output_file)
    proc = subprocess.run(
        [sys.executable, 'build_ontology.py', '--config', config_path, '--run-id', str(run_id)],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"{mode}-mode build exited with code {proc.returncode} "
        f"(config {config_path}); check for OOM/kill or a build error."
    )


def test_graph_db_equivalence(config_setup):
    base_config, temp_configs = config_setup

    graph_filename = 'ontology_graph_clustered_test.nt'
    db_filename = 'ontology_db_clustered_test.nt'

    # Each build in a separate process (memory isolation — see _build_in_subprocess).
    _build_in_subprocess(base_config, 'graph', graph_filename, run_id=0, temp_configs=temp_configs)
    _build_in_subprocess(base_config, 'db', db_filename, run_id=1, temp_configs=temp_configs)

    path_graph = os.path.join(REPO_ROOT, 'ontologies', graph_filename)
    path_db = os.path.join(REPO_ROOT, 'ontologies', db_filename)

    assert os.path.exists(path_graph), f"Graph mode output not found at {path_graph}"
    assert os.path.exists(path_db), f"DB mode output not found at {path_db}"

    # Compare graphs. Pass the ontology so the comparator normalises the arbitrary
    # {rel}{counter} numbering of reified relation-instance subjects (its designed
    # conditional-subject normalisation, keyed on the Property list); without it the
    # comparison spuriously flags order-dependent instance IDs that carry no meaning.
    # light=True skips the memory-heavy degree/union-find analytics — on an 80M+
    # triple ontology those OOM, and they are not part of the equivalence signal.
    comparator = GraphComparator(path_graph, path_db, format1='nt', format2='nt', name1='Graph', name2='Database',
                                 ontology_path=os.path.join(REPO_ROOT, base_config['local_files']['ontology_classes']),
                                 light=True)

    comparator._process_graph(comparator.graph1_path, comparator.format1, 'g1', 1)
    comparator._process_graph(comparator.graph2_path, comparator.format2, 'g2', 2)

    # Check basic stats equality
    stats1 = comparator.stats['g1']
    stats2 = comparator.stats['g2']

    assert stats1['triples'] == stats2['triples'], f"Triple counts differ: {stats1['triples']} vs {stats2['triples']}"
    assert stats1['nodes'] == stats2['nodes'], f"Node counts differ: {stats1['nodes']} vs {stats2['nodes']}"
    assert stats1['relations'] == stats2['relations'], f"Relation counts differ: {stats1['relations']} vs {stats2['relations']}"

    # Check unique samples
    comparator._find_unique_samples()

    unique_g1 = len(comparator.unique_samples['g1'])
    unique_g2 = len(comparator.unique_samples['g2'])

    if unique_g1 > 0:
        print("Unique to Graph Mode:")
        for t in comparator.unique_samples['g1']:
            print(t)

    if unique_g2 > 0:
        print("Unique to DB Mode:")
        for t in comparator.unique_samples['g2']:
            print(t)

    assert unique_g1 == 0, f"Found {unique_g1} triples unique to Graph mode"
    assert unique_g2 == 0, f"Found {unique_g2} triples unique to DB mode"

    print("Graphs are equivalent.")
