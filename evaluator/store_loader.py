import csv
import hashlib
import io
import json
import logging
import pickle
import urllib.parse
from pathlib import Path
from typing import Any, Dict

import pyoxigraph

logger = logging.getLogger(__name__)


def shorten(uri_value: str) -> str:
    return uri_value.split('/')[-1].split('#')[-1]


def get_file_hash(file_path: str) -> str:
    path = Path(file_path)
    if not path.is_file():
        return ""
    stat = path.stat()
    # Key on resolved path + size + mtime. size guards against a content change
    # that preserves mtime (e.g. touch -r, restore-from-archive), which the
    # earlier mtime-only key would have served stale; resolve() normalises `..`
    # so the same physical file yields one key regardless of the invocation cwd.
    return hashlib.md5(
        f"{path.resolve()}_{stat.st_size}_{stat.st_mtime}".encode()
    ).hexdigest()


def parse_csv_to_ntriples(file_path: Path, base_uri: str) -> bytes:
    logger.info("Converting CSV to N-Triples: '%s'", file_path)
    buf = io.BytesIO()
    with file_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue
            try:
                data_json = json.loads(row[4])
                s_str = data_json.get("surfaceStart")
                if not s_str:
                    parts = row[2].split('/')
                    s_str = parts[3] if len(parts) > 3 else row[2]
                o_str = data_json.get("surfaceEnd")
                if not o_str:
                    parts = row[3].split('/')
                    o_str = parts[3] if len(parts) > 3 else row[3]
                p_str = row[1].replace("/r/", "")
                s = f"<{base_uri}concept/{urllib.parse.quote(str(s_str).strip().replace(' ', '_'))}>"
                p = f"<{base_uri}relation/{urllib.parse.quote(str(p_str).strip().replace(' ', '_'))}>"
                o = f"<{base_uri}concept/{urllib.parse.quote(str(o_str).strip().replace(' ', '_'))}>"
                if not all([s, p, o]):
                    continue
                buf.write(f"{s} {p} {o} .\n".encode('utf-8'))
            except Exception:
                continue
    return buf.getvalue()


def load_graph(file_path: str, base_uri: str, cache_dir: str) -> pyoxigraph.Store:
    store = pyoxigraph.Store()
    path = Path(file_path)

    if not path.is_file():
        logger.warning("Ontology file '%s' not found. Returning empty store.", file_path)
        return store

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    stat = path.stat()
    file_hash = hashlib.md5(f"{path.absolute()}_{stat.st_mtime}".encode()).hexdigest()
    cache_file = cache / f"{path.stem}_{file_hash}.pkl"
    rdf_format = (
        pyoxigraph.RdfFormat.TURTLE
        if path.suffix == ".ttl"
        else pyoxigraph.RdfFormat.N_TRIPLES
    )

    if cache_file.exists():
        logger.info("Loading '%s' from cache.", file_path)
        try:
            with cache_file.open('rb') as f:
                data = pickle.load(f)
            if path.suffix == ".csv":
                rdf_format = pyoxigraph.RdfFormat.N_TRIPLES
            store.load(io.BytesIO(data), rdf_format)
            return store
        except Exception as e:
            logger.warning("Cache load failed, reading source: %s", e)

    try:
        # For large RDF files (e.g. the 10 GB+ HOnK build) stream directly from
        # disk into the store. Reading the whole file into a bytes buffer AND
        # pickling it roughly doubles peak memory on top of the store itself and
        # can OOM a 48 GB machine; the pickle cache only pays off for the small,
        # derived (csv) case. Streaming keeps peak at the store's own footprint.
        LARGE = 2 * 1024 ** 3  # 2 GB
        if path.suffix != ".csv" and stat.st_size > LARGE:
            with path.open('rb') as f:
                store.load(f, rdf_format)
            logger.info("Loaded '%s' (streamed, uncached).", file_path)
            return store

        if path.suffix == ".csv":
            data = parse_csv_to_ntriples(path, base_uri)
            rdf_format = pyoxigraph.RdfFormat.N_TRIPLES
        else:
            with path.open('rb') as f:
                data = f.read()
        with cache_file.open('wb') as f:
            pickle.dump(data, f)
        store.load(io.BytesIO(data), rdf_format)
        logger.info("Loaded '%s'.", file_path)
    except Exception as e:
        # Do NOT return a half/empty store: a corrupt or unreadable graph would
        # then be scored as all-zeros and silently corrupt the comparison. Fail
        # loudly so the run aborts rather than reporting fabricated results.
        raise RuntimeError(f"Failed to load ontology '{file_path}': {e}") from e

    return store
