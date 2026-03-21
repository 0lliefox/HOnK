import hashlib
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from .context_builder import format_context
    from .extractor import compute_differential_triples, extract_bridging_triples, extract_subgraph
    from .metrics import compute_deterministic_metrics
    from .reporter import export_paper_tables, export_to_csv, generate_summary_table, report_global_statistics
    from .scorers import compute_embedding_similarity, ensure_models_pulled, query_llm
    from .store_loader import get_file_hash, load_graph
except ImportError:
    from context_builder import format_context  # type: ignore[no-redef]
    from extractor import compute_differential_triples, extract_bridging_triples, extract_subgraph  # type: ignore[no-redef]
    from metrics import compute_deterministic_metrics  # type: ignore[no-redef]
    from reporter import export_paper_tables, export_to_csv, generate_summary_table, report_global_statistics  # type: ignore[no-redef]
    from scorers import compute_embedding_similarity, ensure_models_pulled, query_llm  # type: ignore[no-redef]
    from store_loader import get_file_hash, load_graph  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class HonkEvaluator:

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        eval_cfg = self.config.get('evaluator', {})

        self.base_uri = eval_cfg.get('base_uri', 'http://example.org/ontology/')
        self.output_csv = eval_cfg.get('output_csv', 'evaluation_results.csv')
        self.cache_dir = self.config.get('local_files', {}).get('cache', '.cache')
        self.summary_table_output = eval_cfg.get('summary_table_output', 'evaluator/summary_table.txt')

        self.run_det = eval_cfg.get('run_deterministic_evaluation', True)
        self.run_semantic = eval_cfg.get('run_semantic_evaluation', True)
        self.run_llm = eval_cfg.get('run_llm_evaluation', False)

        self.bridging_limit = eval_cfg.get('bridging_limit', 100)
        self.max_context_triples = eval_cfg.get('max_context_triples', 75)
        self.max_bridge_triples = eval_cfg.get('max_bridge_triples', 20)
        self.max_extraction_workers = eval_cfg.get('max_extraction_workers', 4)

        import transformers
        transformers.logging.set_verbosity_error()
        model_names: List[str] = eval_cfg.get(
            'embedding_models',
            [eval_cfg.get('embedding_model', 'all-MiniLM-L6-v2')]
        )
        self.embedders: Dict[str, SentenceTransformer] = {}
        for name in model_names:
            logger.info("Loading sentence transformer: '%s'", name)
            self.embedders[name] = SentenceTransformer(name)
        transformers.logging.set_verbosity_warning()

        self.embedding_top_k: int = eval_cfg.get('embedding_top_k', 5)

        self.llm_models: List[str] = eval_cfg.get('llms', [])
        self.llm_iterations: int = eval_cfg.get('llm_iterations', 2)
        self.max_llm_workers: int = eval_cfg.get('max_parallel_queries', 2)

        if self.run_llm and self.llm_models:
            ensure_models_pulled(self.llm_models)

        ontologies = eval_cfg.get('ontologies', {})
        baseline_file = ontologies.get('baseline', '')
        honk_file = ontologies.get('honk', '')
        self.baseline_store = load_graph(baseline_file, self.base_uri, self.cache_dir)
        self.honk_store = load_graph(honk_file, self.base_uri, self.cache_dir)
        self._baseline_hash = get_file_hash(baseline_file)
        self._honk_hash = get_file_hash(honk_file)

        self.test_cases: List[Dict[str, Any]] = eval_cfg.get('test_cases', [])

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.is_file():
            logger.error("Configuration file '%s' not found.", config_path)
            sys.exit(1)
        try:
            with path.open('r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("YAML parsing error in '%s': %s", config_path, e)
            sys.exit(1)

    def _extraction_cache_key(self, store_hash: str, keywords: List[str], query_type: str) -> str:
        parts = [store_hash, ",".join(sorted(keywords)), query_type]
        if query_type == "bridging":
            parts.append(str(self.bridging_limit))
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _read_extraction_cache(self, key: str) -> Optional[Any]:
        cache_file = Path(self.cache_dir) / f"extraction_{key}.pkl"
        if cache_file.exists():
            try:
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _write_extraction_cache(self, key: str, value: Any) -> None:
        cache_file = Path(self.cache_dir) / f"extraction_{key}.pkl"
        try:
            with cache_file.open('wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning("Could not write extraction cache: %s", e)

    def _extract_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        keywords = case.get('keywords', [])

        k_b_sub = self._extraction_cache_key(self._baseline_hash, keywords, "subgraph")
        k_h_sub = self._extraction_cache_key(self._honk_hash,     keywords, "subgraph")
        k_b_br  = self._extraction_cache_key(self._baseline_hash, keywords, "bridging")
        k_h_br  = self._extraction_cache_key(self._honk_hash,     keywords, "bridging")

        b_triples = self._read_extraction_cache(k_b_sub)
        h_triples = self._read_extraction_cache(k_h_sub)
        b_bridges = self._read_extraction_cache(k_b_br)
        h_bridges = self._read_extraction_cache(k_h_br)

        missing: Dict[str, Any] = {}
        if b_triples is None: missing['b_sub'] = None
        if h_triples is None: missing['h_sub'] = None
        if b_bridges is None: missing['b_br'] = None
        if h_bridges is None: missing['h_br'] = None

        if missing:
            with ThreadPoolExecutor(max_workers=self.max_extraction_workers) as pool:
                futures = {}
                if 'b_sub' in missing:
                    futures['b_sub'] = pool.submit(extract_subgraph, self.baseline_store, keywords)
                if 'h_sub' in missing:
                    futures['h_sub'] = pool.submit(extract_subgraph, self.honk_store, keywords)
                if 'b_br' in missing:
                    futures['b_br'] = pool.submit(
                        extract_bridging_triples, self.baseline_store, keywords, self.bridging_limit
                    )
                if 'h_br' in missing:
                    futures['h_br'] = pool.submit(
                        extract_bridging_triples, self.honk_store, keywords, self.bridging_limit
                    )
                if 'b_sub' in futures:
                    b_triples = futures['b_sub'].result()
                    self._write_extraction_cache(k_b_sub, b_triples)
                if 'h_sub' in futures:
                    h_triples = futures['h_sub'].result()
                    self._write_extraction_cache(k_h_sub, h_triples)
                if 'b_br' in futures:
                    b_bridges = futures['b_br'].result()
                    self._write_extraction_cache(k_b_br, b_bridges)
                if 'h_br' in futures:
                    h_bridges = futures['h_br'].result()
                    self._write_extraction_cache(k_h_br, h_bridges)
        else:
            logger.debug("Extraction cache hit for keywords: %s", keywords)

        diff_set = compute_differential_triples(b_triples, h_triples)

        return {
            'sentence': case.get('sentence', ''),
            'label': case.get('label', ''),
            'keywords': keywords,
            'b_triples': b_triples,
            'h_triples': h_triples,
            'b_bridges': b_bridges,
            'h_bridges': h_bridges,
            'diff_set': diff_set,
            'b_det': (
                compute_deterministic_metrics(b_triples, keywords, b_bridges)
                if self.run_det else {}
            ),
            'h_det': (
                compute_deterministic_metrics(h_triples, keywords, h_bridges)
                if self.run_det else {}
            ),
            'b_ctx': format_context(
                b_triples, keywords, b_bridges,
                self.max_context_triples, self.max_bridge_triples
            ),
            'h_ctx': format_context(
                h_triples, keywords, h_bridges,
                self.max_context_triples, self.max_bridge_triples
            ),
        }

    def _run_embedding_phase(
        self,
        processed_cases: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        sentence_sim: Dict[Tuple[str, str], Dict[str, float]] = {}
        for model_name, embedder in self.embedders.items():
            short = model_name.split('/')[-1]
            with tqdm(processed_cases, desc=f"Similarity [{short}]", unit="case", ncols=100) as pbar:
                for case in pbar:
                    b_sim = compute_embedding_similarity(
                        embedder, case['sentence'], case['b_ctx'], self.embedding_top_k
                    )
                    h_sim = compute_embedding_similarity(
                        embedder, case['sentence'], case['h_ctx'], self.embedding_top_k
                    )
                    sentence_sim[(case['sentence'], model_name)] = {
                        'baseline': b_sim, 'honk': h_sim
                    }
                    pbar.set_postfix(CN=f"{b_sim:.3f}", HOnK=f"{h_sim:.3f}")
        return sentence_sim

    def _run_llm_phase(
        self,
        processed_cases: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        llm_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for model in self.llm_models:
            logger.info("LLM scoring with %s", model)
            with tqdm(processed_cases, desc=f"LLM [{model}]", unit="case", ncols=100) as pbar:
                for case in pbar:
                    with ThreadPoolExecutor(max_workers=self.max_llm_workers) as executor:
                        b_futures = [
                            executor.submit(
                                query_llm, model, case['sentence'],
                                case['b_ctx'], case['keywords'], case['b_bridges']
                            )
                            for _ in range(self.llm_iterations)
                        ]
                        h_futures = [
                            executor.submit(
                                query_llm, model, case['sentence'],
                                case['h_ctx'], case['keywords'], case['h_bridges']
                            )
                            for _ in range(self.llm_iterations)
                        ]
                        b_results = [f.result() for f in as_completed(b_futures)]
                        h_results = [f.result() for f in as_completed(h_futures)]

                    b_avg = sum(r[1] for r in b_results) / self.llm_iterations
                    h_avg = sum(r[1] for r in h_results) / self.llm_iterations
                    pbar.set_postfix(CN=f"{b_avg:.1f}", HOnK=f"{h_avg:.1f}")

                    llm_scores[(case['sentence'], model)] = {
                        'b_score': round(b_avg, 2),
                        'h_score': round(h_avg, 2),
                        'improvement': round(((h_avg - b_avg) / max(b_avg, 1e-9)) * 100, 2),
                        'b_interp': b_results[0][0].replace('\n', ' | '),
                        'h_interp': h_results[0][0].replace('\n', ' | '),
                    }

        return llm_scores

    def _build_csv_rows(
        self,
        processed_cases: List[Dict[str, Any]],
        sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
        llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        emb_model_names = list(self.embedders.keys()) if self.run_semantic else [None]

        for case in processed_cases:
            sentence = case['sentence']
            b_det, h_det = case['b_det'], case['h_det']

            det_cols: Dict[str, Any] = {
                'Test Sentence': sentence,
                'Baseline Bridging Triples': b_det.get('bridging_triple_count', 0),
                'HOnK Bridging Triples': h_det.get('bridging_triple_count', 0),
                'Baseline Hit Rate (%)': round(b_det.get('hit_rate', 0), 2),
                'HOnK Hit Rate (%)': round(h_det.get('hit_rate', 0), 2),
                'Baseline Vol (Triples)': b_det.get('subgraph_volume', 0),
                'HOnK Vol (Triples)': h_det.get('subgraph_volume', 0),
                'Baseline Relation Diversity': b_det.get('relation_diversity', 0),
                'HOnK Relation Diversity': h_det.get('relation_diversity', 0),
                'Baseline Keyword Connectivity (%)': round(b_det.get('keyword_connectivity', 0), 2),
                'HOnK Keyword Connectivity (%)': round(h_det.get('keyword_connectivity', 0), 2),
                'Baseline Avg Path Length': (
                    round(b_det.get('avg_path_length', float('inf')), 2)
                    if b_det.get('avg_path_length', float('inf')) != float('inf') else 'inf'
                ),
                'HOnK Avg Path Length': (
                    round(h_det.get('avg_path_length', float('inf')), 2)
                    if h_det.get('avg_path_length', float('inf')) != float('inf') else 'inf'
                ),
                'Differential Triples (HOnK extras)': len(case['diff_set']),
            }

            llm_cols: Dict[str, Any] = {}
            if self.run_llm and self.llm_models:
                entries = [v for (s, _), v in llm_scores.items() if s == sentence]
                if entries:
                    avg_b = sum(e['b_score'] for e in entries) / len(entries)
                    avg_h = sum(e['h_score'] for e in entries) / len(entries)
                    llm_cols = {
                        'Avg Baseline LLM Score': round(avg_b, 2),
                        'Avg HOnK LLM Score': round(avg_h, 2),
                        'Avg LLM Score Improvement (%)': round(
                            ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100, 2
                        ),
                    }

            for emb_name in emb_model_names:
                row = dict(det_cols)
                if emb_name is not None:
                    sim = sentence_sim.get((sentence, emb_name), {})
                    b_sim = sim.get('baseline', '')
                    h_sim = sim.get('honk', '')
                    sim_imp = (
                        round(((h_sim - b_sim) / max(b_sim, 1e-9)) * 100, 2)
                        if isinstance(b_sim, float) and b_sim > 0 else ''
                    )
                    row['Embedding Model'] = emb_name.split('/')[-1]
                    row['Baseline Similarity'] = round(b_sim, 4) if isinstance(b_sim, float) else ''
                    row['HOnK Similarity'] = round(h_sim, 4) if isinstance(h_sim, float) else ''
                    row['Similarity Improvement (%)'] = sim_imp
                row.update(llm_cols)
                rows.append(row)

        return rows

    def run_evaluation(self) -> None:
        if not self.test_cases:
            logger.error("No test cases defined in config.yaml.")
            sys.exit(1)

        logger.info("Starting HOnK evaluation — %d test case(s).", len(self.test_cases))

        processed_cases: List[Dict[str, Any]] = []
        with tqdm(self.test_cases, desc="Extracting subgraphs", unit="case", ncols=100) as pbar:
            for case in pbar:
                pbar.set_postfix_str(case.get('sentence', '')[:50])
                result = self._extract_case(case)
                pbar.set_postfix_str(
                    f"CN={len(result['b_triples'])} "
                    f"HOnK={len(result['h_triples'])} "
                    f"bridges={len(result['h_bridges'])}"
                )
                processed_cases.append(result)

        sentence_sim: Dict[Tuple[str, str], Dict[str, float]] = {}
        if self.run_semantic:
            sentence_sim = self._run_embedding_phase(processed_cases)

        llm_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.run_llm and self.llm_models:
            llm_scores = self._run_llm_phase(processed_cases)

        csv_rows = self._build_csv_rows(processed_cases, sentence_sim, llm_scores)
        export_to_csv(csv_rows, self.output_csv, include_llm=self.run_llm and bool(self.llm_models))
        generate_summary_table(
            processed_cases, sentence_sim, llm_scores,
            list(self.embedders.keys()), self.summary_table_output,
        )
        export_paper_tables(
            processed_cases, sentence_sim,
            output_dir=str(Path(self.summary_table_output).parent / "paper_tables"),
            llm_scores=llm_scores if self.run_llm and bool(self.llm_models) else None,
        )
        report_global_statistics(sentence_sim, llm_scores)


if __name__ == "__main__":
    for _candidate in ["config.yaml", "../config.yaml"]:
        if Path(_candidate).is_file():
            _cfg = _candidate
            break
    else:
        logger.error("config.yaml not found. Run from the project root or evaluator/ directory.")
        sys.exit(1)

    HonkEvaluator(_cfg).run_evaluation()
