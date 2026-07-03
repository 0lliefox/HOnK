import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from reporter import _booktabs_table, _append_mean_row, _fmt_path

def _load_sentence_labels() -> dict:
    """sentence -> \\sref label key, from evaluator.test_cases in config.yaml
    (the config is the single source of truth; a hardcoded map here previously
    drifted out of date and silently fell back to raw truncated text)."""
    import yaml
    for candidate in ["config.yaml", "../config.yaml"]:
        if Path(candidate).is_file():
            with open(candidate, encoding='utf-8') as fh:
                cfg = yaml.safe_load(fh)
            cases = cfg.get('evaluator', {}).get('test_cases', [])
            return {c['sentence']: c['label'] for c in cases
                    if c.get('sentence') and c.get('label')}
    logger.warning("config.yaml not found; sentence labels unavailable.")
    return {}


SENTENCE_LABELS = _load_sentence_labels()


def _esc(s: str) -> str:
    for ch, repl in (('_', r'\_'), ('%', r'\%'), ('&', r'\&'), ('#', r'\#')):
        s = s.replace(ch, repl)
    return s


def _label(sentence: str, max_len: int = 40) -> str:
    key = SENTENCE_LABELS.get(sentence)
    if key:
        return f"\\sref{{{key}}}"
    text = sentence[:max_len] + (r'\ldots' if len(sentence) > max_len else '')
    return _esc(text)


def _write_tex(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')
    logger.info("Written %s", path)


def _write_csv(path: Path, headers: list, rows: list) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info("Written %s", path)


def load_csv(path: str):
    sentences = []
    seen = set()
    bridging = {}
    det = {}
    sim = {}
    llm = {}
    models = []
    seen_models = set()

    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            s = row['Test Sentence']
            model = row['Embedding Model']

            if s not in seen:
                sentences.append(s)
                seen.add(s)
                bridging[s] = (
                    int(row['Baseline Bridging Triples']),
                    int(row['HOnK Bridging Triples']),
                )
                det[s] = (
                    {
                        'subgraph_volume':       float(row['Baseline Vol (Triples)']),
                        'hit_rate':              float(row['Baseline Hit Rate (%)']),
                        'relation_diversity':    float(row['Baseline Relation Diversity']),
                        'keyword_connectivity':  float(row['Baseline Keyword Connectivity (%)']),
                        'avg_path_length':       float(row['Baseline Avg Path Length']),
                        'bridging_triple_count': int(row['Baseline Bridging Triples']),
                    },
                    {
                        'subgraph_volume':       float(row['HOnK Vol (Triples)']),
                        'hit_rate':              float(row['HOnK Hit Rate (%)']),
                        'relation_diversity':    float(row['HOnK Relation Diversity']),
                        'keyword_connectivity':  float(row['HOnK Keyword Connectivity (%)']),
                        'avg_path_length':       float(row['HOnK Avg Path Length']),
                        'bridging_triple_count': int(row['HOnK Bridging Triples']),
                    },
                    int(row['Differential Triples (HOnK extras)']),
                )
                b_llm = row.get('Avg Baseline LLM Score', '').strip()
                h_llm = row.get('Avg HOnK LLM Score', '').strip()
                if b_llm and h_llm:
                    llm[s] = (float(b_llm), float(h_llm))

            if model not in seen_models:
                models.append(model)
                seen_models.add(model)

            sim[(s, model)] = {
                'baseline': float(row['Baseline Similarity']),
                'honk':     float(row['HOnK Similarity']),
            }

    logger.info("Loaded %d sentences across %d embedding model(s)", len(sentences), len(models))
    return sentences, bridging, det, sim, llm, models


def write_bridging_table(sentences, bridging, out_dir: Path) -> None:
    rows_tex, rows_csv = [], []
    for s in sentences:
        b, h = bridging[s]
        rows_tex.append(f"{_label(s)} & {b} & {h} \\\\")
        rows_csv.append([s, b, h])

    tex = _booktabs_table(
        caption="Bridging triple counts per test sentence.",
        label="tab:bridges",
        col_spec="Xrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "bridging_counts.tex", tex)
    _write_csv(out_dir / "bridging_counts.csv", ["Sentence", "CN", "HOnK"], rows_csv)


def write_deterministic_table(sentences, det, out_dir: Path) -> None:
    metrics = [
        ('subgraph_volume',       'Volume (triples)'),
        ('hit_rate',              'Hit Rate (\\%)'),
        ('relation_diversity',    'Rel.\\ Diversity'),
        ('keyword_connectivity',  'KW Connectivity (\\%)'),
        ('avg_path_length',       'Avg Path Length'),
        ('bridging_triple_count', 'Bridging Count'),
    ]

    rows_tex, rows_csv = [], []
    for key, label in metrics:
        b_vals = [det[s][0].get(key, 0) for s in sentences]
        h_vals = [det[s][1].get(key, 0) for s in sentences]
        if key == 'avg_path_length':
            b_vals = [v for v in b_vals if v != float('inf')]
            h_vals = [v for v in h_vals if v != float('inf')]
        avg_b = sum(b_vals) / len(b_vals) if b_vals else 0
        avg_h = sum(h_vals) / len(h_vals) if h_vals else 0
        plain = label.replace('\\%', '%').replace('\\ ', ' ')
        rows_tex.append(f"{label} & {avg_b:,.2f} & {avg_h:,.2f} \\\\")
        rows_csv.append([plain, f"{avg_b:.2f}", f"{avg_h:.2f}"])

    diff_vals = [det[s][2] for s in sentences]
    avg_diff = sum(diff_vals) / len(diff_vals) if diff_vals else 0
    rows_tex.append(f"Differential Triples & --- & {avg_diff:,.2f} \\\\")
    rows_csv.append(["Differential Triples", "---", f"{avg_diff:.2f}"])

    tex = _booktabs_table(
        caption="Aggregate deterministic metrics (mean across all test cases).",
        label="tab:det",
        col_spec="Xrr",
        header="\\textbf{Metric} & \\textbf{CN} & \\textbf{HOnK}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "deterministic_metrics.tex", tex)
    _write_csv(out_dir / "deterministic_metrics.csv", ["Metric", "CN", "HOnK"], rows_csv)


def write_embedding_table(sim, models, out_dir: Path) -> None:
    rows_tex, rows_csv = [], []
    all_b, all_h = [], []
    for model in models:
        scores = [v for (_, m), v in sim.items() if m == model]
        if not scores:
            continue
        avg_b = sum(s['baseline'] for s in scores) / len(scores)
        avg_h = sum(s['honk'] for s in scores) / len(scores)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        short = model.split('/')[-1]
        sign = '+' if delta >= 0 else ''
        rows_tex.append(f"\\texttt{{{_esc(short)}}} & {avg_b:.2f} & {avg_h:.2f} & {sign}{delta:.2f}\\% \\\\")
        rows_csv.append([short, f"{avg_b:.2f}", f"{avg_h:.2f}", f"{sign}{delta:.2f}%"])
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows_tex, all_b, all_h)
    tex = _booktabs_table(
        caption="Embedding similarity: mean per-triple top-5 cosine similarity.",
        label="tab:emb",
        col_spec="Xrrr",
        header="\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "embedding_similarity.tex", tex)
    _write_csv(out_dir / "embedding_similarity.csv", ["Model", "CN", "HOnK", "Delta%"], rows_csv)


def write_embedding_sentence_table(sentences, sim, out_dir: Path) -> None:
    rows_tex, rows_csv = [], []
    all_b, all_h = [], []
    for s in sentences:
        scores = [v for (sent, _), v in sim.items() if sent == s]
        lbl = _label(s)
        if not scores:
            rows_tex.append(f"{lbl} & --- & --- & --- \\\\")
            rows_csv.append([s, "---", "---", "---"])
            continue
        avg_b = sum(sc['baseline'] for sc in scores) / len(scores)
        avg_h = sum(sc['honk'] for sc in scores) / len(scores)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        rows_tex.append(f"{lbl} & {avg_b:.2f} & {avg_h:.2f} & ${sign}{delta:.2f}$\\% \\\\")
        rows_csv.append([s, f"{avg_b:.2f}", f"{avg_h:.2f}", f"{sign}{delta:.2f}%"])
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows_tex, all_b, all_h, math_mode=True)
    tex = _booktabs_table(
        caption="Embedding similarity per test sentence (mean across all embedding models).",
        label="tab:emb-sentence",
        col_spec="Xrrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "embedding_similarity_by_sentence.tex", tex)
    _write_csv(out_dir / "embedding_similarity_by_sentence.csv",
               ["Sentence", "CN", "HOnK", "Delta%"], rows_csv)


def load_llm_by_model_csv(path: Path) -> dict:
    by_model = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            model = row['LLM Model']
            by_model.setdefault(model, []).append({
                'b_score': float(row['Baseline Score']),
                'h_score': float(row['HOnK Score']),
            })
    return by_model


def write_llm_by_model_table(by_model: dict, out_dir: Path) -> None:
    rows_tex, rows_csv = [], []
    all_b, all_h = [], []
    for model, entries in by_model.items():
        avg_b = sum(e['b_score'] for e in entries) / len(entries)
        avg_h = sum(e['h_score'] for e in entries) / len(entries)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        short = _esc(model.split('/')[-1])
        rows_tex.append(f"\\texttt{{{short}}} & {avg_b:.2f} & {avg_h:.2f} & {sign}{delta:.2f}\\% \\\\")
        rows_csv.append([model, f"{avg_b:.2f}", f"{avg_h:.2f}", f"{sign}{delta:.2f}%"])
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows_tex, all_b, all_h)
    tex = _booktabs_table(
        caption="LLM relevance scores by model (mean across all test cases and runs).",
        label="tab:llm-model",
        col_spec="Xrrr",
        header="\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "llm_scores_by_model.tex", tex)
    _write_csv(out_dir / "llm_scores_by_model.csv", ["Model", "CN", "HOnK", "Delta%"], rows_csv)


def write_llm_table(sentences, llm, out_dir: Path) -> None:
    rows_tex, rows_csv = [], []
    all_b, all_h = [], []
    for s in sentences:
        lbl = _label(s)
        if s not in llm:
            rows_tex.append(f"{lbl} & --- & --- & --- \\\\")
            rows_csv.append([s, "---", "---", "---"])
            continue
        b, h = llm[s]
        delta = ((h - b) / max(b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        rows_tex.append(f"{lbl} & {b:.2f} & {h:.2f} & ${sign}{delta:.2f}$\\% \\\\")
        rows_csv.append([s, f"{b:.2f}", f"{h:.2f}", f"{sign}{delta:.2f}%"])
        all_b.append(b)
        all_h.append(h)

    _append_mean_row(rows_tex, all_b, all_h, math_mode=True)
    tex = _booktabs_table(
        caption="LLM relevance scores per test sentence (mean across all LLMs and runs).",
        label="tab:llm",
        col_spec="Xrrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body="\n".join(rows_tex) + "\n",
    )
    _write_tex(out_dir / "llm_scores.tex", tex)
    _write_csv(out_dir / "llm_scores.csv", ["Sentence", "CN", "HOnK", "Delta%"], rows_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX and CSV paper tables from evaluation_results.csv."
    )
    parser.add_argument("input", help="Path to evaluation_results.csv")
    parser.add_argument(
        "--output", default="paper_tables",
        help="Output directory (default: paper_tables)",
    )
    args = parser.parse_args()

    if not Path(args.input).is_file():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    sentences, bridging, det, sim, llm, models = load_csv(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_bridging_table(sentences, bridging, out_dir)
    write_deterministic_table(sentences, det, out_dir)
    write_embedding_table(sim, models, out_dir)
    write_embedding_sentence_table(sentences, sim, out_dir)

    if llm:
        write_llm_table(sentences, llm, out_dir)
    else:
        logger.warning("No LLM scores in CSV — skipping llm_scores table")

    input_path = Path(args.input)
    llm_model_csv = input_path.with_name(input_path.stem + '_llm_by_model.csv')
    if llm_model_csv.is_file():
        by_model = load_llm_by_model_csv(llm_model_csv)
        write_llm_by_model_table(by_model, out_dir)
    else:
        logger.warning("No per-model LLM CSV found at %s — skipping llm_scores_by_model table", llm_model_csv)

    logger.info("Done — tables written to %s/", out_dir)


if __name__ == "__main__":
    main()
