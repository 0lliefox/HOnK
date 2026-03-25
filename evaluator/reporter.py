import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _fmt_path(v: float, tex: bool = False) -> str:
    if v == float('inf'):
        return r'$\infty$' if tex else 'inf'
    return f"{v:.2f}"


def _append_mean_row(
    rows: List[str],
    all_b: List[float],
    all_h: List[float],
    math_mode: bool = False,
) -> None:
    if not all_b:
        return
    mb = sum(all_b) / len(all_b)
    mh = sum(all_h) / len(all_h)
    md = ((mh - mb) / max(mb, 1e-9)) * 100
    sign = '+' if md >= 0 else ''
    delta_str = f"${sign}{md:.2f}$\\%" if math_mode else f"{sign}{md:.2f}\\%"
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Mean}} & {mb:.2f} & {mh:.2f} & {delta_str} \\\\")


def export_llm_prompts_table(
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str,
) -> None:
    if not llm_scores:
        return

    sep = "=" * 80
    thin = "-" * 80
    lines = [sep, "LLM PROMPTS AND RESPONSES", sep, ""]

    for (sentence, model), data in llm_scores.items():
        lines += [
            f"SENTENCE : {sentence}",
            f"MODEL    : {model}",
            thin,
        ]

        for kb_label, prompt_key, responses_key in [
            ("Baseline (CN)", "b_prompt", "b_responses"),
            ("HOnK",          "h_prompt", "h_responses"),
        ]:
            lines += [
                f"  [{kb_label}] PROMPT:",
                "",
            ]
            for prompt_line in data.get(prompt_key, "").splitlines():
                lines.append(f"    {prompt_line}")
            lines += ["", f"  [{kb_label}] RESPONSES:"]
            for idx, response in enumerate(data.get(responses_key, []), start=1):
                lines.append(f"    Run {idx}:")
                for resp_line in response.splitlines():
                    lines.append(f"      {resp_line}")
            lines.append("")

        lines += [sep, ""]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LLM prompts and responses written to '%s'", output_path)


def export_llm_by_model_csv(
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    output_csv: str,
) -> None:
    if not llm_scores:
        return
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Sentence', 'LLM Model', 'Baseline Score', 'HOnK Score'])
        writer.writeheader()
        for (sentence, model), scores in llm_scores.items():
            writer.writerow({
                'Sentence': sentence,
                'LLM Model': model,
                'Baseline Score': round(scores['b_score'], 2),
                'HOnK Score': round(scores['h_score'], 2),
            })
    logger.info("LLM per-model data serialised to '%s'", output_csv)


def export_to_csv(
    data: List[Dict[str, Any]],
    output_csv: str,
    include_llm: bool = False,
) -> None:
    if not data:
        return

    headers = [
        'Test Sentence',
        'Baseline Bridging Triples', 'HOnK Bridging Triples',
        'Baseline Hit Rate (%)', 'HOnK Hit Rate (%)',
        'Baseline Vol (Triples)', 'HOnK Vol (Triples)',
        'Baseline Relation Diversity', 'HOnK Relation Diversity',
        'Baseline Keyword Connectivity (%)', 'HOnK Keyword Connectivity (%)',
        'Baseline Avg Path Length', 'HOnK Avg Path Length',
        'Differential Triples (HOnK extras)',
        'Embedding Model',
        'Baseline Similarity', 'HOnK Similarity', 'Similarity Improvement (%)',
    ]
    if include_llm:
        headers += [
            'Avg Baseline LLM Score', 'Avg HOnK LLM Score',
            'Avg LLM Score Improvement (%)',
        ]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    logger.info("Evaluation data serialised to '%s'", output_csv)


def generate_summary_table(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    embedder_names: List[str],
    summary_table_output: str,
) -> None:
    has_llm = bool(llm_scores)
    has_det = any(case.get('b_det') for case in processed_cases)

    col_widths = [28, 6, 8]
    headers = ['Input', 'CN Br', 'HOnK Br']

    if has_det:
        col_widths += [6, 6, 7, 7, 6, 6, 8, 8, 7, 7]
        headers += [
            'CN Vol', 'HK Vol',
            'CN Hit%', 'HK Hit%',
            'CN Div', 'HK Div',
            'CN Conn%', 'HK Conn%',
            'CN Path', 'HK Path',
        ]

    col_widths += [18, 7, 7, 6]
    headers += ['Embedding Model', 'CN Sim', 'HK Sim', 'SimΔ']

    if has_llm:
        col_widths += [7, 7, 6]
        headers += ['CN LLM', 'HK LLM', 'LLMΔ']

    def _sep(widths: List[int]) -> str:
        return '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def _row(cells: List[str], widths: List[int]) -> str:
        return '|' + '|'.join(f' {c:<{w}} ' for c, w in zip(cells, widths)) + '|'

    sep = _sep(col_widths)
    lines = [sep, _row(headers, col_widths), sep]

    for case in processed_cases:
        sentence = case['sentence']
        b_bridges = str(len(case['b_bridges']))
        h_bridges = str(len(case['h_bridges']))
        label = (
            sentence if len(sentence) <= col_widths[0]
            else sentence[:col_widths[0] - 1] + '…'
        )

        det_cells: List[str] = []
        if has_det:
            b_det = case.get('b_det', {})
            h_det = case.get('h_det', {})
            det_cells = [
                str(b_det.get('subgraph_volume', 0)),
                str(h_det.get('subgraph_volume', 0)),
                f"{b_det.get('hit_rate', 0):.1f}",
                f"{h_det.get('hit_rate', 0):.1f}",
                str(b_det.get('relation_diversity', 0)),
                str(h_det.get('relation_diversity', 0)),
                f"{b_det.get('keyword_connectivity', 0):.1f}",
                f"{h_det.get('keyword_connectivity', 0):.1f}",
                _fmt_path(b_det.get('avg_path_length', float('inf'))),
                _fmt_path(h_det.get('avg_path_length', float('inf'))),
            ]

        llm_cells: List[str] = []
        if has_llm:
            entries = [v for (s, _), v in llm_scores.items() if s == sentence]
            if entries:
                avg_b = sum(e['b_score'] for e in entries) / len(entries)
                avg_h = sum(e['h_score'] for e in entries) / len(entries)
                llm_cells = [
                    f"{avg_b:.1f}", f"{avg_h:.1f}",
                    f"{((avg_h - avg_b) / max(avg_b, 1e-9)) * 100:+.1f}%",
                ]
            else:
                llm_cells = ["N/A", "N/A", "N/A"]

        emb_col_idx = len(col_widths) - (6 if has_llm else 3) - 3
        for emb_name in (embedder_names if embedder_names else [None]):
            if emb_name is not None:
                sim = sentence_sim.get((sentence, emb_name), {})
                b_sim = sim.get('baseline')
                h_sim = sim.get('honk')
                if b_sim is not None and h_sim is not None:
                    sim_delta = f"{((h_sim - b_sim) / max(b_sim, 1e-9)) * 100:+.1f}%"
                    b_str, h_str = f"{b_sim:.4f}", f"{h_sim:.4f}"
                else:
                    b_str = h_str = sim_delta = "N/A"
                short_name = emb_name.split('/')[-1][:col_widths[emb_col_idx]]
            else:
                b_str = h_str = sim_delta = short_name = "N/A"

            row_cells = [label, b_bridges, h_bridges] + det_cells + [short_name, b_str, h_str, sim_delta] + llm_cells
            lines.append(_row(row_cells, col_widths))

        lines.append(sep)

    table_text = "\n".join(lines)
    logger.info("Summary Table:\n%s", table_text)

    out_path = Path(summary_table_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_text + "\n", encoding='utf-8')
    logger.info("Summary table written to '%s'", summary_table_output)

    tex_path = out_path.with_suffix('.tex')
    _write_latex_table(
        processed_cases, sentence_sim, llm_scores, embedder_names,
        has_det, has_llm, tex_path,
    )
    logger.info("LaTeX table written to '%s'", tex_path)


def _write_latex_table(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    embedder_names: List[str],
    has_det: bool,
    has_llm: bool,
    tex_path: Path,
) -> None:
    def _esc(s: str) -> str:
        for ch, repl in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'),
                         ('&', r'\&'), ('#', r'\#'), ('{', r'\{'), ('}', r'\}')):
            s = s.replace(ch, repl)
        return s

    col_spec = 'l r r'
    header1_cells = ['Input', r'\multicolumn{2}{c}{Bridging}']
    header2_cells = ['', 'CN', 'HOnK']

    if has_det:
        col_spec += ' r r r r r r r r r r'
        header1_cells += [
            r'\multicolumn{2}{c}{Volume}',
            r'\multicolumn{2}{c}{Hit\%}',
            r'\multicolumn{2}{c}{Rel Div}',
            r'\multicolumn{2}{c}{KW Conn\%}',
            r'\multicolumn{2}{c}{Avg Path}',
        ]
        header2_cells += ['CN', 'HK', 'CN', 'HK', 'CN', 'HK', 'CN', 'HK', 'CN', 'HK']

    col_spec += ' l r r r'
    header1_cells += ['Embedding Model', r'\multicolumn{3}{c}{Similarity}']
    header2_cells += ['', 'CN', 'HK', r'$\Delta$']

    if has_llm:
        col_spec += ' r r r'
        header1_cells += [r'\multicolumn{3}{c}{LLM Score}']
        header2_cells += ['CN', 'HK', r'$\Delta$']

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        ' & '.join(header1_cells) + r' \\',
        r'\cmidrule(lr){2-3}',
    ]
    if has_det:
        lines += [
            r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}'
            r'\cmidrule(lr){10-11}\cmidrule(lr){12-13}',
        ]
    det_end = 13 if has_det else 3
    lines += [
        r'\cmidrule(lr){' + str(det_end + 1) + '-' + str(det_end + 3) + '}',
        ' & '.join(header2_cells) + r' \\',
        r'\midrule',
    ]

    prev_sentence: Optional[str] = None
    for case in processed_cases:
        sentence = case['sentence']
        b_bridges = str(len(case['b_bridges']))
        h_bridges = str(len(case['h_bridges']))
        label = _esc(sentence[:40]) + (r'\ldots' if len(sentence) > 40 else '')

        det_cells: List[str] = []
        if has_det:
            b_det = case.get('b_det', {})
            h_det = case.get('h_det', {})
            det_cells = [
                str(b_det.get('subgraph_volume', 0)),
                str(h_det.get('subgraph_volume', 0)),
                f"{b_det.get('hit_rate', 0):.1f}",
                f"{h_det.get('hit_rate', 0):.1f}",
                str(b_det.get('relation_diversity', 0)),
                str(h_det.get('relation_diversity', 0)),
                f"{b_det.get('keyword_connectivity', 0):.1f}",
                f"{h_det.get('keyword_connectivity', 0):.1f}",
                _fmt_path(b_det.get('avg_path_length', float('inf')), tex=True),
                _fmt_path(h_det.get('avg_path_length', float('inf')), tex=True),
            ]

        llm_cells: List[str] = []
        if has_llm:
            entries = [v for (s, _), v in llm_scores.items() if s == sentence]
            if entries:
                avg_b = sum(e['b_score'] for e in entries) / len(entries)
                avg_h = sum(e['h_score'] for e in entries) / len(entries)
                llm_cells = [
                    f"{avg_b:.1f}", f"{avg_h:.1f}",
                    f"{((avg_h - avg_b) / max(avg_b, 1e-9)) * 100:+.1f}\\%",
                ]
            else:
                llm_cells = ['N/A', 'N/A', 'N/A']

        if prev_sentence is not None and sentence != prev_sentence:
            lines.append(r'\midrule')
        prev_sentence = sentence

        for i, emb_name in enumerate(embedder_names if embedder_names else [None]):
            row_label = label if i == 0 else ''
            row_br_b = b_bridges if i == 0 else ''
            row_br_h = h_bridges if i == 0 else ''
            row_det = det_cells if i == 0 else [''] * len(det_cells)

            if emb_name is not None:
                sim = sentence_sim.get((sentence, emb_name), {})
                b_sim = sim.get('baseline')
                h_sim = sim.get('honk')
                if b_sim is not None and h_sim is not None:
                    sim_delta = f"{((h_sim - b_sim) / max(b_sim, 1e-9)) * 100:+.1f}\\%"
                    b_str, h_str = f"{b_sim:.4f}", f"{h_sim:.4f}"
                else:
                    b_str = h_str = sim_delta = 'N/A'
                short_name = _esc(emb_name.split('/')[-1][:24])
            else:
                b_str = h_str = sim_delta = short_name = 'N/A'

            row_llm = (llm_cells if i == 0 else [''] * 3) if has_llm else []

            cells = [row_label, row_br_b, row_br_h] + row_det + [short_name, b_str, h_str, sim_delta] + row_llm
            lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{HOnK evaluation summary.}',
        r'\label{tab:honk-summary}',
        r'\end{table}',
    ]

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def export_paper_tables(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    output_dir: str,
    llm_scores: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _export_bridging_table(processed_cases, out / "bridging_counts.tex")
    _export_deterministic_table(processed_cases, out / "deterministic_metrics.tex")
    if sentence_sim:
        _export_embedding_table(sentence_sim, out / "embedding_similarity.tex")
        _export_embedding_sentence_table(processed_cases, sentence_sim, out / "embedding_similarity_by_sentence.tex")
    if llm_scores:
        _export_llm_table(processed_cases, llm_scores, out / "llm_scores.tex")
        _export_llm_by_model_table(llm_scores, out / "llm_scores_by_model.tex")

    logger.info("Paper tables written to '%s'", output_dir)


def _booktabs_table(caption: str, label: str, col_spec: str,
                    header: str, body: str) -> str:
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        f"\\begin{{tabularx}}{{\\columnwidth}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n"
    )


def _sentence_ref(case: Dict[str, Any]) -> str:
    lbl = case.get('label', '')
    if lbl:
        return f"\\sref{{{lbl}}}"
    sentence = case['sentence']
    text = (sentence[:38] + r'\ldots') if len(sentence) > 38 else sentence
    return text.replace('_', r'\_').replace('&', r'\&')


def _export_bridging_table(processed_cases: List[Dict[str, Any]], path: Path) -> None:
    rows = []
    for case in processed_cases:
        ref = _sentence_ref(case)
        b = len(case['b_bridges'])
        h = len(case['h_bridges'])
        rows.append(f"{ref} & {b} & {h} \\\\")

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption="Bridging triple counts per test sentence (sentence labels refer to Table~\\ref{tab:sentences}).",
        label="tab:bridges",
        col_spec="Xrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_deterministic_table(processed_cases: List[Dict[str, Any]], path: Path) -> None:
    metrics = [
        ('subgraph_volume',      'Volume (triples)'),
        ('hit_rate',             'Hit Rate (\\%)'),
        ('relation_diversity',   'Rel.\\ Diversity'),
        ('keyword_connectivity', 'KW Connectivity (\\%)'),
        ('avg_path_length',      'Avg Path Length'),
        ('bridging_triple_count','Bridging Count'),
    ]

    rows = []
    for key, label in metrics:
        b_vals = [case['b_det'].get(key, 0) for case in processed_cases if case.get('b_det')]
        h_vals = [case['h_det'].get(key, 0) for case in processed_cases if case.get('h_det')]

        if key == 'avg_path_length':
            b_vals = [v for v in b_vals if v != float('inf')]
            h_vals = [v for v in h_vals if v != float('inf')]

        if not b_vals or not h_vals:
            rows.append(f"{label} & --- & --- \\\\")
            continue

        avg_b = sum(b_vals) / len(b_vals)
        avg_h = sum(h_vals) / len(h_vals)
        rows.append(f"{label} & {avg_b:,.2f} & {avg_h:,.2f} \\\\")

    diff_vals = [len(case.get('diff_set', set())) for case in processed_cases]
    avg_diff = sum(diff_vals) / len(diff_vals) if diff_vals else 0
    rows.append(f"Differential Triples & --- & {avg_diff:,.2f} \\\\")

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption="Aggregate deterministic metrics (mean across all test cases).",
        label="tab:det",
        col_spec="Xrr",
        header="\\textbf{Metric} & \\textbf{CN} & \\textbf{HOnK}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_embedding_table(
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    path: Path,
) -> None:
    by_model: Dict[str, List[Dict[str, float]]] = {}
    for (_, model_name), scores in sentence_sim.items():
        by_model.setdefault(model_name, []).append(scores)

    rows = []
    all_b, all_h = [], []
    for model_name, score_list in by_model.items():
        avg_b = sum(s['baseline'] for s in score_list) / len(score_list)
        avg_h = sum(s['honk'] for s in score_list) / len(score_list)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        short = model_name.split('/')[-1]
        sign = '+' if delta >= 0 else ''
        rows.append(f"\\texttt{{{short}}} & {avg_b:.2f} & {avg_h:.2f} & {sign}{delta:.2f}\\% \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h)
    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption="Embedding similarity: mean per-triple top-5 cosine similarity.",
        label="tab:emb",
        col_spec="Xrrr",
        header="\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_llm_table(
    processed_cases: List[Dict[str, Any]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    path: Path,
) -> None:
    rows = []
    all_b, all_h = [], []
    for case in processed_cases:
        sentence = case['sentence']
        ref = _sentence_ref(case)
        entries = [v for (s, _), v in llm_scores.items() if s == sentence]
        if not entries:
            rows.append(f"{ref} & --- & --- & --- \\\\")
            continue
        avg_b = sum(e['b_score'] for e in entries) / len(entries)
        avg_h = sum(e['h_score'] for e in entries) / len(entries)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        rows.append(f"{ref} & {avg_b:.2f} & {avg_h:.2f} & ${sign}{delta:.2f}$\\% \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h, math_mode=True)
    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "LLM relevance scores per test sentence "
            "(mean across all LLMs and \\texttt{llm\\_iterations} runs; "
            "sentence labels refer to Table~\\ref{tab:sentences})."
        ),
        label="tab:llm",
        col_spec="Xrrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_embedding_sentence_table(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    path: Path,
) -> None:
    rows = []
    all_b, all_h = [], []
    for case in processed_cases:
        sentence = case['sentence']
        ref = _sentence_ref(case)
        scores = [v for (s, _), v in sentence_sim.items() if s == sentence]
        if not scores:
            rows.append(f"{ref} & --- & --- & --- \\\\")
            continue
        avg_b = sum(s['baseline'] for s in scores) / len(scores)
        avg_h = sum(s['honk'] for s in scores) / len(scores)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        rows.append(f"{ref} & {avg_b:.2f} & {avg_h:.2f} & ${sign}{delta:.2f}$\\% \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h, math_mode=True)
    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption="Embedding similarity per test sentence (mean across all embedding models; sentence labels refer to Table~\\ref{tab:sentences}).",
        label="tab:emb-sentence",
        col_spec="Xrrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_llm_by_model_table(
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    path: Path,
) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for (_, model_name), scores in llm_scores.items():
        by_model.setdefault(model_name, []).append(scores)

    rows = []
    all_b, all_h = [], []
    for model_name, entries in by_model.items():
        avg_b = sum(e['b_score'] for e in entries) / len(entries)
        avg_h = sum(e['h_score'] for e in entries) / len(entries)
        delta = ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100
        sign = '+' if delta >= 0 else ''
        short = model_name.replace('_', r'\_')
        rows.append(f"\\texttt{{{short}}} & {avg_b:.2f} & {avg_h:.2f} & {sign}{delta:.2f}\\% \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h)
    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption="LLM relevance scores by model (mean across all test cases and \\texttt{llm\\_iterations} runs).",
        label="tab:llm-model",
        col_spec="Xrrr",
        header="\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def report_global_statistics(
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    if sentence_sim:
        by_model: Dict[str, list] = {}
        for (_, model_name), scores in sentence_sim.items():
            by_model.setdefault(model_name, []).append(scores)
        for model_name, score_list in by_model.items():
            avg_b = sum(s['baseline'] for s in score_list) / len(score_list)
            avg_h = sum(s['honk'] for s in score_list) / len(score_list)
            logger.info(
                "Embedding [%s] — Baseline: %.4f | HOnK: %.4f | Improvement: %+.2f%%",
                model_name.split('/')[-1], avg_b, avg_h,
                ((avg_h - avg_b) / max(avg_b, 1e-9)) * 100,
            )

    if llm_scores:
        all_b = sum(v['b_score'] for v in llm_scores.values()) / len(llm_scores)
        all_h = sum(v['h_score'] for v in llm_scores.values()) / len(llm_scores)
        logger.info(
            "Global LLM Score — Baseline: %.2f | HOnK: %.2f | Improvement: %+.2f%%",
            all_b, all_h, ((all_h - all_b) / max(all_b, 1e-9)) * 100,
        )
