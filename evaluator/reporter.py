import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy.stats import rankdata, spearmanr, wilcoxon
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a declared dependency
    _HAVE_SCIPY = False

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


def _delta_str(avg_b: float, avg_h: float, math_mode: bool = True) -> str:
    """Percentage-change cell, guarding a zero baseline.

    When the baseline has no matching triples (avg_b == 0) a percentage change is
    undefined; render ``new`` (HOnK introduces a link the baseline lacks) rather
    than dividing by a tiny epsilon, which otherwise prints spurious values like
    ``+51068580150%``.
    """
    if avg_b <= 1e-9:
        return "\\emph{new}" if avg_h > 1e-9 else "--"
    delta = ((avg_h - avg_b) / avg_b) * 100
    sign = '+' if delta >= 0 else ''
    return f"${sign}{delta:.2f}$\\%" if math_mode else f"{sign}{delta:.2f}\\%"


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


def _paired_test(baseline: List[float], honk: List[float]) -> Dict[str, Any]:
    """Wilcoxon signed-rank over per-sentence (baseline, honk) pairs.

    Means and 95% CIs are over all n pairs; the test and the matched-pairs
    rank-biserial effect size (Kerby 2014) use the n_eff non-zero differences.
    p and r are None when the test is undefined (all ties, n too small, or
    scipy unavailable) — render as 'n/a', never silently as significance.
    """
    n = len(baseline)
    result: Dict[str, Any] = {
        'n': n, 'n_eff': 0, 'p': None, 'r_rb': None,
        'mean_b': None, 'mean_h': None, 'ci_b': None, 'ci_h': None,
        'delta_pct': None,
    }
    if n == 0:
        return result

    def _mean_ci(vals: List[float]) -> Tuple[float, Optional[float]]:
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, None
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return m, 1.96 * math.sqrt(var / len(vals))

    result['mean_b'], result['ci_b'] = _mean_ci(baseline)
    result['mean_h'], result['ci_h'] = _mean_ci(honk)
    result['delta_pct'] = ((result['mean_h'] - result['mean_b'])
                           / max(result['mean_b'], 1e-9)) * 100

    diffs = [h - b for b, h in zip(baseline, honk)]
    nz = [d for d in diffs if d != 0]
    result['n_eff'] = len(nz)
    if not _HAVE_SCIPY or len(nz) == 0:
        return result

    ranks = rankdata([abs(d) for d in nz])
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    rank_sum = len(nz) * (len(nz) + 1) / 2
    result['r_rb'] = (w_pos - w_neg) / rank_sum

    try:
        result['p'] = float(wilcoxon(baseline, honk, zero_method='wilcox',
                                     alternative='two-sided').pvalue)
    except ValueError:
        result['p'] = None
    return result


def _compute_paired_tests(
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Paired tests for every headline metric.

    emb          per embedding model (primary; scales differ across models)
    emb_overall  per-sentence mean across embedding models (secondary)
    llm_by_judge per judge model (secondary)
    llm_avg      per-sentence mean across judges (primary headline metric)
    """
    tests: Dict[str, Any] = {'emb': {}, 'emb_overall': None,
                             'llm_by_judge': {}, 'llm_avg': None}

    emb_by_model: Dict[str, List[Tuple[float, float]]] = {}
    emb_by_sentence: Dict[str, List[Tuple[float, float]]] = {}
    for (sentence, model), s in sorted(sentence_sim.items()):
        pair = (s['baseline'], s['honk'])
        emb_by_model.setdefault(model, []).append(pair)
        emb_by_sentence.setdefault(sentence, []).append(pair)
    for model, pairs in emb_by_model.items():
        tests['emb'][model] = _paired_test([p[0] for p in pairs], [p[1] for p in pairs])
    if emb_by_sentence:
        b = [sum(p[0] for p in pairs) / len(pairs) for pairs in emb_by_sentence.values()]
        h = [sum(p[1] for p in pairs) / len(pairs) for pairs in emb_by_sentence.values()]
        tests['emb_overall'] = _paired_test(b, h)

    llm_by_judge: Dict[str, List[Tuple[float, float]]] = {}
    llm_by_sentence: Dict[str, List[Tuple[float, float]]] = {}
    for (sentence, model), s in sorted(llm_scores.items()):
        pair = (s['b_score'], s['h_score'])
        llm_by_judge.setdefault(model, []).append(pair)
        llm_by_sentence.setdefault(sentence, []).append(pair)
    for model, pairs in llm_by_judge.items():
        tests['llm_by_judge'][model] = _paired_test([p[0] for p in pairs], [p[1] for p in pairs])
    if llm_by_sentence:
        b = [sum(p[0] for p in pairs) / len(pairs) for pairs in llm_by_sentence.values()]
        h = [sum(p[1] for p in pairs) / len(pairs) for pairs in llm_by_sentence.values()]
        tests['llm_avg'] = _paired_test(b, h)

    return tests


def _krippendorff_alpha_interval(matrix: List[List[float]]) -> Optional[float]:
    """Interval-level Krippendorff's alpha for COMPLETE data (rows = items,
    columns = raters, no missing cells) via the pooled-pairs form, which is
    exact when every item has the same number of raters."""
    items = [row for row in matrix if row]
    if not items:
        return None
    m = len(items[0])
    if m < 2 or any(len(row) != m for row in items):
        return None
    n_items = len(items)

    d_o = 0.0
    for row in items:
        for a in range(m):
            for b in range(m):
                if a != b:
                    d_o += (row[a] - row[b]) ** 2
    d_o /= (n_items * m * (m - 1))

    values = [v for row in items for v in row]
    n = len(values)
    mean = sum(values) / n
    pop_var = sum((v - mean) ** 2 for v in values) / n
    d_e = 2.0 * n * pop_var / (n - 1)
    if d_e == 0:
        return None
    return 1.0 - d_o / d_e


def _compute_llm_agreement(
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Cross-judge agreement over all (sentence x side) score items."""
    judges = sorted({model for (_, model) in llm_scores})
    sentences = sorted({s for (s, _) in llm_scores})
    matrix: List[List[float]] = []
    columns: Dict[str, List[float]] = {j: [] for j in judges}
    for sentence in sentences:
        for side in ('b_score', 'h_score'):
            row = []
            for judge in judges:
                entry = llm_scores.get((sentence, judge))
                if entry is None:
                    row = []
                    break
                row.append(float(entry[side]))
            if row:
                matrix.append(row)
                for judge, value in zip(judges, row):
                    columns[judge].append(value)

    pairwise: Dict[Tuple[str, str], Optional[float]] = {}
    if _HAVE_SCIPY:
        for i, j1 in enumerate(judges):
            for j2 in judges[i + 1:]:
                rho = spearmanr(columns[j1], columns[j2]).statistic if matrix else None
                pairwise[(j1, j2)] = None if rho is None or math.isnan(rho) else float(rho)

    return {'alpha': _krippendorff_alpha_interval(matrix),
            'pairwise': pairwise, 'n_items': len(matrix), 'judges': judges}


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def _fmt_r(r: Optional[float]) -> str:
    return "n/a" if r is None else f"{r:+.2f}"


def _fmt_mean_ci(mean: Optional[float], ci: Optional[float], digits: int = 3) -> str:
    if mean is None:
        return "---"
    if ci is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}\\,{{\\scriptsize$\\pm${ci:.{digits}f}}}"


def _export_llm_agreement_table(agreement: Dict[str, Any], path: Path) -> None:
    rows = [
        f"Krippendorff's $\\alpha$ (interval) & {agreement['alpha']:.3f} \\\\"
        if agreement['alpha'] is not None else
        "Krippendorff's $\\alpha$ (interval) & n/a \\\\"
    ]
    for (j1, j2), rho in sorted(agreement['pairwise'].items()):
        j1s, j2s = j1.replace('_', r'\_'), j2.replace('_', r'\_')
        rho_s = "n/a" if rho is None else f"{rho:.3f}"
        rows.append(f"Spearman $\\rho$ (\\texttt{{{j1s}}} vs \\texttt{{{j2s}}}) & {rho_s} \\\\")

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "Cross-judge agreement over all "
            f"{agreement['n_items']} (sentence $\\times$ system) score items."
        ),
        label="tab:llm-agree",
        col_spec="Xr",
        header="\\textbf{Statistic} & \\textbf{Value}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_provenance_table(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    path: Path,
) -> None:
    """Three-way breakdown by case provenance (hand / differential / neutral).

    The neutral-control row is the selection-bias check: where HOnK has no
    cross-source advantage, its scores should be statistically indistinguishable
    from the baseline, so a large significant gain on the differential set
    reflects genuine integration rather than benchmark construction.
    """
    prov_of = {c['sentence']: (c.get('provenance') or 'unlabelled') for c in processed_cases}

    def _per_sentence_means(pairs_by_sentence):
        b = [sum(p[0] for p in ps) / len(ps) for ps in pairs_by_sentence.values()]
        h = [sum(p[1] for p in ps) / len(ps) for ps in pairs_by_sentence.values()]
        return b, h

    order = ['hand', 'differential', 'neutral']
    label = {'hand': 'Hand-curated', 'differential': 'Differential-seeded',
             'neutral': 'Neutral control'}

    emb_groups: Dict[str, Dict[str, list]] = {}
    for (sentence, _), s in sentence_sim.items():
        g = prov_of.get(sentence, 'unlabelled')
        emb_groups.setdefault(g, {}).setdefault(sentence, []).append((s['baseline'], s['honk']))
    llm_groups: Dict[str, Dict[str, list]] = {}
    for (sentence, _), s in llm_scores.items():
        g = prov_of.get(sentence, 'unlabelled')
        llm_groups.setdefault(g, {}).setdefault(sentence, []).append((s['b_score'], s['h_score']))

    rows = []
    for g in order:
        if g not in emb_groups and g not in llm_groups:
            continue
        eb, eh = _per_sentence_means(emb_groups.get(g, {}))
        lb, lh = _per_sentence_means(llm_groups.get(g, {}))
        et = _paired_test(eb, eh) if eb else None
        lt = _paired_test(lb, lh) if lb else None
        n = len(eb) or len(lb)

        def _cell(t):
            if t is None or t['delta_pct'] is None:
                return "--- & ---"
            sign = '+' if t['delta_pct'] >= 0 else ''
            return f"{sign}{t['delta_pct']:.2f}\\% & {_fmt_p(t['p'])}"

        rows.append(f"{label.get(g, g)} & {n} & {_cell(et)} & {_cell(lt)} \\\\")

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "Downstream gains by case provenance. Differential-seeded cases were "
            "selected for cross-source connectivity in \\gls{onto}; neutral controls "
            "were selected without that criterion. $\\Delta$ is the mean per-sentence "
            "improvement of \\gls{onto} over the ConceptNet baseline; $p$ is a two-sided "
            "Wilcoxon signed-rank test. The neutral controls show no significant "
            "difference, indicating the gains are not an artefact of case selection."
        ),
        label="tab:provenance",
        col_spec="Xrrrrr",
        header=("\\textbf{Provenance} & \\textbf{n} & "
                "\\multicolumn{2}{c}{\\textbf{Embedding}} & "
                "\\multicolumn{2}{c}{\\textbf{\\gls{llm}}} \\\\\n"
                "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
                " & & \\textbf{$\\Delta$\\%} & \\textbf{$p$} & "
                "\\textbf{$\\Delta$\\%} & \\textbf{$p$}"),
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def export_paper_tables(
    processed_cases: List[Dict[str, Any]],
    sentence_sim: Dict[Tuple[str, str], Dict[str, float]],
    output_dir: str,
    llm_scores: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tests = _compute_paired_tests(sentence_sim, llm_scores or {})

    _export_bridging_table(processed_cases, out / "bridging_counts_supp.tex")
    _export_deterministic_table(processed_cases, out / "deterministic_metrics.tex")
    if sentence_sim:
        _export_embedding_table(sentence_sim, out / "embedding_similarity.tex", tests)
        _export_embedding_sentence_table(processed_cases, sentence_sim, out / "embedding_similarity_by_sentence_supp.tex")
    if llm_scores:
        _export_llm_table(processed_cases, llm_scores, out / "llm_scores_supp.tex")
        _export_llm_by_model_table(llm_scores, out / "llm_scores_by_model.tex", tests)
        _export_llm_agreement_table(_compute_llm_agreement(llm_scores), out / "llm_agreement.tex")
    if sentence_sim and llm_scores and any(c.get('provenance') for c in processed_cases):
        _export_provenance_table(processed_cases, sentence_sim, llm_scores, out / "provenance_breakdown.tex")

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


def _longtable(caption: str, label: str, col_spec: str,
               header: str, body: str) -> str:
    """Page-breaking table for the supplementary document (plain column
    types only; requires \\usepackage{longtable})."""
    return (
        "\\begin{longtable}{" + col_spec + "}\n"
        f"\\caption{{{caption}}}\\label{{{label}}} \\\\\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        f"\\multicolumn{{{len(col_spec.replace(' ', ''))}}}{{l}}"
        f"{{\\small\\emph{{(continued)}}}} \\\\\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}"
        "\\end{longtable}\n"
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
    table = _longtable(
        caption="Bridging triple counts per test sentence (sentence labels refer to Supplementary Table~\\ref{tab:sentences-full}).",
        label="tab:bridges-sentence",
        col_spec="lrr",
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
    tests: Optional[Dict[str, Any]] = None,
) -> None:
    tests = tests or _compute_paired_tests(sentence_sim, {})

    rows = []
    n = 0
    for model_name, t in sorted(tests['emb'].items()):
        short = model_name.split('/')[-1]
        sign = '+' if (t['delta_pct'] or 0) >= 0 else ''
        n = t['n']
        rows.append(
            f"\\texttt{{{short}}} & {_fmt_mean_ci(t['mean_b'], t['ci_b'])} & "
            f"{_fmt_mean_ci(t['mean_h'], t['ci_h'])} & {sign}{t['delta_pct']:.2f}\\% & "
            f"{_fmt_p(t['p'])} & {_fmt_r(t['r_rb'])} \\\\"
        )

    overall = tests.get('emb_overall')
    if overall is not None:
        sign = '+' if (overall['delta_pct'] or 0) >= 0 else ''
        rows.append("\\midrule")
        rows.append(
            f"\\textbf{{Mean}} & {_fmt_mean_ci(overall['mean_b'], overall['ci_b'])} & "
            f"{_fmt_mean_ci(overall['mean_h'], overall['ci_h'])} & {sign}{overall['delta_pct']:.2f}\\% & "
            f"{_fmt_p(overall['p'])} & {_fmt_r(overall['r_rb'])} \\\\"
        )

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "Embedding similarity: per-triple top-5 cosine similarity, "
            f"mean $\\pm$95\\% CI over all $n={n}$ test sentences. "
            "$p$ is a two-sided Wilcoxon signed-rank test on the per-sentence "
            "(CN, HOnK) pairs; $r$ is the matched-pairs rank-biserial effect "
            "size. The Mean row tests the per-sentence average across models."
        ),
        label="tab:emb",
        col_spec="Xrrrrr",
        header=("\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & "
                "\\textbf{$\\Delta$\\%} & \\textbf{$p$} & \\textbf{$r$}"),
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
        rows.append(f"{ref} & {avg_b:.2f} & {avg_h:.2f} & {_delta_str(avg_b, avg_h)} \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h, math_mode=True)
    body = "\n".join(rows) + "\n"
    table = _longtable(
        caption=(
            "LLM relevance scores per test sentence "
            "(mean across all LLMs and \\texttt{llm\\_iterations} runs; "
            "sentence labels refer to Supplementary Table~\\ref{tab:sentences-full})."
        ),
        label="tab:llm-sentence",
        col_spec="lrrr",
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
        rows.append(f"{ref} & {avg_b:.2f} & {avg_h:.2f} & {_delta_str(avg_b, avg_h)} \\\\")
        all_b.append(avg_b)
        all_h.append(avg_h)

    _append_mean_row(rows, all_b, all_h, math_mode=True)
    body = "\n".join(rows) + "\n"
    table = _longtable(
        caption="Embedding similarity per test sentence (mean across all embedding models; sentence labels refer to Supplementary Table~\\ref{tab:sentences-full}).",
        label="tab:emb-sentence",
        col_spec="lrrr",
        header="\\textbf{Sentence} & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%}",
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def _export_llm_by_model_table(
    llm_scores: Dict[Tuple[str, str], Dict[str, Any]],
    path: Path,
    tests: Optional[Dict[str, Any]] = None,
) -> None:
    tests = tests or _compute_paired_tests({}, llm_scores)

    rows = []
    n = 0
    for model_name, t in sorted(tests['llm_by_judge'].items()):
        short = model_name.replace('_', r'\_')
        sign = '+' if (t['delta_pct'] or 0) >= 0 else ''
        n = t['n']
        rows.append(
            f"\\texttt{{{short}}} & {_fmt_mean_ci(t['mean_b'], t['ci_b'], digits=2)} & "
            f"{_fmt_mean_ci(t['mean_h'], t['ci_h'], digits=2)} & {sign}{t['delta_pct']:.2f}\\% & "
            f"{_fmt_p(t['p'])} & {_fmt_r(t['r_rb'])} \\\\"
        )

    overall = tests.get('llm_avg')
    if overall is not None:
        sign = '+' if (overall['delta_pct'] or 0) >= 0 else ''
        rows.append("\\midrule")
        rows.append(
            f"\\textbf{{Mean}} & {_fmt_mean_ci(overall['mean_b'], overall['ci_b'], digits=2)} & "
            f"{_fmt_mean_ci(overall['mean_h'], overall['ci_h'], digits=2)} & {sign}{overall['delta_pct']:.2f}\\% & "
            f"{_fmt_p(overall['p'])} & {_fmt_r(overall['r_rb'])} \\\\"
        )

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "\\gls{llm} relevance scores by judge model, mean $\\pm$95\\% CI "
            f"over all $n={n}$ test sentences "
            "(\\texttt{llm\\_iterations} runs each). $p$ is a two-sided "
            "Wilcoxon signed-rank test on the per-sentence (CN, HOnK) pairs; "
            "$r$ is the matched-pairs rank-biserial effect size. The Mean row "
            "tests the per-sentence average across judges (the headline metric)."
        ),
        label="tab:llm",
        col_spec="Xrrrrr",
        header=("\\textbf{Model} & \\textbf{CN} & \\textbf{HOnK} & "
                "\\textbf{$\\Delta$\\%} & \\textbf{$p$} & \\textbf{$r$}"),
        body=body,
    )
    path.write_text(table, encoding='utf-8')


def write_equal_context_table(
    budgets: List[int],
    csv_dir: str,
    out_path: str,
) -> None:
    """Equal-context ablation table: per context budget, the CN/HOnK means and
    the paired significance of the judge-averaged LLM gap, read from the
    per-budget CSVs written by run_context_sweep.py."""
    rows = []
    for budget in budgets:
        results_csv = Path(csv_dir) / f"evaluation_results_ctx{budget}.csv"
        if not results_csv.is_file():
            logger.warning("Equal-context: missing %s, skipping budget %d", results_csv, budget)
            continue

        emb_by_sentence: Dict[str, List[Tuple[float, float]]] = {}
        llm_by_sentence: Dict[str, Tuple[float, float]] = {}
        with results_csv.open(encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                sentence = row['Test Sentence']
                emb_by_sentence.setdefault(sentence, []).append(
                    (float(row['Baseline Similarity']), float(row['HOnK Similarity'])))
                if row.get('Avg Baseline LLM Score'):
                    llm_by_sentence[sentence] = (
                        float(row['Avg Baseline LLM Score']),
                        float(row['Avg HOnK LLM Score']))

        emb_b = [sum(p[0] for p in pairs) / len(pairs) for pairs in emb_by_sentence.values()]
        emb_h = [sum(p[1] for p in pairs) / len(pairs) for pairs in emb_by_sentence.values()]
        emb_t = _paired_test(emb_b, emb_h)
        llm_t = _paired_test([v[0] for v in llm_by_sentence.values()],
                             [v[1] for v in llm_by_sentence.values()])

        def _delta(t: Dict[str, Any]) -> str:
            if t['delta_pct'] is None:
                return "---"
            sign = '+' if t['delta_pct'] >= 0 else ''
            return f"{sign}{t['delta_pct']:.2f}\\%"

        rows.append(
            f"{budget} & {emb_t['mean_b']:.3f} & {emb_t['mean_h']:.3f} & {_delta(emb_t)} & "
            f"{llm_t['mean_b']:.2f} & {llm_t['mean_h']:.2f} & {_delta(llm_t)} & "
            f"{_fmt_p(llm_t['p'])} \\\\"
        )

    body = "\n".join(rows) + "\n"
    table = _booktabs_table(
        caption=(
            "Equal-context ablation: both systems truncated to the same "
            "context budget of $k$ triples. Embedding values are per-sentence "
            "means across embedding models; \\gls{llm} values are "
            "judge-averaged; $p$ is a two-sided Wilcoxon signed-rank test on "
            "the per-sentence \\gls{llm} pairs."
        ),
        label="tab:equalctx",
        col_spec="lrrrrrrr",
        header=("\\textbf{$k$} & \\multicolumn{3}{c}{\\textbf{Embedding}} & "
                "\\multicolumn{4}{c}{\\textbf{\\gls{llm}}} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
                " & \\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%} & "
                "\\textbf{CN} & \\textbf{HOnK} & \\textbf{$\\Delta$\\%} & \\textbf{$p$}"),
        body=body,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(table, encoding='utf-8')
    logger.info("Equal-context table written to '%s'", out_path)


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
