import logging
import re
from urllib.parse import unquote
from typing import List, Tuple

import ollama
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

logger = logging.getLogger(__name__)


def ensure_models_pulled(models: List[str]) -> None:
    try:
        available = {m.model for m in ollama.list().models}
    except Exception as e:
        logger.warning("Could not reach Ollama daemon: %s", e)
        return

    for model in models:
        if model not in available:
            logger.info("Pulling '%s' from Ollama registry...", model)
            try:
                for chunk in ollama.pull(model, stream=True):
                    status = getattr(chunk, 'status', '') or ''
                    completed = getattr(chunk, 'completed', None)
                    total = getattr(chunk, 'total', None)
                    if completed is not None and total:
                        pct = (completed / total) * 100
                        print(f"\r  {model}: {status} {pct:.1f}%", end='', flush=True)
                    elif status:
                        print(f"\r  {model}: {status}            ", end='', flush=True)
                print()
                logger.info("Pulled '%s'.", model)
            except Exception as e:
                logger.error("Failed to pull '%s': %s", model, e)


def compute_embedding_similarity(
    embedder: SentenceTransformer,
    sentence: str,
    context: str,
    top_k: int = 5,
) -> float:
    if not context:
        return 0.0

    # Clean then verbalise each triple into natural language before encoding.
    # Raw arrow notation like "(ferryboat -> eq -> ferry)" embeds poorly under
    # sentence transformers; "ferryboat is equivalent to ferry" maps naturally
    # into the model's representation space and scores much closer to sentences
    # that describe the same relationship.
    cleaned = _clean_context(context)
    verbalized: List[str] = []
    for line in cleaned.splitlines():
        m = _TRIPLE_RE.match(line.strip())
        if m:
            verbalized.append(_verbalize(m.group(1), m.group(2), m.group(3)))
    if not verbalized:
        return 0.0

    sentence_emb = embedder.encode(sentence, convert_to_tensor=True)
    triple_embs = embedder.encode(verbalized, batch_size=64, convert_to_tensor=True)

    sims = st_util.cos_sim(sentence_emb, triple_embs)[0]
    k = min(top_k, len(verbalized))
    return float(sims.topk(k).values.mean().item())


_TRIPLE_RE = re.compile(r'^\((.+?) -> (.+?) -> (.+?)\)$')

_RELATION_GLOSSARY = (
    "Relation glossary: "
    "eq=equivalent/synonym of (same concept, different form), "
    "isA=subtype/instance-of, partOf=component/member-of, "
    "synonymOf/formOf=variant form of the same word, "
    "relatedTo=general semantic association, "
    "derivedFrom=morphological or etymological derivation, "
    "neqTo=explicitly distinct concepts (often confused or compared), "
    "hasContext=belongs to this domain, compositeFormWith=forms a compound expression with, "
    "capableOf=agent can perform, receivesAction=acted upon by."
)


def _clean_term(t: str) -> str:
    """Decode percent-encoding and strip trailing digit suffixes."""
    return re.sub(r'\d+$', '', unquote(t))


# Natural-language templates for verbalising ontology triples before embedding.
# Underscores in subject/object are replaced with spaces after substitution.
_VERBALIZE_TEMPLATES = {
    'eq':               '{s} is equivalent to {o}',
    'isA':              '{s} is a type of {o}',
    'partOf':           '{s} is part of {o}',
    'synonymOf':        '{s} is a synonym of {o}',
    'formOf':           '{s} is a form of {o}',
    'relatedTo':        '{s} is related to {o}',
    'derivedFrom':      '{s} is derived from {o}',
    'neqTo':            '{s} is distinct from {o}',
    'hasContext':       '{s} belongs to the domain of {o}',
    'compositeFormWith': '{s} forms a compound expression with {o}',
    'capableOf':        '{s} is capable of {o}',
    'receivesAction':   '{s} receives the action {o}',
    'mannerOf':         '{s} is a manner of {o}',
    'alternativeOf':    '{s} is an alternative form of {o}',
}


def _verbalize(s: str, p: str, o: str) -> str:
    """Convert a cleaned (s, p, o) triple to a natural-language sentence.

    Falls back to '<s> <p> <o>' for unknown predicates so the term text is
    still present in the embedding rather than the raw arrow notation.
    """
    s_text = s.replace('_', ' ')
    o_text = o.replace('_', ' ')
    template = _VERBALIZE_TEMPLATES.get(p)
    if template:
        return template.format(s=s_text, o=o_text)
    # Unknown predicate: use a generic readable form
    p_text = re.sub(r'([A-Z])', r' \1', p).strip().lower()  # camelCase → words
    return f'{s_text} {p_text} {o_text}'


def _clean_context(context: str) -> str:
    """Decode URIs, strip digit suffixes, normalise predicate case, drop label triples."""
    lines = []
    for raw_line in context.splitlines():
        stripped = raw_line.strip()
        m = _TRIPLE_RE.match(stripped)
        if m:
            s, p, o = m.group(1), m.group(2), m.group(3)
            # Normalise to camelCase start: "DerivedFrom" -> "derivedFrom"
            p_clean = _clean_term(p)
            p_clean = p_clean[0].lower() + p_clean[1:] if p_clean else p_clean
            if p_clean in ('label', 'type'):
                continue  # metadata annotations — carry no semantic information
            lines.append(f"({_clean_term(s)} -> {p_clean} -> {_clean_term(o)})")
        else:
            lines.append(raw_line)
    return '\n'.join(lines)


def query_llm(
    model_name: str,
    sentence: str,
    context: str,
    keywords: List[str],
    bridging_triples: List[Tuple[str, str, str]],
) -> Tuple[str, int, str]:
    kw_str = ", ".join(keywords)
    bridge_count = len(bridging_triples)
    cleaned_context = _clean_context(context) if context else ''

    if bridge_count > 0:
        examples = "\n".join(
            f"  {i + 1}. ({_clean_term(s)} -{_clean_term(p)}-> {_clean_term(o)})"
            for i, (s, p, o) in enumerate(bridging_triples[:5])
        )
        bridge_summary = (
            f"{bridge_count} direct connection(s) found between keywords:\n"
            f"{examples}"
            f"{chr(10) + '  ...' if bridge_count > 5 else ''}\n"
        )
    else:
        bridge_summary = "0 direct connections found between keywords.\n"

    prompt = (
        "You are an ontology quality evaluator. Give ONE overall score (1-10) for how well "
        "the Ontological Context supports understanding the semantic relationship between all "
        "the keywords as used in the sentence. Do NOT score keywords individually.\n\n"
        f'Sentence: "{sentence}"\n'
        f"Keywords: {kw_str}\n\n"
        f"Direct connections between keywords:\n{bridge_summary}\n"
        f"{_RELATION_GLOSSARY}\n\n"
        f"Ontological Context:\n{cleaned_context if cleaned_context else 'NO CONTEXT.'}\n\n"
        "Scoring guide (1-10):\n"
        "- 1-3: Keywords are absent from the context or entirely unconnected.\n"
        "- 4-5: Keywords appear individually but share no meaningful links.\n"
        "- 6-7: Keywords are indirectly connected (e.g. via shared hypernyms or related terms); "
        "relation types are loosely consistent with the sentence meaning.\n"
        "- 8-9: Direct connections exist and the relation type is semantically consistent "
        "with how the keywords relate in the sentence.\n"
        "- 10: Direct connections exist, the relation type precisely matches the sentence intent, "
        "and the context is rich enough to fully resolve the query.\n\n"
        "Consider both connection presence AND semantic type fit. A 'neqTo' link is strong "
        "evidence when the sentence is about distinguishing or confusing two concepts. "
        "A 'relatedTo' link is weaker than 'isA' or 'partOf' for sentences implying hierarchy.\n\n"
        "Respond with exactly two lines:\n"
        "Interpretation: [one sentence summarising the overall fit]\n"
        "Score: [single integer 1-10]"
    )
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        content = response.get('message', {}).get('content', '').strip()
        score_match = re.search(r'Score:\s*(\d+)', content, re.IGNORECASE)
        return content, int(score_match.group(1)) if score_match else 0, prompt
    except Exception as e:
        return f"Error: {e}", 0, prompt
