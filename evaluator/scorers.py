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
    if not context.strip():
        return 0.0

    cleaned = _clean_context(context)
    verbalised: List[str] = []
    for line in cleaned.splitlines():
        m = _TRIPLE_RE.match(line.strip())
        if m:
            verbalised.append(_verbalise(m.group(1), m.group(2), m.group(3)))

    if not verbalised:
        return 0.0

    sentence_emb = embedder.encode(sentence, convert_to_tensor=True)
    triple_embs = embedder.encode(verbalised, batch_size=64, convert_to_tensor=True)

    sims = st_util.cos_sim(sentence_emb, triple_embs)[0]

    k = min(top_k, len(verbalised))
    return float(sims.topk(k).values.mean().item())


_TRIPLE_RE = re.compile(r'^\((.+?) -> (.+?) -> (.+?)\)$')

_RELATION_GLOSSARY = (
    "Relation glossary: "
    "eq/synonym=equivalent/synonym of (same concept, different form), "
    "isA=subtype/instance of, "
    "partOf=component/member of, "
    "formOf=variant form of the same word, "
    "alternativeOf=alternative spelling or regional variant of, "
    "relatedTo=general semantic association, "
    "derivedFrom/etymologicallyRelatedTo=morphological or etymological derivation, "
    "neqTo/antonym/distinctFrom=explicitly distinct or opposite concepts, "
    "similarTo=resembles or is comparable to, "
    "mannerOf=a way or style of performing, "
    "hasContext/atLocation=belongs to this domain or location, "
    "compositeFormWith=forms a compound expression with, "
    "capableOf=agent can perform, receivesAction=acted upon by, "
    "usedFor=the typical purpose or function of, "
    "causes=leads to or produces as a result, "
    "instanceOf=a specific example of (equivalent to isA for individuals), "
    "madeOf=composed of this material or substance, "
    "hasProperty=has this attribute or characteristic, "
    "hasA=possesses or contains."
)


def _clean_term(t: str) -> str:
    """Decode percent-encoding and strip trailing digit suffixes."""
    return re.sub(r'\d+$', '', unquote(t))


# Natural-language templates for verbalising ontology triples before embedding.
# Underscores in subject/object are replaced with spaces after substitution.
_VERBALISE_TEMPLATES = {
    'eq':                       '{s} is equivalent to {o}',
    'synonym':                  '{s} is a synonym of {o}',
    'isA':                      '{s} is a type of {o}',
    'instanceOf':               '{s} is an instance of {o}',
    'partOf':                   '{s} is part of {o}',
    'formOf':                   '{s} is a form of {o}',
    'alternativeOf':            '{s} is an alternative form of {o}',
    'relatedTo':                '{s} is related to {o}',
    'derivedFrom':              '{s} is derived from {o}',
    'etymologicallyRelatedTo':  '{s} is etymologically related to {o}',
    'neqTo':                    '{s} is distinct from {o}',
    'antonym':                  '{s} is the opposite of {o}',
    'distinctFrom':             '{s} is distinct from {o}',
    'similarTo':                '{s} is similar to {o}',
    'hasContext':               '{s} belongs to the domain of {o}',
    'atLocation':               '{s} is found at {o}',
    'compositeFormWith':        '{s} forms a compound expression with {o}',
    'capableOf':                '{s} is capable of {o}',
    'receivesAction':           '{s} receives the action {o}',
    'usedFor':                  '{s} is used for {o}',
    'causes':                   '{s} causes {o}',
    'mannerOf':                 '{s} is a manner of {o}',
    'madeOf':                   '{s} is made of {o}',
    'hasProperty':              '{s} has the property of being {o}',
    'hasA':                     '{s} has {o}',
}


def _verbalise(s: str, p: str, o: str) -> str:
    """Convert a cleaned (s, p, o) triple to a natural-language sentence.

    Falls back to '<s> <p> <o>' for unknown predicates so the term text is
    still present in the embedding rather than the raw arrow notation.
    """
    s_text = s.replace('_', ' ')
    o_text = o.replace('_', ' ')
    template = _VERBALISE_TEMPLATES.get(p)
    if template:
        return template.format(s=s_text, o=o_text)
    # Unknown predicate: use a generic readable form
    p_text = re.sub(r'([A-Z])', r' \1', p).strip().lower()  # camelCase → words
    return f'{s_text} {p_text} {o_text}'


def _clean_context(context: str) -> str:
    lines = []
    seen: set = set()
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
            cleaned = f"({_clean_term(s)} -> {p_clean} -> {_clean_term(o)})"
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(cleaned)
        else:
            lines.append(raw_line)
    return '\n'.join(lines)


def query_llm(
    model_name: str,
    sentence: str,
    context: str,
    keywords: List[str],
    bridging_triples: List[Tuple[str, str, str]],
    temperature: float = 0.0,
) -> Tuple[str, int, str]:
    kw_str = ", ".join(keywords)
    cleaned_context = _clean_context(context) if context else ''

    # Deduplicate bridging triples (case-insensitive) before counting and displaying
    kw_lower = {k.lower() for k in keywords}
    seen_bridges: set = set()
    deduped_bridges: List[Tuple[str, str, str]] = []
    for s, p, o in bridging_triples:
        key = (_clean_term(s).lower(), _clean_term(p).lower(), _clean_term(o).lower())
        if key not in seen_bridges:
            seen_bridges.add(key)
            deduped_bridges.append((s, p, o))

    # Sort so exact keyword-to-keyword bridges appear first, these are the
    # most informative and should prime the LLM's initial impression.
    def _bridge_priority(triple: Tuple[str, str, str]) -> int:
        s_clean, o_clean = _clean_term(triple[0]).lower(), _clean_term(triple[2]).lower()
        exact_matches = (s_clean in kw_lower) + (o_clean in kw_lower)
        return -exact_matches  # higher match count → lower sort key → first

    deduped_bridges.sort(key=_bridge_priority)
    bridge_count = len(deduped_bridges)

    if bridge_count > 0:
        examples = "\n".join(
            f"  {i + 1}. ({_clean_term(s)} -{_clean_term(p)}-> {_clean_term(o)})"
            for i, (s, p, o) in enumerate(deduped_bridges[:5])
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
        "Important: this ontology merges multiple knowledge bases and may contain triples "
        "from unrelated word senses of the same keyword (polysemy). When scoring, focus only "
        "on the connections that are semantically relevant to the sentence; do not penalise "
        "for off-topic sense connections that are clearly unrelated to the sentence meaning.\n\n"
        "Scoring guide (1-10):\n"
        "- 1-3: Keywords are absent from the context or entirely unconnected.\n"
        "- 4-5: Keywords appear individually but share no meaningful links.\n"
        "- 6-7: Keywords are indirectly connected (e.g. via shared hypernyms or related terms); "
        "relation types are loosely consistent with the sentence meaning.\n"
        "- 8: A single, strong direct connection exists (e.g., one 'eq' or 'isA' link), but the context lacks deeper supporting evidence.\n"
        "- 9: Multiple diverse semantic connections exist. The context corroborates the relationship from multiple angles (e.g., a synonym link PLUS a taxonomic or property link).\n"
        "- 10: Direct connections exist, the relation type precisely matches the sentence intent, "
        "and the context is rich enough to fully resolve the query.\n\n"
        "Consider both connection presence AND semantic type fit. "
        "An 'eq' or 'synonym' link is the strongest evidence when the sentence asserts that "
        "two things share the same meaning under different names. "
        "A 'neqTo' link is strong evidence when the sentence is about distinguishing or confusing "
        "two concepts. A 'relatedTo' link is weaker than 'isA', 'partOf', or 'eq' for sentences "
        "implying equivalence or hierarchy.\n\n"
        "Respond with exactly two lines:\n"
        "Interpretation: [one sentence summarising the overall fit]\n"
        "Score: [single integer 1-10]"
    )
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": temperature, "seed": 42},
        )
        content = response.get('message', {}).get('content', '').strip()
        score_match = re.search(r'Score:\s*(\d+)', content, re.IGNORECASE)
        if score_match is None:
            # A malformed response must not be scored as an in-range value: 0 is
            # outside the 1-10 scale and would silently bias the mean. Flag it so
            # the caller can exclude it rather than average a fabricated score.
            logger.warning("No parseable 'Score:' from %s; marking invalid. Response: %r",
                           model_name, content[:200])
            return content, None, prompt
        return content, int(score_match.group(1)), prompt
    except Exception as e:
        logger.error("LLM call to %s failed: %s", model_name, e)
        return f"Error: {e}", None, prompt
