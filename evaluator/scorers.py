import logging
import re
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

    triple_lines = [
        line for line in context.splitlines()
        if line.startswith('(') and '->' in line
    ]
    if not triple_lines:
        return 0.0

    sentence_emb = embedder.encode(sentence, convert_to_tensor=True)
    triple_embs = embedder.encode(triple_lines, batch_size=64, convert_to_tensor=True)

    sims = st_util.cos_sim(sentence_emb, triple_embs)[0]
    k = min(top_k, len(sims))
    return float(sims.topk(k).values.mean().item())


def query_llm(
    model_name: str,
    sentence: str,
    context: str,
    keywords: List[str],
    bridging_triples: List[Tuple[str, str, str]],
) -> Tuple[str, int]:
    kw_str = ", ".join(keywords)
    bridge_count = len(bridging_triples)

    if bridge_count > 0:
        examples = "\n".join(
            f"  {i + 1}. ({s} -> {p} -> {o})"
            for i, (s, p, o) in enumerate(bridging_triples[:5])
        )
        bridge_summary = (
            f"IMPORTANT: {bridge_count} direct connection(s) found between keywords:\n"
            f"{examples}"
            f"{chr(10) + '  ...' if bridge_count > 5 else ''}\n"
        )
    else:
        bridge_summary = "IMPORTANT: 0 direct connections found between keywords.\n"

    prompt = (
        "You are an ontology quality evaluator. Assess how well the Ontological Context "
        "supports semantic resolution of the keywords in the given sentence.\n\n"
        f'Sentence: "{sentence}"\n'
        f"Keywords: {kw_str}\n\n"
        f"{bridge_summary}\n"
        f"Ontological Context:\n{context if context else 'NO CONTEXT.'}\n\n"
        "Evaluation criteria (score 1-10):\n"
        "- 1-3: No relevant links. Keywords absent or unconnected.\n"
        "- 4-6: Some keywords appear but are not linked to each other.\n"
        "- 7-8: Most keywords appear; indirect paths exist between them.\n"
        "- 9-10: Direct semantic connections exist between keywords.\n\n"
        "The direct connections listed above are the strongest signal — "
        "a non-zero count should push the score toward 9-10.\n\n"
        "Format: Interpretation: [one sentence] Score: [integer 1-10]"
    )
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        content = response.get('message', {}).get('content', '').strip()
        score_match = re.search(r'Score:\s*(\d+)', content, re.IGNORECASE)
        return content, int(score_match.group(1)) if score_match else 0
    except Exception as e:
        return f"Error: {e}", 0
