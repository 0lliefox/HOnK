#!/usr/bin/env python3
"""build_match_vocabularies.py — build two small OWL vocabularies + a gold
alignment for the "match stage" experiment (independent test of HOnK's manual
schema mappings against automated ontology matching; reviewers R1-W2/R2-W5).

Outputs (into tools/boomer/match/):
  source.owl      one owl:Class per (KB, source label), rdfs:label = the raw
                  source relation/POS term as a matcher would see it.
  canonical.owl   one owl:Class per canonical HOnK label, rdfs:label = the
                  de-camelCased term, rdfs:comment = a NEUTRAL gloss.
  reference.tsv   the gold alignment: source_iri, canonical_iri, kind, flags.

Fairness notes (documented for reviewers):
  * The canonical glosses are standard, independently-authored definitions of
    each relation / part-of-speech category. They are NOT reverse-engineered
    from the source terms that map to them, so they do not leak the gold: a
    matcher must still bridge e.g. "synonyms" -> "equivalence; same meaning".
  * POS-class glosses are derived mechanically from the class name.
"""

import json
import os
import re

from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "match")
GOLD_JSON = os.path.join(HERE, "match", "gold.json")

SRC = Namespace("http://honk.org/src#")
CAN = Namespace("http://honk.org/canon#")

# Neutral, standard glosses for the canonical *relation* classes (lower-case
# initial). Authored from general relation semantics, independent of the source
# labels. Kept deliberately free of the source terms themselves.
RELATION_GLOSS = {
    "atLocation": "the place where something is typically located or found",
    "capableOf": "the subject is able to perform this action or capability",
    "causes": "one event or state brings about another",
    "causesDesire": "something makes an agent want or wish for something",
    "compositeFormWith": "combines with another element to form a compound term",
    "createdBy": "the agent or process that produces or authors something",
    "definedAs": "a defining description or definition of the term",
    "derivedFrom": "originates from or is derived out of a prior source",
    "desires": "an agent wants or wishes for something",
    "entails": "logically implies or necessarily brings with it",
    "eq": "equivalence; the two terms denote the same thing or sense",
    "etymology": "the historical linguistic origin of a word",
    "formOf": "an inflected, variant or alternative form of a base word",
    "hasA": "possession; the subject has this as a component or attribute",
    "hasContext": "the domain, register or context in which a term is used",
    "hasFirstSubevent": "the initial sub-event within a larger process",
    "hasLastSubevent": "the final sub-event within a larger process",
    "hasPrerequisite": "a precondition that must hold before something occurs",
    "hasProperty": "an attribute, quality or property that the subject has",
    "hasSubevent": "a sub-event that occurs as part of a larger event",
    "instanceOf": "an individual instance belonging to a class",
    "isA": "a type-of relationship; the subject is a kind of the object",
    "locatedNear": "typically situated in the vicinity of something",
    "madeOf": "the material or substance the subject is composed of",
    "mannerOf": "a specific way or manner of performing an action",
    "motivatedByGoal": "an action undertaken in order to achieve a goal",
    "neqTo": "inequality or opposition; the two differ or are opposite in sense",
    "partOf": "a component that belongs to a larger whole",
    "product": "the output or product that results from something",
    "receivesAction": "an action can be carried out upon the subject",
    "relatedTo": "a general, unspecified association between two terms",
    "symbolOf": "serves as a symbol or sign standing for something",
    "usedFor": "the typical purpose, function or use of something",
}


def split_camel(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    s = s.replace(".", " ").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def frag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", text)


def readable(label: str) -> str:
    return split_camel(label)


def canonical_gloss(name: str) -> str:
    if name and name[0].islower():                      # relation class
        return RELATION_GLOSS.get(name, f"the {split_camel(name)} relation")
    words = split_camel(name)                            # POS / lexical class
    return f"part-of-speech category: a {words}"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    gold = json.load(open(GOLD_JSON))                    # [src, tgt, kb, kind, swap, neg]

    src_g = Graph(); can_g = Graph()
    for g in (src_g, can_g):
        g.bind("owl", OWL); g.bind("rdfs", RDFS)
    src_g.bind("src", SRC); can_g.bind("canon", CAN)
    # deeponto/OWLAPI requires a declared ontology IRI on each file
    src_g.add((URIRef("http://honk.org/src"), RDF.type, OWL.Ontology))
    can_g.add((URIRef("http://honk.org/canon"), RDF.type, OWL.Ontology))

    seen_src, seen_can, ref = {}, set(), []
    for source, target, kb, kind, swap, neg in gold:
        s_iri = URIRef(SRC[f"{kb}.{frag(source)}"])
        c_iri = URIRef(CAN[frag(target)])

        if s_iri not in seen_src:
            src_g.add((s_iri, RDF.type, OWL.Class))
            src_g.add((s_iri, RDFS.label, Literal(readable(source))))
            seen_src[s_iri] = True
        if c_iri not in seen_can:
            can_g.add((c_iri, RDF.type, OWL.Class))
            can_g.add((c_iri, RDFS.label, Literal(readable(target))))
            can_g.add((c_iri, RDFS.comment, Literal(canonical_gloss(target))))
            seen_can.add(c_iri)
        ref.append((str(s_iri), str(c_iri), kind,
                    "identity" if source.lower() == target.lower() else "nonidentity",
                    "swap" if swap else "", "neg" if neg else ""))

    src_path = os.path.join(OUT, "source.owl")
    can_path = os.path.join(OUT, "canonical.owl")
    src_g.serialize(destination=src_path, format="xml")
    can_g.serialize(destination=can_path, format="xml")

    ref_path = os.path.join(OUT, "reference.tsv")
    with open(ref_path, "w", encoding="utf-8") as fh:
        fh.write("source_iri\tcanonical_iri\tkind\tclass\tswap\tneg\n")
        for row in ref:
            fh.write("\t".join(row) + "\n")

    print(f"source classes:    {len(seen_src)}  -> {src_path}")
    print(f"canonical classes: {len(seen_can)} -> {can_path}")
    print(f"gold correspondences: {len(ref)}   -> {ref_path}")
    print(f"  non-identity: {sum(1 for r in ref if r[3]=='nonidentity')}"
          f" | swap: {sum(1 for r in ref if r[4])}"
          f" | negated: {sum(1 for r in ref if r[5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
