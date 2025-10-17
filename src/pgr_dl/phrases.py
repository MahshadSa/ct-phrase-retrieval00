# canonical phrases, synonyms, prompts
from __future__ import annotations
from typing import List, Dict, Set

CANONICAL_LIST: List[str] = [
    "liver lesion",
    "renal mass",
    "splenic lesion",
    "lung nodule",
    "enlarged lymph node",
    "bone lesion",
]
CANONICAL: Set[str] = set(CANONICAL_LIST)

SYNONYMS: Dict[str, str] = {
    "hepatic lesion": "liver lesion",
    "renal lesion": "renal mass",
    "kidney tumor": "renal mass",
    "pulmonary nodule": "lung nodule",
}

PROMPTS: List[str] = [
    "A medical CT scan showing a {phrase}.",
    "A CT image with {phrase}.",
    "A radiology CT slice depicting {phrase}.",
    "A diagnostic CT scan containing {phrase}.",
]
# de-duplicate while preserving order
_PROMPTS_UNIQUE: List[str] = list(dict.fromkeys(PROMPTS))


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def canonicalize(q: str) -> str:
    """Map an input query to a canonical phrase if possible."""
    qn = _norm(q)
    if not qn:
        raise ValueError("phrase must be non-empty")
    return qn if qn in CANONICAL else SYNONYMS.get(qn, qn)


def make_prompts(phrase: str) -> List[str]:
    """Return templated prompts for a phrase (canonicalized)."""
    p = canonicalize(phrase)
    return [t.format(phrase=p) for t in _PROMPTS_UNIQUE]


def tags_match_phrase(body_part: str | None, lesion_type: str | None, phrase: str) -> bool:
    """Minimal rule to decide if dataset tags match a canonical phrase."""
    ph = canonicalize(phrase)
    bp = _norm(body_part)
    lt = _norm(lesion_type)

    if ph == "liver lesion":
        return bp == "liver"
    if ph == "renal mass":
        return bp in {"kidney", "renal"}
    if ph == "splenic lesion":
        return bp == "spleen"
    if ph == "lung nodule":
        return bp == "lung"
    if ph == "enlarged lymph node":
        return lt in {"lymph node", "lymphnode", "node"}
    if ph == "bone lesion":
        return bp == "bone"
    return False
