# -*- coding: utf-8 -*-
"""
Llm.Prompting — prompt construction and answer grading for the netlist
QA benchmark.

Kept separate from the model client (Llm.Qwen_Client) because this
part is pure Python — fully testable without loading any model,
exactly like Hybrid.Ocr.Parsing was kept separate from the OCR engine
itself.

Answers are graded by NORMALIZED STRING MATCH, not free-form judgment:
questions are deliberately designed (see Graph.Qa_Generator) to have a
short, canonical answer — a count, "Oui"/"Non", a class label, or a
component value — so the model is instructed to answer in exactly that
form, and grading stays simple and reproducible. A structured-output
schema (CircuitAnswer, already defined) is the natural next step if
free-form answers turn out to need it — start simple, escalate only if
measured to be necessary, same principle used for the OCR engine
choice.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond à des questions sur un circuit "
    "électrique décrit sous forme de netlist. Réponds UNIQUEMENT à "
    "partir des informations de la netlist fournie, jamais par "
    "supposition. Donne une réponse la plus courte possible, sans "
    "phrase, sans explication : "
    "pour un nombre, écris UNIQUEMENT le chiffre (ex: '3', jamais "
    "'trois' ni 'Il y a 3') ; "
    "pour une classe de composant, recopie le mot EXACTEMENT comme il "
    "apparaît dans la netlist (ex: 'resistor', pas de traduction) ; "
    "pour une question oui/non, réponds 'Oui' ou 'Non' ; "
    "pour une valeur, recopie-la exactement comme dans la netlist."
)


def build_prompt(netlist: str, question: str) -> list[dict]:
    """Builds the chat-format messages for one question, grounded in
    one netlist. Returned as a list of {"role", "content"} dicts —
    the format `tokenizer.apply_chat_template` expects."""
    user_content = f"Netlist :\n{netlist}\n\nQuestion : {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


_FRENCH_NUMBERS = {
    "zéro": "0", "zero": "0", "aucun": "0", "aucune": "0",
    "un": "1", "une": "1", "deux": "2", "trois": "3", "quatre": "4",
    "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9",
    "dix": "10",
}


def normalize_answer(text: str) -> str:
    """Normalizes a raw model answer for comparison: strips
    whitespace/punctuation, lowercases, collapses internal whitespace.
    Deliberately does NOT alter unit symbols (Ω, µ) or numbers — only
    presentation noise a model might add around an otherwise-correct
    answer ("La réponse est : 3." -> "3").

    Also maps common French number WORDS to digits ("une" -> "1") as a
    safety net: the system prompt now explicitly asks for digits only,
    but small models do not follow instructions with 100% reliability,
    and this costs nothing to handle defensively.
    """
    t = text.strip()
    # keep only the first line — models sometimes add a trailing
    # explanation despite the instruction not to
    t = t.split("\n")[0].strip()
    # strip common leading filler phrases
    t = re.sub(r"^(la réponse est|réponse\s*:|il y a)\s*:?\s*",
               "", t, flags=re.IGNORECASE)
    t = t.strip().rstrip(".")
    t = re.sub(r"\s+", " ", t)
    t = t.lower()
    return _FRENCH_NUMBERS.get(t, t)


def is_correct(predicted: str, expected: str) -> bool:
    """Grades one answer. Exact match after normalization, with a
    couple of narrowly-scoped equivalences (yes/no phrasing, and a
    value match that tolerates a missing/extra space before the unit —
    "4.7 kΩ" vs "4.7kΩ" — since that is presentation, not content)."""
    p = normalize_answer(predicted)
    e = normalize_answer(expected)
    if p == e:
        return True

    yes_forms = {"oui", "yes", "vrai", "true"}
    no_forms = {"non", "no", "faux", "false"}
    if e in yes_forms and p in yes_forms:
        return True
    if e in no_forms and p in no_forms:
        return True

    if p.replace(" ", "") == e.replace(" ", ""):
        return True

    return False