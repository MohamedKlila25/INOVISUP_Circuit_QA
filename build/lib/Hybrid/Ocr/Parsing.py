# -*- coding: utf-8 -*-
"""
Hybrid.Ocr.Parsing — interprétation du texte OCR brut en (référence, valeur).

Séparé du reste de l'OCR : cette logique est pure Python (aucune
dépendance à easyocr/torch), donc entièrement testable sans GPU ni
lourde installation.
"""
from __future__ import annotations

import re

_REF_RE = re.compile(r"^[A-Za-z]{1,4}\d{0,3}$")

_VALUE_RE = re.compile(
    r"^-?\d+(?:[.,]\d+)?\s*[pnµmkM]?(?:Ω|F|H|V|A|Hz)(?:\s*(?:AC|DC))?$",
    re.IGNORECASE)

_PART_REF_RE = re.compile(r"^[0-9A-Z]{4,10}$")

# Valeur dont l'unité a été perdue ou confondue par l'OCR. Deux
# confusions dominent en conditions réelles :
#   - Ω lu "0" (glyphe proche) ou "Q" (forme arrondie proche, dominant
#     une fois l'alphabet restreint via allowlist alphanumérique)
#   - "1" lu "l" (L minuscule) ou "I" (i majuscule) selon la police —
#     jamais "L" majuscule (qui reste une lettre légitime : "L1",
#     "TL081", "LM358")
_UNIT_DROPPED_RE = re.compile(r"^-?\d+(?:[.,]\d+)?[pnµmkM]$")
_UNIT_GARBLED_RE = re.compile(r"^-?\d+(?:[.,]\d+)?[pnµmkM][0OoQ]$")

_LEADING_DIGIT_FIX = str.maketrans({"l": "1", "I": "1"})

EXPECTED_UNIT: dict[str, str] = {
    "resistor": "Ω",
    "capacitor": "F", "polarized_capacitor": "F",
    "inductor": "H",
    "vsource": "V", "battery": "V", "zener_diode": "V",
}


def fix_leading_digit_confusion(text: str) -> str:
    """Corrects the '1' -> 'l'/'I' confusion, narrowly scoped: only
    replaces lowercase 'l' and uppercase 'I', never uppercase 'L' —
    which stays legitimate in "L1", "TL081", "LM358"."""
    return text.translate(_LEADING_DIGIT_FIX)


def try_repair_value(text: str, expected_unit: str | None) -> str | None:
    if not expected_unit:
        return None
    t = text.strip()
    if _UNIT_GARBLED_RE.match(t):
        return t[:-1] + expected_unit
    if _UNIT_DROPPED_RE.match(t):
        return t + expected_unit
    return None


def looks_like_component_ref(text: str) -> bool:
    return bool(_REF_RE.match(text.strip()))


def looks_like_value_with_unit(text: str) -> bool:
    return bool(_VALUE_RE.match(text.strip().replace(" ", "")))


def looks_like_part_number(text: str) -> bool:
    return bool(_PART_REF_RE.match(text.strip()))


def parse_ocr_lines(lines: list[str],
                    expected_unit: str | None = None
                    ) -> tuple[str | None, str | None]:
    """Interprète les lignes de texte lues par l'OCR près d'un composant."""
    id_text: str | None = None
    value_text: str | None = None
    leftover: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        fixed = fix_leading_digit_confusion(line)
        fixed_is_better_ref = (
            fixed != line and looks_like_component_ref(fixed)
            and any(ch.isdigit() for ch in fixed)
            and not any(ch.isdigit() for ch in line)
        )

        if id_text is None and fixed_is_better_ref:
            id_text = fixed
            continue
        if id_text is None and looks_like_component_ref(line):
            id_text = line
            continue
        if value_text is None and looks_like_value_with_unit(line):
            value_text = line
            continue

        if fixed != line:
            if id_text is None and looks_like_component_ref(fixed):
                id_text = fixed
                continue
            if value_text is None and looks_like_value_with_unit(fixed):
                value_text = fixed
                continue
            leftover.append(fixed)
            continue

        leftover.append(line)

    if value_text is None:
        for line in leftover:
            if looks_like_part_number(line):
                value_text = line
                break

    if value_text is None:
        for line in leftover:
            repaired = try_repair_value(line, expected_unit)
            if repaired is not None:
                value_text = repaired
                leftover.remove(line)
                break

    if id_text is None or value_text is None:
        for line in leftover:
            if line == value_text:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            for tok in tokens:
                if id_text is None and looks_like_component_ref(tok):
                    id_text = tok
                elif value_text is None and (looks_like_value_with_unit(tok)
                                             or looks_like_part_number(tok)):
                    value_text = tok

    if value_text is None and leftover:
        value_text = leftover[0]

    return id_text, value_text
