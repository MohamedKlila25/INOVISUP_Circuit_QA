# -*- coding: utf-8 -*-
"""
Hybrid.Detection.Assign_Ids — attribution des identifiants (R1, C1...).

Choix : numérotation par POSITION (ordre de lecture, haut->bas puis
gauche->droite dans chaque ligne), pas par le graphe de connexions.

Pourquoi pas par le graphe : ça créerait une dépendance dans le mauvais
sens. Si Hybrid.Wires.Tracer se trompe sur une connexion (heuristique,
donc pas infaillible), la numérotation des composants se casserait
aussi, et l'erreur se propagerait à l'étage OCR suivant. En numérotant
par position, chaque étage reste indépendant : même si le traçage des
fils est imparfait, les identifiants restent corrects et stables.
"""
from __future__ import annotations

from Common.Schemas import DetectedComponent

# Préfixe d'ID par classe — aligné sur la convention déjà utilisée dans
# Data_Generation.Circuit_Catalog, pour rester cohérent entre vérité
# terrain (génération) et prédictions (inférence).
ID_PREFIX: dict[str, str] = {
    "resistor": "R", "capacitor": "C", "polarized_capacitor": "C",
    "inductor": "L",
    "diode": "D", "led": "D", "zener_diode": "DZ",
    "npn_transistor": "Q", "pnp_transistor": "Q",
    "vsource": "V", "battery": "V",
    "ground": "GND",
    "switch": "SW", "fuse": "F", "opamp": "U",
    "gate_and": "U", "gate_or": "U", "gate_xor": "U", "gate_not": "U",
    "gate_nand": "U", "gate_nor": "U", "gate_xnor": "U",
}


def _row_clusters(components: list[DetectedComponent],
                  row_tolerance: float) -> list[list[DetectedComponent]]:
    """Regroupe les composants en lignes de lecture : deux composants
    sont sur la même ligne si leurs centres verticaux sont à moins de
    `row_tolerance` px l'un de l'autre. Comparable à la façon dont un
    humain balaie un schéma du regard."""
    by_y = sorted(components, key=lambda c: (c.bbox.y0 + c.bbox.y1) / 2)
    rows: list[list[DetectedComponent]] = []
    for comp in by_y:
        cy = (comp.bbox.y0 + comp.bbox.y1) / 2
        if rows and abs(cy - _row_center(rows[-1])) <= row_tolerance:
            rows[-1].append(comp)
        else:
            rows.append([comp])
    return rows


def _row_center(row: list[DetectedComponent]) -> float:
    ys = [(c.bbox.y0 + c.bbox.y1) / 2 for c in row]
    return sum(ys) / len(ys)


def assign_ids(components: list[DetectedComponent],
               row_tolerance: float = 40.0) -> list[DetectedComponent]:
    """Retourne une NOUVELLE liste de composants, avec `id` assigné par
    ordre de lecture (lignes haut->bas, gauche->droite dans la ligne).
    Numérotation séparée par classe (R1, R2... indépendant de C1, C2...).
    """
    rows = _row_clusters(components, row_tolerance)
    counters: dict[str, int] = {}
    out: list[DetectedComponent] = []

    for row in rows:
        row_sorted = sorted(row, key=lambda c: (c.bbox.x0 + c.bbox.x1) / 2)
        for comp in row_sorted:
            prefix = ID_PREFIX.get(comp.cls, comp.cls[:2].upper())
            counters[prefix] = counters.get(prefix, 0) + 1
            new_comp = comp.model_copy(
                update={"id": f"{prefix}{counters[prefix]}"})
            out.append(new_comp)
    return out
