# -*- coding: utf-8 -*-
"""
validate_dataset.py
Validation complète de la qualité du dataset généré :
  1. FERMETURE : le circuit forme une boucle fermée (flood fill —
     une boucle fermée enclot au moins une région de fond interne)
  2. COMPOSANTS : chaque bbox contient de l'encre (le symbole)
  3. TERMINAUX : chaque point de borne tombe sur un fil (fenêtre 9×9)
  4. TEXTES : chaque bbox de label contient de l'encre (le texte) et
     chevauche moins de 30% de la bbox de son composant
Utilisable sur échantillons pendant le dev, ou sur tout le dataset après
génération :  python validate_dataset.py --dataset data/circuitvqa_v3
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))


def check_closure(ink: np.ndarray) -> bool:
    """Boucle fermée <=> le fond a >=2 composantes connexes
    (l'extérieur + au moins une région interne enclose)."""
    bg = (~ink).astype(np.uint8)
    n, _ = cv2.connectedComponents(bg, connectivity=4)
    return n >= 3   # label 0 réservé + extérieur + interne(s)


def _inter(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _area(a) -> float:
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def check_legibility(record: dict) -> list[str]:
    """Défauts de LISIBILITÉ du schéma.

    Un circuit peut être électriquement juste et parfaitement annoté tout
    en étant graphiquement illisible (symboles empilés, étiquettes les
    unes sur les autres). Ce contrôle attrape ce cas, que les contrôles
    de nets et de bbox laissent passer.

    On ne compare PAS les bboxes de composants entre elles : celles des
    AOP, transistors et portes sont des rectangles autour d'un symbole
    triangulaire, leurs coins vides se recouvrent sans qu'il y ait de
    défaut visible. On teste donc :
      - deux étiquettes qui se recouvrent
      - une étiquette posée sur le symbole d'un AUTRE composant
        (coeur du symbole seulement pour les formes triangulaires)
    """
    issues: list[str] = []
    texts = record.get("texts", [])
    comps = {c["id"]: c for c in record.get("components", [])}

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i]["bbox"], texts[j]["bbox"]
            ov = _inter(a, b) / min(_area(a), _area(b))
            if ov > 0.15:
                issues.append(f"étiquettes {texts[i]['comp_id']}/"
                              f"{texts[j]['comp_id']} superposées")

    for t in texts:
        for cid, c in comps.items():
            if cid == t["comp_id"]:
                continue
            b = c["bbox"]
            if c["class"] == "opamp" or c["class"].startswith("gate_"):
                w, h = b[2] - b[0], b[3] - b[1]
                b = [b[0] + 0.22 * w, b[1] + 0.22 * h,
                     b[2] - 0.22 * w, b[3] - 0.22 * h]
            if _inter(t["bbox"], b) / _area(t["bbox"]) > 0.25:
                issues.append(f"étiquette {t['comp_id']} sur le symbole {cid}")
    return issues


def validate_record(png_path: Path, record: dict) -> list[str]:
    issues = []
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [f"image illisible: {png_path.name}"]
    H, W = img.shape
    ink = img < 128

    # 1. Fermeture de boucle — UNIQUEMENT pour les circuits électriques.
    #    Un schéma logique est OUVERT par nature (entrées à gauche,
    #    sortie à droite) : le contrôle ne s'applique pas.
    domain = (record.get("domain")
              or record.get("circuit", {}).get("domain", "electrical"))
    if domain != "logic" and not check_closure(ink):
        issues.append("boucle NON fermée (aucune région interne enclose)")

    # 2+3. Composants et terminaux
    for comp in record["components"]:
        x0, y0, x1, y1 = map(int, comp["bbox"])
        reg = ink[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if (reg.mean() if reg.size else 0) < 0.01:
            issues.append(f"{comp['id']}: bbox composant vide")
        for t in comp["terminals"]:
            tx, ty = int(t["x"]), int(t["y"])
            win = ink[max(0, ty - 4):ty + 5, max(0, tx - 4):tx + 5]
            if win.size == 0 or not win.any():
                issues.append(f"{comp['id']}.{t['name']}: terminal hors fil")

    # 3bis. Jonctions : sur fil, degré >= 3
    for j in record.get("junctions", []):
        jx, jy = int(j["x"]), int(j["y"])
        win = ink[max(0, jy-4):jy+5, max(0, jx-4):jx+5]
        if win.size == 0 or not win.any():
            issues.append(f"jonction ({jx},{jy}): hors fil")

    # 3ter. NETS : cohérence avec les connexions du catalogue
    #        + anti court-circuit (2 terminaux d'un même composant
    #        2-terminaux jamais dans le même net, sauf masse)
    circuit = record.get("circuit", {})
    nets = record.get("nets", [])
    if circuit and nets:
        cls_map = {c["id"]: c["class"] for c in circuit.get("components", [])}
        comp_nets: dict = {}
        for net in nets:
            in_net: dict = {}
            for m in net["terminals"]:
                comp_nets.setdefault(m["comp_id"], set()).add(net["net_id"])
                in_net.setdefault(m["comp_id"], set()).add(m["terminal"])
            for cid, terms in in_net.items():
                if {"start", "end"} <= terms and cls_map.get(cid) != "ground":
                    issues.append(f"net: court-circuit sur {cid}")
        for conn in circuit.get("connections", []):
            a, b = conn["from"], conn["to"]
            if not (comp_nets.get(a, set()) & comp_nets.get(b, set())):
                issues.append(f"net: connexion {a}-{b} absente")

    # 3quater. LISIBILITÉ du schéma
    issues += check_legibility(record)

    # 4. Textes
    comp_boxes = {c["id"]: c["bbox"] for c in record["components"]}
    comp_class = {c["id"]: c["class"] for c in record["components"]}
    for t in record["texts"]:
        x0, y0, x1, y1 = map(int, t["bbox"])
        reg = ink[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if (reg.mean() if reg.size else 0) < 0.02:
            issues.append(f"texte {t['comp_id']}: bbox texte vide")
        # chevauchement avec la bbox du composant
        cb = comp_boxes.get(t["comp_id"])
        # label DANS le symbole = normal pour les AOP (réf. au centre)
        if cb and comp_class.get(t["comp_id"]) != "opamp":
            ix0, iy0 = max(x0, cb[0]), max(y0, cb[1])
            ix1, iy1 = min(x1, cb[2]), min(y1, cb[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            area_t = max(1, (x1 - x0) * (y1 - y0))
            if inter / area_t > 0.5:
                issues.append(f"texte {t['comp_id']}: chevauche trop "
                              f"son composant ({100*inter/area_t:.0f}%)")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/circuitvqa_v3")
    p.add_argument("--split", default="both", choices=["train", "test", "both"])
    p.add_argument("--max", type=int, default=0, help="0 = tout valider")
    args = p.parse_args()

    root = Path(args.dataset)
    splits = ["train", "test"] if args.split == "both" else [args.split]

    grand_total, grand_bad = 0, 0
    for split in splits:
        ann_dir = root / "annotations" / split
        files = sorted(ann_dir.glob("*.json"))
        if args.max:
            files = files[:args.max]
        n_bad = 0
        for f in files:
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
            png = root / rec["image"]
            issues = validate_record(png, rec)
            if issues:
                n_bad += 1
                if n_bad <= 10:
                    print(f"[{split}] {f.stem} ({rec['template']}): {issues}")
        print(f"[{split}] {len(files)} validés — {n_bad} avec problèmes "
              f"({100*(1 - n_bad/max(1,len(files))):.1f}% propres)")
        grand_total += len(files)
        grand_bad += n_bad

    print(f"\nTOTAL : {grand_total} circuits, {grand_bad} problématiques "
          f"({100*(1 - grand_bad/max(1,grand_total)):.1f}% de qualité)")


if __name__ == "__main__":
    main()
