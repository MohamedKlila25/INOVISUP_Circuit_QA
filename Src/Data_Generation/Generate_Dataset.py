#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py — v3 FINALE
Génération du dataset CircuitVQA v3 :
  - 4500 images train + 500 images test
  - PNG 800px + labels YOLO + annotations JSON complètes
  - Répartition par niveau : 35% lycée / 30% prépa / 35% ingénieur

Structure de sortie :
  <OUT_ROOT>/
    images/
      train/   *.png
      test/    *.png
    labels_yolo/
      train/   *.txt   (format YOLO : class xc yc w h)
      test/    *.txt
    annotations/
      train/   *.json  (bbox + terminaux + OCR + circuit JSON)
      test/    *.json
    circuit.yaml       (config YOLO prête à l'emploi)
    summary.json       (statistiques de génération)

Usage (depuis le dossier notebooks/) :
    python data_generation/generate_dataset.py
    # ou avec paramètres :
    python data_generation/generate_dataset.py --out data/circuitvqa_v3 --seed 42
"""
from __future__ import annotations
import sys
import json
import random
import argparse
import traceback
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Chemin vers les modules du projet ────────────────────────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from Circuit_Catalog import (CATALOG, instantiate, list_templates_by_level,
                             list_templates_by_domain)
from Renderer_Annotated import CircuitRendererAnnotated, CLASS_NAMES
from Renderer import sample_style

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
N_TRAIN  = 4500
N_TEST   = 500
PNG_SIZE = 800     # pixels (côté le plus long)
DPI      = 150
UNIT     = 3.0     # unité Schemdraw (taille relative des éléments)
SEED     = 42

# Répartition par NIVEAU (dans chaque domaine)
LEVEL_WEIGHTS = {"lycee": 0.42, "prepa": 0.43, "ingenieur": 0.15}
# Répartition par DOMAINE : les circuits logiques sont un sous-ensemble
# volontairement minoritaire (les portes ont des layouts plus rigides).
DOMAIN_WEIGHTS = {"electrical": 0.75, "logic": 0.25}
# Part du dataset réservée aux templates contenant des classes rares
# (fournies par au plus deux templates) : garantit assez d'exemples.
RARE_SHARE = 0.30


# ══════════════════════════════════════════════════════════════════════
# ÉCHANTILLONNAGE
# ══════════════════════════════════════════════════════════════════════
def sample_template_names(n: int, levels: dict[str, list[str]],
                          rng: random.Random) -> list[str]:
    """
    Tire n noms de templates.

    Trois niveaux de contrôle, appliqués DANS CET ORDRE :
      1. DOMAINE   : électrique / logique (DOMAIN_WEIGHTS)
      2. RARETÉ    : à l'intérieur de chaque domaine, une part fixe
                     (RARE_SHARE) est réservée aux templates contenant
                     une classe rare — c.-à-d. fournie par au plus deux
                     templates (fusible, transistors, XNOR...). Sans ce
                     quota ces classes n'auraient que quelques dizaines
                     d'exemples sur tout le dataset.
      3. NIVEAU    : lycée / prépa / ingénieur (LEVEL_WEIGHTS) pour le
                     reste du quota de chaque domaine.

    Le quota de rareté est appliqué PAR DOMAINE : appliqué globalement,
    il évincerait les templates logiques courants (le domaine logique
    étant minoritaire), et des classes comme gate_xor tomberaient à zéro.
    """
    domains = list_templates_by_domain()

    # classes fournies par chaque template, et templates par classe
    tmpl_classes = {name: {c["class"] for c in tpl["components"]}
                    for name, tpl in CATALOG.items()}
    providers: dict[str, list[str]] = {}
    for name, classes in tmpl_classes.items():
        for cl in classes:
            providers.setdefault(cl, []).append(name)

    def is_rare(name: str) -> bool:
        return any(len(providers[cl]) <= 2 for cl in tmpl_classes[name])

    names: list[str] = []
    for dom, dw in DOMAIN_WEIGHTS.items():
        pool_dom = domains[dom]
        if not pool_dom:
            continue
        n_dom = int(round(n * dw))
        dom_names: list[str] = []

        # 2. quota des templates rares DE CE DOMAINE (répartition égale)
        rare_pool = sorted(t for t in pool_dom if is_rare(t))
        if rare_pool:
            n_rare = int(round(n_dom * RARE_SHARE))
            dom_names += [rare_pool[i % len(rare_pool)] for i in range(n_rare)]

        # 3. reste réparti par niveau
        n_rest = max(0, n_dom - len(dom_names))
        dom_set = set(pool_dom)
        by_level = {lvl: [t for t in levels[lvl] if t in dom_set]
                    for lvl in LEVEL_WEIGHTS}
        for lvl, lw in LEVEL_WEIGHTS.items():
            pool = by_level[lvl]
            if pool:
                dom_names += [rng.choice(pool)
                              for _ in range(int(n_rest * lw))]
        while len(dom_names) < n_dom:
            dom_names.append(rng.choice(pool_dom))

        names += dom_names[:n_dom]

    all_keys = list(CATALOG.keys())
    while len(names) < n:
        names.append(rng.choice(all_keys))
    names = names[:n]
    rng.shuffle(names)
    return names


# ══════════════════════════════════════════════════════════════════════
# GÉNÉRATION D'UN SPLIT
# ══════════════════════════════════════════════════════════════════════
def generate_split(
    names: list[str],
    split: str,
    out_root: Path,
    renderer: CircuitRendererAnnotated,
) -> tuple[int, int, Counter]:
    """
    Génère toutes les images d'un split (train ou test).
    Retourne (n_ok, n_fail, counter_par_template).
    """
    img_dir = out_root / "images"      / split
    lbl_dir = out_root / "labels_yolo" / split
    ann_dir = out_root / "annotations" / split
    tmp_dir = out_root / "_tmp_svg"

    for d in (img_dir, lbl_dir, ann_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    tmpl_counter: Counter = Counter()

    total = len(names)
    for i, tname in enumerate(names, 1):
        stem     = f"circuit_{i-1:05d}"
        svg_path = str(tmp_dir / f"{stem}.svg")
        png_path = str(img_dir / f"{stem}.png")

        try:
            circuit = instantiate(tname)
            style   = sample_style(random, custom="custom_render" in circuit)
            ann = renderer.render_full(circuit, svg_path, png_path, style=style)

            # Nettoyage SVG temporaire
            Path(svg_path).unlink(missing_ok=True)

            if ann is None:
                failed += 1
                if failed <= 10:
                    print(f"  [!] {split} #{i}: {tname} → rendu None")
                continue

            W, H = ann["image_size"]

            # ── Labels YOLO ─────────────────────────────────────────
            yolo_lines = []
            for comp in ann["components"]:
                x0, y0, x1, y1 = comp["bbox"]
                xc = (x0 + x1) / 2 / W
                yc = (y0 + y1) / 2 / H
                w  = (x1 - x0) / W
                h  = (y1 - y0) / H
                yolo_lines.append(
                    f"{comp['class_idx']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                )
            (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

            # ── Annotation JSON complète ─────────────────────────────
            record = {
                "image":      f"images/{split}/{stem}.png",
                "image_size": ann["image_size"],
                "circuit":    circuit,
                "domain":     circuit.get("domain", "electrical"),
                "components": ann["components"],
                "texts":      ann["texts"],
                "junctions":  ann.get("junctions", []),
                "nets":       ann.get("nets", []),
                "crossovers": ann.get("crossovers", []),
                "template":   tname,
            }
            with open(ann_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            ok += 1
            tmpl_counter[tname] += 1

        except Exception:
            failed += 1
            if failed <= 10:
                print(f"  [!] {split} #{i}: {tname}")
                traceback.print_exc()
            Path(svg_path).unlink(missing_ok=True)

        # Progrès
        if i % 250 == 0 or i == total:
            pct = 100 * ok / i
            print(f"  [{split}] {i}/{total} — {ok} OK, {failed} échecs "
                  f"({pct:.1f}% succès)")

    return ok, failed, tmpl_counter


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Génère le dataset CircuitVQA v3 (4500 train + 500 test)"
    )
    parser.add_argument("--out",    default="data/circuitvqa_v3",
                        help="Dossier de sortie (défaut: data/circuitvqa_v3)")
    parser.add_argument("--train",  type=int, default=N_TRAIN)
    parser.add_argument("--test",   type=int, default=N_TEST)
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--size",   type=int, default=PNG_SIZE,
                        help="Largeur PNG en pixels (défaut: 800)")
    args = parser.parse_args()

    rng      = random.Random(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    levels = list_templates_by_level()
    print("Templates disponibles :")
    for lvl, names in levels.items():
        print(f"  {lvl}: {len(names)} → {names}")
    print(f"\nTotal : {len(CATALOG)} templates")
    print(f"Génération : {args.train} train + {args.test} test "
          f"(seed={args.seed}, PNG={args.size}px)\n")

    renderer = CircuitRendererAnnotated(
        dpi=DPI, unit=UNIT, png_size=args.size
    )

    # ── Tirage des noms de templates ──────────────────────────────────
    train_names = sample_template_names(args.train, levels, rng)
    test_names  = sample_template_names(args.test,  levels, rng)

    print(f"Répartition train (top 5) :")
    for name, cnt in Counter(train_names).most_common(5):
        print(f"  {name}: {cnt}")
    print()

    # ── Génération ────────────────────────────────────────────────────
    ok_tr, fail_tr, ctr_tr = generate_split(
        train_names, "train", out_root, renderer
    )
    ok_te, fail_te, ctr_te = generate_split(
        test_names, "test", out_root, renderer
    )

    # ── YAML YOLO ─────────────────────────────────────────────────────
    yaml_path = out_root / "circuit.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n" +
        "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)) + "\n"
    )

    # ── Résumé ───────────────────────────────────────────────────────
    summary = {
        "seed":        args.seed,
        "n_train_req": args.train,
        "n_test_req":  args.test,
        "n_train_ok":  ok_tr,
        "n_test_ok":   ok_te,
        "n_train_fail":fail_tr,
        "n_test_fail": fail_te,
        "train_by_template": dict(ctr_tr.most_common()),
        "test_by_template":  dict(ctr_te.most_common()),
        "class_names": CLASS_NAMES,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Nettoyage dossier tmp
    tmp_dir = out_root / "_tmp_svg"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)

    print(f"\n{'='*55}")
    print(f"✅ Génération terminée !")
    print(f"   Train : {ok_tr}/{args.train} "
          f"({'%.1f' % (100*ok_tr/args.train)}%)")
    print(f"   Test  : {ok_te}/{args.test}  "
          f"({'%.1f' % (100*ok_te/args.test)}%)")
    print(f"   Sortie: {out_root.resolve()}")
    print(f"   YAML  : {yaml_path.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
