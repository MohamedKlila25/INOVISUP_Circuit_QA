# -*- coding: utf-8 -*-
"""
logic_renders.py
Rendu des CIRCUITS LOGIQUES (domaine "logic"), strictement séparé des
circuits électriques : aucun mélange porte logique / composant passif.

Conforme à la doc officielle schemdraw (elements/logic.html) :
  - `from schemdraw import logic`
  - portes : And, Nand, Or, Nor, Xor, Xnor, Not, Buf
  - anchors : `in1`, `in2`, ... `inN` et `out`
  - paramètre `inputs` pour les portes à plus de 2 entrées

Différences structurelles avec les circuits électriques :
  - un schéma logique est OUVERT (entrées à gauche, sortie à droite),
    il n'y a pas de boucle fermée -> le contrôle de fermeture est
    désactivé pour ce domaine (voir validate_dataset.py)
  - les entrées/sorties sont des ÉTIQUETTES (A, B, Y), pas des composants

Les portes sont TAGUÉES explicitement (`_cvqa_class`) car schemdraw ne
permet pas de les distinguer autrement : Nand a le type-name `And`,
Xor/Nor/Xnor ont le type-name `Or`.
"""
from __future__ import annotations
import schemdraw
from schemdraw import logic

# classe sémantique -> fabrique schemdraw
GATE_FACTORY = {
    "gate_and":  lambda n: logic.And(inputs=n),
    "gate_nand": lambda n: logic.Nand(inputs=n),
    "gate_or":   lambda n: logic.Or(inputs=n),
    "gate_nor":  lambda n: logic.Nor(inputs=n),
    "gate_xor":  lambda n: logic.Xor(),
    "gate_xnor": lambda n: logic.Xnor(),
    "gate_not":  lambda n: logic.Not(),
}


def make_gate(cls: str, inputs: int = 2):
    """Crée une porte TAGUÉE avec sa classe sémantique."""
    g = GATE_FACTORY[cls](inputs)
    g._cvqa_class = cls          # lu par renderer_annotated
    return g


def _val(components, comp_id) -> str:
    """Label d'une porte : son identifiant (+ référence si fournie)."""
    for c in components:
        if c["id"] == comp_id:
            v = c.get("value")
            return f"{comp_id}\n{v}" if v else comp_id
    return comp_id


def _gate_of(components, idx: int) -> dict:
    """idx-ième porte de la liste des composants."""
    gates = [c for c in components if c["class"].startswith("gate_")]
    return gates[idx]


# ─────────────────────────────────────────────────────────────────────
def render_logic(circuit: dict, unit: float = 2.0,
                 style: dict | None = None):
    """
    Rendu paramétrique d'un circuit logique selon circuit['logic_spec'].

    Motifs (`pattern`) :
      single          : une porte, N entrées
      cascade         : G1(in1,in2) -> G2(., in3)
      shared          : G1 et G2 partagent les mêmes entrées (demi-add.)
      inverted_inputs : NOT sur chaque entrée, puis G (De Morgan)

    Retourne (Drawing, dict id->element) comme les autres renderers.
    """
    spec  = circuit["logic_spec"]
    comps = circuit["components"]
    st    = style or {}

    d = schemdraw.Drawing(show=False)
    d.config(unit=unit,
             lw=st.get("lw", 2),
             fontsize=st.get("fontsize", 10),
             font=st.get("font", "sans-serif"))

    pattern = spec["pattern"]
    ins     = spec["inputs"]
    outname = spec.get("output", "Y")
    elems: dict = {}

    lead = unit * 0.9   # longueur des fils d'entrée

    # ── UNE PORTE ────────────────────────────────────────────────────
    if pattern == "single":
        g   = _gate_of(comps, 0)
        n   = len(ins)
        gate = d.add(make_gate(g["class"], n)
                     .label(_val(comps, g["id"]), loc="bottom", ofst=0.45))
        elems[g["id"]] = gate
        for i, lbl in enumerate(ins, start=1):
            anc = getattr(gate, f"in{i}")
            d.add(logic.Line().left(lead).at(anc).label(lbl, loc="left"))
        d.add(logic.Line().right(lead).at(gate.out).label(outname, loc="right"))

    # ── DEUX PORTES EN CASCADE ───────────────────────────────────────
    elif pattern == "cascade":
        g1, g2 = _gate_of(comps, 0), _gate_of(comps, 1)
        n1 = 1 if g1["class"] == "gate_not" else 2

        gate1 = d.add(make_gate(g1["class"], n1)
                      .label(_val(comps, g1["id"]), loc="bottom", ofst=0.45))
        elems[g1["id"]] = gate1
        for i in range(1, n1 + 1):
            anc = getattr(gate1, f"in{i}")
            d.add(logic.Line().left(lead).at(anc)
                  .label(ins[i - 1], loc="left"))

        # G2 ancrée sur in1, alignée sur la sortie de G1
        d.add(logic.Line().right(lead * 0.7).at(gate1.out))
        gate2 = d.add(make_gate(g2["class"], 2).anchor("in1")
                      .label(_val(comps, g2["id"]), loc="bottom", ofst=0.45))
        elems[g2["id"]] = gate2
        # entrée restante de G2 : descend puis part à gauche
        d.add(logic.Line().down(unit * 0.6).at(gate2.in2))
        d.add(logic.Line().left().tox(gate1.in1)
              .label(ins[-1], loc="left"))
        d.add(logic.Line().right(lead).at(gate2.out)
              .label(outname, loc="right"))

    # ── DEUX PORTES À ENTRÉES PARTAGÉES (demi-additionneur) ──────────
    elif pattern == "shared":
        g1, g2 = _gate_of(comps, 0), _gate_of(comps, 1)
        outs   = spec.get("outputs", ["S", "C"])

        gate1 = d.add(make_gate(g1["class"], 2)
                      .label(_val(comps, g1["id"]), loc="bottom", ofst=0.45))
        elems[g1["id"]] = gate1
        d.add(logic.Line().right(lead).at(gate1.out)
              .label(outs[0], loc="right"))

        # G2 sous G1
        gate2 = d.add(make_gate(g2["class"], 2)
                      .at((gate1.in1[0], gate1.in2[1] - unit * 1.5))
                      .label(_val(comps, g2["id"]), loc="bottom", ofst=0.45))
        elems[g2["id"]] = gate2
        d.add(logic.Line().right(lead).at(gate2.out)
              .label(outs[1], loc="right"))

        # bus d'entrées communes : A vers in1 des deux portes, B vers in2
        busx = gate1.in1[0] - lead
        for anc_name, lbl, dx in (("in1", ins[0], 0.0),
                                  ("in2", ins[1], 0.35)):
            a1 = getattr(gate1, anc_name)
            a2 = getattr(gate2, anc_name)
            x  = busx - dx * unit
            d.add(logic.Line().left().at(a1).tox(x))
            d.add(logic.Dot())
            d.push()
            d.add(logic.Line().down().toy(a2))
            d.add(logic.Line().right().tox(a2))
            d.pop()
            d.add(logic.Line().left(lead * 0.5).label(lbl, loc="left"))

    # ── ENTRÉES INVERSÉES (lois de De Morgan) ────────────────────────
    elif pattern == "inverted_inputs":
        gates = [c for c in comps if c["class"].startswith("gate_")]
        nots  = [g for g in gates if g["class"] == "gate_not"]
        main  = [g for g in gates if g["class"] != "gate_not"][0]

        gate = d.add(make_gate(main["class"], 2)
                     .label(_val(comps, main["id"]), loc="bottom", ofst=0.45))
        elems[main["id"]] = gate

        # Les entrées d'une porte ne sont séparées que d'environ une
        # demi-unité : y accrocher directement deux portes NOT les fait
        # se superposer. On écarte donc chaque branche verticalement
        # avant de placer son inverseur.
        spread = unit * 0.75
        for i, (ng, lbl) in enumerate(zip(nots, ins), start=1):
            anc = getattr(gate, f"in{i}")
            d.add(logic.Line().left(lead * 0.35).at(anc))
            if i == 1:
                d.add(logic.Line().up(spread))       # in1 = entrée haute
            else:
                d.add(logic.Line().down(spread))     # in2 = entrée basse
            ne = d.add(make_gate("gate_not", 1).left()
                       .label(_val(comps, ng["id"]), loc="bottom", ofst=0.45))
            elems[ng["id"]] = ne
            d.add(logic.Line().left(lead * 0.6).label(lbl, loc="left"))

        d.add(logic.Line().right(lead).at(gate.out)
              .label(outname, loc="right"))

    else:
        raise ValueError(f"pattern logique inconnu : {pattern}")

    return d, elems


if __name__ == "__main__":
    import sys, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, ".")
    from circuit_catalog import CATALOG, instantiate

    for name, tpl in CATALOG.items():
        if tpl.get("domain") != "logic":
            continue
        c = instantiate(name)
        d, elems = render_logic(c)
        d.save(f"/tmp/logic_{name}.svg")
        plt.close("all")
        print(f"✓ {name}: {len(elems)} portes -> /tmp/logic_{name}.svg")
