# -*- coding: utf-8 -*-
"""
custom_renders.py — v3 FINALE
Rendus dédiés pour les templates qui nécessitent un placement précis
par ancres Schemdraw (AOP, redresseur). Chaque fonction retourne
(schemdraw.Drawing, dict[id→element]) pour permettre l'extraction
des bboxes en aval (renderer_annotated.py).

── Règle de placement des labels (CRITIQUE) ─────────────────────────
ofst=(dx, dy) est appliqué dans le REPÈRE LOCAL de l'élément AVANT
rotation — pas en coordonnées absolues. Vérification numérique (bbox
extension) :
  theta=0°   (.right()) : décal. horizontal global = ofst(dx, 0)
  theta=90°  (.up())    : décal. horizontal global = ofst(0, -dy)
  theta=180° (.left())  : décal. horizontal global = ofst(-dx, 0)
  theta=270° (.down())  : décal. horizontal global = ofst(0, +dy)
On utilise loc='center' + ofst pour tous les labels afin d'éviter
les dépendances à 'top'/'right' qui basculent selon l'orientation.
"""
from __future__ import annotations
import schemdraw
import schemdraw.elements as elm


def _val(components: list[dict], comp_id: str) -> str:
    for c in components:
        if c["id"] == comp_id:
            v = c.get("value")
            return f"{comp_id}\n{v}" if v else comp_id
    return comp_id


# ─────────────────────────────────────────────────────────────────────
def render_half_wave_rectifier(circuit: dict, unit: float = 2.0, style: dict | None = None):
    """
    AC → diode → résistance de charge → retour.
    Pattern officiel Schemdraw, adapté pour l'annotation automatique.
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    st = style or {}
    d.config(unit=unit, lw=st.get("lw", 2),
             fontsize=st.get("fontsize", 10),
             font=st.get("font", "sans-serif"))

    elems: dict = {}

    # Source AC : monte vers le haut (theta=90°)
    # Décalage label vers la gauche (global -x) → ofst local (0, +push)
    src = d.add(elm.SourceSin().up()
                .label(_val(comps, "V1"), loc="center", ofst=(0, -1.3)))
    elems["V1"] = src

    d.add(elm.Line().right().length(unit / 2))

    # Diode horizontale (theta=0°) — label au-dessus (global +y) → ofst(0, +0.6)
    diode = d.add(elm.Diode().right()
                  .label(_val(comps, "D1"), loc="center", ofst=(0, 0.6)))
    elems["D1"] = diode

    # Résistance de charge : .down().toy(src.start)
    # → theta=270° ; décalage global vers la droite (+x) → ofst local (0, +0.9)
    rl = d.add(elm.Resistor().down().toy(src.start)
               .label(_val(comps, "RL1"), loc="center", ofst=(0, 0.9)))
    elems["RL1"] = rl

    d.add(elm.Line().tox(src.start))

    return d, elems


# ─────────────────────────────────────────────────────────────────────
def render_inverting_amplifier(circuit: dict, unit: float = 2.0, style: dict | None = None):
    """
    AOP inverseur. Pattern officiel Schemdraw gallery/opamp.html.
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    st = style or {}
    d.config(unit=unit, lw=st.get("lw", 2),
             fontsize=st.get("fontsize", 10),
             font=st.get("font", "sans-serif"))

    elems: dict = {}

    # Source d'entrée réelle (cohérence vérité-terrain : V1 est déclaré
    # dans le template, il doit apparaître sur l'image). Pattern
    # identique au multistage officiel : Ground + SourceV + Rin + Opamp.
    d.add(elm.Ground(lead=False))
    src = d.add(elm.SourceV()
                .label(_val(comps, "V1"), loc="center", ofst=(0, 1.15)))
    elems["V1"] = src

    rin = d.add(elm.Resistor().right()
                .label(_val(comps, "Rin"), loc="center", ofst=(0, 0.6)))
    d.add(elm.Dot())
    elems["Rin"] = rin

    op = d.add(elm.Opamp(leads=True).anchor("in1")
               .label(_val(comps, "U1"), loc="center", ofst=(0, -0.9)))
    elems["U1"] = op
    d.add(elm.Ground().at(op.in2))

    # Branche de contre-réaction
    d.add(elm.Line().up(unit).at(op.in1))
    rf = d.add(elm.Resistor().tox(op.out)
               .label(_val(comps, "Rf"), loc="center", ofst=(0, 0.6)))
    elems["Rf"] = rf

    d.add(elm.Line().toy(op.out))
    d.add(elm.Dot())
    d.add(elm.Line().right(unit / 2).at(op.out)
          .label("Vout", loc="center", ofst=(0.9, 0)))

    return d, elems


# ─────────────────────────────────────────────────────────────────────
def render_multistage_amplifier(circuit: dict, unit: float = 2.0, style: dict | None = None):
    """
    Deux AOP inverseurs en cascade. Pattern officiel Schemdraw
    gallery/opamp.html (example opamp_1_0).
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    st = style or {}
    d.config(unit=unit, lw=st.get("lw", 2),
             fontsize=st.get("fontsize", 10),
             font=st.get("font", "sans-serif"))

    elems: dict = {}

    d.add(elm.Ground(lead=False))
    src = d.add(elm.SourceV()
                .label(_val(comps, "V1"), loc="center", ofst=(0, 1.15)))
    elems["V1"] = src

    r1 = d.add(elm.Resistor().right()
               .label(_val(comps, "R1"), loc="center", ofst=(0, 0.6)))
    d.add(elm.Dot())
    elems["R1"] = r1

    o1 = d.add(elm.Opamp(leads=True).anchor("in1")
               .label(_val(comps, "U1"), loc="center", ofst=(0, -0.9)))
    elems["U1"] = o1
    d.add(elm.Ground().at(o1.in2))

    d.add(elm.Line().up(unit).at(o1.in1))
    rf1 = d.add(elm.Resistor().tox(o1.out)
                .label(_val(comps, "Rf1"), loc="center", ofst=(0, 0.6)))
    elems["Rf1"] = rf1
    d.add(elm.Line().toy(o1.out))
    d.add(elm.Dot())

    d.add(elm.Line().right(unit * 2.5).at(o1.out))
    o2 = d.add(elm.Opamp(leads=True).anchor("in2")
               .label(_val(comps, "U2"), loc="center", ofst=(0, -0.9)))
    elems["U2"] = o2

    r2 = d.add(
        elm.Resistor().left().at(o2.in1)
        .label(_val(comps, "R2"), loc="center", ofst=(0, -0.6))
    )
    elems["R2"] = r2
    d.add(elm.Dot().at(o2.in1))
    d.add(elm.Ground())

    d.add(elm.Line().up(unit * 0.75).at(o2.in1))
    rf2 = d.add(elm.Resistor().tox(o2.out)
                .label(_val(comps, "Rf2"), loc="center", ofst=(0, 0.6)))
    elems["Rf2"] = rf2
    d.add(elm.Line().toy(o2.out))
    d.add(elm.Dot())

    d.add(elm.Line().right(unit / 2).at(o2.out)
          .label("Vout", loc="center", ofst=(0.9, 0)))

    return d, elems


# ── Registre ─────────────────────────────────────────────────────────
CUSTOM_RENDERERS = {
    "half_wave_rectifier":  render_half_wave_rectifier,
    "inverting_amplifier":  render_inverting_amplifier,
    "multistage_amplifier": render_multistage_amplifier,
}


if __name__ == "__main__":
    import sys, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, ".")
    from circuit_catalog import instantiate

    for name, fn in CUSTOM_RENDERERS.items():
        circuit = instantiate(name)
        d, elems = fn(circuit)
        d.save(f"/tmp/custom_{name}.svg")
        plt.close("all")
        print(f"✓ {name}: {len(elems)} éléments → /tmp/custom_{name}.svg")


# ═════════════════════════════════════════════════════════════════════
#  Templates ajoutés (Plan A) — classes sous-représentées
# ═════════════════════════════════════════════════════════════════════

def _cfg(d, unit, style):
    st = style or {}
    d.config(unit=unit, lw=st.get("lw", 2),
             fontsize=st.get("fontsize", 10),
             font=st.get("font", "sans-serif"))
    return st


def render_npn_common_emitter(circuit: dict, unit: float = 2.0,
                              style: dict | None = None):
    """Transistor NPN en émetteur commun : Vcc - RC - collecteur,
    base polarisée par RB, émetteur à la masse.

    NE PAS insérer de segment supplémentaire entre RC1 et le collecteur :
    cela décale l'émetteur vers le bas et, selon la longueur naturelle
    tirée par le style, il passe sous le rail de masse — le `.toy()` de
    raccordement devrait alors remonter et ne trace rien, rompant la
    connexion émetteur-masse.
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    _cfg(d, unit, style)
    elems: dict = {}

    d.add(elm.Ground(lead=False))
    src = d.add(elm.SourceV().up().label(_val(comps, "V1"),
                                        loc="center", ofst=(0, 1.15)))
    elems["V1"] = src
    start = src.start

    d.add(elm.Line().right(unit * 2.2))
    rc = d.add(elm.Resistor().down().label(_val(comps, "RC1"),
                                          loc="center", ofst=(0, 0.9)))
    elems["RC1"] = rc
    d.add(elm.Dot())
    out_node = d.here

    q = d.add(elm.BjtNpn(circle=True).anchor("collector"))
    elems["Q1"] = q
    d.add(elm.Line().down(unit * 0.4).at(q.emitter))
    d.add(elm.Line().toy(start))
    d.add(elm.Line().tox(start))
    d.add(elm.Ground().at(start))

    rb = d.add(elm.Resistor().left().at(q.base)
               .label(_val(comps, "RB1"), loc="center", ofst=(0, -0.75)))
    elems["RB1"] = rb
    d.add(elm.Line().left(unit * 0.4).label("Vin", loc="left"))

    d.add(elm.Line().right(unit * 0.8).at(out_node)
          .label("Vout", loc="right"))
    return d, elems


def render_pnp_switch(circuit: dict, unit: float = 2.0,
                      style: dict | None = None):
    """Transistor PNP en commutation : émetteur au +Vcc, charge RL entre
    collecteur et masse, base commandée par RB."""
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    _cfg(d, unit, style)
    elems: dict = {}

    d.add(elm.Ground(lead=False))
    src = d.add(elm.SourceV().up().label(_val(comps, "V1"),
                                        loc="center", ofst=(0, 1.15)))
    elems["V1"] = src
    start = src.start

    d.add(elm.Line().right(unit * 2.2))
    q = d.add(elm.BjtPnp(circle=True).anchor("emitter"))
    elems["Q1"] = q

    # Descente par un FIL (se comprime librement) puis charge posée sur
    # le rail du bas : une résistance ne peut pas être comprimée sous sa
    # longueur minimale de tracé, et .toy() la ferait alors dépasser
    # sous le rail de masse (connexion RL1-GND1 rompue).
    d.add(elm.Line().down().at(q.collector).toy(start))
    rl = d.add(elm.Resistor().left().tox(start)
               .label(_val(comps, "RL1"), loc="center", ofst=(0, 0.75)))
    elems["RL1"] = rl
    d.add(elm.Ground().at(start))

    rb = d.add(elm.Resistor().left().at(q.base)
               .label(_val(comps, "RB1"), loc="center", ofst=(0, -0.75)))
    elems["RB1"] = rb
    d.add(elm.Line().left(unit * 0.4).label("Vin", loc="left"))
    return d, elems


def render_full_wave_bridge(circuit: dict, unit: float = 2.0,
                            style: dict | None = None):
    """Pont de Graetz (redressement double alternance).

    Layout rectangulaire (tout horizontal/vertical, aucune rotation) :
      colonne gauche  : N --D2--> A --D1--> P
      colonne droite  : N --D4--> B --D3--> P
      source AC entre A et B (diagonale d'entrée)
      charge RL entre P et N (diagonale de sortie)
    Sens de conduction : D1/D3 vers le +, D2/D4 depuis le -.
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    _cfg(d, unit, style)
    elems: dict = {}

    W = unit * 2.6      # largeur du pont

    # ── colonne gauche : N (bas) -> A (milieu) -> P (haut) ───────────
    d2 = d.add(elm.Diode().up()
               .label(_val(comps, "D2"), loc="center", ofst=(0, -0.85)))
    elems["D2"] = d2
    node_n = d2.start
    node_a = d.here
    d.add(elm.Dot())
    d1 = d.add(elm.Diode().up()
               .label(_val(comps, "D1"), loc="center", ofst=(0, -0.85)))
    elems["D1"] = d1
    node_p = d.here
    d.add(elm.Dot())

    # ── colonne droite, même structure ──────────────────────────────
    d4 = d.add(elm.Diode().up().at((node_n[0] + W, node_n[1]))
               .label(_val(comps, "D4"), loc="center", ofst=(0, 0.85)))
    elems["D4"] = d4
    node_b = d.here
    d.add(elm.Dot())
    d3 = d.add(elm.Diode().up()
               .label(_val(comps, "D3"), loc="center", ofst=(0, 0.85)))
    elems["D3"] = d3
    d.add(elm.Dot())

    # ── rails du pont ───────────────────────────────────────────────
    d.add(elm.Line().at(node_p).tox(node_b[0]))        # rail + (haut)
    d.add(elm.Line().at(node_n).tox(node_b[0]))        # rail - (bas)

    # ── source AC sur la diagonale d'entrée (A <-> B) ────────────────
    src = d.add(elm.SourceSin().at(node_a).right().tox(node_b[0])
                .label(_val(comps, "V1"), loc="center", ofst=(0, 0.85)))
    elems["V1"] = src

    # ── charge sur la diagonale de sortie (P -> N), à droite ────────
    d.add(elm.Line().at(node_p).right(unit * 0.5))
    xr = node_p[0] + W + unit * 0.9
    d.add(elm.Line().tox(xr))
    rl = d.add(elm.Resistor().down().toy(node_n)
               .label(_val(comps, "RL1"), loc="center", ofst=(0, 0.9)))
    elems["RL1"] = rl
    d.add(elm.Line().tox(node_n[0] + W))
    return d, elems


def render_voltage_follower(circuit: dict, unit: float = 2.0,
                            style: dict | None = None):
    """Suiveur de tension : sortie AOP rebouclée sur l'entrée inverseuse,
    charge RL en sortie. Gain unitaire."""
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    _cfg(d, unit, style)
    elems: dict = {}

    d.add(elm.Ground(lead=False))
    src = d.add(elm.SourceV().up().label(_val(comps, "V1"),
                                        loc="center", ofst=(0, 1.15)))
    elems["V1"] = src
    start = src.start

    d.add(elm.Line().right(unit * 0.8))
    op = d.add(elm.Opamp(leads=True).anchor("in2")
               .label(_val(comps, "U1"), loc="center", ofst=(0, -0.9)))
    elems["U1"] = op

    # contre-réaction totale : out -> in1
    d.add(elm.Line().down(unit * 0.5).at(op.in1))
    d.add(elm.Line().tox(op.out))
    d.add(elm.Line().toy(op.out))
    d.add(elm.Dot())

    d.add(elm.Line().right(unit * 0.6).at(op.out))
    rl = d.add(elm.Resistor().down().toy(start)
               .label(_val(comps, "RL1"), loc="center", ofst=(0, 0.9)))
    elems["RL1"] = rl
    d.add(elm.Line().tox(start))
    d.add(elm.Ground().at(start))
    return d, elems


def render_summing_amplifier(circuit: dict, unit: float = 2.0,
                             style: dict | None = None):
    """Sommateur inverseur à deux entrées.

    Motif OFFICIEL schemdraw (gallery/opamp.html, « Inverting Opamp ») :
    l'AOP est placé EN PREMIER, puis TOUT est positionné par `.at()` sur
    ses ancres (in1 / in2 / out). Faire l'inverse — enchaîner les
    éléments avec le curseur de dessin puis raccrocher l'AOP — produit
    des superpositions car la position du curseur n'est plus maîtrisée.

    Topologie : V1-R1 et V2-R2 convergent vers le nœud sommateur
    (entrée inverseuse), contre-réaction par Rf.
    """
    comps = circuit["components"]
    d = schemdraw.Drawing(show=False)
    _cfg(d, unit, style)
    elems: dict = {}

    op = d.add(elm.Opamp(leads=True)
               .label(_val(comps, "U1"), loc="center", ofst=(0, -1.0)))
    elems["U1"] = op

    # entrée non inverseuse à la masse
    d.add(elm.Line().down(unit / 4).at(op.in2))
    d.add(elm.Ground(lead=False))

    # nœud sommateur, en retrait de l'entrée inverseuse
    d.add(elm.Line().left(unit * 0.35).at(op.in1))
    node = d.here
    d.add(elm.Dot())

    # ── branche 1 : R1 puis source V1 ────────────────────────────────
    d.push()
    r1 = d.add(elm.Resistor().left()
               .label(_val(comps, "R1"), loc="center", ofst=(0, 0.65)))
    elems["R1"] = r1
    v1 = d.add(elm.SourceV().down().reverse()
               .label(_val(comps, "V1"), loc="center", ofst=(0, -1.35)))
    elems["V1"] = v1
    d.add(elm.Ground())
    d.pop()

    # ── branche 2 : décalée vers le bas, même structure ──────────────
    d.add(elm.Line().down(unit * 1.6).at(node))
    r2 = d.add(elm.Resistor().left()
               .label(_val(comps, "R2"), loc="center", ofst=(0, 0.65)))
    elems["R2"] = r2
    v2 = d.add(elm.SourceV().down().reverse()
               .label(_val(comps, "V2"), loc="center", ofst=(0, -1.35)))
    elems["V2"] = v2
    d.add(elm.Ground())

    # ── contre-réaction ─────────────────────────────────────────────
    d.add(elm.Line().up(unit).at(op.in1))
    rf = d.add(elm.Resistor().tox(op.out)
               .label(_val(comps, "Rf"), loc="center", ofst=(0, 0.65)))
    elems["Rf"] = rf
    d.add(elm.Line().toy(op.out))
    d.add(elm.Dot())
    d.add(elm.Line().right(unit * 0.4).at(op.out)
          .label("Vout", loc="center", ofst=(0.9, 0)))

    return d, elems


CUSTOM_RENDERERS.update({
    "npn_common_emitter": render_npn_common_emitter,
    "pnp_switch":         render_pnp_switch,
    "full_wave_bridge":   render_full_wave_bridge,
    "voltage_follower":   render_voltage_follower,
    "summing_amplifier":  render_summing_amplifier,
})
