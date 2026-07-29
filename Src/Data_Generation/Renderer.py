# -*- coding: utf-8 -*-
"""
renderer.py — v3.1 CORRIGÉE (après inspection visuelle)
Renderer générique pour les circuits série/parallèle du catalogue.

CORRECTIONS v3.1 (bugs détectés par inspection visuelle des PNG) :
  - _draw_series : boucle TOUJOURS fermée via .tox(start)/.toy(start),
    schéma rectangulaire : source ↑, composants →, descente ↓, retour ←
  - _draw_parallel : retour aligné exactement sur le départ de la source
  - bridge : rendu dédié en vrai pont (2 branches parallèles de 2 R)
  - mixed : rendu série-parallèle correct pour les templates concernés
Patterns conformes à la doc officielle schemdraw (gallery/analog.html) :
fermeture de boucle par .tox()/.toy() sur des points de référence.
"""
from __future__ import annotations
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm


CLASS_TO_ELEM = {
    "resistor":            lambda: elm.Resistor(),
    "capacitor":           lambda: elm.Capacitor(),
    "polarized_capacitor": lambda: elm.Capacitor2(),
    "inductor":            lambda: elm.Inductor2(),
    "diode":               lambda: elm.Diode(),
    "led":                 lambda: elm.LED(),
    "zener_diode":         lambda: elm.Zener(),
    "vsource":             lambda: elm.SourceV(),
    "battery":             lambda: elm.Battery(),
    "switch":              lambda: elm.Switch(),
    "fuse":                lambda: elm.Fuse(),
    "ground":              lambda: elm.Ground(),
    "npn_transistor":      lambda: elm.BjtNpn(circle=True),
    "pnp_transistor":      lambda: elm.BjtPnp(circle=True),
    "opamp":               lambda: elm.Opamp(leads=True),
    "motor":               lambda: elm.Motor(),
}


def _make(cls: str):
    f = CLASS_TO_ELEM.get(cls)
    return f() if f else None


import random as _random


def sample_style(rng: "_random.Random", custom: bool = False) -> dict:
    """Tire un style visuel aléatoire pour diversifier le dataset."""
    return {
        "unit":     rng.uniform(1.9, 2.4) if custom else rng.uniform(2.5, 3.6),
        "lw":       rng.choice([1.5, 2.0, 2.5, 3.0]),
        "fontsize": rng.choice([9, 10, 11, 12]),
        "font":     rng.choice(["sans-serif", "sans-serif", "serif"]),
        "iec":      rng.random() < 0.40,          # résistances IEC 40%
        "hloc":     rng.choice(["top", "bot"]),   # labels horizontaux
        "vloc":     rng.choice(["top", "bot"]),   # labels verticaux
        "spacing":  rng.uniform(0.75, 1.35),      # espacement branches
    }


class CircuitRenderer:
    """Renderer de base — produit un SVG depuis un circuit dict."""

    def __init__(self, dpi: int = 150, unit: float = 3.0, fontsize: int = 10):
        self.dpi = dpi
        self.unit = unit
        self.fontsize = fontsize
        self.style: dict = {}          # style courant (voir sample_style)

    # ─────────────────────────────────────────────────────────────────
    def render(self, circuit: dict, svg_path: str, style: dict | None = None) -> bool:
        self.style = style or {}
        self.last_elems = {}    # id -> element (rendus dédiés uniquement)
        st = self.style
        try:
            # Style de résistance IEC/IEEE (global schemdraw, restauré ensuite)
            if st.get("iec"):
                elm.style(elm.STYLE_IEC)
            try:
                # DOMAINE LOGIQUE : renderer dédié (aucun mélange avec
                # les composants électriques)
                if circuit.get("domain") == "logic":
                    from Logic_Renders import render_logic
                    d, elems = render_logic(circuit,
                                            unit=st.get("unit", self.unit),
                                            style=st)
                    self.last_elems = elems
                    d.save(svg_path)
                    self.last_drawing = d
                    return True
                if "custom_render" in circuit:
                    from Custom_Renders import CUSTOM_RENDERERS
                    fn = CUSTOM_RENDERERS.get(circuit["custom_render"])
                    if fn is None:
                        return False
                    d, elems = fn(circuit, unit=st.get("unit", self.unit),
                                  style=st)
                    self.last_elems = elems
                    d.save(svg_path)
                    self.last_drawing = d
                    return True
                return self._render_generic(circuit, svg_path)
            finally:
                if st.get("iec"):
                    elm.style(elm.STYLE_IEEE)
        except Exception:
            import traceback
            traceback.print_exc()
            plt.close("all")
            return False

    # ─────────────────────────────────────────────────────────────────
    def _render_generic(self, circuit: dict, svg_path: str) -> bool:
        comps   = circuit.get("components", [])
        if not comps:
            return False
        grounds = [c for c in comps if c["class"] == "ground"]
        actives = [c for c in comps if c["class"] != "ground"]
        if not actives:
            return False

        topo = circuit.get("circuit_metadata", {}).get("topology", "series")
        tname = circuit.get("template", "")

        st = self.style
        u  = st.get("unit", self.unit)
        with schemdraw.Drawing(show=False) as d:
            d.config(unit=u,
                     fontsize=st.get("fontsize", self.fontsize),
                     lw=st.get("lw", 2),
                     font=st.get("font", "sans-serif"))
            self._u = u
            self._sp = st.get("spacing", 1.0)
            self._hloc = st.get("hloc", "top")
            self._vloc = st.get("vloc", "bot")

            if circuit.get("procedural"):
                self._draw_procedural(d, circuit)
            elif topo == "bridge":
                self._draw_bridge(d, actives, grounds)
            elif topo == "parallel":
                self._draw_parallel(d, actives, grounds)
            elif topo == "mixed":
                self._draw_mixed(d, actives, grounds, tname,
                                 circuit.get("mixed_spec"))
            else:
                self._draw_series(d, actives, grounds)

            d.save(svg_path)

        self.last_drawing = d   # figure accessible via d.fig.fig
        return True

    # ─────────────────────────────────────────────────────────────────
    def _label_for(self, comp: dict) -> str:
        cid = comp["id"]
        val = comp.get("value")
        return f"{cid}\n{val}" if val else cid

    def _split_src(self, actives):
        sources = [c for c in actives if c["class"] in ("vsource", "battery")]
        others  = [c for c in actives if c["class"] not in ("vsource", "battery")]
        if not sources:
            sources, others = actives[:1], actives[1:]
        return sources[0], others

    # ─────────────────────────────────────────────────────────────────
    def _draw_series(self, d, actives, grounds):
        """
        Boucle rectangulaire TOUJOURS fermée :
          source ↑ | moitié des composants → | descente (1 composant ou fil) ↓
          reste des composants ← | fermeture .tox()/.toy() sur le départ.
        Pattern doc officielle : fermeture via tox/toy sur point de référence.
        """
        src, others = self._split_src(actives)

        # Source : monte (côté gauche)
        s = _make(src["class"])
        src_elem = d.add(s.up().label(self._label_for(src), loc=self._hloc))
        start = src_elem.start          # point de référence de fermeture

        n = len(others)
        if n == 0:
            # Juste la source : petite boucle rectangulaire fermée
            d.add(elm.Line().right(self._u))
            d.add(elm.Line().toy(start))
            d.add(elm.Line().tox(start))
        elif n == 1:
            # 1 composant : en haut, puis descente + retour
            e = _make(others[0]["class"])
            d.add(e.right().label(self._label_for(others[0]), loc=self._hloc))
            d.add(elm.Line().toy(start))
            d.add(elm.Line().tox(start))
        else:
            # ≥2 composants : haut = tous sauf le dernier, descente = dernier
            top_comps  = others[:-1]
            down_comp  = others[-1]
            for comp in top_comps:
                e = _make(comp["class"])
                d.add(e.right().label(self._label_for(comp), loc=self._hloc))
            e = _make(down_comp["class"])
            # composant vertical : label à droite (loc='bot' avec .down()
            # place le texte du côté droit, vérifié visuellement)
            d.add(e.down().toy(start).label(self._label_for(down_comp), loc=self._vloc))
            d.add(elm.Line().tox(start))

        if grounds:
            d.add(elm.Ground())

    # ─────────────────────────────────────────────────────────────────
    def _draw_parallel(self, d, actives, grounds):
        """
        Source ↑ à gauche, branches verticales ↓ alignées, rail bas
        fermé sur le départ EXACT de la source (.tox(start)).
        """
        src, branches = self._split_src(actives)

        s = _make(src["class"])
        src_elem = d.add(s.up().label(self._label_for(src), loc=self._hloc))
        start = src_elem.start
        top_y = src_elem.end

        first_branch = True
        for comp in branches:
            # première branche écartée davantage : sinon son étiquette
            # peut recouvrir celle de la source quand le style tire un
            # espacement serré
            gap = 1.25 if first_branch else 0.9
            d.add(elm.Line().right(self._u * gap * self._sp))
            d.add(elm.Dot())
            first_branch = False
            d.push()
            e = _make(comp["class"])
            d.add(e.down().toy(start).label(self._label_for(comp), loc=self._vloc))
            d.add(elm.Dot())
            d.pop()

        # rail bas : depuis le bas de la dernière branche, retour exact
        d.push()
        # se replacer au bas de la dernière branche
        d.pop()
        d.add(elm.Line().down().toy(start))
        d.add(elm.Line().tox(start))

        if grounds:
            d.add(elm.Ground())

    # ─────────────────────────────────────────────────────────────────
    def _draw_bridge(self, d, actives, grounds):
        """
        Pont de Wheatstone : source à gauche, deux branches verticales
        de 2 résistances chacune, jonctions milieu marquées.
        """
        src, others = self._split_src(actives)
        # ordre attendu : R1, R2 (branche gauche), R3, R4 (branche droite)
        r = others + [None] * (4 - len(others))
        r1, r2, r3, r4 = r[:4]

        s = _make(src["class"])
        src_elem = d.add(s.up().label(self._label_for(src), loc=self._hloc))
        start = src_elem.start

        # rail haut
        d.add(elm.Line().right(self._u * 0.8 * self._sp))
        d.add(elm.Dot())
        top_node = d.here

        # branche gauche : R1 puis R2 vers le bas
        d.push()
        if r1 is not None:
            e = _make(r1["class"])
            d.add(e.down(self._u * 0.5).label(self._label_for(r1), loc=self._vloc))
            d.add(elm.Dot())
        if r2 is not None:
            e = _make(r2["class"])
            d.add(e.down().toy(start).label(self._label_for(r2), loc=self._vloc))
            d.add(elm.Dot())
        d.pop()

        # rail haut vers branche droite
        d.add(elm.Line().right(self._u * 1.4 * self._sp))
        d.add(elm.Dot())
        d.push()
        if r3 is not None:
            e = _make(r3["class"])
            d.add(e.down(self._u * 0.5).label(self._label_for(r3), loc=self._vloc))
            d.add(elm.Dot())
        if r4 is not None:
            e = _make(r4["class"])
            d.add(e.down().toy(start).label(self._label_for(r4), loc=self._vloc))
            d.add(elm.Dot())
        # rail bas : retour au départ de la source (X puis Y pour
        # fermer VRAIMENT la boucle — bug détecté par les nets : le
        # rail flottait 1 unité sous V1.start)
        d.add(elm.Line().tox(start))
        d.add(elm.Line().toy(start))
        d.pop()

        if grounds:
            d.add(elm.Ground().at(start))

    # ─────────────────────────────────────────────────────────────────
    def _draw_procedural(self, d, circuit):
        """
        Chaîne horizontale de slots ; un slot parallèle est dessiné en
        branches empilées (droite / relevée / abaissée) entre deux dots.
        Boucle refermée par toy/tox sur le départ de la source.
        """
        comp_by_id = {c["id"]: c for c in circuit["components"]}
        slots = circuit["procedural"]["slots"]
        src = next(c for c in circuit["components"]
                   if c["class"] in ("vsource", "battery"))

        e = _make(src["class"])
        src_elem = d.add(e.up().label(self._label_for(src), loc=self._hloc))
        start = src_elem.start
        rise = self._u * 0.85   # hauteur des branches relevées/abaissées

        for slot in slots:
            if len(slot) == 1:
                comp = comp_by_id[slot[0]]
                e = _make(comp["class"])
                d.add(e.right().label(self._label_for(comp), loc=self._hloc))
            else:
                d.add(elm.Dot())
                node_in = d.here
                # branche 0 : droite (référence de longueur)
                comp = comp_by_id[slot[0]]
                e = _make(comp["class"])
                mid = d.add(e.right().label(self._label_for(comp),
                                            loc=self._hloc))
                node_out = d.here
                # branches suivantes : relevée puis abaissée
                offsets = [rise, -rise]
                for bi, cid in enumerate(slot[1:]):
                    comp = comp_by_id[cid]
                    off = offsets[bi]
                    d.push()
                    d.move_from(node_in, 0, 0)
                    d.add(elm.Line().up(off) if off > 0
                          else elm.Line().down(-off))
                    e = _make(comp["class"])
                    d.add(e.right().tox(node_out)
                          .label(self._label_for(comp),
                                 loc="top" if off > 0 else "bot"))
                    d.add(elm.Line().toy(node_out))
                    d.pop()
                d.add(elm.Dot())

        # fermeture de la boucle
        d.add(elm.Line().toy(start))
        d.add(elm.Line().tox(start))

    # ─────────────────────────────────────────────────────────────────
    def _draw_mixed(self, d, actives, grounds, tname: str = "",
                    circuit_spec: dict | None = None):
        """
        Topologies mixtes du catalogue :
          voltage_divider_loaded : R1 série puis R2 // RL
          RLC_notch              : (L1-C1 série) // R1
        Rendu : source ↑, élément(s) série →, puis 2 branches // ↓,
        rail bas fermé sur la source.
        """
        src, others = self._split_src(actives)

        s = _make(src["class"])
        src_elem = d.add(s.up().label(self._label_for(src), loc=self._hloc))
        start = src_elem.start

        spec = circuit_spec or {}
        if spec:
            # Spécification générique : N composants en série puis le
            # reste en branches parallèles.
            by_id = {c["id"]: c for c in others}
            for cid in spec.get("series", []):
                comp = by_id.get(cid)
                if comp is None:
                    continue
                e = _make(comp["class"])
                d.add(e.right().label(self._label_for(comp), loc=self._hloc))
            d.add(elm.Dot())
            par = [by_id[c] for c in spec.get("parallel", []) if c in by_id]
            for i, comp in enumerate(par):
                if i:
                    d.add(elm.Line().right(self._u * 1.2 * self._sp))
                    d.add(elm.Dot())
                d.push()
                e = _make(comp["class"])
                d.add(e.down().toy(start)
                      .label(self._label_for(comp), loc=self._vloc))
                d.add(elm.Dot())
                d.pop()
            d.add(elm.Line().down().toy(start))
            d.add(elm.Line().tox(start))
            if grounds:
                d.add(elm.Ground().at(start))
            return

        if tname == "RLC_notch" and len(others) >= 3:
            # branche 1 = L1 + C1 en série (verticale), branche 2 = R1
            l1, c1, r1 = others[0], others[1], others[2]
            d.add(elm.Line().right(self._u * 0.9 * self._sp))
            d.add(elm.Dot())
            d.push()
            e = _make(l1["class"])
            # demi-hauteur : L1 + C1 partagent la branche (sinon C1,
            # étiré par .toy(), est écrasé et déborde sous le rail)
            d.add(e.down(self._u * 0.5).label(self._label_for(l1), loc=self._vloc))
            e = _make(c1["class"])
            d.add(e.down().toy(start).label(self._label_for(c1), loc=self._vloc))
            d.add(elm.Dot())
            d.pop()
            d.add(elm.Line().right(self._u * 1.2 * self._sp))
            d.add(elm.Dot())
            e = _make(r1["class"])
            d.add(e.down().toy(start).label(self._label_for(r1), loc=self._vloc))
            d.add(elm.Dot())
            d.add(elm.Line().tox(start))

        elif tname == "voltage_divider_loaded" and len(others) >= 3:
            # R1 série en haut, puis R2 // RL
            r1, r2, rl = others[0], others[1], others[2]
            e = _make(r1["class"])
            d.add(e.right().label(self._label_for(r1), loc=self._hloc))
            d.add(elm.Dot())
            d.push()
            e = _make(r2["class"])
            d.add(e.down().toy(start).label(self._label_for(r2), loc=self._vloc))
            d.add(elm.Dot())
            d.pop()
            d.add(elm.Line().right(self._u * 1.2 * self._sp))
            d.add(elm.Dot())
            e = _make(rl["class"])
            d.add(e.down().toy(start).label(self._label_for(rl), loc=self._vloc))
            d.add(elm.Dot())
            d.add(elm.Line().tox(start))

        else:
            # fallback générique : tout en parallèle après la source
            d.add(elm.Line().right(self._u * 0.9 * self._sp))
            d.add(elm.Dot())
            first = True
            for comp in others:
                if not first:
                    d.add(elm.Line().right(self._u * 1.2 * self._sp))
                    d.add(elm.Dot())
                d.push()
                e = _make(comp["class"])
                d.add(e.down().toy(start).label(self._label_for(comp), loc=self._vloc))
                d.add(elm.Dot())
                d.pop()
                first = False
            d.add(elm.Line().down().toy(start))
            d.add(elm.Line().tox(start))

        if grounds:
            d.add(elm.Ground().at(start))
