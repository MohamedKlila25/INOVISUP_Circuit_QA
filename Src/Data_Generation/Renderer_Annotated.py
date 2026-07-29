# -*- coding: utf-8 -*-
"""
renderer_annotated.py — v3 FINALE
Rendu Schemdraw + triple annotation :
  1. bbox serrée CARRÉE par composant (pour YOLO)
  2. terminaux précis (start/end en pixels) via anchors Schemdraw
  3. labels texte (id + valeur) avec position approchée (pour OCR)

Usage :
    r = CircuitRendererAnnotated()
    ann = r.render_full(circuit, svg_path, png_path)
    # ann = {'image_size': [W,H], 'components': [...], 'texts': [...]}
    # ou None en cas d'échec
"""
from __future__ import annotations
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import schemdraw
import schemdraw.elements as elm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from Renderer import CircuitRenderer

# ── Classes de composants ─────────────────────────────────────────────
CLASS_NAMES = [
    "resistor", "capacitor", "polarized_capacitor", "inductor",
    "diode", "zener_diode", "led",
    "npn_transistor", "pnp_transistor",
    "vsource", "battery", "ground",
    "switch", "fuse", "opamp",
    "gate_and", "gate_or", "gate_xor", "gate_not",
    "gate_nand", "gate_nor", "gate_xnor",
]
CLASS_MAP = {n: i for i, n in enumerate(CLASS_NAMES)}

# Mapping nom de classe Schemdraw → classe sémantique du catalogue
ELEM_TO_CLASS: dict[str, str] = {
    "ResistorIEC":  "resistor",
    "ResistorIEEE": "resistor",
    "Resistor":     "resistor",
    "Capacitor":    "capacitor",
    "Capacitor2":   "polarized_capacitor",
    "Inductor2":    "inductor",
    "Inductor":     "inductor",
    "Diode":        "diode",
    "LED":          "led",
    "Zener":        "zener_diode",
    "SourceV":      "vsource",
    "SourceSin":    "vsource",
    "Battery":      "battery",
    "Switch":       "switch",
    "FuseIEEE":     "fuse",
    "FuseUS":       "fuse",
    "FuseIEC":      "fuse",
    "Fuse":         "fuse",
    "Ground":       "ground",
    "BjtNpn":       "npn_transistor",
    "BjtPnp":       "pnp_transistor",
    "Opamp":        "opamp",
}


class CircuitRendererAnnotated(CircuitRenderer):
    """
    Rendu avec triple annotation : bbox / terminaux / labels.
    Hérite de CircuitRenderer pour la logique de dessin Schemdraw.
    """

    def __init__(self, dpi: int = 150, unit: float = 3.0,
                 fontsize: int = 10, png_size: int = 800):
        super().__init__(dpi=dpi, unit=unit, fontsize=fontsize)
        self.png_size = png_size

    # ─────────────────────────────────────────────────────────────────
    def render_full(self, circuit: dict,
                    svg_path: str, png_path: str,
                    style: dict | None = None) -> dict | None:
        """
        Rend le circuit, convertit en PNG, extrait les annotations.
        Retourne un dict ou None en cas d'échec.
        """
        records: list[tuple] = []   # (element, class_name, comp_id)
        dot_records: list = []       # éléments Dot (candidats jonction)
        extra_grounds: list = []     # masses au-delà du GND1 du catalogue
        id_map = {c["id"]: c for c in circuit.get("components", [])}

        # File d'attente par classe sémantique (ordre du catalogue)
        remaining: dict[str, list[str]] = {}
        for c in circuit.get("components", []):
            remaining.setdefault(c["class"], []).append(c["id"])
        # 'vsource_dc' a été résolu en 'vsource' ou 'battery' à l'instanciation
        # mais les clés peuvent être 'vsource' → déjà bon

        orig_add  = schemdraw.Drawing.add

        def _tracking_add(d_self, element, **kwargs):
            result = orig_add(d_self, element, **kwargs)
            ecls = type(element).__name__
            if ecls == "Dot":
                dot_records.append(result)
                return result
            # Tag explicite prioritaire : schemdraw ne distingue pas
            # Nand/And (type `And`) ni Or/Nor/Xor/Xnor (type `Or`).
            cls  = getattr(element, "_cvqa_class", None) or ELEM_TO_CLASS.get(ecls)
            if cls is None:
                # Le style (IEEE / IEC / US) change le nom de classe
                # concret : ResistorIEC, FuseIEC, FuseUS... On retombe
                # sur le nom de base en retirant le suffixe de norme.
                for suf in ("IEEE", "IEC", "US"):
                    if ecls.endswith(suf):
                        cls = ELEM_TO_CLASS.get(ecls[: -len(suf)])
                        break
            if cls:
                ids_left = remaining.get(cls, [])
                comp_id  = ids_left.pop(0) if ids_left else None
                if comp_id is None and cls == "ground":
                    # Masse surnuméraire : même nœud électrique que GND1
                    # (convention) — gardée pour le calcul des nets.
                    extra_grounds.append(result)
                else:
                    records.append((result, cls, comp_id))
            return result

        schemdraw.Drawing.add = _tracking_add

        try:
            # ── Rendu SVG ────────────────────────────────────────────
            ok = self.render(circuit, svg_path, style=style)
            if not ok:
                print("[render_full] échec du rendu SVG")
                return None
            if not records:
                print("[render_full] aucun composant tracké")
                return None

            # Figure récupérée DIRECTEMENT depuis la Drawing schemdraw
            # (robuste en Jupyter : plt.gcf() peut pointer sur une autre
            # figure avec le backend inline).
            drawing = getattr(self, "last_drawing", None)
            sfig    = getattr(drawing, "fig", None)   # schemdraw Figure
            fig     = getattr(sfig, "fig", None)      # matplotlib Figure
            ax      = getattr(sfig, "ax",  None)      # matplotlib Axes
            if fig is None or ax is None:
                print("[render_full] figure schemdraw inaccessible")
                return None

            # ── Figure → PNG (rendu matplotlib direct) ────────────────
            # On N'UTILISE PAS le SVG comme source du PNG : schemdraw
            # l'écrit en « bbox tight », donc sa taille diffère de celle
            # de la figure dès qu'une étiquette déborde. L'échelle
            # déduite des dimensions figure serait alors fausse (~1 %) et
            # les terminaux tomberaient à côté des fils. En rasterisant
            # la figure elle-même, la correspondance coordonnées
            # d'affichage -> pixels est exacte par construction.
            dpi_out = self.png_size / fig.get_figwidth()
            fig.savefig(png_path, dpi=dpi_out, facecolor="white",
                        edgecolor="none", bbox_inches=None)
            with Image.open(png_path) as _im:
                W, H = _im.size

            scale = dpi_out / fig.dpi     # display px -> pixels PNG

            def to_px(x_data: float, y_data: float) -> tuple[float, float]:
                xd, yd = ax.transData.transform((x_data, y_data))
                return xd * scale, H - yd * scale

            # ── Bboxes RÉELLES des textes (labels) via matplotlib ─────
            # Canvas Agg attaché EXPLICITEMENT : sur certains environnements
            # (cluster/Jupyter), fig.canvas est un FigureCanvasBase sans
            # get_renderer(). FigureCanvasAgg fonctionne partout.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            agg_canvas = FigureCanvasAgg(fig)
            agg_canvas.draw()
            mpl_renderer = agg_canvas.get_renderer()
            text_boxes = []   # (contenu, [x0,y0,x1,y1])
            for txt_obj in ax.texts:
                content = txt_obj.get_text()
                if not content.strip():
                    continue
                tb = txt_obj.get_window_extent(mpl_renderer)
                tx0 = tb.x0 * scale
                tx1 = tb.x1 * scale
                ty0 = H - tb.y1 * scale
                ty1 = H - tb.y0 * scale
                text_boxes.append((content, [round(tx0, 1), round(ty0, 1),
                                             round(tx1, 1), round(ty1, 1)]))

            # ── Extraction annotations ─────────────────────────────────
            # Les rendus dédiés (AOP, transistors, pont, logique)
            # retournent une table id -> element : c'est la SOURCE DE
            # VÉRITÉ des identifiants. L'appariement par file d'attente
            # de classe suppose que l'ordre de dessin suit l'ordre du
            # catalogue, ce qui n'est pas garanti dans ces rendus.
            id_by_obj = {id(v): k
                         for k, v in getattr(self, "last_elems", {}).items()}
            if id_by_obj:
                records = [(el, cls, id_by_obj.get(id(el), cid))
                           for (el, cls, cid) in records]

            components_out = []
            texts_out      = []

            for elem, cls_name, comp_id in records:
                if comp_id is None:
                    continue

                # Bounding box sans texte
                try:
                    bb = elem.get_bbox(transform=True, includetext=False)
                except Exception:
                    continue

                x0p, y0p = to_px(bb.xmin, bb.ymin)
                x1p, y1p = to_px(bb.xmax, bb.ymax)
                px0, px1  = sorted([x0p, x1p])
                py0, py1  = sorted([y0p, y1p])
                bw = px1 - px0
                bh = py1 - py0

                # ── bbox CORPS du composant ────────────────────────────
                # Doc schemdraw : istart/iend = points internes du corps,
                # avant les extensions de leads. Le corps s'étend de
                # istart à iend le long de l'axe, et sur toute la hauteur
                # de la bbox brute en perpendiculaire. Le composant est
                # donc ENTIÈREMENT contenu dans cette boîte.
                absanchors = getattr(elem, "absanchors", None)
                MARGIN = 8  # marge px : englobe trait épais + antialiasing
                if (absanchors and "istart" in absanchors
                        and "iend" in absanchors):
                    isx, isy = to_px(absanchors["istart"].x,
                                     absanchors["istart"].y)
                    iex, iey = to_px(absanchors["iend"].x,
                                     absanchors["iend"].y)
                    ax0, ax1 = sorted([isx, iex])
                    ay0, ay1 = sorted([isy, iey])
                    # axe quasi-horizontal : X depuis istart/iend,
                    # Y depuis la bbox brute (hauteur du zigzag) — et
                    # inversement pour un composant vertical.
                    if (ax1 - ax0) >= (ay1 - ay0):
                        sq_x0, sq_x1 = ax0, ax1
                        sq_y0, sq_y1 = py0, py1
                    else:
                        sq_x0, sq_x1 = px0, px1
                        sq_y0, sq_y1 = ay0, ay1
                else:
                    # Éléments sans istart/iend (opamp, ground) :
                    # bbox brute complète (le symbole entier)
                    sq_x0, sq_y0, sq_x1, sq_y1 = px0, py0, px1, py1

                sq_x0 = max(0, sq_x0 - MARGIN)
                sq_y0 = max(0, sq_y0 - MARGIN)
                sq_x1 = min(W, sq_x1 + MARGIN)
                sq_y1 = min(H, sq_y1 + MARGIN)
                if sq_x1 - sq_x0 < 4 or sq_y1 - sq_y0 < 4:
                    continue
                cx = (sq_x0 + sq_x1) / 2
                cy = (sq_y0 + sq_y1) / 2
                half = max(sq_x1 - sq_x0, sq_y1 - sq_y0) / 2

                # ── Terminaux (start / end, ou anchors spécifiques) ────
                terminals: list[dict] = []
                absanchors = getattr(elem, "absanchors", None)
                if absanchors and (cls_name == "opamp"
                                   or cls_name.startswith("gate_")):
                    # AOP et portes logiques : entrées inN + sortie out
                    innames = sorted(
                        (k for k in absanchors if k.startswith("in")),
                        key=lambda k: k)
                    for name in innames + ["out"]:
                        if name in absanchors:
                            pt = absanchors[name]
                            tx, ty = to_px(pt.x, pt.y)
                            terminals.append({"name": name, "x": round(tx, 1),
                                              "y": round(ty, 1)})
                elif absanchors and cls_name in ("npn_transistor",
                                                 "pnp_transistor"):
                    # Transistors : 3 terminaux réels
                    for name in ("base", "collector", "emitter"):
                        if name in absanchors:
                            pt = absanchors[name]
                            tx, ty = to_px(pt.x, pt.y)
                            terminals.append({"name": name, "x": round(tx, 1),
                                              "y": round(ty, 1)})
                elif absanchors and "start" in absanchors and "end" in absanchors:
                    for name in ("start", "end"):
                        pt = absanchors[name]
                        tx, ty = to_px(pt.x, pt.y)
                        terminals.append({"name": name, "x": round(tx, 1),
                                          "y": round(ty, 1)})
                else:
                    # Composant sans start/end (ex: ground) → deux points sur les bords
                    if bw >= bh:
                        terminals = [
                            {"name": "start", "x": round(px0, 1), "y": round(cy, 1)},
                            {"name": "end",   "x": round(px1, 1), "y": round(cy, 1)},
                        ]
                    else:
                        terminals = [
                            {"name": "start", "x": round(cx, 1), "y": round(py0, 1)},
                            {"name": "end",   "x": round(cx, 1), "y": round(py1, 1)},
                        ]

                # ── Label texte : bbox RÉELLE, appariée par ID ─────────
                comp_real = id_map.get(comp_id, {})
                value = comp_real.get("value")

                components_out.append({
                    "id":        comp_id,
                    "class":     cls_name,
                    "class_idx": CLASS_MAP.get(cls_name, -1),
                    "bbox":      [round(sq_x0, 1), round(sq_y0, 1),
                                  round(sq_x1, 1), round(sq_y1, 1)],
                    "terminals": terminals,
                })

                # Trouver le texte dont la 1re ligne est exactement l'ID
                if value is not None:
                    for content, tbox in text_boxes:
                        first_line = content.split("\n")[0].strip()
                        if first_line == comp_id:
                            texts_out.append({
                                "comp_id":    comp_id,
                                "text":       content.replace("\n", " "),
                                "id_text":    comp_id,
                                "value_text": str(value),
                                "bbox":       tbox,
                            })
                            break

            # ── Jonctions : Dots avec ≥3 branches de fil ─────────────
            # Un Dot posé à un coin (2 branches) n'est PAS une jonction
            # électrique — on mesure le degré en comptant les directions
            # (N,S,E,O) contenant de l'encre à ~14px du centre.
            import numpy as np
            with Image.open(png_path) as _im:
                g = np.array(_im.convert("L"))
            ink_arr = g < 128
            junctions_out = []
            seen = set()
            for dot in dot_records:
                aa = getattr(dot, "absanchors", None)
                if not aa or "center" not in aa:
                    continue
                jx, jy = to_px(aa["center"].x, aa["center"].y)
                key = (round(jx / 4), round(jy / 4))   # dédoublonnage ~4px
                if key in seen:
                    continue
                seen.add(key)
                jxi, jyi = int(jx), int(jy)
                R, T = 14, 4   # rayon d'échantillonnage, demi-épaisseur
                deg = 0
                # N, S, E, O : bande perpendiculaire à chaque direction
                probes = [
                    ink_arr[max(0, jyi-R-3):max(0, jyi-R+3), max(0, jxi-T):jxi+T],
                    ink_arr[jyi+R-3:jyi+R+3,                 max(0, jxi-T):jxi+T],
                    ink_arr[max(0, jyi-T):jyi+T,             jxi+R-3:jxi+R+3],
                    ink_arr[max(0, jyi-T):jyi+T,             max(0, jxi-R-3):max(0, jxi-R+3)],
                ]
                for pr in probes:
                    if pr.size and pr.any():
                        deg += 1
                if deg >= 3:
                    junctions_out.append({
                        "x": round(jx, 1), "y": round(jy, 1),
                        "degree": deg, "type": "junction",
                    })

            # ── NETS : ground truth des connexions terminal-à-terminal ─
            # Masque de fils = encre − bboxes composants − bboxes textes.
            # Deux terminaux sur la même composante connexe du masque
            # appartiennent au même nœud électrique (net).
            import cv2 as _cv2
            wire_mask = ink_arr.copy()

            def _erase(bx0, by0, bx1, by1, pad: int = 2):
                # Arrondi vers l'EXTÉRIEUR + marge : int() tronquait les
                # bords (384.6→384) et laissait des résidus de corps de
                # composant auxquels les terminaux s'accrochaient.
                ex0 = max(0, int(np.floor(bx0)) - pad)
                ey0 = max(0, int(np.floor(by0)) - pad)
                ex1 = min(W, int(np.ceil(bx1)) + pad)
                ey1 = min(H, int(np.ceil(by1)) + pad)
                wire_mask[ey0:ey1, ex0:ex1] = False

            for comp in components_out:
                bx0, by0, bx1, by1 = comp["bbox"]
                if comp["class"] == "ground":
                    # La masse est un NŒUD conducteur : son encre reste
                    # dans le masque de fils (convention netlist).
                    continue
                if comp["class"] in ("npn_transistor", "pnp_transistor"):
                    # Transistor : effacer un DISQUE couvrant le corps du
                    # symbole (le cercle), ce qui sépare les trois
                    # électrodes tout en préservant leurs fils. Le symbole
                    # peut être tourné (collecteur et émetteur alors au
                    # même x) : réduire sur un seul axe ne suffirait pas.
                    # Rayon calibré empiriquement (0.48 : ni court-circuit
                    # collecteur-émetteur, ni terminal isolé).
                    cxb, cyb = (bx0 + bx1) / 2, (by0 + by1) / 2
                    rad = 0.48 * min(bx1 - bx0, by1 - by0)
                    yy_d, xx_d = np.ogrid[:H, :W]
                    wire_mask[((yy_d - cyb) ** 2
                               + (xx_d - cxb) ** 2) <= rad ** 2] = False
                    continue
                elif comp["class"].startswith("gate_"):
                    # Porte logique : entrées à gauche, sortie à droite ;
                    # effacer le corps en préservant les fils de broche.
                    txs = [t["x"] for t in comp["terminals"]]
                    if txs:
                        nb0, nb1 = min(txs) + 8, max(txs) - 8
                        if nb1 > nb0:
                            bx0, bx1 = nb0, nb1
                if comp["class"] == "opamp":
                    # N'effacer que le triangle : préserver les fils
                    # verticaux qui longent les bords (contre-réaction)
                    # et les dots aux anchors in/out.
                    txs = [t["x"] for t in comp["terminals"]]
                    if txs:
                        bx0 = min(txs) + 12
                        bx1 = max(txs) - 12
                _erase(bx0, by0, bx1, by1)
            text_regions = []
            for _, tbox in text_boxes:
                _erase(*tbox, pad=1)
                text_regions.append(tbox)

            # RECONNEXION DES FILS SOUS LES TEXTES.
            # Un label peut chevaucher un fil : l'effacer couperait le
            # conducteur en deux blobs et casserait le net. Pour chaque
            # boîte de texte, si de l'encre existe des DEUX côtés opposés
            # à la même ligne (ou colonne), c'est un fil qui traverse :
            # on rétablit un pont d'un pixel de large.
            for (tx0, ty0, tx1, ty1) in text_regions:
                ex0 = max(0, int(np.floor(tx0)) - 1)
                ey0 = max(0, int(np.floor(ty0)) - 1)
                ex1 = min(W, int(np.ceil(tx1)) + 1)
                ey1 = min(H, int(np.ceil(ty1)) + 1)
                PROBE = 4
                # fils horizontaux traversant la boîte
                lx0, lx1 = max(0, ex0 - PROBE), ex0
                rx0, rx1 = ex1, min(W, ex1 + PROBE)
                if lx1 > lx0 and rx1 > rx0:
                    for yy in range(ey0, ey1):
                        if (wire_mask[yy, lx0:lx1].any()
                                and wire_mask[yy, rx0:rx1].any()):
                            wire_mask[yy, ex0:ex1] = True
                # fils verticaux traversant la boîte
                ty_a0, ty_a1 = max(0, ey0 - PROBE), ey0
                ty_b0, ty_b1 = ey1, min(H, ey1 + PROBE)
                if ty_a1 > ty_a0 and ty_b1 > ty_b0:
                    for xx in range(ex0, ex1):
                        if (wire_mask[ty_a0:ty_a1, xx].any()
                                and wire_mask[ty_b0:ty_b1, xx].any()):
                            wire_mask[ey0:ey1, xx] = True

            # PROTECTION DES NŒUDS : restaurer l'encre dans un disque de
            # 6px autour de chaque terminal. À un terminal, l'encre est
            # toujours du CONDUCTEUR (dot de jonction, extrémité de lead)
            # — jamais du corps de composant. Sans cela, les styles à
            # petite unité (leads courts) voient leur dot de nœud
            # entièrement effacé par les bboxes voisines.
            PROT = 6
            yy_g, xx_g = np.ogrid[-PROT:PROT + 1, -PROT:PROT + 1]
            disk = (yy_g ** 2 + xx_g ** 2) <= PROT ** 2
            for comp in components_out:
                for t in comp["terminals"]:
                    txi, tyi = int(round(t["x"])), int(round(t["y"]))
                    y0p_, y1p_ = max(0, tyi - PROT), min(H, tyi + PROT + 1)
                    x0p_, x1p_ = max(0, txi - PROT), min(W, txi + PROT + 1)
                    dsub = disk[PROT - (tyi - y0p_):PROT + (y1p_ - tyi),
                                PROT - (txi - x0p_):PROT + (x1p_ - txi)]
                    region = wire_mask[y0p_:y1p_, x0p_:x1p_]
                    region |= ink_arr[y0p_:y1p_, x0p_:x1p_] & dsub

            n_labels, labels_img = _cv2.connectedComponents(
                wire_mask.astype(np.uint8), connectivity=8)

            def _blobs_near(px: float, py: float, radius: int = 40,
                            away_from=None) -> list[int]:
                """TOUS les labels de blobs de fil à portée du point.
                away_from=(ox,oy) : ne garder que les pixels du demi-plan
                opposé à ce point (l'autre terminal du composant), pour ne
                pas fusionner les fils des deux côtés d'un composant court
                (court-circuit). Marge 14px : couvre les fils perpendiculaires au niveau du terminal et le stub de lead du composant lui-même (~10px)
                passant au niveau exact du terminal."""
                pxi, pyi = int(round(px)), int(round(py))
                dirx = diry = 0.0
                if away_from is not None:
                    dirx, diry = px - away_from[0], py - away_from[1]
                    norm = (dirx * dirx + diry * diry) ** 0.5
                    if norm > 1e-6:
                        dirx, diry = dirx / norm, diry / norm
                    else:
                        away_from = None
                y0s, y1s = max(0, pyi - radius), min(H, pyi + radius + 1)
                x0s, x1s = max(0, pxi - radius), min(W, pxi + radius + 1)
                sub = labels_img[y0s:y1s, x0s:x1s]
                found = {}
                ys, xs = np.nonzero(sub)
                for yy, xx in zip(ys, xs):
                    dx = xx + x0s - pxi
                    dy = yy + y0s - pyi
                    d = dx * dx + dy * dy
                    if d > radius ** 2:
                        continue
                    if away_from is not None and (dx * dirx + dy * diry) < -14:
                        continue   # du côté de l'autre terminal : exclu
                    lab = int(sub[yy, xx])
                    if lab not in found or d < found[lab]:
                        found[lab] = d
                return sorted(found, key=found.get)

            # Union-find sur les blobs
            parent: dict[int, int] = {}
            def _find(a):
                parent.setdefault(a, a)
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def _union(a, b):
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[rb] = ra

            term_blob: list[tuple] = []   # (comp_id, terminal, blob_repr)
            for comp in components_out:
                terms = comp["terminals"]
                for t in terms:
                    # L'autre terminal du composant (filtre demi-plan) :
                    # uniquement pour les 2-terminaux non-masse.
                    away = None
                    if len(terms) == 2 and comp["class"] != "ground":
                        other = terms[1] if t is terms[0] else terms[0]
                        away = (other["x"], other["y"])
                    blobs = _blobs_near(t["x"], t["y"], away_from=away)
                    if not blobs:
                        continue
                    first = blobs[0]
                    for b in blobs[1:]:
                        _union(first, b)
                    term_blob.append((comp["id"], t["name"], first))

            net_members: dict[int, list] = {}
            for comp_id, tname_, blob in term_blob:
                root = _find(blob)
                net_members.setdefault(root, []).append(
                    {"comp_id": comp_id, "terminal": tname_})

            # Masses surnuméraires -> membres 'GND1' dans leur blob
            gnd_id = next((c["id"] for c in components_out
                           if c["class"] == "ground"), "GND")
            for g in extra_grounds:
                aa = getattr(g, "absanchors", None)
                if not aa:
                    continue
                pt = aa.get("start") or aa.get("center") or aa.get("xy")
                if pt is None:
                    continue
                gx, gy = to_px(pt.x, pt.y)
                blobs = _blobs_near(gx, gy)
                if blobs:
                    first = blobs[0]
                    for b in blobs[1:]:
                        _union(first, b)
                    net_members.setdefault(_find(first), []).append(
                        {"comp_id": gnd_id, "terminal": "gnd"})

            # Convention netlist : toutes les masses forment UN SEUL net
            # global (les symboles de masse séparés = même nœud électrique)
            ground_ids = {c["id"] for c in components_out
                          if c["class"] == "ground"}
            gnd_blobs = [b for b, ms in net_members.items()
                         if any(m["comp_id"] in ground_ids for m in ms)]
            if len(gnd_blobs) > 1:
                target = gnd_blobs[0]
                for b in gnd_blobs[1:]:
                    net_members[target].extend(net_members.pop(b))

            nets_out = []
            for i, (blob, members) in enumerate(sorted(net_members.items())):
                if len(members) >= 2:   # un net relie au moins 2 terminaux
                    # dédoublonner (comp_id, terminal)
                    seen_m = set()
                    uniq = []
                    for m in members:
                        key = (m["comp_id"], m["terminal"])
                        if key not in seen_m:
                            seen_m.add(key)
                            uniq.append(m)
                    nets_out.append({"net_id": i, "terminals": uniq})

            return {
                "image_size": [W, H],
                "domain":     circuit.get("domain", "electrical"),
                "components": components_out,
                "texts":      texts_out,
                "junctions":  junctions_out,
                "nets":       nets_out,
                "crossovers": [],   # catalogue actuel 100% planaire :
                                    # aucun croisement sans connexion
            }

        except Exception:
            import traceback
            traceback.print_exc()
            return None

        finally:
            schemdraw.Drawing.add = orig_add
            plt.close("all")
