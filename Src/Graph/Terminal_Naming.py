# -*- coding: utf-8 -*-
"""
Graph.Terminal_Naming — derives human-meaningful pin names from
terminal geometry and component class.

The pipeline knows WHERE each terminal is (x, y) but not what to CALL
it. A user asking "what is R1's left pin wired to?" needs names, not
coordinates.

Naming rules, in order of specificity:
  1. Semantic names for polarised / multi-pin parts, where the pin has
     a real electrical role: diodes (anode/cathode), transistors
     (base/collector/emitter), op-amps (in+/in-/out).
  2. Otherwise, positional names derived from the terminal's position
     relative to the component's bounding-box centre: a horizontal
     component gets "gauche"/"droite", a vertical one "haut"/"bas".

On diode orientation: schemdraw exposes only generic `start`/`end`
anchors (verified — no `anode`/`cathode` anchors exist), but by its
drawing convention current flows start -> end, so start is the anode
and end the cathode. Since the pipeline loses anchor names by the time
nets are built, orientation is recovered here from geometry instead:
for a horizontal diode the left terminal is the anode, for a vertical
one the top terminal is. This holds for the generated dataset, where
diodes are drawn in the conducting direction; it would NOT be safe to
assume on arbitrary hand-drawn schematics.
"""
from __future__ import annotations

# Classes whose pins carry a real electrical role, not just a position.
POLARISED_CLASSES = {"diode", "zener_diode", "led"}
THREE_PIN_CLASSES = {"npn_transistor", "pnp_transistor", "opamp"}


def is_horizontal(bbox) -> bool:
    """True if the component is drawn wider than tall."""
    return (bbox.x1 - bbox.x0) >= (bbox.y1 - bbox.y0)


def orientation_from_terminals(terminals: list[tuple[float, float]]) -> str:
    """Decides 'h' or 'v' from how the terminals are SPREAD, not from
    the bounding-box shape.

    Measured failure of the bbox rule: a voltage source renders in a
    near-square box (105x105 px), so width>=height classified it as
    horizontal and both of its terminals — which are actually stacked
    vertically — got named "gauche". Comparing the terminals' own x vs
    y spread is what actually reflects how the part is wired.
    """
    if len(terminals) < 2:
        return "h"
    xs = [t[0] for t in terminals]
    ys = [t[1] for t in terminals]
    return "h" if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else "v"


def positional_name(x: float, y: float, bbox,
                    orientation: str | None = None) -> str:
    """Names a terminal by its side of the component's centre.
    `orientation` ('h'/'v') should be passed when the component's
    terminals are known — it is more reliable than the bbox shape."""
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2
    horiz = (orientation == "h") if orientation else is_horizontal(bbox)
    if horiz:
        return "gauche" if x < cx else "droite"
    return "haut" if y < cy else "bas"


def semantic_name(cls: str, x: float, y: float, bbox,
                  orientation: str | None = None) -> str:
    """Names a terminal by its electrical role where one exists,
    falling back to a positional name otherwise."""
    pos = positional_name(x, y, bbox, orientation)

    if cls in POLARISED_CLASSES:
        # conducting direction: left->right, or top->bottom
        if pos in ("gauche", "haut"):
            return "anode"
        return "cathode"

    return pos


def name_terminal(cls: str, x: float, y: float, bbox,
                  orientation: str | None = None) -> str:
    """Public entry point: best available name for one terminal."""
    return semantic_name(cls, x, y, bbox, orientation)
