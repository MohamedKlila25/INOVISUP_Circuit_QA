# -*- coding: utf-8 -*-
"""
Common.Schemas — contrats de données de tout le pipeline.

Placé dans Src/Common/ (et non directement sous Src/) précisément pour
éviter le risque de collision de noms génériques déjà démontré avec
Config.py : un module isolé nommé "Schemas" ou "Config" directement au
sommet de Src/ deviendrait un module Python global (`import Schemas`),
exposé à toute collision avec un autre paquet du même nom. En le
rangeant dans le paquet Common, il ne devient accessible que via
`Common.Schemas` — sans ambiguïté possible.

Note de convention : les DOSSIERS et FICHIERS suivent PascalCase (choix
du projet), mais les identifiants Python (fonctions, variables) restent
en snake_case — seule convention PEP8 qui ne concerne pas les noms de
fichiers et qui reste attendue par l'écosystème (pydantic, pytest,
autocomplétion). Seules les classes sont en PascalCase, ce qui est de
toute façon la norme partout.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComponentClass = Literal[
    "resistor", "capacitor", "polarized_capacitor", "inductor",
    "diode", "zener_diode", "led",
    "npn_transistor", "pnp_transistor",
    "vsource", "battery", "ground",
    "switch", "fuse", "opamp",
    "gate_and", "gate_or", "gate_xor", "gate_not",
    "gate_nand", "gate_nor", "gate_xnor",
]


class Domain(str, Enum):
    ELECTRICAL = "electrical"
    LOGIC = "logic"


class BBox(BaseModel):
    """Boîte englobante en pixels, coin haut-gauche / bas-droit."""
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="before")
    @classmethod
    def _accept_list(cls, v):
        if isinstance(v, (list, tuple)):
            if len(v) != 4:
                raise ValueError(f"une boîte attend 4 valeurs, reçu {len(v)}")
            return dict(zip(("x0", "y0", "x1", "y1"), v))
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"boîte dégénérée ou inversée : {self}")
        return self

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        return ((self.x0 + self.x1) / 2 / img_w,
                (self.y0 + self.y1) / 2 / img_h,
                (self.x1 - self.x0) / img_w,
                (self.y1 - self.y0) / img_h)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Terminal(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    x: float
    y: float


class DetectedComponent(BaseModel):
    """Composant détecté dans une image — sortie de l'étage YOLO.

    C'est le contrat entre Hybrid.Detection et tout ce qui suit
    (traçage des fils, OCR, construction du graphe).
    """
    id: str | None = None          # assigné après détection (ex: "R1")
    cls: ComponentClass | str = Field(alias="class")
    class_idx: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    terminals: list[Terminal] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class Net(BaseModel):
    net_id: int
    terminals: list[dict] = Field(min_length=2)


class CircuitAnnotation(BaseModel):
    """Vérité terrain du dataset généré — contrat de sortie de
    Data_Generation et contrat d'entrée de l'évaluation."""
    model_config = ConfigDict(populate_by_name=True)

    image: str
    image_size: tuple[int, int]
    template: str
    domain: Domain
    components: list[DetectedComponent]
    nets: list[Net] = Field(default_factory=list)

    def net_of(self, comp_id: str, terminal: str) -> int | None:
        for net in self.nets:
            for ref in net.terminals:
                if ref.get("comp_id") == comp_id and ref.get("terminal") == terminal:
                    return net.net_id
        return None


# ═══════════════════════════════════════════════════════════════════
#  Circuit graph — the pivot representation of the project
# ═══════════════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    """A component in the graph, with its OCR-read value."""
    id: str
    cls: ComponentClass | str = Field(alias="class")
    value: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(populate_by_name=True)


class GraphEdge(BaseModel):
    """An edge = two components sharing an electrical node."""
    source: str
    target: str
    net_id: int | None = None


class CircuitGraph(BaseModel):
    """Circuit representation consumed by the LLM stage.

    This is the pivot of the whole project: everything upstream
    (detection, terminals, wires, OCR) exists to produce it, and
    everything downstream consumes it. Giving it an explicit schema is
    what makes a predicted graph directly comparable to the
    ground-truth graph, term by term.
    """
    circuit_id: str
    domain: Domain = Domain.ELECTRICAL
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    source_image: str | None = None

    @model_validator(mode="after")
    def _edges_reference_nodes(self) -> "CircuitGraph":
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            unknown = {e.source, e.target} - ids
            if unknown:
                raise ValueError(f"edge references unknown node: {sorted(unknown)}")
        return self

    def to_netlist(self) -> str:
        """Compact textual netlist — THIS is what gets fed to the LLM,
        not the raw JSON: it is shorter and much closer to what the
        model has seen during pre-training (SPICE decks, textbook
        circuit descriptions) than a nested JSON structure would be.
        """
        lines = [f"# circuit {self.circuit_id} ({self.domain.value})"]
        for n in self.nodes:
            value = f" = {n.value}" if n.value else ""
            lines.append(f"{n.id}: {n.cls}{value}")
        seen: set[tuple[str, str]] = set()
        for e in self.edges:
            key = tuple(sorted((e.source, e.target)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{e.source} -- {e.target}")
        return "\n".join(lines)

    def component_count(self) -> dict[str, int]:
        """Components per class — answers 'how many resistors?' exactly,
        by graph traversal rather than by asking the LLM to count."""
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.cls] = counts.get(n.cls, 0) + 1
        return counts

    def neighbours(self, comp_id: str) -> set[str]:
        out: set[str] = set()
        for e in self.edges:
            if e.source == comp_id:
                out.add(e.target)
            elif e.target == comp_id:
                out.add(e.source)
        return out
