# -*- coding: utf-8 -*-
"""
Graph.Qa_Generator — generates question/answer pairs FROM a CircuitGraph,
with exact, automatic ground truth.

Same principle as Data_Generation: since every fact in a CircuitGraph
is already known exactly (it was either generated with known ground
truth, or extracted by the pipeline), question/answer pairs can be
derived directly by TRAVERSING the graph — no manual labeling needed,
and the answer is guaranteed correct by construction.

This produces the benchmark used to decide whether a general-purpose
LLM (prompted, no fine-tuning) is already good enough at reasoning over
a netlist before committing to fine-tuning — see the reasoning in
Src/Common/Schemas.py's CircuitGraph docstring for why this task
(reasoning over an already-extracted, compact netlist) is fundamentally
easier than raw image QA, and worth measuring before assuming
fine-tuning is required.
"""
from __future__ import annotations

import random

from Common.Schemas import CircuitGraph, Domain


class QaPair:
    __slots__ = ("question", "answer", "question_type", "circuit_id")

    def __init__(self, question: str, answer: str, question_type: str,
                circuit_id: str):
        self.question = question
        self.answer = answer
        self.question_type = question_type
        self.circuit_id = circuit_id

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
               "question_type": self.question_type,
               "circuit_id": self.circuit_id}


_CLASS_LABEL_FR = {
    "resistor": "résistance", "capacitor": "condensateur",
    "polarized_capacitor": "condensateur polarisé", "inductor": "inductance",
    "diode": "diode", "zener_diode": "diode Zener", "led": "LED",
    "npn_transistor": "transistor NPN", "pnp_transistor": "transistor PNP",
    "vsource": "source de tension", "battery": "pile", "ground": "masse",
    "switch": "interrupteur", "fuse": "fusible", "opamp": "amplificateur opérationnel",
    "gate_and": "porte AND", "gate_or": "porte OR", "gate_xor": "porte XOR",
    "gate_not": "porte NOT", "gate_nand": "porte NAND", "gate_nor": "porte NOR",
    "gate_xnor": "porte XNOR",
}


def _label(cls: str) -> str:
    return _CLASS_LABEL_FR.get(cls, cls)


def generate_qa_pairs(graph: CircuitGraph, rng: random.Random | None = None
                      ) -> list[QaPair]:
    """Generates a battery of question/answer pairs from one graph.
    Every answer is derived by traversal, never guessed — exact ground
    truth by construction."""
    rng = rng or random.Random(0)
    pairs: list[QaPair] = []
    cid = graph.circuit_id

    # ── count questions ────────────────────────────────────────────
    counts = graph.component_count()
    for cls, n in counts.items():
        q = f"Combien de {_label(cls)}(s) y a-t-il dans ce circuit ?"
        pairs.append(QaPair(q, str(n), "count", cid))

    q = "Combien de composants (nœuds) au total y a-t-il dans ce circuit ? Ne compte pas les connexions."
    pairs.append(QaPair(q, str(len(graph.nodes)), "count", cid))

    # ── identify questions ──────────────────────────────────────────
    # Expected answer = the RAW class token, exactly as it appears in
    # the netlist (n.cls), NOT a French translation. Asking for a
    # translation the model was never instructed to produce created a
    # spurious failure mode: the model correctly read "battery" from
    # the netlist and answered "Battery" — accurate, just not what a
    # strict French-label match expected.
    for node in graph.nodes:
        q = (f"Quelle est la classe du composant {node.id}, "
             f"telle qu'elle apparaît dans la netlist ?")
        pairs.append(QaPair(q, node.cls, "identify", cid))

    # ── value questions ──────────────────────────────────────────────
    for node in graph.nodes:
        if node.value:
            q = f"Quelle est la valeur de {node.id} ?"
            pairs.append(QaPair(q, node.value, "value", cid))

    # ── connectivity questions (balanced: real + fake pairs) ─────────
    node_ids = [n.id for n in graph.nodes]
    if len(node_ids) >= 2:
        real_pairs = [(e.source, e.target) for e in graph.edges]
        seen = {tuple(sorted(p)) for p in real_pairs}
        for a, b in real_pairs[: min(5, len(real_pairs))]:
            q = f"{a} est-il directement connecté à {b} ?"
            pairs.append(QaPair(q, "Oui", "connectivity", cid))

        # Cap at the actual number of non-connected pairs available —
        # a dense graph (e.g. 9 of 10 possible pairs already edges)
        # may have very few, or even zero, negative examples to draw.
        all_pairs = {tuple(sorted((a, b)))
                    for i, a in enumerate(node_ids)
                    for b in node_ids[i + 1:]}
        available_negatives = list(all_pairs - seen)
        rng.shuffle(available_negatives)
        for a, b in available_negatives[:5]:
            q = f"{a} est-il directement connecté à {b} ?"
            pairs.append(QaPair(q, "Non", "connectivity", cid))

    # ── topology question ────────────────────────────────────────────
    if graph.domain == Domain.ELECTRICAL:
        q = "Ce circuit contient-il une masse (ground) ?"
        has_gnd = any(n.cls == "ground" for n in graph.nodes)
        pairs.append(QaPair(q, "Oui" if has_gnd else "Non", "topology", cid))

    return pairs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from Graph.Builder import from_annotation

    graph = from_annotation(sys.argv[1] if len(sys.argv) > 1
                            else "/tmp/eval_test/circuit_00000.json")
    pairs = generate_qa_pairs(graph)
    print(f"{len(pairs)} questions générées pour {graph.circuit_id}")
    for p in pairs:
        print(f"  [{p.question_type}] {p.question} -> {p.answer}")