# -*- coding: utf-8 -*-
"""
Graph.Builder — turns pipeline output (or ground truth) into a
CircuitGraph, the pivot representation consumed by the LLM stage.

Two entry points, deliberately producing the SAME type:
  - from_pipeline()   : predicted graph, from HybridPipeline output
  - from_annotation() : ground-truth graph, from a Data_Generation JSON

Sharing the type is what makes evaluation honest: predicted and
reference graphs are compared term by term, not through two different
ad-hoc formats.
"""
from __future__ import annotations

import json
from pathlib import Path

from Common.Schemas import (CircuitGraph, GraphEdge, GraphNode, Domain,
                            DetectedComponent)


def _edges_from_nets(nets: list[set[str]],
                     net_ids: list[int] | None = None) -> list[GraphEdge]:
    """One edge per pair of components sharing a net.

    A net with k components yields k(k-1)/2 edges — a net is a
    hyperedge (it can join more than two components, e.g. a ground
    rail), and expanding it into pairwise edges keeps the graph a
    plain graph, which is what traversal and comparison expect.
    """
    edges: list[GraphEdge] = []
    for i, net in enumerate(nets):
        members = sorted(net)
        nid = net_ids[i] if net_ids and i < len(net_ids) else i
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                edges.append(GraphEdge(source=members[a],
                                      target=members[b], net_id=nid))
    return edges


def from_pipeline(components: list[DetectedComponent],
                  nets: list[set[str]],
                  values: dict[str, str | None],
                  circuit_id: str,
                  source_image: str | None = None,
                  domain: Domain = Domain.ELECTRICAL) -> CircuitGraph:
    """Predicted graph, from HybridPipeline results.

    Only components that actually appear in the graph's node list can
    be referenced by edges, so nets mentioning an unknown id (possible
    if a detection was dropped downstream) are filtered rather than
    allowed to raise — a partially recovered graph is more useful than
    none, and the missing component is already penalised by the
    detection metric.
    """
    nodes = [
        GraphNode(id=c.id, **{"class": c.cls}, value=values.get(c.id),
                  confidence=c.confidence)
        for c in components if c.id is not None
    ]
    known = {n.id for n in nodes}
    clean_nets = [{i for i in net if i in known} for net in nets]
    clean_nets = [net for net in clean_nets if len(net) >= 2]

    return CircuitGraph(
        circuit_id=circuit_id,
        domain=domain,
        nodes=nodes,
        edges=_edges_from_nets(clean_nets),
        source_image=source_image,
    )


def from_annotation(annotation_path: str | Path,
                    circuit_id: str | None = None) -> CircuitGraph:
    """Ground-truth graph, from a Data_Generation annotation JSON."""
    path = Path(annotation_path)
    ann = json.loads(path.read_text(encoding="utf-8"))

    values = {t["comp_id"]: t["value_text"] for t in ann.get("texts", [])}
    nodes = [
        GraphNode(id=c["id"], **{"class": c["class"]},
                  value=values.get(c["id"]))
        for c in ann["circuit"]["components"]
    ] if "circuit" in ann else [
        GraphNode(id=c["id"], **{"class": c["class"]},
                  value=values.get(c["id"]))
        for c in ann["components"]
    ]

    known = {n.id for n in nodes}
    nets = []
    net_ids = []
    for n in ann.get("nets", []):
        members = {m["comp_id"] for m in n["terminals"]} & known
        if len(members) >= 2:
            nets.append(members)
            net_ids.append(n.get("net_id"))

    domain_value = ann.get("domain", "electrical")
    return CircuitGraph(
        circuit_id=circuit_id or path.stem,
        domain=Domain(domain_value),
        nodes=nodes,
        edges=_edges_from_nets(nets, net_ids),
        source_image=ann.get("image"),
    )


def from_pipeline_with_terminals(components: list[DetectedComponent],
                                 raw_nets: list,
                                 values: dict[str, str | None],
                                 circuit_id: str,
                                 source_image: str | None = None,
                                 domain: Domain = Domain.ELECTRICAL) -> CircuitGraph:
    """Builds a graph whose edges carry PIN NAMES, from the tracer's raw
    nets (which keep each terminal's x/y) rather than from flattened
    id-only nets.

    `raw_nets` are Hybrid.Wires.Tracer Net objects, whose `.members`
    are dicts of {comp_id, x, y}. Terminal names are derived from that
    geometry — see Graph.Terminal_Naming.
    """
    from collections import defaultdict
    from Graph.Terminal_Naming import name_terminal, orientation_from_terminals

    comp_by_id = {c.id: c for c in components if c.id is not None}

    nodes = [
        GraphNode(id=c.id, **{"class": c.cls}, value=values.get(c.id),
                  confidence=c.confidence)
        for c in components if c.id is not None
    ]

    # Orientation needs ALL of a component's terminals, across every
    # net — a terminal's name depends on where it sits relative to the
    # component's other terminals, not just its own net.
    positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for net in raw_nets:
        for m in net.members:
            positions[m["comp_id"]].append((m["x"], m["y"]))
    orientations = {cid: orientation_from_terminals(pts)
                   for cid, pts in positions.items()}

    edges: list[GraphEdge] = []
    for net in raw_nets:
        members = [m for m in net.members if m["comp_id"] in comp_by_id]
        named = []
        for m in members:
            c = comp_by_id[m["comp_id"]]
            tname = name_terminal(c.cls, m["x"], m["y"], c.bbox,
                                 orientations.get(m["comp_id"]))
            named.append((m["comp_id"], tname))
        for a in range(len(named)):
            for b in range(a + 1, len(named)):
                (id_a, t_a), (id_b, t_b) = named[a], named[b]
                if id_a == id_b:
                    continue
                edges.append(GraphEdge(
                    source=id_a, target=id_b, net_id=net.net_id,
                    source_terminal=t_a, target_terminal=t_b))

    return CircuitGraph(circuit_id=circuit_id, domain=domain, nodes=nodes,
                       edges=edges, source_image=source_image)


def graph_edit_similarity(predicted: CircuitGraph,
                          reference: CircuitGraph) -> dict:
    """Node/edge/value agreement between a predicted and a reference
    graph.

    Not a true graph edit distance: node identity is taken from the
    ids, which the evaluation step has already aligned by IoU (see
    Scripts/Evaluate_Hybrid.build_id_map). This measures the three
    things a downstream question-answering stage actually depends on —
    are the components right, are the connections right, are the
    values right.
    """
    pred_nodes = {n.id: n for n in predicted.nodes}
    ref_nodes = {n.id: n for n in reference.nodes}

    node_correct = sum(1 for i, n in ref_nodes.items()
                       if i in pred_nodes and pred_nodes[i].cls == n.cls)

    pred_edges = {tuple(sorted((e.source, e.target))) for e in predicted.edges}
    ref_edges = {tuple(sorted((e.source, e.target))) for e in reference.edges}

    value_total = sum(1 for n in ref_nodes.values() if n.value)
    value_correct = sum(
        1 for i, n in ref_nodes.items()
        if n.value and i in pred_nodes
        and (pred_nodes[i].value or "").strip() == n.value.strip()
    )

    return {
        "node_accuracy": node_correct / len(ref_nodes) if ref_nodes else 0.0,
        "edge_precision": (len(pred_edges & ref_edges) / len(pred_edges)
                           if pred_edges else 0.0),
        "edge_recall": (len(pred_edges & ref_edges) / len(ref_edges)
                        if ref_edges else 0.0),
        "value_accuracy": (value_correct / value_total
                           if value_total else 0.0),
        "n_nodes_ref": len(ref_nodes),
        "n_edges_ref": len(ref_edges),
    }
