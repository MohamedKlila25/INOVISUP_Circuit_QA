# -*- coding: utf-8 -*-
"""
Graph.Tools — deterministic graph queries, exposed as LLM tool-calling
functions.

Design, per the project's decision: structural questions (count,
neighbours, ground connections, topology, floating pins...) are
answered by EXECUTING these functions against the graph — exact,
free, instant — never by asking the LLM to read a netlist and guess.
The LLM's only job is (1) picking which tool answers the question and
its arguments, and (2) turning the tool's raw result into a natural
sentence. See Llm.Tool_Agent for the two-turn orchestration.

Each tool is bound to ONE specific CircuitGraph via `build_tools`,
returned as a list of plain Python functions with type hints and
docstrings — the format `tokenizer.apply_chat_template(tools=...)`
expects to auto-generate JSON schemas from.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

from Common.Schemas import CircuitGraph

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


def _nets_from_graph(graph: CircuitGraph) -> dict[int, set[str]]:
    """Reconstructs net_id -> {component ids} from the graph's edges.
    Needed because CircuitGraph.edges is the PAIRWISE EXPANSION of each
    net (a net of 3 components becomes 3 edges) — topology
    classification needs the original net groupings back, not the
    expanded clique, otherwise a single 3-way shared node looks
    identical to three independent series links.
    """
    nets: dict[int, set[str]] = defaultdict(set)
    for e in graph.edges:
        nid = e.net_id if e.net_id is not None else -1
        nets[nid].add(e.source)
        nets[nid].add(e.target)
    return dict(nets)


def classify_topology(graph: CircuitGraph) -> str:
    """Classifies the circuit as 'série', 'parallèle', or 'mixte',
    from the net structure (not the expanded edge graph — see
    _nets_from_graph)."""
    nets = _nets_from_graph(graph)
    sizes = [len(members) for members in nets.values()]
    if not sizes:
        return "indéterminé (aucune connexion trouvée)"

    max_net_size = max(sizes)
    n_branch_nets = sum(1 for s in sizes if s >= 3)

    if max_net_size <= 2:
        return "série"
    if n_branch_nets >= 1 and all(s >= 3 or s == 2 for s in sizes):
        # a mix of small (series-like, 2-member) and large (branch,
        # 3+ member) nets is the signature of a mixed topology; if
        # EVERY net is a branch net, treat it as parallel
        if all(s >= 3 for s in sizes):
            return "parallèle"
        return "mixte"
    return "mixte"


def _to_networkx(graph: CircuitGraph) -> nx.Graph:
    G = nx.Graph()
    for n in graph.nodes:
        G.add_node(n.id)
    for e in graph.edges:
        G.add_edge(e.source, e.target)
    return G


def _find_source_id(graph: CircuitGraph) -> str | None:
    for n in graph.nodes:
        if n.cls in ("vsource", "battery"):
            return n.id
    return None


def build_tools(graph: CircuitGraph) -> list:
    """Returns the list of tool functions bound to this specific
    graph, ready to pass to tokenizer.apply_chat_template(tools=...)."""

    def count_total_components() -> int:
        """Retourne le nombre total de composants dans le circuit."""
        return len(graph.nodes)

    def count_components_by_type() -> dict:
        """Retourne, pour chaque type de composant, le nombre d'exemplaires ET la liste de leurs identifiants+valeurs (ex: résistances -> 3, avec R1=1kΩ, R2=4.7kΩ)."""
        by_type: dict[str, list[str]] = defaultdict(list)
        for n in graph.nodes:
            label = f"{n.id}={n.value}" if n.value else n.id
            by_type[_CLASS_LABEL_FR.get(n.cls, n.cls)].append(label)
        return {cls: {"quantité": len(items), "détail": items}
                for cls, items in by_type.items()}

    def get_component_class(component_id: str) -> str:
        """
        Retourne la classe/le type d'un composant donné son identifiant.

        Args:
            component_id: L'identifiant du composant (ex: 'R1').
        """
        for n in graph.nodes:
            if n.id == component_id:
                return _CLASS_LABEL_FR.get(n.cls, n.cls)
        return f"composant '{component_id}' introuvable"

    def get_component_value(component_id: str) -> str:
        """
        Retourne la valeur d'un composant donné son identifiant.

        Args:
            component_id: L'identifiant du composant (ex: 'R1'), dont on veut la valeur (ex: '4.7kΩ').
        """
        for n in graph.nodes:
            if n.id == component_id:
                return n.value or "aucune valeur connue"
        return f"composant '{component_id}' introuvable"

    def get_neighbours(component_id: str) -> list:
        """
        Retourne la liste des composants directement connectés (voisins électriques) à un composant donné.

        Args:
            component_id: L'identifiant du composant dont on veut les voisins (ex: 'R1').
        """
        return sorted(graph.neighbours(component_id))

    def is_connected_to_ground(component_id: str) -> bool:
        """
        Indique si un composant est directement relié à la masse (ground).

        Args:
            component_id: L'identifiant du composant à vérifier (ex: 'R1').
        """
        ground_ids = {n.id for n in graph.nodes if n.cls == "ground"}
        return bool(graph.neighbours(component_id) & ground_ids)

    def list_components_connected_to_ground() -> list:
        """Retourne la liste de tous les composants directement reliés à la masse."""
        ground_ids = {n.id for n in graph.nodes if n.cls == "ground"}
        connected = set()
        for gid in ground_ids:
            connected |= graph.neighbours(gid)
        return sorted(connected - ground_ids)

    def get_circuit_topology() -> str:
        """Détermine si le circuit est monté en série, en parallèle (dérivation), ou en assemblage mixte."""
        return classify_topology(graph)

    def find_floating_components() -> list:
        """Retourne la liste des composants qui n'ont AUCUNE connexion électrique (isolés/flottants)."""
        connected_ids = {n.id for n in graph.nodes for _ in graph.neighbours(n.id)}
        all_ids = {n.id for n in graph.nodes}
        return sorted(all_ids - connected_ids)

    def _source_side_groups() -> list:
        """Components wired to the source, grouped BY NET — i.e. one
        group per source terminal.

        Returning groups rather than a flat list matters: a source
        terminal is often shared by several components (in a regulator,
        the low side carries the load, the zener AND ground at once).
        A flat neighbour list loses which terminal each one sits on, so
        picking "the first component" from it would be an arbitrary
        alphabetical choice rather than a real answer.
        """
        src = _find_source_id(graph)
        if src is None:
            return []
        ground_ids = {n.id for n in graph.nodes if n.cls == "ground"}
        groups: dict[int, set[str]] = {}
        for e in graph.edges:
            if src not in (e.source, e.target):
                continue
            other = e.target if e.source == src else e.source
            groups.setdefault(e.net_id if e.net_id is not None else -1,
                             set()).add(other)
        out = []
        for _, members in sorted(groups.items()):
            visible = sorted(m for m in members if m not in ground_ids)
            out.append(visible or sorted(members))
        return out

    def get_first_component() -> str:
        """
        Retourne le premier composant du circuit, c'est-à-dire celui directement relié à la première borne de la source d'alimentation.
        """
        if _find_source_id(graph) is None:
            return "aucune source de tension trouvée dans ce circuit"
        groups = _source_side_groups()
        if not groups:
            return "aucun composant connecté à la source"
        return ", ".join(groups[0])

    def get_last_component() -> str:
        """
        Retourne le dernier composant du circuit, c'est-à-dire celui directement relié à la borne de retour de la source d'alimentation.
        """
        if _find_source_id(graph) is None:
            return "aucune source de tension trouvée dans ce circuit"
        groups = _source_side_groups()
        if not groups:
            return "aucun composant connecté à la source"
        return ", ".join(groups[-1])

    def get_components_directly_on_source() -> list:
        """
        Retourne les composants directement reliés aux deux bornes de la source d'alimentation (le premier et le dernier composant du circuit).
        """
        src = _find_source_id(graph)
        if src is None:
            return ["aucune source de tension trouvée dans ce circuit"]
        return sorted(graph.neighbours(src))

    def get_signal_path(target_component_id: str = "") -> list:
        """
        Retourne le chemin (liste ordonnée de composants) entre la source de tension et un composant cible.

        Args:
            target_component_id: L'identifiant du composant cible (ex: 'R1'). Si vide, utilise automatiquement le composant le plus éloigné de la source (typiquement la sortie).
        """
        src = _find_source_id(graph)
        if src is None:
            return ["aucune source de tension trouvée dans ce circuit"]
        G = _to_networkx(graph)
        target = target_component_id or get_last_component()
        if target not in G:
            return [f"composant '{target}' introuvable"]
        try:
            return nx.shortest_path(G, src, target)
        except nx.NetworkXNoPath:
            return [f"aucun chemin trouvé entre {src} et {target}"]

    return [
        count_total_components, count_components_by_type,
        get_component_class, get_component_value, get_neighbours,
        is_connected_to_ground, list_components_connected_to_ground,
        get_circuit_topology, find_floating_components,
        get_first_component, get_last_component, get_signal_path,
        get_components_directly_on_source,
    ]