# -*- coding: utf-8 -*-
"""
Llm.Tool_Agent — two-turn tool-calling orchestration.

Turn 1: the LLM sees the question + the list of available tools (NOT
the raw netlist), and picks which tool to call with which arguments.
Turn 2: we EXECUTE that tool ourselves (exact, deterministic, free —
never asking the model to compute or guess), then feed the raw result
back to the LLM to phrase a natural-sounding sentence.

This is the architecture change from the earlier prompted-netlist
approach: there, the LLM read the whole netlist and answered directly
(measured 99% on structural questions, but with zero guarantee of
correctness — a small model CAN misread a 20-line netlist). Here,
structural questions are answered by Graph.Tools functions — 100%
correct by construction — and the LLM's only job is choosing the
right tool and writing a nice sentence around its result.

IMPORTANT — not executed end to end in the environment used to write
this file (torch unavailable in this sandbox, same constraint as
throughout the LLM/OCR work). The tools= parameter of
apply_chat_template and the <tool_call> parsing follow Qwen2.5's
current, documented function-calling format, verified via a fresh
fetch — validate on a handful of real questions before trusting it.
"""
from __future__ import annotations

import json
import re

SYSTEM_PROMPT_TOOLS = (
    "Tu es un assistant qui répond à des questions sur un circuit "
    "électrique. Tu DOIS utiliser un outil pour obtenir chaque "
    "information — ne réponds JAMAIS de mémoire ou par supposition."
)

SYSTEM_PROMPT_REPHRASE = (
    "Tu reformules le résultat d'un calcul en une phrase française "
    "naturelle et concise. Tu ne dois utiliser QUE les informations "
    "présentes dans le résultat fourni — n'invente jamais de composant, "
    "de valeur, d'unité ou de grandeur physique qui n'y figure pas "
    "(par exemple : ne parle jamais de 'vitesse' pour une tension). "
    "Recopie les valeurs exactement telles qu'elles apparaissent. "
    "Si le résultat est vide ou nul, dis simplement qu'il n'y en a aucun."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Keyword fallback, used ONLY when the model fails to emit a tool call.
# Measured need: a 1.5B model sometimes starts reasoning in prose and
# never produces the <tool_call> block, leaving the question
# unanswered. Matching a few unambiguous keywords is far better than
# returning the model's truncated musing — and it stays deterministic.
_KEYWORD_TOOLS: list[tuple[tuple[str, ...], str]] = [
    # ORDER MATTERS: matched top-down, first hit wins. Specific,
    # multi-word phrases must come BEFORE broad single words —
    # measured: "quels composants sont en parallèle ?" was being
    # captured by the topology rule (which also lists "parallèle"),
    # and "combien de nœuds électriques" by the generic "combien" rule.
    (("broche", "patte", "pin"), "get_terminal_connections"),
    (("anode", "cathode", "orientation", "polarité", "quel sens", "sens est"), "get_diode_orientation"),
    (("nomenclature", "liste d'achat", "bom", "matériel"), "get_bill_of_materials"),
    (("en parallèle", "montés en parallèle"), "find_parallel_components"),
    (("nœud", "noeud"), "count_electrical_nodes"),
    (("flottant", "non connecté", "en l'air", "isolé"), "find_unconnected_terminals"),
    (("série", "parallèle", "mixte", "topologie", "dérivation"), "get_circuit_topology"),
    (("type", "types", "quantité", "catégorie"), "count_components_by_type"),
    (("valeur",), "get_component_value"),
    (("voisin", "connecté à", "relié à"), "get_neighbours"),
    (("masse", "ground"), "list_components_connected_to_ground"),
    (("premier",), "get_first_component"),
    (("dernier",), "get_last_component"),
    (("chemin", "parcours", "trajet"), "get_signal_path"),
    (("combien", "nombre", "total"), "count_total_components"),
]

_COMPONENT_ID_RE = re.compile(r"\b([A-Z]{1,3}\d{1,3})\b")


def guess_tool_from_question(question: str, tools_by_name: dict) -> dict | None:
    """Fallback tool selection by keyword, when the model didn't emit a
    tool call. Returns a call dict in the same shape as parse_tool_call,
    or None if no keyword matches confidently."""
    q = question.lower()
    for keywords, tool_name in _KEYWORD_TOOLS:
        if any(k in q for k in keywords) and tool_name in tools_by_name:
            args = {}
            # tools that need a component id: extract it from the question
            if tool_name in ("get_component_value", "get_component_class",
                            "get_neighbours", "is_connected_to_ground",
                            "get_terminal_connections", "get_diode_orientation"):
                m = _COMPONENT_ID_RE.search(question)
                if not m:
                    continue   # can't call it without an id — try next rule
                args = {"component_id": m.group(1)}
            return {"name": tool_name, "arguments": args}
    return None


def parse_tool_call(raw_text: str) -> dict | None:
    """Extracts the first <tool_call>{"name":..., "arguments":...}
    </tool_call> block from the model's raw generation. Returns None
    if the model didn't call a tool (e.g. answered directly, or the
    output doesn't match the expected format)."""
    m = _TOOL_CALL_RE.search(raw_text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def execute_tool_call(call: dict, tools_by_name: dict) -> str:
    """Executes a parsed tool call against the bound tool functions.
    Returns a string result (or an error message) — never raises, so
    a bad tool call from the model degrades to a clear message rather
    than crashing the whole answer flow."""
    name = call.get("name")
    args = call.get("arguments", {}) or {}
    fn = tools_by_name.get(name)
    if fn is None:
        return f"Erreur : outil '{name}' inconnu."
    try:
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
    except Exception as e:
        return f"Erreur lors de l'exécution de l'outil : {e}"


class ToolAgent:
    """Ties a QwenClient-like model to a set of graph tools for one
    circuit. Construct once per graph (tools are bound to it), then
    call `.answer(question)` repeatedly."""

    def __init__(self, client, tools: list):
        self.client = client
        self.tools = tools
        self.tools_by_name = {t.__name__: t for t in tools}

    def answer(self, question: str) -> dict:
        """Runs the full two-turn flow. Returns a dict with the raw
        tool call, the tool's result, and the final natural-language
        answer — kept separate so a caller can grade the STRUCTURAL
        correctness (did it call the right tool?) independently from
        the PHRASING quality."""
        self.client._ensure_model()  # noqa: SLF001 — same lazy-load pattern as elsewhere

        # ── Turn 1 : pick a tool ─────────────────────────────────────
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
            {"role": "user", "content": question},
        ]
        text = self.client._tokenizer.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=True, tokenize=False)
        # Larger budget for turn 1: a complete <tool_call> block can run
        # 30-50 tokens on its own, and a small model often writes a
        # sentence of reasoning first. Measured failure at 32 tokens:
        # the model got cut off mid-sentence before ever emitting the
        # tool call, so parsing found nothing and the tool never ran.
        raw_turn1 = self._generate_raw(text, max_new_tokens=192)

        call = parse_tool_call(raw_turn1)
        if call is None:
            # The model didn't emit a tool call — fall back to keyword
            # matching rather than returning its (often truncated) prose.
            call = guess_tool_from_question(question, self.tools_by_name)
            if call is None:
                return {"tool_call": None, "tool_result": None,
                       "answer": raw_turn1.strip()}

        tool_result = execute_tool_call(call, self.tools_by_name)

        # Empty results are answered directly, WITHOUT asking the LLM to
        # rephrase. Measured failure mode: given "[]", a 1.5B model
        # invented plausible-sounding content ("les composants reliés à
        # la masse sont les électrodes de l'appareil") instead of saying
        # "none". There is nothing to rephrase when the result is empty,
        # so a deterministic answer is strictly better here.
        if tool_result in ("[]", "{}", "", "null"):
            return {"tool_call": call, "tool_result": tool_result,
                   "answer": "Aucun élément ne correspond à cette requête "
                             "dans ce circuit."}

        # ── Turn 2 : rephrase the tool result naturally ──────────────
        messages2 = [
            {"role": "system", "content": SYSTEM_PROMPT_REPHRASE},
            {"role": "user", "content": f"Question : {question}"},
            {"role": "user", "content": f"Résultat du calcul : {tool_result}"},
            {"role": "user", "content": "Formule la réponse en une phrase, en français."},
        ]
        text2 = self.client._tokenizer.apply_chat_template(
            messages2, add_generation_prompt=True, tokenize=False)
        # Scale the budget with the result size: a rich result (e.g. all
        # component types with their values) needs far more than the 32
        # tokens that suffice for "5". Measured failure: a multi-type
        # breakdown got cut off mid-sentence at 32 tokens.
        rephrase_budget = max(64, min(256, len(tool_result) // 2 + 48))
        final_answer = self._generate_raw(text2, max_new_tokens=rephrase_budget)

        return {"tool_call": call, "tool_result": tool_result,
               "answer": final_answer.strip()}

    def _generate_raw(self, text: str, max_new_tokens: int | None = None) -> str:
        import torch
        model_inputs = self.client._tokenizer(
            [text], return_tensors="pt").to(self.client._device)
        with torch.no_grad():
            generated_ids = self.client._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens or self.client._max_new_tokens)
        generated_ids = [out[len(inp):] for inp, out in
                         zip(model_inputs.input_ids, generated_ids)]
        return self.client._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
