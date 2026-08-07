# -*- coding: utf-8 -*-
"""
circuit_catalog.py  — v3 FINALE
Catalogue fermé de templates de circuits électroniques pédagogiques.
Niveaux : lycée / prépa / ingénieur.

Règles :
  - Topologie FIXE par template (jamais générée librement).
  - Valeurs tirées aléatoirement dans des plages E12 réalistes.
  - Convention de polarité : (A, B) = (pôle+/anode, pôle-/cathode).
  - Tout composant orienté respecte le sens de conduction physique réel.
"""
from __future__ import annotations
import random

# ── Séries normalisées E12 ───────────────────────────────────────────
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]

UNITS = {
    "resistor":  ("Ω",  [("",  1),      ("k", 1e3),  ("M", 1e6)]),
    "capacitor": ("F",  [("p", 1e-12),  ("n", 1e-9), ("µ", 1e-6)]),
    "inductor":  ("H",  [("µ", 1e-6),   ("m", 1e-3), ("",  1)]),
}


def pick_e12(cls: str, lo: float, hi: float) -> str:
    unit_sym, prefixes = UNITS[cls]
    for _ in range(300):
        sym, mult = random.choice(prefixes)
        base = random.choice(E12)
        val = base * mult
        if lo <= val <= hi:
            disp = int(base) if base == int(base) else base
            return f"{disp}{sym}{unit_sym}"
    mid = (lo + hi) / 2
    return _fmt_raw(mid, cls)


def _fmt_raw(val: float, cls: str) -> str:
    u, _ = UNITS[cls]
    if val >= 1e6:  return f"{val/1e6:.1f}M{u}"
    if val >= 1e3:  return f"{val/1e3:.1f}k{u}"
    if val >= 1:    return f"{val:.1f}{u}"
    if val >= 1e-3: return f"{val*1e3:.1f}m{u}"
    if val >= 1e-6: return f"{val*1e6:.1f}µ{u}"
    return f"{val*1e9:.1f}n{u}"


def pick_voltage() -> str:
    return f"{random.choice([1.5, 3, 5, 6, 9, 12, 15, 24])}V"


def pick_ac_voltage() -> str:
    return f"{random.choice([12, 24, 48, 110, 220])}V AC"


def pick_source_class() -> str:
    return random.choice(["vsource", "battery"])


def pick_diode_ref() -> str:
    return random.choice(["1N4148", "1N4001", "1N4007", "1N5819"])


def pick_zener_voltage() -> str:
    return f"{random.choice([3.3, 5.1, 6.2, 9.1, 12])}V"


def pick_opamp_ref() -> str:
    return random.choice(["LM741", "TL081", "LM358", "NE5532"])


def pick_transistor_ref(kind: str) -> str:
    if kind == "npn_transistor":
        return random.choice(["2N2222", "BC547", "2N3904"])
    return random.choice(["2N2907", "BC557", "2N3906"])


def pick_gate_ref(cls: str) -> str:
    """Référence de circuit intégré réel pour une porte logique."""
    refs = {
        "gate_and":  ["74HC08", "74LS08", "CD4081"],
        "gate_nand": ["74HC00", "74LS00", "CD4011"],
        "gate_or":   ["74HC32", "74LS32", "CD4071"],
        "gate_nor":  ["74HC02", "74LS02", "CD4001"],
        "gate_xor":  ["74HC86", "74LS86", "CD4070"],
        "gate_xnor": ["74HC266", "CD4077"],
        "gate_not":  ["74HC04", "74LS04", "CD4069"],
    }
    return random.choice(refs.get(cls, ["74HC00"]))


def T(id_, cls, value_fn=None):
    return {"id": id_, "class": cls, "_value_fn": value_fn}


# ══════════════════════════════════════════════════════════════════════
#  CATALOGUE
# ══════════════════════════════════════════════════════════════════════
CATALOG: dict = {

    # ── Lycée ─────────────────────────────────────────────────────────
    "R_seul": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",   lambda: pick_e12("resistor", 10, 10_000)),
        ],
        "connections": [("V1","R1"), ("R1","V1")],
        "description": "Circuit résistif simple — loi d'Ohm U=RI.",
    },

    "RC_series_charge": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",   lambda: pick_e12("resistor",  100, 100_000)),
            T("C1", "capacitor",  lambda: pick_e12("capacitor", 1e-9, 1e-6)),
        ],
        "connections": [("V1","R1"),("R1","C1"),("C1","V1")],
        "description": "RC série charge/décharge, τ=RC.",
    },

    "RL_series": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",   lambda: pick_e12("resistor", 10, 1_000)),
            T("L1", "inductor",   lambda: pick_e12("inductor", 1e-3, 1)),
        ],
        "connections": [("V1","R1"),("R1","L1"),("L1","V1")],
        "description": "RL série, établissement du courant, τ=L/R.",
    },

    "RLC_series_transient": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",   lambda: pick_e12("resistor",  10, 1_000)),
            T("L1", "inductor",   lambda: pick_e12("inductor",  1e-3, 1)),
            T("C1", "capacitor",  lambda: pick_e12("capacitor", 1e-9, 1e-4)),
        ],
        "connections": [("V1","R1"),("R1","L1"),("L1","C1"),("C1","V1")],
        "description": "RLC série régime transitoire, oscillations amorties.",
    },

    "R_parallel_divider": {
        "level": "lycee", "topology": "parallel",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("R2", "resistor", lambda: pick_e12("resistor", 100, 10_000)),
        ],
        "connections": [("V1","R1"),("V1","R2"),("R1","V1"),("R2","V1")],
        "description": "Deux résistances en parallèle, diviseur de courant.",
    },

    "RC_parallel": {
        "level": "lycee", "topology": "parallel",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",  lambda: pick_e12("resistor",  100, 10_000)),
            T("C1", "capacitor", lambda: pick_e12("capacitor", 1e-9, 1e-6)),
        ],
        "connections": [("V1","R1"),("V1","C1"),("R1","V1"),("C1","V1")],
        "description": "Résistance et condensateur en parallèle.",
    },

    "RL_parallel": {
        "level": "lycee", "topology": "parallel",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor", 10, 1_000)),
            T("L1", "inductor", lambda: pick_e12("inductor", 1e-3, 1)),
        ],
        "connections": [("V1","R1"),("V1","L1"),("R1","V1"),("L1","V1")],
        "description": "Résistance et inductance en parallèle.",
    },

    "voltage_divider": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("R2", "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
        ],
        "connections": [("V1","R1"),("R1","R2"),("R2","V1")],
        "description": "Diviseur de tension, Vout=V·R2/(R1+R2).",
    },

    "voltage_divider_loaded": {
        "level": "lycee", "topology": "mixed",
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("R2",  "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("RL1", "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("R1","R2"),("R1","RL1"),
                        ("R2","GND1"),("RL1","GND1"),("GND1","V1")],
        "description": "Diviseur de tension chargé par RL en parallèle sur R2.",
    },

    # ── Prépa / L1-L2 ─────────────────────────────────────────────────
    "RC_lowpass": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",  lambda: pick_e12("resistor",  1_000, 100_000)),
            T("C1", "capacitor", lambda: pick_e12("capacitor", 1e-9, 1e-6)),
        ],
        "connections": [("V1","R1"),("R1","C1"),("C1","V1")],
        "description": "Filtre passe-bas RC 1er ordre, fc=1/(2πRC).",
    },

    "RC_highpass": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("C1", "capacitor", lambda: pick_e12("capacitor", 1e-9, 1e-6)),
            T("R1", "resistor",  lambda: pick_e12("resistor",  1_000, 100_000)),
        ],
        "connections": [("V1","C1"),("C1","R1"),("R1","V1")],
        "description": "Filtre passe-haut RC 1er ordre, fc=1/(2πRC).",
    },

    "RL_lowpass": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor", 10, 1_000)),
            T("L1", "inductor", lambda: pick_e12("inductor", 1e-3, 1)),
        ],
        "connections": [("V1","R1"),("R1","L1"),("L1","V1")],
        "description": "Filtre passe-bas RL 1er ordre, fc=R/(2πL).",
    },

    "RL_highpass": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("L1", "inductor", lambda: pick_e12("inductor", 1e-3, 1)),
            T("R1", "resistor", lambda: pick_e12("resistor", 10, 1_000)),
        ],
        "connections": [("V1","L1"),("L1","R1"),("R1","V1")],
        "description": "Filtre passe-haut RL 1er ordre, fc=R/(2πL).",
    },

    "RLC_bandpass": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor",  lambda: pick_e12("resistor",  10, 1_000)),
            T("L1", "inductor",  lambda: pick_e12("inductor",  1e-3, 1)),
            T("C1", "capacitor", lambda: pick_e12("capacitor", 1e-9, 1e-6)),
        ],
        "connections": [("V1","R1"),("R1","L1"),("L1","C1"),("C1","V1")],
        "description": "Filtre passe-bande RLC série, f0=1/(2π√LC).",
    },

    "RLC_notch": {
        "level": "prepa", "topology": "mixed",
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("L1",  "inductor",   lambda: pick_e12("inductor",  1e-3, 1)),
            T("C1",  "capacitor",  lambda: pick_e12("capacitor", 1e-9, 1e-6)),
            T("R1",  "resistor",   lambda: pick_e12("resistor",  10, 1_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","L1"),("L1","C1"),("V1","R1"),
                        ("C1","GND1"),("R1","GND1"),("GND1","V1")],
        "description": "Filtre coupe-bande RLC, atténue f0=1/(2π√LC).",
    },

    "wheatstone_bridge": {
        "level": "prepa", "topology": "bridge",
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("R2",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("R3",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("R4",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("V1","R3"),("R1","R2"),("R3","R4"),
                        ("R2","GND1"),("R4","GND1"),("GND1","V1")],
        "description": "Pont de Wheatstone 4 résistances.",
    },

    "RLC_resonant_series": {
        "level": "prepa", "topology": "series",
        "components": [
            T("V1", "vsource",  pick_ac_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor",  10, 100)),
            T("L1", "inductor", lambda: pick_e12("inductor",  1e-3, 100e-3)),
            T("C1", "capacitor",lambda: pick_e12("capacitor", 1e-9, 1e-6)),
        ],
        "connections": [("V1","R1"),("R1","L1"),("L1","C1"),("C1","V1")],
        "description": "RLC série en résonance, Q=(1/R)√(L/C).",
    },


    # ── Nouveaux templates : classes sous-représentées ────────────────
    "led_with_resistor": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("R1", "resistor", lambda: pick_e12("resistor", 100, 2_200)),
            T("D1", "led"),
        ],
        "connections": [("V1","R1"),("R1","D1"),("D1","V1")],
        "description": "LED avec résistance de limitation de courant.",
    },

    "switch_led_circuit": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("SW1", "switch"),
            T("R1",  "resistor", lambda: pick_e12("resistor", 100, 2_200)),
            T("D1",  "led"),
        ],
        "connections": [("V1","SW1"),("SW1","R1"),("R1","D1"),("D1","V1")],
        "description": "Commande d'une LED par interrupteur.",
    },

    "fuse_protected_load": {
        "level": "lycee", "topology": "series",
        "components": [
            T("V1", "vsource_dc", pick_voltage),
            T("F1", "fuse"),
            T("R1", "resistor", lambda: pick_e12("resistor", 10, 1_000)),
        ],
        "connections": [("V1","F1"),("F1","R1"),("R1","V1")],
        "description": "Charge protégée par fusible.",
    },

    "switch_two_loads": {
        "level": "lycee", "topology": "parallel",
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("R2",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
        ],
        "connections": [("V1","R1"),("V1","R2"),("R1","V1"),("R2","V1")],
        "description": "Deux charges en parallèle sur une même source.",
    },

    "smoothing_filter": {
        "level": "prepa", "topology": "mixed",
        "mixed_spec": {"series": ["R1"], "parallel": ["C1", "RL1"]},
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 10, 1_000)),
            T("C1",  "polarized_capacitor",
              lambda: pick_e12("capacitor", 1e-6, 1e-3)),
            T("RL1", "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("R1","C1"),("R1","RL1"),
                        ("C1","GND1"),("RL1","GND1"),("GND1","V1")],
        "description": "Filtrage par condensateur electrochimique "
                        "(polarise) en parallele sur la charge.",
    },

    "zener_regulator": {
        "level": "prepa", "topology": "mixed",
        "mixed_spec": {"series": ["R1"], "parallel": ["DZ1", "RL1"]},
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 100, 10_000)),
            T("DZ1", "zener_diode", pick_zener_voltage),
            T("RL1", "resistor", lambda: pick_e12("resistor", 470, 47_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("R1","DZ1"),("R1","RL1"),
                        ("DZ1","GND1"),("RL1","GND1"),("GND1","V1")],
        "description": "Régulateur de tension à diode Zener avec charge.",
    },

    "led_current_limiter": {
        "level": "prepa", "topology": "mixed",
        "mixed_spec": {"series": ["R1"], "parallel": ["D1", "R2"]},
        "components": [
            T("V1",  "vsource_dc", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 100, 2_200)),
            T("D1",  "led"),
            T("R2",  "resistor", lambda: pick_e12("resistor", 1_000, 47_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("R1","D1"),("R1","R2"),
                        ("D1","GND1"),("R2","GND1"),("GND1","V1")],
        "description": "LED et résistance de rappel alimentées par R1.",
    },

    "npn_common_emitter": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "npn_common_emitter",
        "components": [
            T("V1",  "vsource", pick_voltage),
            T("RC1", "resistor", lambda: pick_e12("resistor", 470, 10_000)),
            T("RB1", "resistor", lambda: pick_e12("resistor", 4_700, 470_000)),
            T("Q1",  "npn_transistor", lambda: pick_transistor_ref("npn_transistor")),
            T("GND1","ground"),
        ],
        "connections": [("V1","RC1"),("RC1","Q1"),("RB1","Q1"),
                        ("Q1","GND1"),("GND1","V1")],
        "description": "Transistor NPN en émetteur commun (amplificateur).",
    },

    "pnp_switch": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "pnp_switch",
        "components": [
            T("V1",  "vsource", pick_voltage),
            T("RB1", "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("RL1", "resistor", lambda: pick_e12("resistor", 100, 4_700)),
            T("Q1",  "pnp_transistor", lambda: pick_transistor_ref("pnp_transistor")),
            T("GND1","ground"),
        ],
        "connections": [("V1","Q1"),("RB1","Q1"),("Q1","RL1"),
                        ("RL1","GND1"),("GND1","V1")],
        "description": "Transistor PNP en commutation de charge.",
    },

    "full_wave_bridge": {
        "level": "ingenieur", "topology": "bridge",
        "custom_render": "full_wave_bridge",
        "components": [
            T("V1",  "vsource", pick_ac_voltage),
            T("D1",  "diode", pick_diode_ref),
            T("D2",  "diode", pick_diode_ref),
            T("D3",  "diode", pick_diode_ref),
            T("D4",  "diode", pick_diode_ref),
            T("RL1", "resistor", lambda: pick_e12("resistor", 100, 10_000)),
        ],
        "connections": [("V1","D1"),("V1","D3"),("D1","RL1"),("D3","RL1"),
                        ("RL1","D2"),("RL1","D4"),("D2","V1"),("D4","V1")],
        "description": "Redresseur double alternance en pont de Graetz.",
    },

    "voltage_follower": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "voltage_follower",
        "components": [
            T("V1",  "vsource", pick_voltage),
            T("U1",  "opamp", pick_opamp_ref),
            T("RL1", "resistor", lambda: pick_e12("resistor", 1_000, 100_000)),
            T("GND1","ground"),
        ],
        "connections": [("V1","U1"),("U1","RL1"),("RL1","GND1"),("GND1","V1")],
        "description": "Suiveur de tension (buffer) à AOP, gain unitaire.",
    },

    "summing_amplifier": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "summing_amplifier",
        "components": [
            T("V1",  "vsource", pick_voltage),
            T("V2",  "vsource", pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor", 1_000, 47_000)),
            T("R2",  "resistor", lambda: pick_e12("resistor", 1_000, 47_000)),
            T("Rf",  "resistor", lambda: pick_e12("resistor", 10_000, 220_000)),
            T("U1",  "opamp", pick_opamp_ref),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("V2","R2"),("R1","U1"),("R2","U1"),
                        ("U1","Rf"),("Rf","U1"),("U1","GND1"),("GND1","V1")],
        "description": "Amplificateur sommateur inverseur à deux entrées.",
    },
    # ── Ingénieur (rendu custom) ───────────────────────────────────────
    "half_wave_rectifier": {
        "level": "ingenieur", "topology": "series",
        "custom_render": "half_wave_rectifier",
        "components": [
            T("V1",  "vsource",  pick_ac_voltage),
            T("D1",  "diode",    pick_diode_ref),
            T("RL1", "resistor", lambda: pick_e12("resistor", 100, 10_000)),
        ],
        "connections": [("V1","D1"),("D1","RL1"),("RL1","V1")],
        "description": "Redresseur simple alternance : diode + résistance de charge.",
    },

    "inverting_amplifier": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "inverting_amplifier",
        "components": [
            T("V1",  "vsource",  pick_voltage),
            T("Rin", "resistor", lambda: pick_e12("resistor",   1_000, 10_000)),
            T("Rf",  "resistor", lambda: pick_e12("resistor",  10_000, 1_000_000)),
            T("U1",  "opamp",    pick_opamp_ref),
            T("GND1","ground"),
        ],
        "connections": [("V1","Rin"),("Rin","U1"),("U1","Rf"),
                        ("Rf","U1"),("U1","GND1"),("GND1","V1")],
        "description": "Amplificateur inverseur AOP, gain Av=-Rf/Rin.",
    },

    "multistage_amplifier": {
        "level": "ingenieur", "topology": "mixed",
        "custom_render": "multistage_amplifier",
        "components": [
            T("V1",  "vsource",  pick_voltage),
            T("R1",  "resistor", lambda: pick_e12("resistor",  10_000, 30_000)),
            T("Rf1", "resistor", lambda: pick_e12("resistor",  80_000, 120_000)),
            T("U1",  "opamp",    pick_opamp_ref),
            T("R2",  "resistor", lambda: pick_e12("resistor",  20_000, 40_000)),
            T("Rf2", "resistor", lambda: pick_e12("resistor",  80_000, 120_000)),
            T("U2",  "opamp",    pick_opamp_ref),
            T("GND1","ground"),
        ],
        "connections": [("V1","R1"),("R1","U1"),("U1","Rf1"),("Rf1","U1"),
                        ("U1","U2"),("R2","U2"),("U2","Rf2"),("Rf2","U2"),
                        ("R2","GND1"),("U1","GND1"),("GND1","V1")],
        "description": "Ampli-op inverseur deux étages en cascade.",
    },

    "logic_and2": {
        "domain": "logic", "level": "lycee", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_and", lambda: pick_gate_ref("gate_and"))],
        "connections": [],
        "description": "Porte AND à 2 entrées, table de vérité de base.",
    },
    "logic_or2": {
        "domain": "logic", "level": "lycee", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_or", lambda: pick_gate_ref("gate_or"))],
        "connections": [],
        "description": "Porte OR à 2 entrées, table de vérité de base.",
    },
    "logic_not1": {
        "domain": "logic", "level": "lycee", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A'], "output": "Y"},
        "components": [T("U1", "gate_not", lambda: pick_gate_ref("gate_not"))],
        "connections": [],
        "description": "Porte NOT à 1 entrées, table de vérité de base.",
    },
    "logic_nand2": {
        "domain": "logic", "level": "lycee", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_nand", lambda: pick_gate_ref("gate_nand"))],
        "connections": [],
        "description": "Porte NAND à 2 entrées, table de vérité de base.",
    },
    "logic_nor2": {
        "domain": "logic", "level": "lycee", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_nor", lambda: pick_gate_ref("gate_nor"))],
        "connections": [],
        "description": "Porte NOR à 2 entrées, table de vérité de base.",
    },
    "logic_xor2": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_xor", lambda: pick_gate_ref("gate_xor"))],
        "connections": [],
        "description": "Porte XOR à 2 entrées, table de vérité de base.",
    },
    "logic_xnor2": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B'], "output": "Y"},
        "components": [T("U1", "gate_xnor", lambda: pick_gate_ref("gate_xnor"))],
        "connections": [],
        "description": "Porte XNOR à 2 entrées, table de vérité de base.",
    },
    "logic_and3": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B', 'C'], "output": "Y"},
        "components": [T("U1", "gate_and", lambda: pick_gate_ref("gate_and"))],
        "connections": [],
        "description": "Porte AND à 3 entrées, table de vérité de base.",
    },
    "logic_nand3": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B', 'C'], "output": "Y"},
        "components": [T("U1", "gate_nand", lambda: pick_gate_ref("gate_nand"))],
        "connections": [],
        "description": "Porte NAND à 3 entrées, table de vérité de base.",
    },
    "logic_or3": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "single", "inputs": ['A', 'B', 'C'], "output": "Y"},
        "components": [T("U1", "gate_or", lambda: pick_gate_ref("gate_or"))],
        "connections": [],
        "description": "Porte OR à 3 entrées, table de vérité de base.",
    },
    "logic_and_then_or": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "cascade", "inputs": ["A","B","C"], "output": "Y"},
        "components": [T("U1", "gate_and", lambda: pick_gate_ref("gate_and")), T("U2", "gate_or", lambda: pick_gate_ref("gate_or"))],
        "connections": [("U1","U2")],
        "description": "Combinaison en cascade : AND puis OR.",
    },
    "logic_or_then_and": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "cascade", "inputs": ["A","B","C"], "output": "Y"},
        "components": [T("U1", "gate_or", lambda: pick_gate_ref("gate_or")), T("U2", "gate_and", lambda: pick_gate_ref("gate_and"))],
        "connections": [("U1","U2")],
        "description": "Combinaison en cascade : OR puis AND.",
    },
    "logic_xor_then_and": {
        "domain": "logic", "level": "prepa", "topology": "logic",
        "logic_spec": {"pattern": "cascade", "inputs": ["A","B","C"], "output": "Y"},
        "components": [T("U1", "gate_xor", lambda: pick_gate_ref("gate_xor")), T("U2", "gate_and", lambda: pick_gate_ref("gate_and"))],
        "connections": [("U1","U2")],
        "description": "Combinaison en cascade : XOR puis AND.",
    },
    "logic_nand_then_nor": {
        "domain": "logic", "level": "ingenieur", "topology": "logic",
        "logic_spec": {"pattern": "cascade", "inputs": ["A","B","C"], "output": "Y"},
        "components": [T("U1", "gate_nand", lambda: pick_gate_ref("gate_nand")), T("U2", "gate_nor", lambda: pick_gate_ref("gate_nor"))],
        "connections": [("U1","U2")],
        "description": "Combinaison en cascade : NAND puis NOR.",
    },
    "logic_half_adder": {
        "domain": "logic", "level": "ingenieur", "topology": "logic",
        "logic_spec": {"pattern": "shared", "inputs": ["A","B"],
                        "outputs": ["S","C"]},
        "components": [T("U1", "gate_xor", lambda: pick_gate_ref("gate_xor")),
                        T("U2", "gate_and", lambda: pick_gate_ref("gate_and"))],
        "connections": [],
        "description": "Demi-additionneur : somme S = A XOR B, retenue C = A AND B.",
    },

    "logic_demorgan": {
        "domain": "logic", "level": "ingenieur", "topology": "logic",
        "logic_spec": {"pattern": "inverted_inputs", "inputs": ["A","B"],
                        "output": "Y"},
        "components": [T("U1", "gate_not", lambda: pick_gate_ref("gate_not")),
                        T("U2", "gate_not", lambda: pick_gate_ref("gate_not")),
                        T("U3", "gate_or", lambda: pick_gate_ref("gate_or"))],
        "connections": [("U1","U3"),("U2","U3")],
        "description": "Loi de De Morgan : (NOT A) OR (NOT B) equivaut a NOT(A AND B).",
    },
}


def instantiate(template_name: str, shuffle: bool = True) -> dict:
    """Génère un circuit concret depuis un template (valeurs aléatoires).

    shuffle=True : mélange l'ordre de dessin des composants pour
    diversifier les images (les connexions sont RECONSTRUITES pour
    rester la vérité terrain exacte du dessin) :
      - series   : ordre de la chaîne mélangé, connexions re-chaînées
      - parallel : ordre des branches mélangé (connexions inchangées)
      - mixed/bridge/custom : ordre fixe (layout par ancres)
    """
    tpl = CATALOG[template_name]
    components = []
    for comp in tpl["components"]:
        value = comp["_value_fn"]() if comp["_value_fn"] else None
        cls = comp["class"]
        if cls == "vsource_dc":
            cls = pick_source_class()
        components.append({"id": comp["id"], "class": cls, "value": value})

    topo = tpl["topology"]
    is_custom = "custom_render" in tpl
    domain = tpl.get("domain", "electrical")
    if domain == "logic":
        is_custom = True   # layout logique fixe : jamais de shuffle

    if shuffle and not is_custom and topo in ("series", "parallel"):
        srcs   = [c for c in components if c["class"] in ("vsource", "battery")]
        gnds   = [c for c in components if c["class"] == "ground"]
        others = [c for c in components
                  if c["class"] not in ("vsource", "battery", "ground")]
        random.shuffle(others)
        components = srcs + others + gnds
        if topo == "series":
            # reconstruire la chaîne : V1 -> o1 -> ... -> on -> V1
            chain = [srcs[0]["id"]] + [c["id"] for c in others] if srcs else                     [c["id"] for c in others]
            conns = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
            conns.append((chain[-1], chain[0]))
            connections = [{"from": a, "to": b} for (a, b) in conns]
        else:
            connections = [{"from": a, "to": b} for (a, b) in tpl["connections"]]
    else:
        connections = [{"from": a, "to": b} for (a, b) in tpl["connections"]]

    result = {
        "template": template_name,
        "domain": domain,
        "components": components,
        "connections": connections,
        "circuit_metadata": {"topology": tpl["topology"], "domain": domain},
    }
    if "custom_render" in tpl:
        result["custom_render"] = tpl["custom_render"]
    if "logic_spec" in tpl:
        result["logic_spec"] = tpl["logic_spec"]
    if "mixed_spec" in tpl:
        result["mixed_spec"] = tpl["mixed_spec"]
    return result


def list_templates_by_level() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"lycee": [], "prepa": [], "ingenieur": []}
    for name, tpl in CATALOG.items():
        out[tpl["level"]].append(name)
    return out


def list_templates_by_domain() -> dict[str, list[str]]:
    """Sépare circuits électriques et circuits logiques."""
    out: dict[str, list[str]] = {"electrical": [], "logic": []}
    for name, tpl in CATALOG.items():
        out[tpl.get("domain", "electrical")].append(name)
    return out


if __name__ == "__main__":
    import json
    levels = list_templates_by_level()
    for lvl, names in levels.items():
        print(f"{lvl}: {len(names)} templates -> {names}")
    print(f"\nTotal : {len(CATALOG)} templates")
    print("\n--- RC_lowpass ---")
    print(json.dumps(instantiate("RC_lowpass"), ensure_ascii=False, indent=2))
