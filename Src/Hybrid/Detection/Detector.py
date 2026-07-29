# -*- coding: utf-8 -*-
"""
Hybrid.Detection.Detector — inférence YOLO sur un schéma de circuit.

Sépare volontairement l'ENTRAÎNEMENT (voir Train.py, exécuté une fois,
en tâche longue) de l'INFÉRENCE (cette classe, appelée à chaque image
du pipeline hybride — traçage de fils, OCR, construction du graphe).
"""
from __future__ import annotations

from pathlib import Path

from Common.Config import HybridConfig
from Common.Schemas import BBox, DetectedComponent

# Ordre EXACT des classes déclarées dans circuit.yaml par Data_Generation
# (Renderer_Annotated.CLASS_NAMES) — doit rester synchronisé avec lui.
CLASS_NAMES = [
    "resistor", "capacitor", "polarized_capacitor", "inductor",
    "diode", "zener_diode", "led",
    "npn_transistor", "pnp_transistor",
    "vsource", "battery", "ground",
    "switch", "fuse", "opamp",
    "gate_and", "gate_or", "gate_xor", "gate_not",
    "gate_nand", "gate_nor", "gate_xnor",
]


class Detector:
    """Détecteur de composants — charge un modèle YOLO entraîné et
    produit des `DetectedComponent` (contrat partagé avec les étages
    suivants du pipeline hybride)."""

    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.50):
        from ultralytics import YOLO   # import différé : pas besoin de
                                        # torch/ultralytics pour le reste
                                        # du paquet (datagen, schemas...)
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou

    @classmethod
    def from_config(cls, cfg: HybridConfig, weights: str | Path) -> "Detector":
        return cls(weights=weights, conf=cfg.yolo.conf, iou=cfg.yolo.iou)

    def detect(self, image_path: str | Path) -> list[DetectedComponent]:
        """Détecte les composants d'une image, triés par confiance
        décroissante. Les `id` (R1, C1...) ne sont PAS assignés ici —
        c'est le rôle de l'étage suivant, une fois les positions
        connues (numérotation par position, gauche à droite)."""
        results = self.model.predict(
            str(image_path), conf=self.conf, iou=self.iou, verbose=False)
        r = results[0]

        detections: list[DetectedComponent] = []
        for box in r.boxes:
            cls_idx = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "unknown"
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            detections.append(DetectedComponent(
                **{"class": cls_name},
                class_idx=cls_idx,
                confidence=float(box.conf.item()),
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
            ))
        detections.sort(key=lambda d: -d.confidence)
        return detections

    def detect_batch(self, image_paths: list[str | Path]
                     ) -> dict[str, list[DetectedComponent]]:
        """Détection sur plusieurs images — plus efficace qu'un appel
        par image (un seul batch GPU)."""
        results = self.model.predict(
            [str(p) for p in image_paths], conf=self.conf, iou=self.iou,
            verbose=False)
        out: dict[str, list[DetectedComponent]] = {}
        for path, r in zip(image_paths, results):
            dets = []
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "unknown"
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                dets.append(DetectedComponent(
                    **{"class": cls_name}, class_idx=cls_idx,
                    confidence=float(box.conf.item()),
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1)))
            dets.sort(key=lambda d: -d.confidence)
            out[str(path)] = dets
        return out
