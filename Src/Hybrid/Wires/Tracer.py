# -*- coding: utf-8 -*-
"""
Hybrid.Wires.Tracer — reconstruction des connexions électriques.

Principe (repris de l'algorithme validé côté génération du dataset,
Data_Generation.Renderer_Annotated, où les nets de vérité terrain sont
extraits de la même façon — mais LÀ-BAS on connaissait les terminaux
exacts (anchors schemdraw). ICI, à l'inférence, on ne connaît que les
bboxes détectées par YOLO : il faut d'abord DÉDUIRE les points de
connexion depuis la géométrie, avant de les regrouper en nets.

Étapes :
  1. Binariser l'image (encre = pixels sombres).
  2. Effacer les bboxes des composants détectés (+ marge) -> masque de
     fils. La masse ("ground") n'est PAS effacée : c'est un nœud
     conducteur, comme dans le générateur.
  3. Étiqueter les composantes connexes du masque de fils (chaque blob
     = un nœud électrique potentiel).
  4. Pour chaque composant, chercher les points où de l'encre touche
     le PÉRIMÈTRE de sa bbox (+ petite marge) : ce sont les terminaux
     déduits. Un composant à 2 broches en a normalement 2 ; un AOP,
     transistor ou porte peut en avoir plus (base/collecteur/émetteur,
     in1/in2/out) mais ceux-ci restent indistingués ici par nature
     (aucun ordre connu sans régression de terminaux dédiée).
  5. Regrouper les terminaux par blob (union-find) -> nets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from Common.Config import HybridConfig
from Common.Schemas import DetectedComponent


class Net:
    """Nœud électrique : ensemble de (comp_id, terminal_index, x, y)."""

    __slots__ = ("net_id", "members")

    def __init__(self, net_id: int, members: list[dict]):
        self.net_id = net_id
        self.members = members  # [{"comp_id": "R1", "x":.., "y":..}, ...]

    def component_ids(self) -> set[str]:
        return {m["comp_id"] for m in self.members}

    def __repr__(self) -> str:
        ids = [m["comp_id"] for m in self.members]
        return f"Net({self.net_id}: {ids})"


class WireTracer:
    """Reconstruit les nets électriques à partir d'une image et des
    composants détectés (bboxes YOLO)."""

    def __init__(self, bbox_margin: int = 6, touch_dist: int = 10,
                min_net_pixels: int = 30):
        self.bbox_margin = bbox_margin
        self.touch_dist = touch_dist
        self.min_net_pixels = min_net_pixels

    @classmethod
    def from_config(cls, cfg: HybridConfig) -> "WireTracer":
        wt = cfg.wire_tracer
        return cls(bbox_margin=wt.bbox_margin, touch_dist=wt.touch_dist,
                  min_net_pixels=wt.min_net_pixels)

    def build_wire_mask(self, ink: np.ndarray,
                        components: list[DetectedComponent]) -> np.ndarray:
        """Masque de fils = encre moins les bboxes des composants
        (marge configurable), sauf la masse (nœud conducteur)."""
        H, W = ink.shape
        mask = ink.copy()
        for comp in components:
            if comp.cls == "ground":
                continue
            b = comp.bbox
            x0 = max(0, int(b.x0) - self.bbox_margin)
            y0 = max(0, int(b.y0) - self.bbox_margin)
            x1 = min(W, int(b.x1) + self.bbox_margin)
            y1 = min(H, int(b.y1) + self.bbox_margin)
            mask[y0:y1, x0:x1] = False
        return mask

    def find_terminal_points(self, comp: DetectedComponent,
                             ink: np.ndarray, img_shape: tuple[int, int]
                             ) -> list[tuple[float, float]]:
        """Points où l'encre touche le périmètre de la bbox (+ marge) —
        ce sont les terminaux déduits, sans connaître leur ordre/rôle."""
        H, W = img_shape
        b = comp.bbox
        m = self.bbox_margin
        x0 = max(0, int(b.x0) - m); y0 = max(0, int(b.y0) - m)
        x1 = min(W, int(b.x1) + m); y1 = min(H, int(b.y1) + m)

        band = 3
        touches: list[tuple[int, int]] = []
        for y in range(max(0, y0 - band), min(H, y1 + band)):
            on_top = y0 - band <= y < y0
            on_bot = y1 <= y < y1 + band
            if on_top or on_bot:
                row = np.where(ink[y, x0:x1])[0]
                touches += [(x0 + x, y) for x in row]
        for x in range(max(0, x0 - band), min(W, x1 + band)):
            on_left = x0 - band <= x < x0
            on_right = x1 <= x < x1 + band
            if on_left or on_right:
                col = np.where(ink[y0:y1, x])[0]
                touches += [(x, y0 + y) for y in col]

        if not touches:
            return []

        pts = np.array(touches, dtype=float)
        clusters: list[list[np.ndarray]] = []
        for p in pts:
            placed = False
            for cl in clusters:
                if np.linalg.norm(cl[-1] - p) < self.touch_dist:
                    cl.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])
        return [tuple(np.mean(cl, axis=0)) for cl in clusters]

    def trace(self, image_path: str | Path,
             components: list[DetectedComponent]) -> list[Net]:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)
        H, W = img.shape
        ink = img < 128

        wire_mask = self.build_wire_mask(ink, components)

        n_labels, labels_img = cv2.connectedComponents(
            wire_mask.astype(np.uint8), connectivity=8)

        sizes = np.bincount(labels_img.ravel())
        valid_blobs = {i for i in range(1, n_labels)
                      if sizes[i] >= self.min_net_pixels}

        parent: dict[int, int] = {}

        def find(a: int) -> int:
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        net_members: dict[int, list[dict]] = {}
        for comp in components:
            if comp.id is None:
                continue
            pts = self.find_terminal_points(comp, ink, (H, W))
            for (x, y) in pts:
                xi, yi = int(round(x)), int(round(y))
                r = self.touch_dist
                y0s, y1s = max(0, yi - r), min(H, yi + r + 1)
                x0s, x1s = max(0, xi - r), min(W, xi + r + 1)
                sub = labels_img[y0s:y1s, x0s:x1s]
                found = sorted({int(v) for v in np.unique(sub)
                               if v in valid_blobs})
                if not found:
                    continue
                first = found[0]
                for b in found[1:]:
                    union(first, b)
                net_members.setdefault(find(first), []).append(
                    {"comp_id": comp.id, "x": round(x, 1), "y": round(y, 1)})

        nets = []
        for i, (blob, members) in enumerate(sorted(net_members.items())):
            if len({m["comp_id"] for m in members}) >= 2:
                nets.append(Net(net_id=i, members=members))
        return nets
