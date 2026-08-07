# -*- coding: utf-8 -*-
"""
Hybrid.Ocr.Line_Segmenter — splits a crop into individual text-line
images, via connected-component clustering.

Needed because TrOCR (unlike EasyOCR/PaddleOCR) is RECOGNITION ONLY:
it expects one pre-cropped text line, with no built-in detection
stage.

A naive row-by-row ink projection was tried first and measured to
fail: wire stubs connecting the component to the rest of the circuit
extend slightly past its bounding box into the search-margin crop, and
their ink rows get mixed into the projection, splitting or merging
text bands incorrectly. Connected-component clustering is robust to
this: a text glyph is a small, roughly square blob, while a wire
segment is a single long, thin, high-aspect-ratio blob — the two are
easy to tell apart by shape, so wire fragments are filtered out before
clustering the remaining glyphs into lines by vertical proximity.
"""
from __future__ import annotations

import cv2
import numpy as np


def find_text_lines(gray: np.ndarray, ink_threshold: int = 128,
                    max_wire_aspect: float = 6.0,
                    row_cluster_gap: float = 12.0,
                    pad: int = 3) -> list[tuple[int, int]]:
    """Finds (y0, y1) bands of a grayscale crop likely to contain text.

    1. Connected components on the ink.
    2. Drop components shaped like a wire fragment (very elongated —
       aspect ratio beyond `max_wire_aspect` in either direction).
    3. Cluster the remaining components by vertical centre proximity
       (within `row_cluster_gap` px) — each cluster is one text line.

    Returns bands top to bottom, matching the order our labels are
    always rendered in ("id" over "value").
    """
    ink = (gray < ink_threshold).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)

    glyphs = []   # (y_center, y0, y1)
    for i in range(1, n):   # skip background label 0
        x, y, w, h, area = stats[i]
        if area < 2:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > max_wire_aspect:
            continue   # wire fragment, not a text glyph
        glyphs.append((centroids[i][1], y, y + h))

    if not glyphs:
        return []

    glyphs.sort(key=lambda g: g[0])

    clusters: list[list[tuple[float, int, int]]] = [[glyphs[0]]]
    for g in glyphs[1:]:
        if g[0] - clusters[-1][-1][0] <= row_cluster_gap:
            clusters[-1].append(g)
        else:
            clusters.append([g])

    H = gray.shape[0]
    bands = []
    for cluster in clusters:
        y0 = min(c[1] for c in cluster)
        y1 = max(c[2] for c in cluster)
        bands.append((max(0, y0 - pad), min(H, y1 + pad)))
    return bands


def crop_lines(img: np.ndarray, bands: list[tuple[int, int]]) -> list[np.ndarray]:
    """Crops each (y0, y1) band out of the full-width image."""
    return [img[y0:y1, :] for (y0, y1) in bands]
