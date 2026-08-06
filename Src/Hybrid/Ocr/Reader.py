# -*- coding: utf-8 -*-
"""
Hybrid.Ocr.Reader — component text reading, via EasyOCR.

Decision, measured: three engines were tried for this project.
  - EasyOCR (this file), after fixing the two dominant systematic
    character confusions (Ω<->Q/0, '1'<->'l'/'I') in Parsing.py:
    value exact-match 0.51 on 500 test circuits.
  - PaddleOCR: abandoned — its dependency chain (paddlex -> modelscope
    -> modelscope_hub -> ...) required an isolated venv + subprocess
    IPC bridge, and the downloaded PP-OCRv6 model was incompatible
    with the paddlepaddle version available (PIR "strides" attribute
    mismatch).
  - TrOCR (microsoft/trocr-base-printed): native PyTorch, no venv
    needed — but measured WORSE (0.42 exact-match on 50 circuits),
    with 91% of remaining errors in the diffuse "other/multi-char"
    bucket (even more scattered than EasyOCR's). Plausible cause:
    TrOCR is a generative model trained on natural English text
    (receipts, sentences); its language-model priors likely work
    AGAINST it on short, non-linguistic strings like "R1", "4.7kΩ",
    "TL081" — the opposite of what a classification-based OCR (like
    EasyOCR) needs to get right.

EasyOCR is therefore the one actually wired into the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from Common.Config import HybridConfig
from Common.Schemas import DetectedComponent
from Hybrid.Ocr.Parsing import parse_ocr_lines, EXPECTED_UNIT


class ComponentTextReader:
    """Reads the reference and value printed near each detected
    component, via EasyOCR applied to a search region around each
    bbox."""

    def __init__(self, search_margin: int = 55, gpu: bool = True,
                languages: list[str] | None = None):
        self.search_margin = search_margin
        self._reader = None
        self._gpu = gpu
        self._languages = languages or ["en"]

    @classmethod
    def from_config(cls, cfg: HybridConfig) -> "ComponentTextReader":
        return cls(search_margin=cfg.ocr.search_margin, gpu=cfg.ocr.gpu)

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def _crop_region(self, img: np.ndarray,
                     comp: DetectedComponent) -> tuple[np.ndarray, int, int]:
        H, W = img.shape[:2]
        m = self.search_margin
        x0 = max(0, int(comp.bbox.x0) - m)
        y0 = max(0, int(comp.bbox.y0) - m)
        x1 = min(W, int(comp.bbox.x1) + m)
        y1 = min(H, int(comp.bbox.y1) + m)
        return img[y0:y1, x0:x1], x0, y0

    def read_component(self, img: np.ndarray,
                       comp: DetectedComponent) -> dict:
        crop, _, _ = self._crop_region(img, comp)
        if crop.size == 0:
            return {"id_text": None, "value_text": None, "raw": []}

        results = self.reader.readtext(
            crop,
            allowlist="0123456789.,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzΩµ")
        results_sorted = sorted(results, key=lambda r: r[0][0][1])
        lines = [text for (_, text, conf) in results_sorted]

        id_text, value_text = parse_ocr_lines(
            lines, expected_unit=EXPECTED_UNIT.get(comp.cls))
        return {
            "id_text": id_text,
            "value_text": value_text,
            "raw": lines,
            "confidences": [float(c) for (_, _, c) in results_sorted],
        }

    def read_all(self, image_path: str | Path,
                components: list[DetectedComponent]) -> dict[str, dict]:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)

        out: dict[str, dict] = {}
        for comp in components:
            if comp.id is None:
                continue
            out[comp.id] = self.read_component(img, comp)
        return out