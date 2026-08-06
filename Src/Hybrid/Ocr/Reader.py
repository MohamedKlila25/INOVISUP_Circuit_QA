# -*- coding: utf-8 -*-
"""
Hybrid.Ocr.Reader — component text reading, via TrOCR (Hugging Face
transformers), running natively in pytorch251.

Why TrOCR over PaddleOCR: PaddleOCR/paddlex pulled a dependency chain
deep enough (paddlex -> modelscope -> modelscope_hub -> aistudio_sdk
-> ...) that it needed a separate venv to avoid destabilizing
numpy/torch, which in turn needed a subprocess/IPC layer to bridge the
two environments — and even then, the downloaded PP-OCRv6 model turned
out incompatible with the paddlepaddle version installed
(pd_op IR "strides" attribute error). TrOCR is PyTorch-native: it
reuses torch, already installed and GPU-configured in pytorch251, with
`transformers` as the only new dependency. No venv, no subprocess, no
framework-version matching.

Trade-off, stated plainly: TrOCR is RECOGNITION ONLY, unlike EasyOCR
and PaddleOCR which bundle text DETECTION too. Our crops need to be
pre-split into individual text lines before TrOCR can read them — see
Hybrid.Ocr.Line_Segmenter, validated separately on real generated
crops (correctly isolates "R1" / "1kΩ" as two clean lines, robust to
wire stubs crossing the crop, which a naive row-projection was NOT).

IMPORTANT — not executed end to end in the environment used to write
this file: torch itself could not be installed in the sandbox used to
validate the rest of this pipeline (disk constraints hit earlier with
easyocr). The line-segmentation step IS validated on real crops. The
TrOCR call itself follows the current, stable Hugging Face API
(TrOCRProcessor + VisionEncoderDecoderModel, microsoft/trocr-base-printed)
verified via a fresh documentation fetch — but has not been run here.
Validate on 5 real images before the full evaluation run, same
discipline as every previous OCR change in this project.

External interface is UNCHANGED (read_component, read_all) — nothing
else in the pipeline needs to change.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from Common.Config import HybridConfig
from Common.Schemas import DetectedComponent
from Hybrid.Ocr.Parsing import parse_ocr_lines, EXPECTED_UNIT
from Hybrid.Ocr.Line_Segmenter import find_text_lines, crop_lines

DEFAULT_MODEL = "microsoft/trocr-base-printed"


class ComponentTextReader:
    """Reads the reference and value printed near each detected
    component, via TrOCR applied to individually segmented text lines
    within a search region around each bbox."""

    def __init__(self, search_margin: int = 55, gpu: bool = True,
                model_name: str = DEFAULT_MODEL):
        self.search_margin = search_margin
        self._gpu = gpu
        self._model_name = model_name
        self._processor = None
        self._model = None
        self._device = None

    @classmethod
    def from_config(cls, cfg: HybridConfig) -> "ComponentTextReader":
        return cls(search_margin=cfg.ocr.search_margin, gpu=cfg.ocr.gpu)

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._device = "cuda" if (self._gpu and torch.cuda.is_available()) else "cpu"
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self._model_name).to(self._device)
        self._model.eval()

    def _read_line(self, line_img: np.ndarray) -> str:
        """Recognizes ONE pre-cropped text line. TrOCR expects an RGB
        PIL-like image; our crops are grayscale, so replicate to 3
        channels rather than reconverting through a color space that
        has no real color information to begin with."""
        import torch
        from PIL import Image

        if line_img.size == 0:
            return ""
        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self._processor(
            images=pil_img, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, max_length=32)
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _crop_region(self, img: np.ndarray,
                     comp: DetectedComponent) -> tuple[np.ndarray, int, int, int, int]:
        H, W = img.shape[:2]
        m = self.search_margin
        x0 = max(0, int(comp.bbox.x0) - m)
        y0 = max(0, int(comp.bbox.y0) - m)
        x1 = min(W, int(comp.bbox.x1) + m)
        y1 = min(H, int(comp.bbox.y1) + m)
        return img[y0:y1, x0:x1], x0, y0, x1, y1

    def read_component(self, img: np.ndarray,
                       comp: DetectedComponent) -> dict:
        self._ensure_model()

        crop, cx0, cy0, cx1, cy1 = self._crop_region(img, comp)
        if crop.size == 0:
            return {"id_text": None, "value_text": None, "raw": []}

        # Erase the component's OWN body before segmenting lines —
        # without this, the symbol's own ink (zigzag resistor, coil
        # loops...) gets picked up as spurious text-shaped blobs.
        crop_work = crop.copy()
        lx0 = max(0, int(comp.bbox.x0) - cx0)
        ly0 = max(0, int(comp.bbox.y0) - cy0)
        lx1 = min(crop.shape[1], int(comp.bbox.x1) - cx0)
        ly1 = min(crop.shape[0], int(comp.bbox.y1) - cy0)
        crop_work[ly0:ly1, lx0:lx1] = 255

        bands = find_text_lines(crop_work)
        line_imgs = crop_lines(crop_work, bands)

        lines = [self._read_line(li) for li in line_imgs]
        lines = [l for l in lines if l]

        id_text, value_text = parse_ocr_lines(
            lines, expected_unit=EXPECTED_UNIT.get(comp.cls))
        return {"id_text": id_text, "value_text": value_text, "raw": lines}

    def read_all(self, image_path: str | Path,
                components: list[DetectedComponent]) -> dict[str, dict]:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)

        out: dict[str, dict] = {}
        for comp in components:
            if comp.id is None:
                continue
            out[comp.id] = self.read_component(img, comp)
        return out