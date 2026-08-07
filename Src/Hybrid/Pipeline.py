# -*- coding: utf-8 -*-
"""
Hybrid.Pipeline — orchestrates the full hybrid pipeline on one image:

    Detector.detect()        -> raw component boxes, no id yet
    Assign_Ids.assign_ids()  -> ids assigned by reading-order position
    WireTracer.trace()       -> electrical nets between component ids
    ComponentTextReader      -> id_text / value_text per component

Each stage stays independently testable and independently replaceable
(e.g. swapping EasyOCR for a stronger model later only touches this
file's OCR step, nothing upstream or downstream).
"""
from __future__ import annotations

from pathlib import Path

from Common.Config import HybridConfig
from Common.Schemas import CircuitGraph, DetectedComponent
from Hybrid.Detection.Detector import Detector
from Hybrid.Detection.Assign_Ids import assign_ids
from Hybrid.Wires.Tracer import WireTracer, Net
from Hybrid.Ocr.Reader import ComponentTextReader
from Graph.Builder import from_pipeline


class PipelineResult:
    """Everything the pipeline produced for one image — kept as plain
    attributes rather than a single merged dict, so each stage's
    output stays inspectable on its own (useful when only one stage
    needs debugging)."""

    def __init__(self, components: list[DetectedComponent],
                nets: list[Net], ocr: dict[str, dict]):
        self.components = components
        self.nets = nets
        self.ocr = ocr

    def net_sets(self) -> list[set[str]]:
        """Nets as plain sets of component ids — the format expected
        by Evaluation.Metrics.net_pairwise_prf1."""
        return [n.component_ids() for n in self.nets]

    def values(self) -> dict[str, str | None]:
        """Component id -> OCR-extracted value, ready for
        Evaluation.Metrics.value_accuracy."""
        return {cid: r["value_text"] for cid, r in self.ocr.items()}

    def to_dict(self) -> dict:
        """JSON-serializable summary, for logging or manual inspection."""
        return {
            "components": [
                {"id": c.id, "class": c.cls, "confidence": c.confidence,
                 "bbox": c.bbox.to_list()}
                for c in self.components
            ],
            "nets": [sorted(n.component_ids()) for n in self.nets],
            "values": self.values(),
        }

    def to_graph(self, circuit_id: str,
                source_image: str | None = None,
                with_terminals: bool = True) -> CircuitGraph:
        """Assembles this pipeline's output into a CircuitGraph — the
        pivot representation consumed by the LLM stage.

        `with_terminals=True` (default) keeps WHICH PIN of each
        component sits on each node, which per-pin questions need
        ("what is R1's left pin wired to?", "which way is D1 facing?").
        Set it False for the plain component-level graph.
        """
        if with_terminals:
            from Graph.Builder import from_pipeline_with_terminals
            return from_pipeline_with_terminals(
                components=self.components,
                raw_nets=self.nets,
                values=self.values(),
                circuit_id=circuit_id,
                source_image=source_image,
            )
        return from_pipeline(
            components=self.components,
            nets=self.net_sets(),
            values=self.values(),
            circuit_id=circuit_id,
            source_image=source_image,
        )


class HybridPipeline:
    """Ties Detector, Assign_Ids, WireTracer and ComponentTextReader
    together. Construct once (loads the YOLO weights + lazily prepares
    the OCR reader), then call `.run(image_path)` per image."""

    def __init__(self, detector: Detector, wire_tracer: WireTracer,
                text_reader: ComponentTextReader):
        self.detector = detector
        self.wire_tracer = wire_tracer
        self.text_reader = text_reader

    @classmethod
    def from_config(cls, cfg: HybridConfig, weights: str | Path) -> "HybridPipeline":
        return cls(
            detector=Detector.from_config(cfg, weights=weights),
            wire_tracer=WireTracer.from_config(cfg),
            text_reader=ComponentTextReader.from_config(cfg),
        )

    def run(self, image_path: str | Path) -> PipelineResult:
        raw_components = self.detector.detect(image_path)
        components = assign_ids(raw_components)
        nets = self.wire_tracer.trace(image_path, components)
        ocr = self.text_reader.read_all(image_path, components)
        return PipelineResult(components=components, nets=nets, ocr=ocr)
