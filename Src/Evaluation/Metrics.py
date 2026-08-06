# -*- coding: utf-8 -*-
"""
Evaluation.Metrics — per-stage and end-to-end metrics for the hybrid
pipeline (Detection -> Wires -> OCR -> Graph).

Design principle: every stage gets its OWN metric, decoupled from the
others. This matters for debugging and for the report: a low
end-to-end score is meaningless on its own — you need to know whether
the failure comes from detection (wrong/missing components), wire
tracing (wrong connections), or OCR (wrong values), because each has a
completely different fix. Reporting only a single aggregate accuracy
would hide exactly the information a reader (or you, six months from
now) needs to improve the system.

Metrics implemented:
  - detection_prf1        : precision/recall/F1 on detected components,
                             matched to ground truth by IoU + class.
  - net_pairwise_prf1      : precision/recall/F1 on electrical
                             connectivity, computed pairwise over all
                             component-id pairs sharing a net.
  - value_accuracy         : exact-match rate of OCR-extracted values
                             against ground truth, keyed by component id.
  - evaluate_circuit        : combines all of the above for one circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from Common.Schemas import DetectedComponent


# ═══════════════════════════════════════════════════════════════════
#  Stage 1 — Detection
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PRF1:
    """Precision / recall / F1, plus the raw counts that produced them
    — always keep the raw counts alongside the ratios: a P=1.0 on 2
    samples and a P=1.0 on 200 samples are not the same claim, and a
    report should be able to tell them apart."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


def _safe_prf1(tp: int, fp: int, fn: int) -> PRF1:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return PRF1(precision, recall, f1, tp, fp, fn)


def match_components_by_iou(
    predicted: list[DetectedComponent],
    ground_truth: list[DetectedComponent],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching between predicted and ground-truth
    components. A match requires the SAME class and an IoU above the
    threshold. Greedy by descending IoU — standard practice for
    detection matching (same principle as COCO/YOLO evaluation).

    Returns (matched_pairs, unmatched_predicted_indices,
    unmatched_ground_truth_indices), all as index lists into the two
    input lists.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(ground_truth):
            if p.cls != g.cls:
                continue
            iou = p.bbox.iou(g.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        pairs.append((pi, gi))

    unmatched_pred = [i for i in range(len(predicted)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in matched_gt]
    return pairs, unmatched_pred, unmatched_gt


def detection_prf1(
    predicted: list[DetectedComponent],
    ground_truth: list[DetectedComponent],
    iou_threshold: float = 0.5,
) -> PRF1:
    """Precision/recall/F1 on component detection: a true positive is
    a predicted box matched (same class, IoU >= threshold) to a
    ground-truth box. Complements YOLO's own mAP (which is
    confidence-threshold-independent) with a single operating-point
    metric — the one that matters once the detector is actually
    deployed at a fixed confidence threshold."""
    pairs, unmatched_pred, unmatched_gt = match_components_by_iou(
        predicted, ground_truth, iou_threshold)
    return _safe_prf1(tp=len(pairs), fp=len(unmatched_pred),
                      fn=len(unmatched_gt))


# ═══════════════════════════════════════════════════════════════════
#  Stage 2 — Wire tracing / connectivity
# ═══════════════════════════════════════════════════════════════════

def _all_pairs(ids: list[str]) -> set[frozenset[str]]:
    return {frozenset((a, b)) for i, a in enumerate(ids)
            for b in ids[i + 1:] if a != b}


def net_pairwise_prf1(
    predicted_nets: list[set[str]],
    ground_truth_nets: list[set[str]],
) -> PRF1:
    """Precision/recall/F1 on electrical connectivity, computed
    PAIRWISE over all component-id pairs that share a net.

    Why pairwise rather than comparing net-to-net directly: nets don't
    have a natural identity to match on (unlike detections, which can
    be matched by IoU) — only their *membership* matters. Reducing
    each net to the set of component pairs it implies turns "did we
    reconstruct the same connectivity" into a standard set-comparison
    problem, robust to a net being split in two or wrongly merged
    (each such error only costs the specific pairs affected, not the
    whole net).
    """
    pred_pairs: set[frozenset[str]] = set()
    for net in predicted_nets:
        pred_pairs |= _all_pairs(sorted(net))

    gt_pairs: set[frozenset[str]] = set()
    for net in ground_truth_nets:
        gt_pairs |= _all_pairs(sorted(net))

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    return _safe_prf1(tp, fp, fn)


# ═══════════════════════════════════════════════════════════════════
#  Stage 3 — OCR (id + value)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValueAccuracy:
    exact_match_rate: float
    n_correct: int
    n_total: int
    n_missing: int   # ground truth expected a value, OCR returned None
    mismatches: list[tuple[str, str | None, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "exact_match_rate": round(self.exact_match_rate, 4),
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "mismatches": self.mismatches,
        }


def value_accuracy(
    predicted_values: dict[str, str | None],
    ground_truth_values: dict[str, str],
) -> ValueAccuracy:
    """Exact-match rate of OCR-extracted values, keyed by component id.
    Only components with a ground-truth value are scored (e.g. ground
    symbols have none, and are correctly excluded rather than counted
    as failures). `mismatches` lists (comp_id, predicted, expected) for
    every miss — this is what goes into an error-analysis table for
    the report, not just the aggregate rate.
    """
    correct = 0
    missing = 0
    mismatches: list[tuple[str, str | None, str]] = []
    total = 0

    for comp_id, expected in ground_truth_values.items():
        if expected is None:
            continue
        total += 1
        got = predicted_values.get(comp_id)
        if got is None:
            missing += 1
            mismatches.append((comp_id, got, expected))
        elif got.strip() == expected.strip():
            correct += 1
        else:
            mismatches.append((comp_id, got, expected))

    rate = correct / total if total else 0.0
    return ValueAccuracy(rate, correct, total, missing, mismatches)


# ═══════════════════════════════════════════════════════════════════
#  End-to-end
# ═══════════════════════════════════════════════════════════════════

def evaluate_circuit(
    predicted_components: list[DetectedComponent],
    ground_truth_components: list[DetectedComponent],
    predicted_nets: list[set[str]],
    ground_truth_nets: list[set[str]],
    predicted_values: dict[str, str | None],
    ground_truth_values: dict[str, str],
    iou_threshold: float = 0.5,
) -> dict:
    """Combined report for a single circuit — one dict, ready to log
    to W&B or aggregate across a test set. Keeping the three stages as
    separate sub-dicts (rather than flattening into one score) is
    deliberate: it is the decomposition that makes an error-analysis
    section of a report possible.
    """
    return {
        "detection": detection_prf1(
            predicted_components, ground_truth_components, iou_threshold
        ).as_dict(),
        "connectivity": net_pairwise_prf1(
            predicted_nets, ground_truth_nets
        ).as_dict(),
        "value_ocr": value_accuracy(
            predicted_values, ground_truth_values
        ).as_dict(),
    }
