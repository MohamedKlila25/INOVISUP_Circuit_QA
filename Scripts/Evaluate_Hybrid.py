#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts/Evaluate_Hybrid.py — end-to-end evaluation of the hybrid
pipeline against the test split, with per-stage metrics.

    python Scripts/Evaluate_Hybrid.py --config Configs/hybrid.yaml \
        --weights Runs/hybrid/yolo11n_circuits/weights/best.pt

Reports detection / connectivity / OCR metrics SEPARATELY (see
Evaluation.Metrics for why), aggregated over the whole test set, and
logs them to W&B for the report alongside a per-circuit breakdown
table — the aggregate score alone would hide which circuits and which
stage are actually driving any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Src"))

from Common.Config import HybridConfig
from Common.Schemas import DetectedComponent
from Hybrid.Pipeline import HybridPipeline
from Evaluation.Metrics import (detection_prf1, net_pairwise_prf1,
                                value_accuracy, match_components_by_iou)


def build_id_map(predicted: list[DetectedComponent],
                 ground_truth: list[DetectedComponent],
                 iou_threshold: float = 0.5) -> dict[str, str]:
    """Maps each predicted component id to the ground-truth id of the
    SAME physical component (matched by IoU + class).

    Necessary because Assign_Ids numbers components purely by reading
    position (R1, R2, R3... per class), while several catalog
    templates use semantic ids instead (RL1, DZ1, Rf, RB1...). Without
    this remapping, connectivity and value scoring would compare by
    string identity and mostly fail on id disagreement rather than on
    actual connectivity or OCR errors — this was measured directly: it
    dropped OCR exact-match from a real ~90%+ down to 9.6% on a first
    20-image run, for a reason having nothing to do with OCR quality.
    """
    pairs, _, _ = match_components_by_iou(predicted, ground_truth, iou_threshold)
    return {predicted[pi].id: ground_truth[gi].id for pi, gi in pairs}


def remap_nets(nets: list[set[str]], id_map: dict[str, str]) -> list[set[str]]:
    """Translates predicted-pipeline ids inside each net to their
    matched ground-truth ids (unmatched predicted ids are left as-is:
    they correspond to false-positive detections, which should simply
    fail to line up with any ground-truth pair — no special-casing
    needed downstream)."""
    return [{id_map.get(i, i) for i in net} for net in nets]


def remap_values(values: dict[str, str | None],
                 id_map: dict[str, str]) -> dict[str, str | None]:
    """Translates predicted-pipeline ids to ground-truth ids for OCR
    value scoring. Predicted components with no ground-truth match are
    dropped (nothing to compare them against)."""
    out: dict[str, str | None] = {}
    for pred_id, val in values.items():
        gt_id = id_map.get(pred_id)
        if gt_id is not None:
            out[gt_id] = val
    return out


def load_ground_truth(annotation_path: Path
                      ) -> tuple[list[DetectedComponent], list[set[str]],
                                dict[str, str]]:
    """Reads one Data_Generation annotation JSON and extracts the three
    things needed for evaluation: ground-truth components (for
    detection scoring), nets as plain id sets (for connectivity
    scoring), and values keyed by component id (for OCR scoring)."""
    ann = json.loads(annotation_path.read_text(encoding="utf-8"))

    components = [
        DetectedComponent(id=c["id"], **{"class": c["class"]},
                          class_idx=0, confidence=1.0, bbox=c["bbox"])
        for c in ann["components"]
    ]
    nets = [{m["comp_id"] for m in n["terminals"]} for n in ann.get("nets", [])]
    values = {t["comp_id"]: t["value_text"] for t in ann.get("texts", [])}

    return components, nets, values


def categorize_mismatch(predicted: str | None, expected: str) -> str:
    """Buckets a value mismatch by likely cause, so error patterns show
    up systematically across the whole test set instead of being
    spotted one sampled image at a time. A single-character bucket
    like "'1' read instead of 'V'" is directly actionable (another
    allowlist/normalization fix); "other/multi-char" usually is not,
    and is worth inspecting individually instead."""
    if predicted is None:
        return "missing (OCR found nothing)"
    if len(predicted) == len(expected):
        diffs = [(a, b) for a, b in zip(predicted, expected) if a != b]
        if len(diffs) == 1:
            got, exp = diffs[0]
            return f"single-char: {got!r} read instead of {exp!r}"
    return "other/multi-char"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("Configs/hybrid.yaml"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None,
                    help="limit for a quick smoke test")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = HybridConfig.from_yaml(args.config)
    dataset_dir = cfg.resolve_path(cfg.data.dataset_dir)
    ann_dir = dataset_dir / "annotations" / args.split
    img_dir = dataset_dir / "images" / args.split

    ann_files = sorted(ann_dir.glob("*.json"))
    if args.max_images:
        ann_files = ann_files[:args.max_images]
    if not ann_files:
        sys.exit(f"No annotation files found in {ann_dir}")

    pipeline = HybridPipeline.from_config(cfg, weights=args.weights)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.evaluation.wandb_project,
                  dir=cfg.evaluation.wandb_dir,
                  job_type="hybrid_eval",
                  config={"weights": str(args.weights), "split": args.split,
                         "n_images": len(ann_files),
                         "iou_threshold": args.iou_threshold})

    per_circuit_reports: list[dict] = []
    per_circuit_rows: list[list] = []   # for the wandb table
    all_mismatches: list[tuple[str, str, str | None, str]] = []
    # (circuit_stem, comp_id, predicted, expected)

    for ann_path in ann_files:
        stem = ann_path.stem
        image_path = img_dir / f"{stem}.png"
        if not image_path.is_file():
            continue

        gt_components, gt_nets, gt_values = load_ground_truth(ann_path)

        result = pipeline.run(image_path)

        det = detection_prf1(result.components, gt_components,
                             iou_threshold=args.iou_threshold)

        # Remap predicted ids (position-assigned) to ground-truth ids
        # (sometimes semantic, e.g. RL1/DZ1/Rf) BEFORE scoring
        # connectivity and OCR — see build_id_map for why this matters.
        id_map = build_id_map(result.components, gt_components,
                             iou_threshold=args.iou_threshold)
        remapped_nets = remap_nets(result.net_sets(), id_map)
        remapped_values = remap_values(result.values(), id_map)

        conn = net_pairwise_prf1(remapped_nets, gt_nets)
        val = value_accuracy(remapped_values, gt_values)

        for comp_id, predicted, expected in val.mismatches:
            all_mismatches.append((stem, comp_id, predicted, expected))

        report = {"detection": det.as_dict(), "connectivity": conn.as_dict(),
                 "value_ocr": val.as_dict()}
        per_circuit_reports.append(report)
        per_circuit_rows.append([
            stem, det.f1, conn.f1, val.exact_match_rate,
        ])

    n = len(per_circuit_reports)
    mean_det_f1 = sum(r["detection"]["f1"] for r in per_circuit_reports) / n
    mean_conn_f1 = sum(r["connectivity"]["f1"] for r in per_circuit_reports) / n
    mean_val_acc = sum(r["value_ocr"]["exact_match_rate"]
                       for r in per_circuit_reports) / n

    print(f"\nEvaluated {n} circuits from split='{args.split}'")
    print(f"  Detection   F1 (mean over circuits): {mean_det_f1:.4f}")
    print(f"  Connectivity F1 (mean over circuits): {mean_conn_f1:.4f}")
    print(f"  Value OCR exact-match (mean):         {mean_val_acc:.4f}")

    # Categorized error breakdown: which failure patterns actually
    # dominate across the WHOLE test set, not just a handful of
    # manually sampled images.
    from collections import Counter
    categories = Counter(
        categorize_mismatch(pred, exp) for _, _, pred, exp in all_mismatches)
    print(f"\n  {len(all_mismatches)} value mismatches, by category:")
    for cat, count in categories.most_common():
        print(f"    {count:4d}  {cat}")

    if use_wandb:
        wandb.log({
            "eval/detection_f1_mean": mean_det_f1,
            "eval/connectivity_f1_mean": mean_conn_f1,
            "eval/value_accuracy_mean": mean_val_acc,
            "eval/n_circuits": n,
            "eval/n_mismatches": len(all_mismatches),
        })
        table = wandb.Table(
            columns=["circuit", "detection_f1", "connectivity_f1",
                    "value_accuracy"],
            data=per_circuit_rows,
        )
        wandb.log({"eval/per_circuit": table})

        mismatch_table = wandb.Table(
            columns=["circuit", "comp_id", "predicted", "expected", "category"],
            data=[[stem, cid, pred, exp, categorize_mismatch(pred, exp)]
                  for stem, cid, pred, exp in all_mismatches],
        )
        wandb.log({"eval/ocr_mismatches": mismatch_table})

        cat_table = wandb.Table(
            columns=["category", "count"],
            data=[[cat, count] for cat, count in categories.most_common()],
        )
        wandb.log({"eval/ocr_mismatch_categories": cat_table})

        wandb.finish()


if __name__ == "__main__":
    main()