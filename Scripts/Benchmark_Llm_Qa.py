#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts/Benchmark_Llm_Qa.py — measures a PROMPTED (not fine-tuned)
Qwen2.5-Instruct on circuit netlist QA, to decide whether fine-tuning
is actually needed before committing to it.

Two modes:
  --source ground_truth : netlist built from Data_Generation's ground
      truth (Graph.Builder.from_annotation) — measures the LLM's
      reasoning ceiling, isolated from any pipeline extraction error.
  --source pipeline : netlist built from the hybrid pipeline's actual
      predictions (Detector + Wires + OCR) — measures the realistic,
      end-to-end score, including upstream extraction errors.

Run ground_truth FIRST: if the LLM struggles even with a perfect
netlist, fine-tuning the LLM itself is the right lever. If it does
well on ground truth but poorly on pipeline output, the extraction
pipeline is the bottleneck, not the LLM — a very different fix.

    python Scripts/Benchmark_Llm_Qa.py --source ground_truth --max-circuits 30
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Src"))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from Common.Config import HybridConfig
from Graph.Builder import from_annotation, from_pipeline
from Graph.Qa_Generator import generate_qa_pairs
from Llm.Prompting import build_prompt, is_correct
from Llm.Qwen_Client import QwenClient, DEFAULT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("Configs/hybrid.yaml"))
    ap.add_argument("--source", choices=["ground_truth", "pipeline"],
                    default="ground_truth")
    ap.add_argument("--weights", type=Path, default=None,
                    help="YOLO weights, required if --source pipeline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-circuits", type=int, default=30,
                    help="start small: each circuit yields ~20 questions")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.source == "pipeline" and args.weights is None:
        sys.exit("--source pipeline requires --weights")

    cfg = HybridConfig.from_yaml(args.config)
    dataset_dir = cfg.resolve_path(cfg.data.dataset_dir)
    ann_dir = dataset_dir / "annotations" / args.split
    ann_files = sorted(ann_dir.glob("*.json"))[:args.max_circuits]
    if not ann_files:
        sys.exit(f"No annotation files found in {ann_dir}")

    pipeline = None
    if args.source == "pipeline":
        from Hybrid.Pipeline import HybridPipeline
        pipeline = HybridPipeline.from_config(cfg, weights=args.weights)

    client = QwenClient(model_name=args.model)
    rng = random.Random(args.seed)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.evaluation.wandb_project,
                  dir=cfg.evaluation.wandb_dir,
                  job_type="llm_qa_benchmark",
                  config={"source": args.source, "model": args.model,
                         "n_circuits": len(ann_files)})

    per_type_total: dict[str, int] = defaultdict(int)
    per_type_correct: dict[str, int] = defaultdict(int)
    rows: list[list] = []

    iterator = tqdm(ann_files, desc=f"Benchmark LLM ({args.source})") if tqdm else ann_files
    for ann_path in iterator:
        if args.source == "ground_truth":
            graph = from_annotation(ann_path, circuit_id=ann_path.stem)
        else:
            import json
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            image_path = dataset_dir / ann["image"]
            result = pipeline.run(image_path)
            graph = result.to_graph(circuit_id=ann_path.stem,
                                   source_image=str(image_path))

        netlist = graph.to_netlist()
        qa_pairs = generate_qa_pairs(graph, rng=rng)

        for qa in qa_pairs:
            messages = build_prompt(netlist, qa.question)
            raw_answer = client.ask(messages)
            correct = is_correct(raw_answer, qa.answer)

            per_type_total[qa.question_type] += 1
            if correct:
                per_type_correct[qa.question_type] += 1

            rows.append([ann_path.stem, qa.question_type, qa.question,
                        qa.answer, raw_answer, correct])

        if tqdm:
            done = sum(per_type_total.values())
            ok = sum(per_type_correct.values())
            iterator.set_postfix(acc=f"{ok/max(1,done):.3f}")

    print(f"\n{args.source} — {len(ann_files)} circuits, "
         f"{sum(per_type_total.values())} questions")
    for qtype in sorted(per_type_total):
        t, c = per_type_total[qtype], per_type_correct[qtype]
        print(f"  {qtype:14s}: {c}/{t}  ({c/t:.3f})")
    overall = sum(per_type_correct.values()) / max(1, sum(per_type_total.values()))
    print(f"  {'OVERALL':14s}: {overall:.4f}")

    if use_wandb:
        for qtype in per_type_total:
            t, c = per_type_total[qtype], per_type_correct[qtype]
            wandb.log({f"llm_qa/{qtype}_accuracy": c / t})
        wandb.log({"llm_qa/overall_accuracy": overall,
                  "llm_qa/n_questions": sum(per_type_total.values())})
        table = wandb.Table(
            columns=["circuit", "type", "question", "expected", "predicted", "correct"],
            data=rows)
        wandb.log({"llm_qa/details": table})
        wandb.finish()


if __name__ == "__main__":
    main()
