#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts/Train_Yolo.py — entraînement du détecteur de composants,
avec suivi Weights & Biases pensé pour un rapport / article.

    python Scripts/Train_Yolo.py --config Configs/hybrid.yaml
    python Scripts/Train_Yolo.py --config Configs/hybrid.yaml --no-wandb

IMPORTANT — sur l'intégration W&B :
`wandb.integration.ultralytics.add_wandb_callback` n'est testée par
wandb que jusqu'à ultralytics v8.0.238 (leur propre avertissement).
Sur des versions plus récentes (ex. 8.4.x), elle plante en important
des éléments internes d'ultralytics qui ont bougé (`RANK` introuvable).
On logge donc MANUELLEMENT via le système de callbacks natif
d'ultralytics (`model.add_callback`, stable et premier·e partie) plutôt
que via cette colle tierce qui doit resynchroniser sa compatibilité à
chaque refonte d'ultralytics — plus robuste, et on garde un contrôle
total sur ce qui est loggé.

Ce qui est loggé, et pourquoi :
  - config complète (hyperparamètres) + hash git : reproductibilité.
  - composition du dataset : contexte indispensable pour interpréter
    les métriques dans un rapport.
  - métriques de CHAQUE époque (pas juste la fin) : courbes complètes.
  - images clés générées par ultralytics (matrice de confusion,
    courbes PR, exemples de prédictions) : preuve visuelle qualitative.
  - tableau mAP50 par classe, trié : identifie les classes faibles
    d'un coup d'œil (les portes logiques OR/NOR/XOR/XNOR, visuellement
    proches, se sont confondues bien plus que le reste dans nos essais).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Src"))

from Common.Config import HybridConfig


def check_dataset_layout(dataset_dir: Path) -> None:
    problems = []
    for split in ("train", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.is_dir():
            problems.append(f"dossier d'images manquant : {img_dir}")
        elif not lbl_dir.is_dir():
            problems.append(
                f"dossier de labels manquant : {lbl_dir} "
                f"(vérifiez qu'il ne s'appelle pas encore 'labels_yolo')")
    if problems:
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        sys.exit(1)


def git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()
        h = out.stdout.strip()
        return f"{h}+dirty" if dirty else h
    except Exception:
        return "unknown"


def dataset_composition(dataset_dir: Path) -> dict:
    summary_path = dataset_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dataset/n_train": summary.get("n_train_ok"),
        "dataset/n_test": summary.get("n_test_ok"),
        "dataset/n_templates": len(summary.get("train_by_template", {})),
    }


def _attach_wandb_callbacks(model, wandb) -> None:
    """Callbacks manuels, en s'appuyant UNIQUEMENT sur l'API stable
    d'ultralytics (`trainer.metrics`, `trainer.epoch`) — jamais sur des
    internes susceptibles de bouger d'une version à l'autre."""

    def on_fit_epoch_end(trainer):
        metrics = {k: float(v) for k, v in trainer.metrics.items()
                  if isinstance(v, (int, float))}
        wandb.log({**metrics, "epoch": trainer.epoch})

    def on_train_end(trainer):
        # images clés déjà générées par ultralytics sur disque
        save_dir = Path(trainer.save_dir)
        for name in ("results.png", "confusion_matrix.png",
                     "confusion_matrix_normalized.png", "PR_curve.png",
                     "labels.jpg"):
            f = save_dir / name
            if f.is_file():
                wandb.log({f"plots/{f.stem}": wandb.Image(str(f))})
        for f in sorted(save_dir.glob("val_batch*_pred.jpg"))[:4]:
            wandb.log({f"predictions/{f.stem}": wandb.Image(str(f))})

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def _log_per_class_table(metrics, wandb) -> None:
    """Tableau mAP50 par classe, trié du pire au meilleur. Le nom exact
    de l'attribut par-classe peut varier d'une version d'ultralytics à
    l'autre : si `ap50` n'existe pas, ce bloc échoue sans casser le run."""
    try:
        names = metrics.names
        ap50_per_class = metrics.box.ap50
        rows = sorted(
            ((names[i], float(ap)) for i, ap in enumerate(ap50_per_class)),
            key=lambda r: r[1])
        table = wandb.Table(columns=["classe", "mAP50"], data=rows)
        wandb.log({"mAP50_par_classe": table})
    except Exception as e:
        print(f"(tableau par classe non généré : {e})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("Configs/hybrid.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = HybridConfig.from_yaml(args.config)
    dataset_dir = cfg.resolve_path(cfg.data.dataset_dir)
    circuit_yaml = cfg.resolve_path(cfg.data.circuit_yaml)

    if not circuit_yaml.is_file():
        sys.exit(f"circuit.yaml introuvable : {circuit_yaml}")
    check_dataset_layout(dataset_dir)

    from ultralytics import YOLO

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        import wandb

        run_config = {
            "model": args.weights or cfg.yolo.model,
            "epochs": args.epochs or cfg.yolo.epochs,
            "imgsz": cfg.yolo.imgsz,
            "batch": cfg.yolo.batch,
            "conf": cfg.yolo.conf,
            "iou": cfg.yolo.iou,
            "git_commit": git_commit_hash(),
            **dataset_composition(dataset_dir),
        }
        wandb.init(
            project=cfg.evaluation.wandb_project,
            dir=cfg.evaluation.wandb_dir,
            name=args.run_name,
            job_type="eval" if args.eval_only else "train",
            tags=["hybrid", "yolo", "detection"],
            config=run_config,
        )

    model = YOLO(args.weights or cfg.yolo.model)
    if use_wandb:
        _attach_wandb_callbacks(model, wandb)

    if args.eval_only:
        metrics = model.val(data=str(circuit_yaml), imgsz=cfg.yolo.imgsz)
    else:
        model.train(
            data=str(circuit_yaml),
            epochs=args.epochs or cfg.yolo.epochs,
            imgsz=cfg.yolo.imgsz,
            batch=cfg.yolo.batch,
            project=str(cfg.resolve_path(cfg.yolo.project)),
            name=cfg.yolo.name,
            plots=True,
        )
        metrics = model.val(
           data=str(circuit_yaml), imgsz=cfg.yolo.imgsz,
           project=str(cfg.resolve_path(cfg.yolo.project)),
           name=cfg.yolo.name, exist_ok=True)

    print(f"\nmAP50 : {metrics.box.map50:.4f} | mAP50-95 : {metrics.box.map:.4f}")

    if use_wandb:
        _log_per_class_table(metrics, wandb)
        wandb.finish()

    if not args.eval_only:
        print(f"Poids : {cfg.resolve_path(cfg.yolo.project)}/{cfg.yolo.name}/weights/best.pt")


if __name__ == "__main__":
    main()